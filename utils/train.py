"""
Training, evaluation, checkpoint loading, and seed utilities.
"""
import os
import time
import copy
import random
import inspect
import json
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader as PyGDataLoader

from .metrics import *

# ============================================================
# Compact optical-response GINE training utilities
# ============================================================

def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    os.environ["PYTHONHASHSEED"] = str(int(seed))

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def flat2d_tensor(x):
    x = x.to(torch.float32)

    if x.dim() == 1:
        return x.view(1, -1)

    return x.view(x.shape[0], -1)


def get_axis_from_graph_batch(batch):
    axis = getattr(batch, "axis_grid", None)

    if axis is None:
        axis = getattr(batch, "spectral_axis", None)

    if axis is None:
        return None

    return flat2d_tensor(axis)


def maybe_shared_axis_array(axis_2d):
    if axis_2d is None:
        return None

    if torch.is_tensor(axis_2d):
        if axis_2d.ndim != 2:
            return axis_2d.detach().cpu().numpy()

        if axis_2d.shape[0] == 1:
            return axis_2d[0].detach().cpu().numpy()

        first = axis_2d[0:1].expand_as(axis_2d)
        if torch.allclose(axis_2d, first):
            return axis_2d[0].detach().cpu().numpy()

        return axis_2d.detach().cpu().numpy()

    return axis_2d


def autocast_context_for_device(device, use_amp):
    if not use_amp or device.type != "cuda":
        return nullcontext()

    return torch.amp.autocast(device_type="cuda", enabled=True)


def make_grad_scaler_for_device(device, use_amp):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=use_amp)

    return None


def build_model_from_signature(
    model_class,
    node_in_dim,
    edge_in_dim,
    out_dim,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.10,
    graph_attr_dim=0,
    use_graph_attr=False,
):
    init_sig = inspect.signature(model_class.__init__)
    pset = set(init_sig.parameters.keys())

    name_map = {
        "node_in_dim": node_in_dim,
        "in_dim": node_in_dim,
        "num_node_features": node_in_dim,
        "x_dim": node_in_dim,

        "edge_in_dim": edge_in_dim,
        "edge_dim": edge_in_dim,
        "num_edge_features": edge_in_dim,

        "out_dim": out_dim,
        "output_dim": out_dim,
        "num_outputs": out_dim,
        "spectrum_dim": out_dim,
        "target_dim": out_dim,

        "hidden_dim": hidden_dim,
        "hidden_channels": hidden_dim,
        "channels": hidden_dim,
        "emb_dim": hidden_dim,
        "embed_dim": hidden_dim,

        "latent_dim": latent_dim,

        "num_layers": num_layers,
        "n_layers": num_layers,
        "gnn_layers": num_layers,
        "num_convs": num_layers,

        "dropout": dropout,
        "dropout_rate": dropout,

        "graph_attr_dim": graph_attr_dim,
        "use_graph_attr": use_graph_attr,
        "attach_graph_attr": use_graph_attr,
    }

    kwargs = {k: v for k, v in name_map.items() if k in pset}

    try:
        return model_class(**kwargs)
    except Exception:
        pass

    fallback_args = [
        (node_in_dim, edge_in_dim, out_dim),
        (node_in_dim, edge_in_dim, hidden_dim, out_dim),
        (node_in_dim, edge_in_dim, hidden_dim, latent_dim, out_dim),
    ]

    for args in fallback_args:
        try:
            return model_class(*args)
        except Exception:
            pass

    raise RuntimeError("Could not initialize the optical model from the provided model_class.")


def extract_prediction_from_output(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ["pred", "prediction", "y_hat", "out", "output", "spectrum", "logits"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        for value in output.values():
            if torch.is_tensor(value):
                return value

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

            if isinstance(item, dict):
                pred = extract_prediction_from_output(item)

                if torch.is_tensor(pred):
                    return pred

    raise RuntimeError("Could not extract a prediction tensor from model output.")


def call_model_safely(model, batch):
    try:
        pnames = list(inspect.signature(model.forward).parameters.keys())
    except Exception:
        pnames = []

    if len(pnames) == 1 and pnames[0] in ["data", "batch", "graph", "g"]:
        return model(batch)

    available = {
        "x": batch.x,
        "edge_index": batch.edge_index,
        "edge_attr": getattr(batch, "edge_attr", None),
        "batch": batch.batch,
        "graph_attr": getattr(batch, "graph_attr", None),
        "spectral_axis": getattr(batch, "spectral_axis", None),
        "axis_grid": getattr(batch, "axis_grid", None),
    }

    kwargs = {k: v for k, v in available.items() if k in pnames and v is not None}

    trials = [
        lambda: model(**kwargs) if len(kwargs) > 0 else (_ for _ in ()).throw(RuntimeError()),
        lambda: model(batch),
        lambda: model(batch.x, batch.edge_index, batch.edge_attr, batch.batch),
        lambda: model(batch.x, batch.edge_index, batch.batch),
    ]

    for trial in trials:
        try:
            return trial()
        except Exception:
            pass

    raise RuntimeError("Calling model.forward failed.")


def load_partial_checkpoint(model, checkpoint_path, device="cpu", verbose=True):
    """
    Load matching weights only. Useful for SSL-init checkpoints with partially
    different heads.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict", None)
            or ckpt.get("state_dict", None)
            or ckpt.get("model", None)
            or ckpt
        )
    else:
        state = ckpt

    model_state = model.state_dict()
    matched = {}
    skipped = []

    for k, v in state.items():
        kk = k.replace("module.", "")

        if kk in model_state and tuple(model_state[kk].shape) == tuple(v.shape):
            matched[kk] = v
        else:
            skipped.append(k)

    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)

    if verbose:
        print(f"Loaded partial checkpoint: {checkpoint_path}")
        print(f"Matched tensors          : {len(matched)}")
        print(f"Skipped tensors          : {len(skipped)}")

    return model


def train_optical_gine_single_target(
    target_key,
    variant_name,
    train_ds,
    val_ds,
    test_ds,
    model_class,
    output_dir,
    device=None,
    seeds=(42, 123, 2025),
    batch_size=16,
    num_epochs=100,
    early_stop_patience=15,
    lr=5e-4,
    weight_decay=1e-4,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.10,
    peak_weight=2.0,
    smoothness_weight=0.03,
    sched_factor=0.5,
    sched_patience=4,
    min_lr=1e-6,
    clip_grad_norm=5.0,
    num_workers=0,
    pin_memory=None,
    use_amp=None,
    pretrained_ckpt_path=None,
    run_name=None,
):
    """
    Train one optical-response model for one target channel.

    This function is intentionally explicit:
    - pass epsR_train_ds / epsI_train_ds directly
    - do not rely on old global names such as opt_train_graph_ds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    if use_amp is None:
        use_amp = bool(torch.cuda.is_available())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"{target_key}_{variant_name}".replace("/", "_")

    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    sample0 = train_ds[0]

    node_in_dim = int(sample0.x.shape[-1])
    edge_in_dim = int(sample0.edge_attr.shape[-1]) if getattr(sample0, "edge_attr", None) is not None else 0
    out_dim = int(sample0.y.view(-1).shape[0])

    has_graph_attr = getattr(sample0, "graph_attr", None) is not None
    graph_attr_dim = int(sample0.graph_attr.view(-1).shape[0]) if has_graph_attr else 0

    sample_axis = get_axis_from_graph_batch(sample0)
    shared_spectral_axis = maybe_shared_axis_array(sample_axis) if sample_axis is not None else None

    spec_mean, spec_std, spec_stats_df = compute_spectrum_stats(train_ds, device="cpu")

    if spec_mean.numel() != out_dim or spec_std.numel() != out_dim:
        raise RuntimeError(
            f"spec_mean/spec_std do not match out_dim: {spec_mean.numel()}, {spec_std.numel()} vs {out_dim}"
        )

    spec_stats_path = run_output_dir / f"spectrum_train_stats_{target_key}_{variant_name}.csv"

    if spec_stats_df is not None:
        spec_stats_df.to_csv(spec_stats_path, index=False)

    spec_mean_dev = spec_mean.to(device)
    spec_std_dev = spec_std.to(device)

    print("=" * 100)
    print(f"TRAIN OPTICAL GINE | target={target_key} | variant={variant_name}")
    print("=" * 100)
    print("device              :", device)
    print("run_output_dir      :", run_output_dir)
    print("train/val/test      :", len(train_ds), len(val_ds), len(test_ds))
    print("node_in_dim         :", node_in_dim)
    print("edge_in_dim         :", edge_in_dim)
    print("out_dim             :", out_dim)
    print("graph_attr_dim      :", graph_attr_dim if has_graph_attr else "<none>")
    print("seeds               :", list(seeds))
    print("batch_size          :", batch_size)
    print("num_epochs          :", num_epochs)
    print("AMP                 :", use_amp)

    if sample_axis is not None:
        axis_np = sample_axis.view(-1).detach().cpu().numpy()
        print("axis_min            :", float(axis_np.min()))
        print("axis_max            :", float(axis_np.max()))
        if len(axis_np) > 1:
            print("axis_step           :", float(axis_np[1] - axis_np[0]))

    print("Mean(mean_train)    :", float(spec_mean.mean()))
    print("Mean(std_train)     :", float(spec_std.mean()))
    print("allow_negative      :", infer_allow_negative_from_target_key(target_key))
    print("negative_weight     :", get_default_negative_weight(target_key, fallback=0.02))

    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = PyGDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = PyGDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    def normalize_local(y):
        return (y - spec_mean_dev.view(1, -1)) / (spec_std_dev.view(1, -1) + EPS_SPECTRUM)

    def compute_loss_local(pred_denorm, true_denorm, mask):
        pred_denorm = flat2d_tensor(pred_denorm)
        true_denorm = flat2d_tensor(true_denorm)
        mask = flat2d_tensor(mask)

        pred_norm = normalize_local(pred_denorm)
        true_norm = normalize_local(true_denorm)

        return spectrum_loss(
            pred_norm=pred_norm,
            target_norm=true_norm,
            mask=mask,
            pred_denorm=pred_denorm,
            target_denorm=true_denorm,
            target_key=target_key,
            peak_weight=peak_weight,
            peak_mode="denorm",
            smoothness_weight=smoothness_weight,
            smoothness_domain="norm",
            negative_weight=None,
            allow_negative=None,
        )

    def compute_metrics_local(pred_denorm, true_denorm, mask, axis_grid=None):
        return compute_spectrum_metrics(
            pred_denorm=flat2d_tensor(pred_denorm),
            true_denorm=flat2d_tensor(true_denorm),
            mask=flat2d_tensor(mask),
            axis_grid=axis_grid,
        )

    with torch.no_grad():
        y0 = flat2d_tensor(sample0.y).to(device)
        m0 = flat2d_tensor(sample0.y_mask).to(device)
        a0 = get_axis_from_graph_batch(sample0)
        if a0 is not None:
            a0 = a0.to(device)
        smoke_metrics = compute_metrics_local(y0, y0, m0, axis_grid=a0)

    print("\nSmoke-test metrics on perfect prediction:")
    for k, v in smoke_metrics.items():
        print(f"{k}: {v:.6f}")

    def train_one_epoch(model, loader, optimizer, scaler=None):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            target = flat2d_tensor(batch.y)
            mask = flat2d_tensor(batch.y_mask)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context_for_device(device, use_amp):
                pred = flat2d_tensor(extract_prediction_from_output(call_model_safely(model, batch)))

                if pred.shape != target.shape:
                    raise RuntimeError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

                loss, _ = compute_loss_local(pred, target, mask)

            if scaler is not None and use_amp and device.type == "cuda":
                scaler.scale(loss).backward()

                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate_epoch(model, loader, return_predictions=False):
        model.eval()

        total_loss = 0.0
        n_batches = 0

        preds_all = []
        tgts_all = []
        masks_all = []
        axis_all = []

        base_idx_all = []
        sample_idx_all = []
        view_idx_all = []

        for batch in loader:
            batch = batch.to(device)

            target = flat2d_tensor(batch.y)
            mask = flat2d_tensor(batch.y_mask)

            axis_grid = get_axis_from_graph_batch(batch)
            if axis_grid is not None:
                axis_grid = axis_grid.to(device)

            pred = flat2d_tensor(extract_prediction_from_output(call_model_safely(model, batch)))

            if pred.shape != target.shape:
                raise RuntimeError(f"[EVAL] pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

            loss, _ = compute_loss_local(pred, target, mask)
            total_loss += float(loss.detach().item())
            n_batches += 1

            preds_all.append(pred.detach().cpu())
            tgts_all.append(target.detach().cpu())
            masks_all.append(mask.detach().cpu())

            if axis_grid is not None:
                axis_all.append(flat2d_tensor(axis_grid).detach().cpu())

            if hasattr(batch, "base_idx"):
                base_idx_all.append(batch.base_idx.detach().cpu())
            if hasattr(batch, "sample_idx"):
                sample_idx_all.append(batch.sample_idx.detach().cpu())
            if hasattr(batch, "view_idx"):
                view_idx_all.append(batch.view_idx.detach().cpu())

        preds_all = torch.cat(preds_all, dim=0)
        tgts_all = torch.cat(tgts_all, dim=0)
        masks_all = torch.cat(masks_all, dim=0)

        axis_cat = torch.cat(axis_all, dim=0) if len(axis_all) > 0 else None

        metrics = compute_metrics_local(preds_all, tgts_all, masks_all, axis_grid=axis_cat)
        metrics["loss"] = total_loss / max(n_batches, 1)

        if return_predictions:
            pack = {
                "pred": preds_all.numpy(),
                "true": tgts_all.numpy(),
                "mask": masks_all.numpy(),
                "spectral_axis": maybe_shared_axis_array(axis_cat) if axis_cat is not None else shared_spectral_axis,
                "axis_grid": maybe_shared_axis_array(axis_cat) if axis_cat is not None else shared_spectral_axis,
            }

            if len(base_idx_all) > 0:
                pack["base_idx"] = torch.cat(base_idx_all, dim=0).view(-1).numpy()
            if len(sample_idx_all) > 0:
                pack["sample_idx"] = torch.cat(sample_idx_all, dim=0).view(-1).numpy()
            if len(view_idx_all) > 0:
                pack["view_idx"] = torch.cat(view_idx_all, dim=0).view(-1).numpy()

            return metrics, pack

        return metrics

    summary_rows = []

    for seed in seeds:
        print("\n" + "=" * 118)
        print(f"RUN | target={target_key} | variant={variant_name} | seed={seed}")
        print("=" * 118)

        set_all_seeds(seed)

        model = build_model_from_signature(
            model_class=model_class,
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            graph_attr_dim=graph_attr_dim,
            use_graph_attr=has_graph_attr,
        ).to(device)

        if pretrained_ckpt_path is not None:
            model = load_partial_checkpoint(
                model=model,
                checkpoint_path=pretrained_ckpt_path,
                device=device,
                verbose=True,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_factor,
            patience=sched_patience,
            min_lr=min_lr,
        )

        scaler = make_grad_scaler_for_device(device, use_amp)

        best_state = None
        best_epoch = -1
        best_val = float("inf")
        patience_counter = 0
        history = []

        ckpt_path = run_output_dir / f"best_optical_{target_key}_{variant_name}_seed{seed}.pt"
        val_pred_path = run_output_dir / f"val_predictions_{target_key}_{variant_name}_seed{seed}.npz"
        test_pred_path = run_output_dir / f"test_predictions_{target_key}_{variant_name}_seed{seed}.npz"
        hist_path = run_output_dir / f"history_optical_{target_key}_{variant_name}_seed{seed}.csv"

        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_info = train_one_epoch(model, train_loader, optimizer, scaler)
            val_metrics = evaluate_epoch(model, val_loader, return_predictions=False)

            scheduler.step(val_metrics["MAE_mean"])

            row = {
                "target_key": target_key,
                "variant": variant_name,
                "seed": seed,
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **train_info,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)

            if val_metrics["MAE_mean"] < best_val:
                best_val = val_metrics["MAE_mean"]
                best_epoch = epoch
                patience_counter = 0

                best_state = {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "seed": seed,
                    "target_key": target_key,
                    "variant": variant_name,
                    "val_metrics": val_metrics,
                    "config": {
                        "target_key": target_key,
                        "variant": variant_name,
                        "node_in_dim": node_in_dim,
                        "edge_in_dim": edge_in_dim,
                        "out_dim": out_dim,
                        "graph_attr_dim": graph_attr_dim,
                        "hidden_dim": hidden_dim,
                        "latent_dim": latent_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "peak_weight": peak_weight,
                        "smoothness_weight": smoothness_weight,
                        "clip_grad_norm": clip_grad_norm,
                        "pretrained_ckpt_path": str(pretrained_ckpt_path) if pretrained_ckpt_path is not None else None,
                    },
                }

                torch.save(best_state, ckpt_path)

            else:
                patience_counter += 1

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"[target={target_key} | variant={variant_name} | seed={seed}] "
                    f"Epoch {epoch:03d} | train_loss={train_info['train_loss']:.5f} | "
                    f"val_MAE={val_metrics['MAE_mean']:.5f} | "
                    f"val_RMSE={val_metrics['RMSE_mean']:.5f} | "
                    f"val_Cos={val_metrics['CosineSim_mean']:.5f} | "
                    f"best_epoch={best_epoch:03d}"
                )

            if patience_counter >= early_stop_patience:
                print(f"[target={target_key} | variant={variant_name} | seed={seed}] Early stopping at epoch {epoch}.")
                break

        if best_state is None:
            raise RuntimeError("best_state is None. Training failed.")

        model.load_state_dict(best_state["model_state_dict"])

        val_best_metrics, val_pack = evaluate_epoch(model, val_loader, return_predictions=True)
        test_best_metrics, test_pack = evaluate_epoch(model, test_loader, return_predictions=True)

        np.savez_compressed(val_pred_path, **val_pack)
        np.savez_compressed(test_pred_path, **test_pack)
        pd.DataFrame(history).to_csv(hist_path, index=False)

        elapsed = time.time() - t0

        row = {
            "target_key": target_key,
            "variant": variant_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,

            "val_loss": val_best_metrics["loss"],
            "val_MAE_mean": val_best_metrics["MAE_mean"],
            "val_RMSE_mean": val_best_metrics["RMSE_mean"],
            "val_CosineSim_mean": val_best_metrics["CosineSim_mean"],
            "val_PeakMAE_mean": val_best_metrics["PeakMAE_mean"],
            "val_PeakBinShift_mean": val_best_metrics["PeakBinShift_mean"],
            "val_PeakAxisError_mean": val_best_metrics.get("PeakAxisError_mean", np.nan),
            "val_PeakIndexError_mean": val_best_metrics.get("PeakIndexError_mean", np.nan),
            "val_NegativeBinRate_mean": val_best_metrics["NegativeBinRate_mean"],

            "test_loss": test_best_metrics["loss"],
            "test_MAE_mean": test_best_metrics["MAE_mean"],
            "test_RMSE_mean": test_best_metrics["RMSE_mean"],
            "test_CosineSim_mean": test_best_metrics["CosineSim_mean"],
            "test_PeakMAE_mean": test_best_metrics["PeakMAE_mean"],
            "test_PeakBinShift_mean": test_best_metrics["PeakBinShift_mean"],
            "test_PeakAxisError_mean": test_best_metrics.get("PeakAxisError_mean", np.nan),
            "test_PeakIndexError_mean": test_best_metrics.get("PeakIndexError_mean", np.nan),
            "test_NegativeBinRate_mean": test_best_metrics["NegativeBinRate_mean"],

            "ckpt_path": str(ckpt_path),
            "val_pred_path": str(val_pred_path),
            "test_pred_path": str(test_pred_path),
            "history_path": str(hist_path),
        }

        summary_rows.append(row)

        print("-" * 118)
        print(f"[target={target_key} | variant={variant_name} | seed={seed}] BEST EPOCH = {best_epoch}")
        print("VAL :", {k: round(v, 6) if isinstance(v, float) else v for k, v in val_best_metrics.items()})
        print("TEST:", {k: round(v, 6) if isinstance(v, float) else v for k, v in test_best_metrics.items()})

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = run_output_dir / f"summary_optical_{target_key}_{variant_name}.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg_cols = [
        "val_MAE_mean", "val_RMSE_mean", "val_CosineSim_mean", "val_PeakMAE_mean",
        "val_PeakBinShift_mean", "val_PeakAxisError_mean", "val_PeakIndexError_mean",
        "val_NegativeBinRate_mean",
        "test_MAE_mean", "test_RMSE_mean", "test_CosineSim_mean", "test_PeakMAE_mean",
        "test_PeakBinShift_mean", "test_PeakAxisError_mean", "test_PeakIndexError_mean",
        "test_NegativeBinRate_mean",
    ]

    agg_summary = {}

    for col in agg_cols:
        if col in summary_df.columns:
            agg_summary[col + "_mean"] = float(summary_df[col].mean())
            agg_summary[col + "_std"] = float(summary_df[col].std(ddof=0))

    agg_json = run_output_dir / f"aggregate_optical_{target_key}_{variant_name}.json"

    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_key": target_key,
                "variant": variant_name,
                "n_seeds": len(list(seeds)),
                "seeds": list(seeds),
                "aggregate": agg_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_config = {
        "target_key": target_key,
        "variant": variant_name,
        "run_name": run_name,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "out_dim": out_dim,
        "graph_attr_dim": graph_attr_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "early_stop_patience": early_stop_patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "peak_weight": peak_weight,
        "smoothness_weight": smoothness_weight,
        "clip_grad_norm": clip_grad_norm,
        "use_amp": use_amp,
        "device": str(device),
        "pretrained_ckpt_path": str(pretrained_ckpt_path) if pretrained_ckpt_path is not None else None,
        "axis_available": sample_axis is not None,
        "axis_range_eV": [
            float(sample_axis.view(-1).detach().cpu().numpy().min()),
            float(sample_axis.view(-1).detach().cpu().numpy().max()),
        ] if sample_axis is not None else None,
        "axis_points": int(out_dim),
        "allow_negative": infer_allow_negative_from_target_key(target_key),
        "negative_weight": get_default_negative_weight(target_key, fallback=0.02),
    }

    with open(run_output_dir / f"train_config_{target_key}_{variant_name}.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    print("\nFINAL SUMMARY")
    summary_cols = [
        "target_key", "variant", "seed", "best_epoch",
        "val_MAE_mean", "val_RMSE_mean", "val_CosineSim_mean",
        "val_PeakMAE_mean", "val_PeakBinShift_mean", "val_PeakAxisError_mean",
        "test_MAE_mean", "test_RMSE_mean", "test_CosineSim_mean",
        "test_PeakMAE_mean", "test_PeakBinShift_mean", "test_PeakAxisError_mean",
    ]
    summary_cols = [c for c in summary_cols if c in summary_df.columns]

    print(summary_df[summary_cols].to_string(index=False))

    artifacts = {
        "run_output_dir": str(run_output_dir),
        "summary_csv": str(summary_csv),
        "aggregate_json": str(agg_json),
        "spectrum_stats_csv": str(spec_stats_path),
        "aggregate": agg_summary,
    }

    return summary_df, artifacts
