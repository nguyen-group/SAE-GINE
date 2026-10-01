"""
Low-data experiment utilities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from .models import OpticalResponseGINE
from .train import train_optical_gine_single_target

# ============================================================
# Low-data SSL-init vs scratch experiment utility
# Added for Q1_PAPER3 low-data ablation
# ============================================================

def run_lowdata_ssl_vs_scratch(
    targets,
    root,
    ssl_ckpt,
    label_fraction=0.05,
    fraction_tag=None,
    split_seed=2025,
    seeds=(42, 123, 2025),
    model_class=None,
    train_kwargs=None,
    metric_cols=None,
    save_outputs=True,
    show=True,
):
    """
    Run low-data UE-GINE scratch vs SSL-init UE-GINE experiments.

    Parameters
    ----------
    targets : dict
        Mapping from target key to (train_ds, val_ds, test_ds), for example:
        {
            "epsI_0": (epsI_enhanced_train_ds, epsI_enhanced_val_ds, epsI_enhanced_test_ds),
            "epsR_0": (epsR_enhanced_train_ds, epsR_enhanced_val_ds, epsR_enhanced_test_ds),
        }
    root : str or pathlib.Path
        Output root, usually D:/TB3/processed/paired_training.
    ssl_ckpt : str or pathlib.Path
        SSL-init backbone checkpoint for enhanced 19/24 graphs.
    label_fraction : float
        Fraction of labeled training data to keep.
    fraction_tag : str or None
        Tag used in output names. If None, generated from label_fraction.
    split_seed : int
        Random seed for choosing the low-data subset.
    seeds : tuple/list
        Training seeds for each scratch/SSL-init run.
    model_class : class or None
        Model class. Defaults to OpticalResponseGINE.
    train_kwargs : dict or None
        Extra keyword arguments for train_optical_gine_single_target.
    metric_cols : list or None
        Metrics to aggregate. Defaults to common validation/test metrics.
    save_outputs : bool
        Save summary/meanstd/gain CSV files.
    show : bool
        Display tables in notebook if IPython is available.

    Returns
    -------
    dict
        summary, meanstd, gain DataFrames and output paths.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Subset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ssl_ckpt = Path(ssl_ckpt)
    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_ckpt}")

    if model_class is None:
        model_class = OpticalResponseGINE

    if fraction_tag is None:
        fraction_tag = f"{int(round(float(label_fraction) * 100))}pct_enhanced19_24"

    if train_kwargs is None:
        train_kwargs = {}

    device = train_kwargs.pop("device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    default_train_kwargs = dict(
        model_class=model_class,
        output_dir=root,
        device=device,
        seeds=list(seeds),
        batch_size=16,
        num_epochs=100,
        early_stop_patience=15,
        lr=5e-4,
        weight_decay=5e-5,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.05,
        peak_weight=2.0,
        smoothness_weight=0.03,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        use_amp=bool(torch.cuda.is_available()),
    )
    default_train_kwargs.update(train_kwargs)

    if metric_cols is None:
        metric_cols = [
            "test_MAE_mean",
            "test_RMSE_mean",
            "test_CosineSim_mean",
            "test_PeakMAE_mean",
            "test_PeakBinShift_mean",
            "test_PeakAxisError_mean",
            "val_MAE_mean",
            "val_RMSE_mean",
            "val_CosineSim_mean",
        ]

    def _check_enhanced(ds, name):
        g = ds[0]
        node_dim = int(g.x.shape[1])
        edge_dim = int(g.edge_attr.shape[1]) if getattr(g, "edge_attr", None) is not None else None

        print(f"{name}: n={len(ds)} | node={node_dim} | edge={edge_dim}")

        if node_dim != 19 or edge_dim != 24:
            raise RuntimeError(
                f"{name} is not enhanced 19/24. Found node={node_dim}, edge={edge_dim}."
            )

    def _make_subset(ds, frac, seed):
        if float(frac) >= 1.0:
            idx = np.arange(len(ds), dtype=np.int64)
            return ds, idx

        n_keep = max(1, int(round(len(ds) * float(frac))))
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(len(ds), size=n_keep, replace=False)).astype(np.int64)
        return Subset(ds, idx.tolist()), idx

    print("=" * 100)
    print("LOW-DATA UE-GINE vs SSL-init UE-GINE")
    print("=" * 100)
    print("root           :", root)
    print("ssl_ckpt       :", ssl_ckpt)
    print("label_fraction :", label_fraction)
    print("fraction_tag   :", fraction_tag)
    print("split_seed     :", split_seed)
    print("seeds          :", list(seeds))
    print("device         :", device)

    rows = []

    for target_key, datasets in targets.items():
        if len(datasets) != 3:
            raise ValueError(f"targets[{target_key!r}] must be (train_ds, val_ds, test_ds).")

        train_full, val_ds, test_ds = datasets

        print("\n" + "=" * 100)
        print("Target:", target_key)
        print("=" * 100)

        _check_enhanced(train_full, f"{target_key} train")
        _check_enhanced(val_ds, f"{target_key} val")
        _check_enhanced(test_ds, f"{target_key} test")

        train_low, selected_idx = _make_subset(train_full, label_fraction, split_seed)
        print(f"Low-data subset: {len(train_low)} / {len(train_full)} = {100 * len(train_low) / len(train_full):.2f}%")

        idx_path = root / f"lowdata_{fraction_tag}_{target_key}_indices_seed{split_seed}.npy"
        if save_outputs:
            np.save(idx_path, selected_idx)
            print("Saved low-data indices:", idx_path)

        for init_type, ckpt in [("scratch", None), ("ssl_init", ssl_ckpt)]:
            variant = f"lowdata_{fraction_tag}_{init_type}"
            run_name = f"{target_key}_{variant}"

            print("\n" + "-" * 100)
            print(f"RUN | target={target_key} | init={init_type} | variant={variant}")
            print("-" * 100)

            df, artifacts = train_optical_gine_single_target(
                target_key=target_key,
                variant_name=variant,
                train_ds=train_low,
                val_ds=val_ds,
                test_ds=test_ds,
                pretrained_ckpt_path=ckpt,
                run_name=run_name,
                **default_train_kwargs,
            )

            df["label_fraction"] = float(label_fraction)
            df["label_fraction_tag"] = fraction_tag
            df["init_type"] = init_type
            df["lowdata_split_seed"] = int(split_seed)
            df["n_train_lowdata"] = int(len(train_low))
            df["n_train_full"] = int(len(train_full))
            df["node_dim"] = 19
            df["edge_dim"] = 24
            df["lowdata_indices_path"] = str(idx_path)
            df["run_output_dir"] = artifacts.get("run_output_dir", "") if isinstance(artifacts, dict) else ""

            rows.append(df)

    summary = pd.concat(rows, ignore_index=True)

    out_summary = root / f"lowdata_{fraction_tag}_ue_gine_vs_ssl_init_summary.csv"
    out_meanstd = root / f"lowdata_{fraction_tag}_ue_gine_vs_ssl_init_meanstd.csv"
    out_gain = root / f"lowdata_{fraction_tag}_ssl_init_gain.csv"

    available_metric_cols = [c for c in metric_cols if c in summary.columns]

    mean_aggs = {f"{c}_avg": (c, "mean") for c in available_metric_cols}
    std_aggs = {f"{c}_std": (c, lambda s: float(s.std(ddof=0))) for c in available_metric_cols}

    meanstd = (
        summary
        .groupby(["target_key", "init_type", "variant", "label_fraction_tag"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            label_fraction=("label_fraction", "first"),
            n_train_lowdata=("n_train_lowdata", "first"),
            n_train_full=("n_train_full", "first"),
            node_dim=("node_dim", "first"),
            edge_dim=("edge_dim", "first"),
            **mean_aggs,
            **std_aggs,
        )
        .reset_index()
    )

    gain_rows = []
    for target_key in targets.keys():
        sub = meanstd[meanstd["target_key"] == target_key].copy()
        scratch = sub[sub["init_type"] == "scratch"]
        ssl = sub[sub["init_type"] == "ssl_init"]

        if len(scratch) == 1 and len(ssl) == 1 and "test_MAE_mean_avg" in sub.columns:
            scratch = scratch.iloc[0]
            ssl = ssl.iloc[0]

            scratch_mae = float(scratch["test_MAE_mean_avg"])
            ssl_mae = float(ssl["test_MAE_mean_avg"])
            scratch_rmse = float(scratch.get("test_RMSE_mean_avg", np.nan))
            ssl_rmse = float(ssl.get("test_RMSE_mean_avg", np.nan))

            gain_rows.append({
                "target_key": target_key,
                "label_fraction": float(label_fraction),
                "label_fraction_tag": fraction_tag,
                "n_train_lowdata": int(scratch["n_train_lowdata"]),
                "n_train_full": int(scratch["n_train_full"]),
                "scratch_test_MAE": scratch_mae,
                "ssl_init_test_MAE": ssl_mae,
                "MAE_reduction_abs": scratch_mae - ssl_mae,
                "MAE_reduction_percent": 100.0 * (scratch_mae - ssl_mae) / scratch_mae if scratch_mae > 0 else np.nan,
                "scratch_test_RMSE": scratch_rmse,
                "ssl_init_test_RMSE": ssl_rmse,
                "RMSE_reduction_abs": scratch_rmse - ssl_rmse,
            })

    gain = pd.DataFrame(gain_rows)

    if save_outputs:
        summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
        meanstd.to_csv(out_meanstd, index=False, encoding="utf-8-sig")
        gain.to_csv(out_gain, index=False, encoding="utf-8-sig")

        print("\nSaved:")
        print(out_summary)
        print(out_meanstd)
        print(out_gain)

    if show:
        try:
            from IPython.display import display
            display(summary)
            display(meanstd)
            display(gain)
        except Exception:
            print(summary)
            print(meanstd)
            print(gain)

    print("\nIMPORTANT CHECK:")
    print("The training log must show node_in_dim = 19 and edge_in_dim = 24.")
    print("If it shows 11/16, stop and rebuild enhanced datasets.")

    return {
        "summary": summary,
        "meanstd": meanstd,
        "gain": gain,
        "summary_csv": str(out_summary),
        "meanstd_csv": str(out_meanstd),
        "gain_csv": str(out_gain),
    }


# Helpers extracted from TASK3 notebook Cell 40
def standardize_lowdata_df(df):
    df = df.copy()

    # target column
    if "target_key" not in df.columns and "target" in df.columns:
        df["target_key"] = df["target"]

    # fraction column
    if "fraction" not in df.columns:
        if "label_fraction" in df.columns:
            df["fraction"] = df["label_fraction"]
        else:
            raise RuntimeError("Cannot find fraction or label_fraction column.")

    # fraction tag column
    if "fraction_tag" not in df.columns:
        if "label_fraction_tag" in df.columns:
            df["fraction_tag"] = df["label_fraction_tag"]
        else:
            df["fraction_tag"] = df["fraction"].apply(lambda x: f"{int(round(float(x) * 100))}pct")

    # seed column
    if "seed" not in df.columns:
        raise RuntimeError("Cannot find seed column.")

    # make sure fraction is numeric
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")

    # clean 5pct_enhanced19_24 tag to 5pct for paper table
    df["fraction_tag"] = df["fraction"].apply(lambda x: f"{int(round(float(x) * 100))}pct")

    return df

def get_model_name(v):
    v = str(v).lower()
    if "ssl_pretrained" in v or "ssl_init" in v or "ssl" in v:
        return "SSL-init UE-GINE"
    if "scratch" in v:
        return "UE-GINE"
    return "Unknown"

def get_initialization(model_name):
    if model_name == "SSL-init UE-GINE":
        return "SSL-initialized"
    return "Scratch"

def get_target_name(x):
    if x == "epsR_0":
        return "ε₁(E)"
    if x == "epsI_0":
        return "ε₂(E)"
    return x

def get_fraction_label(x):
    try:
        return f"{int(round(float(x) * 100))}%"
    except Exception:
        x = str(x)
        return x.replace("pct", "%").replace("_enhanced19_24", "")
