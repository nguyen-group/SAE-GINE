"""Helpers extracted from notebook 2; imports are local to each operation."""


def summarize_baselines(runs):
    import pandas as pd
    from IPython.display import display
    all_summary = pd.concat([df.assign(target_key=target, variant=variant) for (target, variant), df in runs.items()], ignore_index=True)
    benchmark_metrics = [
        "test_MAE_mean",
        "test_RMSE_mean",
        "test_CosineSim_mean",
        "test_PeakAxisError_mean",
    ]

    benchmark_summary = (
        all_summary
        .groupby(["target_key", "variant"])[benchmark_metrics]
        .agg(["mean", "std"])
        .round(6)
    )

    decision_rows = []

    for target_key in ["epsI_0", "epsR_0"]:
        target_df = (
            all_summary[all_summary["target_key"] == target_key]
            .groupby("variant")[benchmark_metrics]
            .mean()
        )

        baseline_mae = target_df.loc["baseline", "test_MAE_mean"]
        enhanced_mae = target_df.loc["enhanced", "test_MAE_mean"]
        ssl_mae = target_df.loc["ssl_init_baseline_11_16", "test_MAE_mean"]

        ssl_gain_vs_baseline = (baseline_mae - ssl_mae) / baseline_mae * 100
        enhanced_gain_vs_baseline = (baseline_mae - enhanced_mae) / baseline_mae * 100
        ssl_gap_vs_enhanced = (ssl_mae - enhanced_mae) / enhanced_mae * 100

        decision_rows.append(
            {
                "target_key": target_key,
                "baseline_MAE": baseline_mae,
                "enhanced_MAE": enhanced_mae,
                "ssl_init_baseline_11_16_MAE": ssl_mae,
                "ssl_gain_vs_baseline_%": ssl_gain_vs_baseline,
                "enhanced_gain_vs_baseline_%": enhanced_gain_vs_baseline,
                "ssl_gap_vs_enhanced_%": ssl_gap_vs_enhanced,
            }
        )

    decision_summary = pd.DataFrame(decision_rows).round(4)

    display(benchmark_summary)
    display(decision_summary)
    return (benchmark_summary, decision_summary)


def pretrain_enhanced(TB3, PROCESSED_DIR, epsI_train_enh, device):
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader as PyGDataLoader
    SSL_ENHANCED_DIR = PROCESSED_DIR / "ssl_enhanced_19_24"
    SSL_ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    SSL_CKPT_ENHANCED = SSL_ENHANCED_DIR / "ssl_enhanced_19_24_graphdesc_best.pt"
    ssl_train_ds = epsI_train_enh
    g0 = ssl_train_ds[0]

    ssl_dims = TB3.check_ssl_graph_dims(
        g0,
        expected_node_dim=19,
        expected_edge_dim=24,
    )

    ssl_loader = PyGDataLoader(
        ssl_train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    ssl_model = TB3.OpticalResponseGINE(
        node_in_dim=ssl_dims["node_dim"],
        edge_in_dim=ssl_dims["edge_dim"],
        out_dim=ssl_dims["ssl_out_dim"],
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.10,
    ).to(device)

    optimizer = torch.optim.AdamW(
        ssl_model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=80,
    )
    best_ssl_loss = float("inf")
    best_ssl_epoch = -1
    ssl_history = []

    start_time = time.time()

    for epoch in range(1, 81):
        ssl_model.train()
        epoch_losses = []

        for batch in ssl_loader:
            batch = batch.to(device)

            target = TB3.graph_desc_target(batch)
            x_aug, edge_attr_aug = TB3.corrupt_graph(
                batch,
                p_node=0.15,
                noise=0.02,
            )

            pred = ssl_model(
                x=x_aug,
                edge_index=batch.edge_index,
                edge_attr=edge_attr_aug,
                batch=batch.batch,
            )

            loss = F.smooth_l1_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        scheduler.step()

        ssl_loss = float(np.mean(epoch_losses))
        ssl_history.append(
            {
                "epoch": epoch,
                "ssl_loss": ssl_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if ssl_loss < best_ssl_loss:
            best_ssl_loss = ssl_loss
            best_ssl_epoch = epoch

            torch.save(
                {
                    "model_state_dict": ssl_model.state_dict(),
                    "best_epoch": best_ssl_epoch,
                    "best_ssl_loss": best_ssl_loss,
                    "node_in_dim": ssl_dims["node_dim"],
                    "edge_in_dim": ssl_dims["edge_dim"],
                    "ssl_out_dim": ssl_dims["ssl_out_dim"],
                    "ssl_method": "enhanced_graph_descriptor_prediction",
                },
                SSL_CKPT_ENHANCED,
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[SSL enhanced 19/24] epoch {epoch:03d} | "
                f"loss={ssl_loss:.6f} | best={best_ssl_loss:.6f}"
            )
    ssl_history_df = pd.DataFrame(ssl_history)
    ssl_history_df.to_csv(
        SSL_ENHANCED_DIR / "history_ssl_enhanced_19_24_graphdesc.csv",
        index=False,
    )
    ssl_enhanced_summary = {
        "train_samples": len(ssl_train_ds),
        "node_features": ssl_dims["node_dim"],
        "edge_features": ssl_dims["edge_dim"],
        "ssl_out_dim": ssl_dims["ssl_out_dim"],
        "best_epoch": best_ssl_epoch,
        "best_ssl_loss": best_ssl_loss,
        "checkpoint": str(SSL_CKPT_ENHANCED),
        "time_min": (time.time() - start_time) / 60,
    }
    ssl_enhanced_summary
    SSL_CKPT_ENHANCED_BACKBONE = SSL_CKPT_ENHANCED.with_name(
        "ssl_enhanced_19_24_backbone_only_for_finetune.pt"
    )

    ssl_ckpt = torch.load(SSL_CKPT_ENHANCED, map_location="cpu")
    ssl_state = ssl_ckpt["model_state_dict"]

    backbone_state = {
        k: v for k, v in ssl_state.items()
        if not k.startswith("head.")
    }

    torch.save(
        {
            "model_state_dict": backbone_state,
            "source_checkpoint": str(SSL_CKPT_ENHANCED),
            "best_epoch": ssl_ckpt["best_epoch"],
            "best_ssl_loss": ssl_ckpt["best_ssl_loss"],
            "node_in_dim": 19,
            "edge_in_dim": 24,
            "ssl_method": "enhanced_graphdesc_backbone_only",
        },
        SSL_CKPT_ENHANCED_BACKBONE,
    )

    checkpoint_summary = {
        "checkpoint": str(SSL_CKPT_ENHANCED_BACKBONE),
        "num_tensors": len(backbone_state),
    }

    checkpoint_summary
    return (SSL_CKPT_ENHANCED_BACKBONE, ssl_enhanced_summary, checkpoint_summary)


def summarize_linear_runs(linear_runs):
    import pandas as pd
    from IPython.display import display
    runs = [
        ("epsI_0", "ε₂(E) / epsI₀", "SSL-init Linear Ensemble UE-GINE", "SSL-init", linear_runs[('epsI_0', 'SSL-init')]),
        ("epsI_0", "ε₂(E) / epsI₀", "Scratch Linear Ensemble UE-GINE", "Scratch", linear_runs[('epsI_0', 'Scratch')]),
        ("epsR_0", "ε₁(E) / epsR₀", "SSL-init Linear Ensemble UE-GINE", "SSL-init", linear_runs[('epsR_0', 'SSL-init')]),
        ("epsR_0", "ε₁(E) / epsR₀", "Scratch Linear Ensemble UE-GINE", "Scratch", linear_runs[('epsR_0', 'Scratch')]),
    ]

    rows = []

    for target_key, target_label, model_name, init_type, df in runs:
        rows.append(
            {
                "target_key": target_key,
                "Target": target_label,
                "Model": model_name,
                "Init": init_type,
                "Graph": "20/24",
                "MAE_mean": df["test_MAE_mean"].mean(),
                "MAE_std": df["test_MAE_mean"].std(ddof=1),
                "RMSE_mean": df["test_RMSE_mean"].mean(),
                "RMSE_std": df["test_RMSE_mean"].std(ddof=1),
                "Cosine_mean": df["test_CosineSim_mean"].mean(),
                "Cosine_std": df["test_CosineSim_mean"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(rows)

    paper_table = pd.DataFrame(
        {
            "Target": summary["Target"],
            "Model": summary["Model"],
            "Init": summary["Init"],
            "Graph": summary["Graph"],
            "MAE ↓": summary.apply(lambda r: f"{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}", axis=1),
            "RMSE ↓": summary.apply(lambda r: f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f}", axis=1),
            "Cosine ↑": summary.apply(lambda r: f"{r['Cosine_mean']:.3f} ± {r['Cosine_std']:.3f}", axis=1),
        }
    )

    gain_rows = []

    for target_key, target_label in [("epsI_0", "ε₂(E) / epsI₀"), ("epsR_0", "ε₁(E) / epsR₀")]:
        scratch = summary[(summary["target_key"] == target_key) & (summary["Init"] == "Scratch")].iloc[0]
        ssl = summary[(summary["target_key"] == target_key) & (summary["Init"] == "SSL-init")].iloc[0]

        gain_rows.append(
            {
                "target_key": target_key,
                "target_label": target_label,
                "scratch_test_MAE": scratch["MAE_mean"],
                "ssl_init_test_MAE": ssl["MAE_mean"],
                "MAE_reduction_abs": scratch["MAE_mean"] - ssl["MAE_mean"],
                "MAE_reduction_percent": (scratch["MAE_mean"] - ssl["MAE_mean"]) / scratch["MAE_mean"] * 100,
                "scratch_test_RMSE": scratch["RMSE_mean"],
                "ssl_init_test_RMSE": ssl["RMSE_mean"],
                "RMSE_reduction_abs": scratch["RMSE_mean"] - ssl["RMSE_mean"],
                "scratch_test_Cosine": scratch["Cosine_mean"],
                "ssl_init_test_Cosine": ssl["Cosine_mean"],
                "Cosine_gain_abs": ssl["Cosine_mean"] - scratch["Cosine_mean"],
            }
        )
    gain_table = pd.DataFrame(gain_rows).round(6)

    display(paper_table)
    display(gain_table)
    return (paper_table, gain_table)


def top_contributions(node_df, edge_df):
    import pandas as pd
    from IPython.display import display
    target_labels = {
        "epsR_0": "ε₁(E) / epsR₀",
        "epsI_0": "ε₂(E) / epsI₀",
    }

    def make_top_table(df, feature_type, top_k=5):
        rows = []

        for target_key in ["epsR_0", "epsI_0"]:
            sub = (
                df[df["target_key"] == target_key]
                .sort_values("weight_mean", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )

            for rank, row in sub.iterrows():
                rows.append({
                    "Target": target_labels[target_key],
                    "Model": "SSL-init Linear Ensemble UE-GINE",
                    "Rank": rank + 1,
                    "Feature type": feature_type,
                    "Feature": row["feature"],
                    "Contribution weight": f"{row['weight_mean']:.4f} ± {row['weight_std']:.4f}",
                })

        return pd.DataFrame(rows)

    node_top5 = make_top_table(node_df, "node")
    edge_top5 = make_top_table(edge_df, "edge")

    display(node_top5)
    display(edge_top5)
    return (node_top5, edge_top5)


def plot_linear_gain(training_dir):
    from pathlib import Path
    import pandas as pd
    from IPython.display import display
    import utils_tb as TB3

    ROOT = Path(training_dir)
    OUT_DIR = ROOT / "linear_ensemble_contribution_20_24"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_SEEDS = {42, 123, 2025}

    runs = {
        "epsI_0": ("epsI_0_scratch_linear_ue_20_24_epsI", "epsI_0_ssl_init_linear_ue_20_24", r"$\varepsilon_2(E)$"),
        "epsR_0": ("epsR_0_scratch_linear_ue_20_24_epsR", "epsR_0_ssl_init_linear_ue_20_24_epsR", r"$\varepsilon_1(E)$"),
    }

    rows = []
    for target_key, (scratch_dir, ssl_dir, label) in runs.items():
        scratch = pd.read_csv(next((ROOT / scratch_dir).glob("summary_optical_*.csv")))
        ssl = pd.read_csv(next((ROOT / ssl_dir).glob("summary_optical_*.csv")))

        if set(scratch["seed"].astype(int)) != EXPECTED_SEEDS or set(ssl["seed"].astype(int)) != EXPECTED_SEEDS:
            raise ValueError(f"{target_key}: unexpected seeds found.")

        scratch_mae = scratch["test_MAE_mean"]
        ssl_mae = ssl["test_MAE_mean"]

        rows.append({
            "target_key": target_key,
            "Target": label,
            "Scratch_MAE_mean": scratch_mae.mean(),
            "Scratch_MAE_std": scratch_mae.std(ddof=1),
            "SSL_MAE_mean": ssl_mae.mean(),
            "SSL_MAE_std": ssl_mae.std(ddof=1),
            "MAE_reduction_percent": 100 * (scratch_mae.mean() - ssl_mae.mean()) / scratch_mae.mean(),
        })

    gain_table = pd.DataFrame(rows)

    print("All four runs contain seeds 42, 123, 2025.\n")
    display(gain_table.round(6))

    gain_table.to_csv(
        OUT_DIR / "linear_ensemble_20_24_ssl_gain_vs_scratch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    TB3.plot_linear_ensemble_ssl_gain_dotplot_final(
        root=ROOT,
        save_prefix="fig_linear_ensemble_ssl_gain_verified",
    )
    return gain_table


def lowdata_tables(TB3, PROCESSED_DIR):
    import pandas as pd
    from IPython.display import display

    ROOT = PROCESSED_DIR / "paired_training"

    meanstd_files = sorted(ROOT.glob(
        "lowdata_*pct_enhanced19_24_ue_gine_vs_ssl_init_meanstd.csv"
    ))
    gain_files = sorted(ROOT.glob(
        "lowdata_*pct_enhanced19_24_ssl_init_gain.csv"
    ))

    if len(meanstd_files) != 5 or len(gain_files) != 5:
        raise FileNotFoundError(
            f"Expected 5 mean/std and 5 gain files, found "
            f"{len(meanstd_files)} and {len(gain_files)}."
        )

    paper_table = pd.concat(
        [pd.read_csv(f) for f in meanstd_files],
        ignore_index=True,
    )

    gain_table = pd.concat(
        [pd.read_csv(f) for f in gain_files],
        ignore_index=True,
    )

    paper_table["target_name"] = paper_table["target_key"].map(TB3.get_target_name)
    paper_table["model_name"] = paper_table["init_type"].map(TB3.get_model_name)
    paper_table["initialization"] = paper_table["model_name"].map(TB3.get_initialization)
    paper_table["fraction_label"] = paper_table["label_fraction"].map(TB3.get_fraction_label)

    for name, mean_col, std_col in [
        ("MAE", "test_MAE_mean_avg", "test_MAE_mean_std"),
        ("RMSE", "test_RMSE_mean_avg", "test_RMSE_mean_std"),
        ("Cosine", "test_CosineSim_mean_avg", "test_CosineSim_mean_std"),
    ]:
        paper_table[name] = paper_table.apply(
            lambda r: f"{r[mean_col]:.3f} ± {r[std_col]:.3f}",
            axis=1,
        )

    gain_table["target_name"] = gain_table["target_key"].map(TB3.get_target_name)
    gain_table["fraction_label"] = gain_table["label_fraction"].map(TB3.get_fraction_label)

    paper_table.to_csv(
        ROOT / "table_lowdata_ssl_init_ue_gine_meanstd.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gain_table.to_csv(
        ROOT / "table_lowdata_ssl_init_ue_gine_gain.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(paper_table)
    display(gain_table)
    return (paper_table, gain_table)


def lowdata_main_table(TB3, PROCESSED_DIR):
    import pandas as pd
    from pathlib import Path
    from IPython.display import display

    ROOT = PROCESSED_DIR / "paired_training"

    wanted = {
        "5pct": "5%",
        "10pct": "10%",
        "25pct": "25%",
        "50pct": "50%",
    }

    rows = []

    for tag, label in wanted.items():
        f = ROOT / f"lowdata_{tag}_enhanced19_24_ue_gine_vs_ssl_init_meanstd.csv"

        if not f.exists():
            print("Missing:", f.name)
            continue

        df = pd.read_csv(f)

        mae_col  = [c for c in df.columns if "test_MAE" in c and "mean" in c][0]
        rmse_col = [c for c in df.columns if "test_RMSE" in c and "mean" in c][0]

        for _, r in df.iterrows():
            variant = str(r["variant"])
            init = "SSL-init" if "ssl_init" in variant else "Scratch"
            target = "ε₁(E)" if r["target_key"] == "epsR_0" else "ε₂(E)"

            rows.append({
                "Target": target,
                "Labeled data": label,
                "Model": "UE-GINE",
                "Initialization": init,
                "MAE": round(r[mae_col], 4),
                "RMSE": round(r[rmse_col], 4),
            })

    paper_table_main = pd.DataFrame(rows)

    order_frac = {"5%": 0, "10%": 1, "25%": 2, "50%": 3}
    order_target = {"ε₁(E)": 0, "ε₂(E)": 1}
    order_init = {"Scratch": 0, "SSL-init": 1}

    paper_table_main["frac_order"] = paper_table_main["Labeled data"].map(order_frac)
    paper_table_main["target_order"] = paper_table_main["Target"].map(order_target)
    paper_table_main["init_order"] = paper_table_main["Initialization"].map(order_init)

    paper_table_main = (
        paper_table_main
        .sort_values(["frac_order", "target_order", "init_order"])
        .drop(columns=["frac_order", "target_order", "init_order"])
        .reset_index(drop=True)
    )

    display(paper_table_main)

    out_csv = ROOT / "paper_table_lowdata_ssl_init_ue_gine_CLEAN.csv"
    paper_table_main.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", out_csv)
    return paper_table_main


def label_efficiency_stats(TB3, PROCESSED_DIR):
    from pathlib import Path
    import pandas as pd
    from IPython.display import display

    ROOT = PROCESSED_DIR / "paired_training"

    summary_files = sorted(
        ROOT.glob("lowdata_*pct_enhanced19_24_ue_gine_vs_ssl_init_summary.csv")
    )

    if len(summary_files) != 5:
        raise FileNotFoundError(
            f"Expected 5 low-data summary files, found {len(summary_files)}."
        )

    cols = [
        "target_key",
        "init_type",
        "seed",
        "label_fraction",
        "test_MAE_mean",
    ]

    raw = (
        pd.concat(
            [pd.read_csv(path)[cols] for path in summary_files],
            ignore_index=True,
        )
        .drop_duplicates(
            ["target_key", "init_type", "seed", "label_fraction"]
        )
    )

    figure3_stats = (
        raw.groupby(
            ["target_key", "init_type", "label_fraction"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            test_MAE_mean=("test_MAE_mean", "mean"),
            test_MAE_std=("test_MAE_mean", lambda x: x.std(ddof=1)),
        )
        .sort_values(["target_key", "init_type", "label_fraction"])
        .reset_index(drop=True)
    )

    if not figure3_stats["n_seeds"].eq(3).all():
        raise RuntimeError("Each Figure 3 point must contain exactly three seeds.")

    OUT_CSV = ROOT / "figure3_label_efficiency_meanstd_ddof1.csv"
    figure3_stats.to_csv(OUT_CSV, index=False)

    display(figure3_stats)
    print(f"Saved: {OUT_CSV}")
    return figure3_stats


def plot_saved_examples(TB3, training_dir):
    from pathlib import Path

    ROOT = Path(training_dir)
    FIG_DIR = ROOT / "final_benchmark_tables" / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    epsI_npz = (
        ROOT
        / "epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"
        / "test_predictions_epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed_seed123.npz"
    )

    epsI_results = TB3.plot_pred_true_cases(
        npz_path=epsI_npz,
        main_title=r"SSL-init UE-GINE prediction of $\varepsilon_2(E)$",
        y_label=r"$\varepsilon_2(E)$",
        case_mode="paper",
        save_path=FIG_DIR / "fig_epsI0_ssl_init_ue_gine_prediction_examples.png",
    )

    mae_epsI = epsI_results["mae_each"]
    idxs_epsI = epsI_results["selected_indices"]
    epsR_npz = (
        ROOT
        / "epsR_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"
        / "test_predictions_epsR_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed_seed123.npz"
    )

    epsR_results = TB3.plot_pred_true_cases(
        npz_path=epsR_npz,
        main_title=r"SSL-init UE-GINE prediction of $\varepsilon_1(E)$",
        y_label=r"$\varepsilon_1(E)$",
        case_mode="paper",
        save_path=FIG_DIR / "fig_epsR0_ssl_init_ue_gine_prediction_examples.png",
    )

    mae_epsR = epsR_results["mae_each"]
    idxs_epsR = epsR_results["selected_indices"]
    alpha_dir = (
        ROOT
        / "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed"
    )

    alpha_npz = sorted(alpha_dir.glob("test_predictions_*.npz"))[0]

    alpha_results = TB3.plot_pred_true_cases(
        npz_path=alpha_npz,
        main_title=r"Direct-alpha ablation on log-scaled $\alpha(E)$",
        y_label=r"$\log_{10}(1+\alpha)$",
        case_mode="paper",
        save_path=FIG_DIR / "fig_direct_alpha_ssl_init_ablation_examples.png",
    )

    mae_alpha = alpha_results["mae_each"]
    idxs_alpha = alpha_results["selected_indices"]
    return (epsI_results, epsR_results, alpha_results)


def plot_derived_example(PROCESSED_DIR):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    ROOT = PROCESSED_DIR / "paired_training"
    OUT_DIR = ROOT / "paper_outputs"
    FIG_DIR = OUT_DIR / "figures"
    DERIVED_DIR = OUT_DIR / "derived_optical_properties"

    DERIVED_NPZ = DERIVED_DIR / "derived_optical_test_ensemble_ssl_init_ue_gine_100pct.npz"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    assert DERIVED_NPZ.exists(), DERIVED_NPZ

    data = np.load(DERIVED_NPZ, allow_pickle=True)

    E = data["energy_eV"]
    sample_ids = data["sample_ids"].astype(str)

    preferred_id = "4054"
    if preferred_id in sample_ids:
        sample_idx = int(np.where(sample_ids == preferred_id)[0][0])
    else:
        err = (
            np.nanmean(np.abs(data["eps1_pred"] - data["eps1_true"]), axis=1)
            + np.nanmean(np.abs(data["eps2_pred"] - data["eps2_true"]), axis=1)
        )
        sample_idx = int(np.argsort(err)[len(err) // 2])

    print("Selected sample ID:", sample_ids[sample_idx])

    alpha_true = data["alpha_true_cm1"] / 1e6
    alpha_pred = data["alpha_pred_cm1"] / 1e6

    panels = [
        ("(a)", "Real dielectric function", r"$\varepsilon_1(E)$", data["eps1_true"][sample_idx], data["eps1_pred"][sample_idx]),
        ("(b)", "Imaginary dielectric function", r"$\varepsilon_2(E)$", data["eps2_true"][sample_idx], data["eps2_pred"][sample_idx]),
        ("(c)", "Refractive index", r"$n(E)$", data["n_true"][sample_idx], data["n_pred"][sample_idx]),
        ("(d)", "Extinction coefficient", r"$k(E)$", data["k_true"][sample_idx], data["k_pred"][sample_idx]),
        ("(e)", "Absorption coefficient", r"$\alpha(E)$ ($10^6$ cm$^{-1}$)", alpha_true[sample_idx], alpha_pred[sample_idx]),
        ("(f)", "Reflectance", r"$R(E)$", data["R_true"][sample_idx], data["R_pred"][sample_idx]),
    ]

    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), dpi=300)

    for ax, (panel, title, ylabel, y_true, y_pred) in zip(axes.ravel(), panels):
        ax.plot(E, y_true, lw=2.2, label="Reference")
        ax.plot(E, y_pred, lw=2.2, ls="--", label="SSL-init UE-GINE")
        ax.set_title(title)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.94, panel, transform=ax.transAxes, fontsize=15, va="top", ha="left")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Representative dielectric-first prediction of optical properties",
        fontsize=16,
        y=1.07,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = FIG_DIR / "fig_dielectric_to_optical_prediction.png"
    out_pdf = FIG_DIR / "fig_dielectric_to_optical_prediction.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved:")
    print(out_png)
    print(out_pdf)
    return (out_png, out_pdf)


def matched_full_rerun(TB3, PROCESSED_DIR, dataset):
    # Matched full-data comparison: Scratch vs SSL-init
    # Enhanced graph representation: 19 node / 24 edge features

    from pathlib import Path
    import importlib
    from IPython.display import display

    import utils_tb as TB3
    import utils_tb.features as FEAT



    ROOT = PROCESSED_DIR / "paired_training"

    SSL_CKPT = (
        PROCESSED_DIR
        / "ssl_enhanced_19_24"
        / "ssl_enhanced_19_24_backbone_only_for_finetune.pt"
    )

    assert SSL_CKPT.exists(), SSL_CKPT

    # ============================================================
    # Enhanced graph datasets
    # ============================================================

    epsR_graph, epsI_graph, *_ = FEAT.build_paired_dielectric_graph_datasets_enhanced(
        dataset,
        key_real="epsR_0",
        key_imag="epsI_0",
        verbose=False,
    )

    print("=" * 100)
    print("ENHANCED GRAPH CHECK")
    print("=" * 100)
    print(
        "epsR_0:",
        epsR_graph[0].x.shape[-1],
        "node features /",
        epsR_graph[0].edge_attr.shape[-1],
        "edge features",
    )
    print(
        "epsI_0:",
        epsI_graph[0].x.shape[-1],
        "node features /",
        epsI_graph[0].edge_attr.shape[-1],
        "edge features",
    )

    assert (
        epsR_graph[0].x.shape[-1],
        epsR_graph[0].edge_attr.shape[-1],
    ) == (19, 24)

    assert (
        epsI_graph[0].x.shape[-1],
        epsI_graph[0].edge_attr.shape[-1],
    ) == (19, 24)

    # Fixed paired split
    # ============================================================

    split = FEAT.build_fixed_paired_split(
        epsR_graph,
        epsI_graph,
        processed_dir=PROCESSED_DIR,
        seed=2025,
        overwrite=False,
    )

    targets = {
        "epsR_0": (
            split["epsR_train_ds"],
            split["epsR_val_ds"],
            split["epsR_test_ds"],
        ),
        "epsI_0": (
            split["epsI_train_ds"],
            split["epsI_val_ds"],
            split["epsI_test_ds"],
        ),
    }

    print("=" * 100)
    print("FIXED DATASET SIZES")
    print("=" * 100)

    for target_key, (train_ds, val_ds, test_ds) in targets.items():
        print(
            f"{target_key}: "
            f"train={len(train_ds)}, "
            f"val={len(val_ds)}, "
            f"test={len(test_ds)}"
        )
    # ============================================================
    # Matched Scratch vs SSL-init training
    # ============================================================

    results_matched_full = TB3.run_lowdata_ssl_vs_scratch(
        targets=targets,
        root=ROOT,
        ssl_ckpt=SSL_CKPT,
        label_fraction=1.0,
        fraction_tag="100pct_matched_full_rerun_enhanced19_24",
        split_seed=777,
        seeds=(42, 123, 2025),
        train_kwargs={"lr": 3e-4},
        save_outputs=True,
        show=False,
    )

    # ============================================================
    # Final tables
    # ============================================================

    print("\n" + "=" * 100)
    print("PER-SEED RESULTS")
    print("=" * 100)
    display(results_matched_full["summary"])

    print("\n" + "=" * 100)
    print("MEAN AND STANDARD DEVIATION")
    print("=" * 100)
    display(results_matched_full["meanstd"])

    print("\n" + "=" * 100)
    print("SSL GAIN RELATIVE TO SCRATCH")
    print("=" * 100)
    display(results_matched_full["gain"])
    return results_matched_full


def export_historical_figure2(project_root, font_paths=None):
    from pathlib import Path
    from datetime import datetime
    from tempfile import mkdtemp
    import hashlib, json, shutil, zipfile
    import numpy as np
    import pandas as pd
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    import pypdfium2 as pdfium

    # ============================================================
    # 1. Original prediction files
    # ============================================================

    ROOT = Path(project_root)
    TRAIN = ROOT / "processed" / "paired_training"
    SEEDS = (42, 123, 2025)
    PERCENTILES = (10, 50, 90)

    EPS2_RUN = "epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"

    FILES = {
        "eps1": [
            TRAIN / "epsR_0_enhanced"
            / f"test_predictions_epsR_0_enhanced_seed{seed}.npz"
            for seed in SEEDS
        ],
        "eps2": [
            TRAIN / EPS2_RUN
            / f"test_predictions_{EPS2_RUN}_seed{seed}.npz"
            for seed in SEEDS
        ],
    }

    # Original six examples: verify that the same materials are retained.
    EXPECTED_BASE_IDS = {
        "eps1": [2566, 4373, 4813],
        "eps2": [5035, 10115, 3812],
    }

    results = {}
    source_records = []
    all_metrics = []

    for target, paths in FILES.items():
        predictions = []
        masks = []
        reference = None
        energy = None
        identity = None

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing original prediction file:\n{path}")

            with np.load(path, allow_pickle=False) as archive:
                true = np.asarray(archive["true"], dtype=float)
                pred = np.asarray(archive["pred"], dtype=float)
                raw_mask = np.asarray(archive["mask"])

                if not np.isfinite(raw_mask).all():
                    raise ValueError(f"Nonfinite mask in {path.name}")
                if not np.isin(raw_mask, [0, 1]).all():
                    raise ValueError(f"Unexpected mask values in {path.name}")

                mask = raw_mask.astype(bool)
                axis = np.asarray(archive["spectral_axis"], dtype=float).reshape(-1)
                ids = {
                    key: np.asarray(archive[key]).reshape(-1)
                    for key in ("base_idx", "sample_idx", "view_idx")
                }

                if "axis_grid" in archive.files:
                    np.testing.assert_allclose(
                        axis,
                        np.asarray(archive["axis_grid"]).reshape(-1),
                        rtol=0, atol=1e-10,
                    )

            if true.shape != (992, 2001):
                raise ValueError(f"Unexpected spectrum shape in {path.name}: {true.shape}")
            if pred.shape != true.shape or mask.shape != true.shape:
                raise ValueError(f"Prediction/mask shape mismatch in {path.name}")
            if (
                axis.shape != (2001,)
                or not np.isfinite(axis).all()
                or not np.all(np.diff(axis) > 0)
                or not np.allclose(axis[[0, -1]], [0, 20])
            ):
                raise ValueError(f"Unexpected energy grid in {path.name}")
            if any(values.size != true.shape[0] for values in ids.values()):
                raise ValueError(f"Identity-array shape mismatch in {path.name}")

            if reference is None:
                reference = true
                energy = axis
                identity = ids
            else:
                np.testing.assert_allclose(
                    true, reference, rtol=0, atol=0, equal_nan=True
                )
                np.testing.assert_allclose(axis, energy, rtol=0, atol=1e-10)
                for key in identity:
                    np.testing.assert_array_equal(ids[key], identity[key])

            predictions.append(pred)
            masks.append(mask)
            source_records.append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

        prediction = np.mean(np.stack(predictions), axis=0)
        valid = (
            np.logical_and.reduce(masks)
            & np.isfinite(reference)
            & np.isfinite(prediction)
        )
        counts = valid.sum(axis=1)

        if np.any(counts == 0):
            raise ValueError(f"{target}: a structure has no valid spectral points.")

        difference = prediction - reference
        mae = np.where(valid, np.abs(difference), 0).sum(axis=1) / counts
        rmse = np.sqrt(
            np.where(valid, difference**2, 0).sum(axis=1) / counts
        )

        if not np.isfinite(mae).all() or not np.isfinite(rmse).all():
            raise ValueError(f"{target}: nonfinite metrics.")

        indices = np.array([
            np.argmin(np.abs(mae - np.percentile(mae, percentile)))
            for percentile in PERCENTILES
        ], dtype=int)

        np.testing.assert_array_equal(
            identity["base_idx"][indices],
            EXPECTED_BASE_IDS[target],
            err_msg="Selected materials differ from the original six examples.",
        )

        if not valid[indices].all():
            raise ValueError(f"{target}: a selected example has invalid spectral points.")

        results[target] = {
            "energy": energy,
            "reference": reference,
            "prediction": prediction,
            "base_idx": identity["base_idx"],
            "MAE": mae,
            "RMSE": rmse,
            "indices": indices,
        }

        for index in range(len(mae)):
            all_metrics.append({
                "response": target,
                "test_index": index,
                "base_idx": int(identity["base_idx"][index]),
                "valid_energy_points": int(counts[index]),
                "MAE": float(mae[index]),
                "RMSE": float(rmse[index]),
                "selected_for_figure": bool(index in indices),
            })

    np.testing.assert_allclose(
        results["eps1"]["energy"],
        results["eps2"]["energy"],
        rtol=0, atol=1e-10,
    )

    print("Original seed files / identities / ensemble / original six examples: PASS")



    META=ROOT/'processed/fixed_splits/fixed_split_paired_epsR_epsI_seed2025_v1.csv'
    metadata=pd.read_csv(META).set_index('base_idx')
    records=[];curves=[]
    for row,target in enumerate(('eps1','eps2')):
        data=results[target]
        for col,(i,q) in enumerate(zip(data['indices'],PERCENTILES)):
            panel='abcdef'[row*3+col];bid=int(data['base_idx'][i]);m=metadata.loc[bid]
            assert m['split']=='test'
            records.append(dict(panel=panel,response=target,requested_percentile=q,test_index=int(i),base_idx=bid,formula=m.formula,material_id=m.mat_id,MAE=data['MAE'][i],RMSE=data['RMSE'][i]))
            curves.append(pd.DataFrame(dict(panel=panel,energy_eV=data['energy'],reference=data['reference'][i],prediction=data['prediction'][i])))
    selected=pd.DataFrame(records);plotted=pd.concat(curves,ignore_index=True)
    OUT=Path(mkdtemp(prefix='Figure2_material_names_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_',dir=str(ROOT)))
    if font_paths is None:
        from matplotlib.font_manager import FontProperties, findfont
        font_paths = (findfont(FontProperties(family="DejaVu Sans")),
                      findfont(FontProperties(family="DejaVu Sans", weight="bold")))
    pdfmetrics.registerFont(TTFont('Arial', str(font_paths[0])))
    pdfmetrics.registerFont(TTFont('Arial-Bold', str(font_paths[1])))
    c=canvas.Canvas(str(OUT/'Figure2_FINAL.pdf'),pagesize=(170*72/25.4,155*72/25.4))
    c.setTitle('Representative dielectric spectra and pointwise prediction errors')
    BLUE,ORANGE,INK='#246781','#C76532','#20262C'
    def text(x,y,s,size=8.5,bold=False,align='left'):
        c.setFillColor(HexColor(INK));c.setFont('Arial-Bold' if bold else 'Arial',size)
        {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,y,s)
    def line(x,y,xx,yy,color=INK,width=.65,dash=None):
        c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setDash(dash or []);c.line(x,y,xx,yy);c.setDash([])
    def trace(xs,ys,color,dash=None,width=1.1):
        p=c.beginPath();p.moveTo(xs[0],ys[0])
        for x,y in zip(xs[1:],ys[1:]):p.lineTo(x,y)
        c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.setDash(dash or []);c.drawPath(p);c.setDash([])
    def ylabel(x,y,index=None):
        c.saveState();c.translate(x,y);c.rotate(90)
        if index is None:text(0,0,'|Δε|',9,align='center')
        else:
            parts=[('ε',9.5,0),(str(index),7,-2.5),('(E)',9.5,0)]
            width=sum(pdfmetrics.stringWidth(s,'Arial',fs) for s,fs,rise in parts)
            t=c.beginText(-width/2,0);c.setFillColor(HexColor(INK))
            for s,fs,rise in parts:t.setFont('Arial',fs);t.setRise(rise);t.textOut(s)
            c.drawText(t)
        c.restoreState()
    def formula(value,x,y,align):
        parts=[(ch,6.5,-2) if ch.isdigit() else (ch,9.5,0) for ch in value]
        width=sum(pdfmetrics.stringWidth(ch,'Arial',size) for ch,size,dy in parts)
        if align=='right':x-=width
        for ch,size,dy in parts:
            text(x,y+dy,ch,size);x+=pdfmetrics.stringWidth(ch,'Arial',size)

    for x,label,col,dash in [(145,'Reference',BLUE,None),(255,'Prediction',ORANGE,[4,2.5])]:
        line(x,420,x+24,420,col,1.3,dash);text(x+30,417,label,9.5)
    for n,r in enumerate(selected.itertuples()):
        row,col=divmod(n,3);x0=34+155*col;pw=126;ph=98;eh=32
        my,ey=(279,229) if row==0 else (82,32)
        ymin,ymax=(-3.5,14) if row==0 else (-.4,7.5)
        df=plotted[plotted.panel==r.panel]
        E=df.energy_eV.to_numpy();ref=df.reference.to_numpy();pred=df.prediction.to_numpy();err=abs(ref-pred)
        assert min(ref.min(),pred.min())>ymin and max(ref.max(),pred.max())<ymax and err.max()<3
        xs=x0+E/20*pw
        text(x0-23,my+ph+10,r.panel,11,True,'center')
        text(x0+pw/2,my+ph+10,f'{r.requested_percentile}th MAE percentile',8.5,True,'center')
        # Place panel-f metrics over its empty upper-left region.
        tx,align=(x0+5,'left') if r.panel=='f' else (x0+pw-3,'right')
        text(tx,my+ph-11,f'MAE = {r.MAE:.3f}',8,align=align)
        text(tx,my+ph-22,f'RMSE = {r.RMSE:.3f}',8,align=align)
        formula(r.formula,x0+pw-5 if r.panel=='f' else (x0+38 if r.panel=='a' else x0+5),my+ph-13,'right' if r.panel=='f' else 'left')
        line(x0,my,x0+pw,my);line(x0,my,x0,my+ph)
        line(x0,my+ph,x0+pw,my+ph);line(x0+pw,my,x0+pw,my+ph)
        for yt in ([0,4,8,12] if row==0 else [0,2,4,6]):
            yy=my+(yt-ymin)/(ymax-ymin)*ph;line(x0-2.5,yy,x0,yy);text(x0-5,yy-2.7,str(yt),8,align='right')
        trace(xs,my+(ref-ymin)/(ymax-ymin)*ph,BLUE)
        trace(xs,my+(pred-ymin)/(ymax-ymin)*ph,ORANGE,[4,2.5])
        line(x0,ey,x0+pw,ey);line(x0,ey,x0,ey+eh)
        line(x0,ey+eh,x0+pw,ey+eh);line(x0+pw,ey,x0+pw,ey+eh)
        for yt in [0,1.5,3]:
            yy=ey+yt/3*eh;line(x0-2.5,yy,x0,yy);text(x0-5,yy-2.5,f'{yt:g}',7.5,align='right')
        trace(xs,ey+err/3*eh,ORANGE,width=.85)
        for xt in [0,5,10,15,20]:
            xx=x0+xt/20*pw;line(xx,my,xx,my-2.5);line(xx,ey,xx,ey-2.5);text(xx,ey-12,str(xt),8,align='center')
        text(x0+pw/2,ey-25,'Photon energy (eV)',8.5,align='center')
        if col==0:ylabel(x0-23,my+ph/2,row+1);ylabel(x0-23,ey+eh/2)
    c.save()
    with pdfium.PdfDocument(str(OUT/'Figure2_FINAL.pdf')) as pdf:
        pdf[0].render(scale=2.8).to_pil().save(OUT/'Figure2_preview.png')
        pdf[0].render(scale=600/72).to_pil().save(OUT/'Figure2_FINAL_600dpi.png',dpi=(600,600))
    caption = (
        "Figure 2. Representative held-out dielectric spectra and prediction "
        "errors. (a–c) Real dielectric spectra from the three-seed AE-GINE "
        "ensemble; (d–f) imaginary dielectric spectra from the three-seed "
        "SAE-GINE ensemble. For each target, examples were selected nearest "
        "to the 10th, 50th and 90th percentiles of the per-structure ensemble "
        "mean absolute error (MAE) distribution. Solid blue and dashed orange "
        "curves denote reference and predicted spectra, respectively. Spectral "
        "limits are identical within each row. Lower traces show pointwise "
        "absolute errors, |Δε| = |prediction − reference|, on a common scale "
        "across all six examples. MAE and root mean squared error (RMSE) are "
        "computed over the same 2001 energy points from 0 to 20 eV. RMSE is "
        "a complementary diagnostic and was not used to select examples. "
        "The two rows contain independently selected structures, not paired materials."
    )
    (OUT / "caption.txt").write_text(caption, encoding="utf-8")


    selected.to_csv(OUT/'selected_samples_metrics.csv',index=False)
    plotted.to_csv(OUT/'plotted_spectra.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'sources':source_records,'metadata':str(META),'metadata_sha256':hashlib.sha256(META.read_bytes()).hexdigest(),'original_six_materials_verified':True},indent=2),encoding='utf-8')
    print(selected.to_string(index=False))
    print('OUTPUT',OUT)

    # Display the figure inline when run in Jupyter.
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display, Image
            display(Image(filename=str(OUT / "Figure2_preview.png")))
    except ImportError:
        pass
    return OUT

