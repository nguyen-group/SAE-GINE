"""
Benchmark table and weighted-R2 utilities for TASK3.
"""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None

# ============================================================
# Final benchmark table utilities for TASK3
# ============================================================

def build_task3_final_benchmark_tables(
    root=r"D:\TB3\processed\paired_training",
    out_subdir="final_benchmark_tables",
    r2_filename="weighted_r2_all_main_models_average.csv",
    display_tables=True,
):
    """
    Build Q1-style benchmark tables for TASK3, including SSL-init Gated UE-GINE.

    Returns
    -------
    dict with raw/display tables and saved CSV paths.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    try:
        from IPython.display import display
    except Exception:
        display = None

    ROOT = Path(root)
    OUT = ROOT / out_subdir
    OUT.mkdir(parents=True, exist_ok=True)

    r2_path = OUT / r2_filename
    if r2_path.exists():
        r2 = pd.read_csv(r2_path, header=[0, 1], index_col=0)
        r2.columns = [f"{a}_{b}" for a, b in r2.columns]
    else:
        print("Missing R2 table:", r2_path)
        r2 = pd.DataFrame()

    models = [
        # epsI_0
        ("epsI_0 | GCN", "ε₂(E) [epsI₀]", "GCN", "Basic", "Scratch", "epsI_0_gcn"),
        ("epsI_0 | GraphSAGE", "ε₂(E) [epsI₀]", "GraphSAGE", "Basic", "Scratch", "epsI_0_graphsage"),
        ("epsI_0 | GINE scratch 11/16", "ε₂(E) [epsI₀]", "GINE scratch", "Basic", "Scratch", "epsI_0_baseline"),
        ("epsI_0 | GINE enhanced 19/24", "ε₂(E) [epsI₀]", "UE-GINE", "Enhanced", "Scratch", "epsI_0_enhanced"),
        ("epsI_0 | SSL-init baseline 11/16", "ε₂(E) [epsI₀]", "SSL-init GINE", "Basic", "SSL-pretrained", "epsI_0_ssl_init_baseline"),
        ("epsI_0 | Optimized SSL-init enhanced 19/24", "ε₂(E) [epsI₀]", "SSL-init UE-GINE", "Enhanced", "SSL-pretrained", "epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"),
        ("epsI_0 | SSL-init Gated UE-GINE", "ε₂(E) [epsI₀]", "SSL-init Gated UE-GINE", "Enhanced + Gated", "SSL-pretrained", "epsI_0_SSL_init_gated_UE_GINE_epsI_0_3seed"),

        # epsR_0
        ("epsR_0 | GCN", "ε₁(E) [epsR₀]", "GCN", "Basic", "Scratch", "epsR_0_gcn"),
        ("epsR_0 | GraphSAGE", "ε₁(E) [epsR₀]", "GraphSAGE", "Basic", "Scratch", "epsR_0_graphsage"),
        ("epsR_0 | GINE scratch 11/16", "ε₁(E) [epsR₀]", "GINE scratch", "Basic", "Scratch", "epsR_0_baseline"),
        ("epsR_0 | GINE enhanced 19/24", "ε₁(E) [epsR₀]", "UE-GINE", "Enhanced", "Scratch", "epsR_0_enhanced"),
        ("epsR_0 | SSL-init baseline 11/16", "ε₁(E) [epsR₀]", "SSL-init GINE", "Basic", "SSL-pretrained", "epsR_0_ssl_init_baseline"),
        ("epsR_0 | Optimized SSL-init enhanced 19/24", "ε₁(E) [epsR₀]", "SSL-init UE-GINE", "Enhanced", "SSL-pretrained", "epsR_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"),
        ("epsR_0 | SSL-init Gated UE-GINE", "ε₁(E) [epsR₀]", "SSL-init Gated UE-GINE", "Enhanced + Gated", "SSL-pretrained", "epsR_0_SSL_init_gated_UE_GINE_epsR_0_3seed"),

        # direct-alpha ablation
        ("alpha_log | Direct-alpha GINE scratch", "log-scaled α(E)", "Direct-alpha GINE", "Enhanced", "Scratch", "epsI_alpha_log10_1p_direct_alpha_enhanced_scratch_3seed"),
        ("alpha_log | Direct-alpha SSL-init", "log-scaled α(E)", "Direct-alpha SSL-init GINE", "Enhanced", "SSL-pretrained", "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed"),
    ]

    rows = []
    for key, target, model, graph, init, folder in models:
        files = list((ROOT / folder).glob("summary_optical_*.csv"))
        if not files:
            print("Missing summary:", folder)
            continue

        s = pd.read_csv(files[0])
        mae_values = s["test_MAE_mean"]

        rows.append({
            "R2_key": key,
            "Target": target,
            "Model": model,
            "Graph": graph,
            "Initialization": init,
            "MAE_m": mae_values.mean(),
            "MAE_s": mae_values.std(),
            "WR2_m": r2.loc[key, "weighted_R2_mean"] if key in r2.index else np.nan,
            "WR2_s": r2.loc[key, "weighted_R2_std"] if key in r2.index else np.nan,
            "MR2_m": r2.loc[key, "mean_R2_mean"] if key in r2.index else np.nan,
            "MR2_s": r2.loc[key, "mean_R2_std"] if key in r2.index else np.nan,
        })

    df = pd.DataFrame(rows)

    def _metric(mean, std, digits=3):
        if pd.isna(mean):
            return "N/A"
        if pd.isna(std):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    show = df[["Target", "Model", "Graph", "Initialization"]].copy()
    show["MAE ↓"] = df.apply(lambda r: _metric(r.MAE_m, r.MAE_s), axis=1)
    show["Weighted R² ↑"] = df.apply(lambda r: _metric(r.WR2_m, r.WR2_s), axis=1)
    show["Mean R² ↑"] = df.apply(lambda r: _metric(r.MR2_m, r.MR2_s), axis=1)

    main_targets = ["ε₂(E) [epsI₀]", "ε₁(E) [epsR₀]"]
    main_show = show[show["Target"].isin(main_targets)].reset_index(drop=True)
    main_raw = df[df["Target"].isin(main_targets)].reset_index(drop=True)
    alpha_show = show[show["Target"].eq("log-scaled α(E)")].reset_index(drop=True)
    alpha_raw = df[df["Target"].eq("log-scaled α(E)")].reset_index(drop=True)

    def _style(show_df, raw_df, caption):
        def highlight(row):
            css = [""] * len(row)
            raw = raw_df.loc[row.name]
            sub = raw_df[raw_df["Target"] == raw["Target"]]

            if raw["Model"] in ["UE-GINE", "SSL-init UE-GINE", "SSL-init Gated UE-GINE"]:
                css[1] = "font-weight: bold;"
            if np.isclose(raw["MAE_m"], sub["MAE_m"].min(), equal_nan=False):
                css[4] = "font-weight: bold; background-color: #fff2cc;"
            if not pd.isna(raw["WR2_m"]) and np.isclose(raw["WR2_m"], sub["WR2_m"].max(skipna=True)):
                css[5] = "font-weight: bold; background-color: #d9ead3;"
            if not pd.isna(raw["MR2_m"]) and np.isclose(raw["MR2_m"], sub["MR2_m"].max(skipna=True)):
                css[6] = "font-weight: bold; background-color: #d9ead3;"
            if raw["Model"] == "SSL-init Gated UE-GINE":
                css = [c + " border-top: 2px solid #666;" for c in css]
            return css

        return (
            show_df.style
            .apply(highlight, axis=1)
            .hide(axis="index")
            .set_caption(caption)
            .set_table_styles([
                {"selector": "caption", "props": "caption-side: top; font-weight: bold; font-size: 13pt;"},
                {"selector": "th", "props": "font-weight: bold; text-align: center; border-bottom: 1px solid black;"},
                {"selector": "td", "props": "text-align: center; padding: 5px 8px;"},
                {"selector": "td:nth-child(2)", "props": "text-align: left;"},
            ])
        )

    main_csv = OUT / "main_dielectric_first_benchmark_Q1_style_with_gated.csv"
    alpha_csv = OUT / "direct_alpha_ablation_Q1_style.csv"
    main_show.to_csv(main_csv, index=False, encoding="utf-8-sig")
    alpha_show.to_csv(alpha_csv, index=False, encoding="utf-8-sig")

    main_style = _style(main_show, main_raw, "Performance comparison of graph neural networks for dielectric-spectrum prediction")
    alpha_style = _style(alpha_show, alpha_raw, "Direct-alpha prediction ablation study")

    if display_tables and display is not None:
        display(main_style)
        display(alpha_style)

    missing_r2 = df[df[["WR2_m", "MR2_m"]].isna().any(axis=1)][["Target", "Model", "R2_key"]]
    if len(missing_r2) > 0:
        print("\nModels missing R2 values:")
        if display is not None:
            display(missing_r2)
        else:
            print(missing_r2)

    print("Saved:")
    print(main_csv)
    print(alpha_csv)

    return {
        "raw": df,
        "main_show": main_show,
        "alpha_show": alpha_show,
        "main_style": main_style,
        "alpha_style": alpha_style,
        "missing_r2": missing_r2,
        "main_csv": main_csv,
        "alpha_csv": alpha_csv,
    }


# ============================================================
# TASK3 benchmark R2 utilities
# ============================================================

def parse_task3_seed_from_name(name):
    """Extract seed number from a prediction filename such as test_predictions_seed2025.npz."""
    import re
    m = re.search(r"seed(\d+)", str(name))
    return int(m.group(1)) if m else None


def masked_weighted_r2_task3(y_true, y_pred, mask=None):
    """Compute weighted R2 and mean bin-wise R2 for full-spectrum prediction."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)

    if mask is None:
        mask = np.ones_like(y_true, dtype=np.float64)
    else:
        mask = np.asarray(mask, dtype=np.float64).reshape(len(y_true), -1)

    wsum = mask.sum(axis=0)
    valid_mean = wsum > 0

    mean_true = np.zeros(y_true.shape[1], dtype=np.float64)
    mean_true[valid_mean] = (
        y_true[:, valid_mean] * mask[:, valid_mean]
    ).sum(axis=0) / wsum[valid_mean]

    sse = (((y_true - y_pred) ** 2) * mask).sum(axis=0)
    sst = (((y_true - mean_true[None, :]) ** 2) * mask).sum(axis=0)
    valid = sst > 1e-12

    weighted_r2 = 1.0 - sse[valid].sum() / sst[valid].sum()

    r2_bins = np.full_like(sst, np.nan, dtype=np.float64)
    r2_bins[valid] = 1.0 - sse[valid] / sst[valid]
    mean_r2 = np.nanmean(r2_bins)

    return float(weighted_r2), float(mean_r2)


def get_task3_r2_model_folders(include_gated=True):
    """Folder map used to compute weighted R2 for all main TASK3 benchmark models."""
    folders = {
        # epsI_0
        "epsI_0 | GCN": "epsI_0_gcn",
        "epsI_0 | GraphSAGE": "epsI_0_graphsage",
        "epsI_0 | GINE scratch 11/16": "epsI_0_baseline",
        "epsI_0 | GINE enhanced 19/24": "epsI_0_enhanced",
        "epsI_0 | SSL-init baseline 11/16": "epsI_0_ssl_init_baseline",
        "epsI_0 | Optimized SSL-init enhanced 19/24": "epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed",

        # epsR_0
        "epsR_0 | GCN": "epsR_0_gcn",
        "epsR_0 | GraphSAGE": "epsR_0_graphsage",
        "epsR_0 | GINE scratch 11/16": "epsR_0_baseline",
        "epsR_0 | GINE enhanced 19/24": "epsR_0_enhanced",
        "epsR_0 | SSL-init baseline 11/16": "epsR_0_ssl_init_baseline",
        "epsR_0 | Optimized SSL-init enhanced 19/24": "epsR_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed",

        # direct-alpha
        "alpha_log | Direct-alpha GINE scratch": "epsI_alpha_log10_1p_direct_alpha_enhanced_scratch_3seed",
        "alpha_log | Direct-alpha SSL-init": "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed",
    }

    if include_gated:
        folders.update({
            "epsI_0 | SSL-init Gated UE-GINE": "epsI_0_SSL_init_gated_UE_GINE_epsI_0_3seed",
            "epsR_0 | SSL-init Gated UE-GINE": "epsR_0_SSL_init_gated_UE_GINE_epsR_0_3seed",
        })

    return folders


def compute_task3_weighted_r2_tables(
    root,
    out_dir=None,
    include_gated=True,
    display_tables=True,
):
    """
    Compute and save weighted R2 tables for TASK3 main benchmark models.

    Outputs
    -------
    weighted_r2_all_main_models_per_seed.csv
    weighted_r2_all_main_models_average.csv
    """
    from pathlib import Path

    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "final_benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    folders = get_task3_r2_model_folders(include_gated=include_gated)

    for label, folder_name in folders.items():
        folder = root / folder_name

        if not folder.exists():
            print("SKIP missing:", folder_name)
            continue

        pred_files = sorted(folder.glob("test_predictions*.npz"))
        if len(pred_files) == 0:
            print("SKIP no test prediction:", folder_name)
            continue

        for p in pred_files:
            z = np.load(p, allow_pickle=True)

            if not {"pred", "true"}.issubset(set(z.keys())):
                print("SKIP bad keys:", p.name, list(z.keys()))
                continue

            weighted_r2, mean_r2 = masked_weighted_r2_task3(
                y_true=z["true"],
                y_pred=z["pred"],
                mask=z["mask"] if "mask" in z.keys() else None,
            )

            rows.append({
                "label": label,
                "folder": folder_name,
                "seed": parse_task3_seed_from_name(p.name),
                "weighted_R2": weighted_r2,
                "mean_R2": mean_r2,
                "prediction_file": p.name,
            })

    r2_all_df = pd.DataFrame(rows)

    if len(r2_all_df) == 0:
        print("No valid prediction files found.")
        return {"per_seed": r2_all_df, "average": pd.DataFrame()}

    r2_all_df = r2_all_df.sort_values(["label", "seed"]).reset_index(drop=True)
    r2_avg_df = (
        r2_all_df
        .groupby("label")[["weighted_R2", "mean_R2"]]
        .agg(["mean", "std"])
        .round(6)
    )

    out_seed = out_dir / "weighted_r2_all_main_models_per_seed.csv"
    out_avg = out_dir / "weighted_r2_all_main_models_average.csv"

    r2_all_df.to_csv(out_seed, index=False)
    r2_avg_df.to_csv(out_avg)

    print("\n================ Weighted mean R2 | per seed ================")
    if display_tables and "display" in globals():
        display(r2_all_df[["label", "seed", "weighted_R2", "mean_R2"]].round(6))
    else:
        print(r2_all_df[["label", "seed", "weighted_R2", "mean_R2"]].round(6))

    print("\n================ Weighted mean R2 | average ================")
    if display_tables and "display" in globals():
        display(r2_avg_df)
    else:
        print(r2_avg_df)

    print("Saved per-seed:", out_seed)
    print("Saved average :", out_avg)

    return {
        "per_seed": r2_all_df,
        "average": r2_avg_df,
        "per_seed_csv": out_seed,
        "average_csv": out_avg,
    }
