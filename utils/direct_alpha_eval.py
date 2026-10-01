"""
GNNOpt-style direct-alpha evaluation utilities for TASK3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


def gnnopt_weighted_mean(
    spectra,
    energy,
):
    """
    Compute the GNNOpt-style energy-weighted mean:
        integral[W(E) * E dE] / integral[W(E) dE]
    """
    spectra = np.asarray(
        spectra,
        dtype=np.float64,
    )

    energy = np.asarray(
        energy,
        dtype=np.float64,
    )

    numerator = np.trapezoid(
        spectra
        * energy[None, :],
        energy,
        axis=1,
    )

    denominator = np.trapezoid(
        spectra,
        energy,
        axis=1,
    )

    result = np.full(
        spectra.shape[0],
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 1e-30)
    )

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return result


def run_gnnopt_style_direct_alpha_test(
    project_root=r"D:\TB3",
    npz_file=None,
    output_dir=None,
    expected_test_n=992,
    gnnopt_alpha_r2=0.93,
    gnnopt_alpha_relerr_lt10=0.79,
    show=True,
):
    """
    Evaluate held-out direct-alpha predictions using the GNNOpt-style metric.

    The stored model target is:
        y = log10(1 + alpha_cm^-1)

    The physical absorption coefficient is reconstructed as:
        alpha_cm^-1 = 10^y - 1

    No band-gap cleanup is applied.

    Returns
    -------
    dict
        Metrics, audit dataframe, arrays, figure, and saved output paths.
    """
    project_root = Path(project_root)

    root = (
        project_root
        / "processed"
        / "paired_training"
    )

    if npz_file is None:
        npz_file = (
            root
            / "paper_outputs"
            / "FINAL_HELDOUT_TEST_V6"
            / "final_heldout_test_predictions_direct_alpha.npz"
        )
    else:
        npz_file = Path(npz_file)

    if output_dir is None:
        output_dir = (
            root
            / "paper_outputs"
            / "FINAL_direct_alpha_SLME_500nm"
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not npz_file.exists():
        raise FileNotFoundError(
            f"Missing held-out test prediction file:\n{npz_file}"
        )

    data = np.load(
        npz_file,
        allow_pickle=True,
    )

    required_keys = [
        "log_alpha_true",
        "log_alpha_pred",
        "energy_ev",
    ]

    missing = [
        key
        for key in required_keys
        if key not in data.files
    ]

    if missing:
        raise KeyError(
            f"Missing keys: {missing}\n"
            f"Available keys: {data.files}"
        )

    log_alpha_true = np.asarray(
        data["log_alpha_true"],
        dtype=np.float64,
    )

    log_alpha_pred = np.asarray(
        data["log_alpha_pred"],
        dtype=np.float64,
    )

    energy_ev = np.asarray(
        data["energy_ev"],
        dtype=np.float64,
    )

    if (
        log_alpha_true.shape
        != log_alpha_pred.shape
    ):
        raise RuntimeError(
            "Reference/prediction shape mismatch."
        )

    if (
        log_alpha_true.ndim != 2
        or log_alpha_true.shape[0] != int(expected_test_n)
    ):
        raise RuntimeError(
            "Unexpected held-out test shape:\n"
            f"{log_alpha_true.shape}"
        )

    if (
        log_alpha_true.shape[1]
        != len(energy_ev)
    ):
        raise RuntimeError(
            "Energy-axis length mismatch."
        )

    if not np.all(
        np.diff(energy_ev) > 0
    ):
        raise RuntimeError(
            "Energy axis must be strictly increasing."
        )

    print("=" * 105)
    print("DIRECT-ALPHA GNNOpt-STYLE TEST — PHYSICAL ALPHA")
    print("=" * 105)
    print(
        "NPZ        :",
        npz_file,
    )
    print(
        "Test shape :",
        log_alpha_true.shape,
    )
    print(
        "Energy     :",
        f"{energy_ev.min():.3f} to "
        f"{energy_ev.max():.3f} eV",
    )

    alpha_true = (
        np.power(
            10.0,
            log_alpha_true,
        )
        - 1.0
    )

    alpha_pred = (
        np.power(
            10.0,
            log_alpha_pred,
        )
        - 1.0
    )

    alpha_true = np.maximum(
        alpha_true,
        0.0,
    )

    alpha_pred = np.maximum(
        alpha_pred,
        0.0,
    )

    print()
    print(
        "Reference alpha range:",
        f"{np.nanmin(alpha_true):.6e}",
        "to",
        f"{np.nanmax(alpha_true):.6e}",
        "cm^-1",
    )
    print(
        "Predicted alpha range:",
        f"{np.nanmin(alpha_pred):.6e}",
        "to",
        f"{np.nanmax(alpha_pred):.6e}",
        "cm^-1",
    )

    alpha_bar_true = gnnopt_weighted_mean(
        alpha_true,
        energy_ev,
    )

    alpha_bar_pred = gnnopt_weighted_mean(
        alpha_pred,
        energy_ev,
    )

    valid = (
        np.isfinite(alpha_bar_true)
        & np.isfinite(alpha_bar_pred)
    )

    n_valid = int(
        valid.sum()
    )

    if n_valid != int(expected_test_n):
        raise RuntimeError(
            f"Expected {expected_test_n} valid samples, "
            f"found {n_valid}."
        )

    y_true = alpha_bar_true[valid]
    y_pred = alpha_bar_pred[valid]

    r2_gnnopt_style = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    pearson = float(
        pearsonr(
            y_true,
            y_pred,
        ).statistic
    )

    mae_weighted_ev = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    relative_error = (
        np.abs(
            y_pred
            - y_true
        )
        /
        np.maximum(
            np.abs(y_true),
            1e-12,
        )
    )

    relerr_lt10 = float(
        np.mean(
            relative_error < 0.10
        )
    )

    flat_true = alpha_true.reshape(-1)
    flat_pred = alpha_pred.reshape(-1)

    flat_valid = (
        np.isfinite(flat_true)
        & np.isfinite(flat_pred)
    )

    r2_pointwise_physical = float(
        r2_score(
            flat_true[flat_valid],
            flat_pred[flat_valid],
        )
    )

    log_flat_true = (
        log_alpha_true.reshape(-1)
    )

    log_flat_pred = (
        log_alpha_pred.reshape(-1)
    )

    log_valid = (
        np.isfinite(log_flat_true)
        & np.isfinite(log_flat_pred)
    )

    r2_pointwise_log = float(
        r2_score(
            log_flat_true[log_valid],
            log_flat_pred[log_valid],
        )
    )

    delta_r2 = (
        r2_gnnopt_style
        - float(gnnopt_alpha_r2)
    )

    if delta_r2 > 0.005:
        assessment = "HIGHER than GNNOpt"
    elif delta_r2 < -0.005:
        assessment = "LOWER than GNNOpt"
    else:
        assessment = "COMPARABLE to GNNOpt"

    audit_df = pd.DataFrame(
        {
            "test_row":
                np.arange(
                    int(expected_test_n)
                ),

            "alpha_bar_true_eV":
                alpha_bar_true,

            "alpha_bar_pred_eV":
                alpha_bar_pred,

            "relative_error":
                np.abs(
                    alpha_bar_pred
                    - alpha_bar_true
                )
                /
                np.maximum(
                    np.abs(alpha_bar_true),
                    1e-12,
                ),
        }
    )

    audit_csv = (
        output_dir
        / "Direct_alpha_test992_GNNOpt_style_PHYSICAL_alpha_audit.csv"
    )

    audit_df.to_csv(
        audit_csv,
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(5.0, 5.0),
        dpi=250,
    )

    ax.scatter(
        y_true,
        y_pred,
        s=18,
        alpha=0.55,
    )

    lo = float(
        min(
            np.min(y_true),
            np.min(y_pred),
        )
    )

    hi = float(
        max(
            np.max(y_true),
            np.max(y_pred),
        )
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlim(
        lo,
        hi,
    )

    ax.set_ylim(
        lo,
        hi,
    )

    ax.set_xlabel(
        r"Reference weighted $\alpha$ (eV)",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Predicted weighted $\alpha$ (eV)",
        fontsize=12,
    )

    ax.text(
        0.05,
        0.94,
        rf"$R^2={r2_gnnopt_style:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )

    ax.text(
        0.05,
        0.86,
        rf"$N={n_valid}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    fig.tight_layout()

    fig_png = (
        output_dir
        / "Direct_alpha_test992_GNNOpt_style_PHYSICAL_alpha.png"
    )

    fig_pdf = (
        output_dir
        / "Direct_alpha_test992_GNNOpt_style_PHYSICAL_alpha.pdf"
    )

    fig.savefig(
        fig_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        fig_pdf,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print()
    print("=" * 105)
    print("FINAL GNNOpt-COMPARABLE RESULT — PHYSICAL ALPHA")
    print("=" * 105)
    print(
        "Test samples                   :",
        n_valid,
    )
    print(
        "Energy range                   :",
        f"{energy_ev.min():.3f}-"
        f"{energy_ev.max():.3f} eV",
    )

    print()
    print("-" * 105)
    print("PRIMARY — GNNOpt Eq. (7)")
    print("-" * 105)
    print(
        f"Our Direct-alpha GINE R2       : "
        f"{r2_gnnopt_style:.6f}"
    )
    print(
        f"GNNOpt published alpha R2      : "
        f"{float(gnnopt_alpha_r2):.6f}"
    )
    print(
        f"Delta R2                       : "
        f"{delta_r2:+.6f}"
    )
    print(
        f"Assessment                     : "
        f"{assessment}"
    )
    print(
        f"Pearson r                      : "
        f"{pearson:.6f}"
    )
    print(
        f"Weighted-mean MAE              : "
        f"{mae_weighted_ev:.6f} eV"
    )

    print()
    print("-" * 105)
    print("RELATIVE ERROR")
    print("-" * 105)
    print(
        f"Our |error| <10%               : "
        f"{100.0 * relerr_lt10:.2f}%"
    )
    print(
        f"GNNOpt published alpha         : "
        f"{100.0 * float(gnnopt_alpha_relerr_lt10):.2f}%"
    )

    print()
    print("-" * 105)
    print("SECONDARY DIAGNOSTICS")
    print("-" * 105)
    print(
        f"Physical-alpha pointwise R2    : "
        f"{r2_pointwise_physical:.6f}"
    )
    print(
        f"Log-target pointwise R2        : "
        f"{r2_pointwise_log:.6f}"
    )
    print(
        "Only the PRIMARY Eq. (7) R2 should be compared "
        "with GNNOpt R2=0.93."
    )

    print()
    print("=" * 105)
    print("SAVED")
    print("=" * 105)
    print(
        "Audit CSV:",
        audit_csv,
    )
    print(
        "PNG      :",
        fig_png,
    )
    print(
        "PDF      :",
        fig_pdf,
    )

    metrics = {
        "N_VALID":
            n_valid,

        "R2_GNNOPT_STYLE":
            r2_gnnopt_style,

        "PEARSON":
            pearson,

        "MAE_WEIGHTED_eV":
            mae_weighted_ev,

        "RELERR_LT10":
            relerr_lt10,

        "R2_POINTWISE_PHYSICAL":
            r2_pointwise_physical,

        "R2_POINTWISE_LOG":
            r2_pointwise_log,

        "DELTA_R2":
            delta_r2,

        "assessment":
            assessment,
    }

    return {
        "metrics": metrics,
        "audit_df": audit_df,
        "energy_ev": energy_ev,
        "log_alpha_true": log_alpha_true,
        "log_alpha_pred": log_alpha_pred,
        "alpha_true": alpha_true,
        "alpha_pred": alpha_pred,
        "alpha_bar_true": alpha_bar_true,
        "alpha_bar_pred": alpha_bar_pred,
        "figure": fig,
        "axis": ax,
        "audit_csv": audit_csv,
        "figure_png": fig_png,
        "figure_pdf": fig_pdf,
        "npz_file": npz_file,
        "output_dir": output_dir,
    }
