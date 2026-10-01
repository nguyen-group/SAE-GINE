"""
Publication figure for held-out Direct-alpha -> SLME validation.

Reads the saved sample-level SLME CSV and creates:
(a) reference vs predicted SLME with an explicit 35% ceiling
(b) top-k ranking preservation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_column(columns, include_terms):
    for column in list(columns):
        low = str(column).lower()

        if all(
            term.lower() in low
            for term in include_terms
        ):
            return column

    return None


def _compute_correlations(
    reference,
    prediction,
):
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_value = float(
            pearsonr(
                reference,
                prediction,
            )[0]
        )

        spearman_value = float(
            spearmanr(
                reference,
                prediction,
            )[0]
        )

    except Exception:
        pearson_value = float(
            np.corrcoef(
                reference,
                prediction,
            )[0, 1]
        )

        spearman_value = float(
            pd.Series(
                reference
            ).corr(
                pd.Series(
                    prediction
                ),
                method="spearman",
            )
        )

    return (
        pearson_value,
        spearman_value,
    )


def _compute_topk_overlap(
    reference,
    prediction,
    top_k_values,
):
    reference_order = np.argsort(
        -reference,
        kind="mergesort",
    )

    prediction_order = np.argsort(
        -prediction,
        kind="mergesort",
    )

    rows = []

    for k in top_k_values:
        k_effective = min(
            int(k),
            len(reference),
        )

        reference_top = set(
            reference_order[
                :k_effective
            ]
        )

        prediction_top = set(
            prediction_order[
                :k_effective
            ]
        )

        overlap_count = len(
            reference_top.intersection(
                prediction_top
            )
        )

        overlap_percent = (
            100.0
            * overlap_count
            / k_effective
        )

        rows.append(
            {
                "k":
                    k_effective,

                "overlap_count":
                    overlap_count,

                "overlap_percent":
                    overlap_percent,
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_final_slme_parity_topk(
    project_root=r"D:\TB3",
    samples_csv=None,
    output_dir=None,
    top_k_values=(10, 20, 50, 100),
    axis_max=36.5,
    ceiling_percent=35.0,
    show=True,
):
    """
    Create the final two-panel held-out SLME figure.

    Panel (a)
    ---------
    Reference vs predicted SLME with ideal y=x line and an explicit
    35% physical-ceiling visualization.

    Panel (b)
    ---------
    Top-k overlap between reference and predicted rankings.

    Returns the dataframe, metrics, top-k table, figure, axes, and paths.
    """
    project_root = Path(
        project_root
    )

    default_root = (
        project_root
        / "processed"
        / "paired_training"
        / "paper_outputs"
        / "FINAL_direct_alpha_SLME_500nm"
    )

    if output_dir is None:
        output_dir = default_root
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if samples_csv is None:
        samples_csv = (
            default_root
            / "FINAL_direct_alpha_SLME_test_samples.csv"
        )
    else:
        samples_csv = Path(
            samples_csv
        )

    if not samples_csv.exists():
        raise FileNotFoundError(
            f"Missing file:\n{samples_csv}"
        )

    dataframe = pd.read_csv(
        samples_csv
    )

    reference_column = (
        _find_column(
            dataframe.columns,
            [
                "reference",
                "slme",
            ],
        )
        or _find_column(
            dataframe.columns,
            [
                "reference",
                "eta",
            ],
        )
        or _find_column(
            dataframe.columns,
            [
                "reference",
            ],
        )
    )

    predicted_column = (
        _find_column(
            dataframe.columns,
            [
                "predicted",
                "slme",
            ],
        )
        or _find_column(
            dataframe.columns,
            [
                "predicted",
                "eta",
            ],
        )
        or _find_column(
            dataframe.columns,
            [
                "predicted",
            ],
        )
    )

    if (
        reference_column is None
        or predicted_column is None
    ):
        raise ValueError(
            "Could not identify reference/predicted SLME columns in the CSV.\n"
            f"Columns found:\n{list(dataframe.columns)}"
        )

    reference = pd.to_numeric(
        dataframe[
            reference_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    prediction = pd.to_numeric(
        dataframe[
            predicted_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            reference
        )
        & np.isfinite(
            prediction
        )
    )

    reference = (
        reference[
            valid
        ]
    )

    prediction = (
        prediction[
            valid
        ]
    )

    if reference.size < 2:
        raise RuntimeError(
            "Not enough valid samples to plot."
        )

    pearson_value, spearman_value = _compute_correlations(
        reference,
        prediction,
    )

    ss_res = float(
        np.sum(
            (
                reference
                - prediction
            )**2
        )
    )

    ss_tot = float(
        np.sum(
            (
                reference
                - np.mean(
                    reference
                )
            )**2
        )
    )

    r2_value = (
        1.0
        - ss_res
        / ss_tot
        if ss_tot > 0.0
        else np.nan
    )

    topk_df = _compute_topk_overlap(
        reference,
        prediction,
        top_k_values,
    )

    reference_max = float(
        np.max(
            reference
        )
    )

    predicted_max = float(
        np.max(
            prediction
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.8,
            4.7,
        ),
        dpi=300,
        constrained_layout=True,
    )

    ax = axes[
        0
    ]

    ax.axhspan(
        ceiling_percent,
        axis_max,
        color="lightgray",
        alpha=0.35,
        zorder=0,
    )

    ax.axvspan(
        ceiling_percent,
        axis_max,
        color="lightgray",
        alpha=0.35,
        zorder=0,
    )

    ax.scatter(
        reference,
        prediction,
        s=15,
        alpha=0.55,
        color="#5A9E6F",
        zorder=2,
    )

    ax.plot(
        [
            0.0,
            axis_max,
        ],
        [
            0.0,
            axis_max,
        ],
        "--",
        color="black",
        linewidth=1.4,
        zorder=3,
    )

    ax.axhline(
        ceiling_percent,
        linestyle=":",
        linewidth=1.4,
        color="crimson",
    )

    ax.axvline(
        ceiling_percent,
        linestyle=":",
        linewidth=1.4,
        color="crimson",
    )

    ax.text(
        ceiling_percent - 0.3,
        ceiling_percent + 0.55,
        f"{ceiling_percent:.0f}% ceiling",
        color="crimson",
        fontsize=9.5,
        ha="right",
    )

    ax.text(
        0.06,
        0.93,
        (
            f"Pearson = {pearson_value:.3f}\n"
            f"Spearman = {spearman_value:.3f}\n"
            rf"$R^2$ = {r2_value:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
    )

    ax.text(
        0.06,
        0.72,
        (
            rf"$\eta_{{\rm ref,max}}$ = {reference_max:.2f}%"
            "\n"
            rf"$\eta_{{\rm pred,max}}$ = {predicted_max:.2f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
    )

    ax.set_xlim(
        0.0,
        axis_max,
    )

    ax.set_ylim(
        0.0,
        axis_max,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        r"Reference SLME, $\eta$ (%)",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Predicted SLME, $\eta$ (%)",
        fontsize=12,
    )

    ax.text(
        -0.15,
        1.04,
        "a",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[
        1
    ]

    ax.plot(
        topk_df[
            "k"
        ],
        topk_df[
            "overlap_percent"
        ],
        marker="o",
        markersize=5.8,
        linewidth=1.8,
        color="#D95F02",
    )

    for _, row in topk_df.iterrows():
        ax.text(
            row[
                "k"
            ],
            row[
                "overlap_percent"
            ]
            + 2.5,
            (
                f"{int(row['overlap_count'])}/"
                f"{int(row['k'])}"
            ),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlim(
        5,
        105,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xticks(
        list(
            top_k_values
        )
    )

    ax.set_xlabel(
        r"Top-$k$ materials",
        fontsize=12,
    )

    ax.set_ylabel(
        r"Overlap with reference top-$k$ (%)",
        fontsize=12,
    )

    ax.text(
        -0.13,
        1.04,
        "b",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    for axis in axes:
        axis.tick_params(
            labelsize=10.5,
        )

        for spine in axis.spines.values():
            spine.set_linewidth(
                1.0
            )

    output_png = (
        output_dir
        / "FINAL_FIG_SLME_parity_topk_with_35pct_ceiling.png"
    )

    output_pdf = (
        output_dir
        / "FINAL_FIG_SLME_parity_topk_with_35pct_ceiling.pdf"
    )

    output_svg = (
        output_dir
        / "FINAL_FIG_SLME_parity_topk_with_35pct_ceiling.svg"
    )

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        output_svg,
        bbox_inches="tight",
        facecolor="white",
    )

    if show:
        plt.show()
    else:
        plt.close(
            fig
        )

    print(
        "Reference maximum :",
        f"{reference_max:.3f}%",
    )

    print(
        "Predicted maximum :",
        f"{predicted_max:.3f}%",
    )

    print(
        "Reference > 35%   :",
        int(
            np.sum(
                reference
                > ceiling_percent
            )
        ),
    )

    print(
        "Predicted > 35%   :",
        int(
            np.sum(
                prediction
                > ceiling_percent
            )
        ),
    )

    print()
    print("Saved:")
    print(
        output_png
    )
    print(
        output_pdf
    )
    print(
        output_svg
    )

    return {
        "dataframe":
            dataframe,

        "reference_column":
            reference_column,

        "predicted_column":
            predicted_column,

        "reference":
            reference,

        "prediction":
            prediction,

        "topk_df":
            topk_df,

        "metrics": {
            "Pearson":
                pearson_value,

            "Spearman":
                spearman_value,

            "R2":
                float(
                    r2_value
                ),

            "reference_max_percent":
                reference_max,

            "predicted_max_percent":
                predicted_max,

            "reference_gt_ceiling":
                int(
                    np.sum(
                        reference
                        > ceiling_percent
                    )
                ),

            "predicted_gt_ceiling":
                int(
                    np.sum(
                        prediction
                        > ceiling_percent
                    )
                ),
        },

        "figure":
            fig,

        "axes":
            axes,

        "output_png":
            output_png,

        "output_pdf":
            output_pdf,

        "output_svg":
            output_svg,
    }



def plot_final_slme_parity_topk_journal(
    project_root=r"D:\TB3",
    samples_csv=None,
    output_dir=None,
    top_k_values=(10, 20, 50, 100),
    show=True,
):
    """
    Create the journal-style two-panel held-out SLME figure.

    Panel (a): reference vs predicted SLME.
    Panel (b): top-k ranking preservation.
    """
    project_root = Path(
        project_root
    )

    root = (
        project_root
        / "processed"
        / "paired_training"
        / "paper_outputs"
        / "FINAL_direct_alpha_SLME_500nm"
    )

    if samples_csv is None:
        samples_csv = (
            root
            / "FINAL_direct_alpha_SLME_test_samples.csv"
        )
    else:
        samples_csv = Path(
            samples_csv
        )

    if output_dir is None:
        output_dir = root
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not samples_csv.exists():
        raise FileNotFoundError(
            f"Missing file:\n{samples_csv}"
        )

    dataframe = pd.read_csv(
        samples_csv
    )

    required_columns = [
        "reference_SLME_percent",
        "predicted_SLME_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    reference = dataframe[
        "reference_SLME_percent"
    ].to_numpy(
        dtype=float
    )

    prediction = dataframe[
        "predicted_SLME_percent"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            reference
        )
        & np.isfinite(
            prediction
        )
    )

    reference = (
        reference[
            valid
        ]
    )

    prediction = (
        prediction[
            valid
        ]
    )

    if len(
        reference
    ) < 2:
        raise RuntimeError(
            "Not enough valid samples to plot."
        )

    mae = float(
        np.mean(
            np.abs(
                prediction
                - reference
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    - reference
                )**2
            )
        )
    )

    ss_res = float(
        np.sum(
            (
                prediction
                - reference
            )**2
        )
    )

    ss_tot = float(
        np.sum(
            (
                reference
                - reference.mean()
            )**2
        )
    )

    r2 = (
        1.0
        - ss_res
        / ss_tot
        if ss_tot > 0.0
        else np.nan
    )

    top_k = np.asarray(
        top_k_values,
        dtype=int,
    )

    reference_rank = np.argsort(
        -reference,
        kind="mergesort",
    )

    prediction_rank = np.argsort(
        -prediction,
        kind="mergesort",
    )

    overlap = np.asarray(
        [
            len(
                set(
                    reference_rank[
                        :k
                    ]
                )
                & set(
                    prediction_rank[
                        :k
                    ]
                )
            )
            for k in top_k
        ],
        dtype=int,
    )

    overlap_percent = (
        100.0
        * overlap
        / top_k
    )

    rc = {
        "font.family":
            "Arial",

        "font.size":
            13,

        "axes.labelsize":
            15,

        "xtick.labelsize":
            12.5,

        "ytick.labelsize":
            12.5,

        "axes.linewidth":
            1.15,

        "xtick.major.width":
            1.15,

        "ytick.major.width":
            1.15,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    }

    with plt.rc_context(
        rc
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                10.8,
                5.3,
            ),
            dpi=300,
            constrained_layout=True,
        )

        ax = axes[
            0
        ]

        axis_max = 35.0

        ax.scatter(
            reference,
            prediction,
            s=18,
            alpha=0.62,
            edgecolors="none",
        )

        ax.plot(
            [
                0.0,
                axis_max,
            ],
            [
                0.0,
                axis_max,
            ],
            "--",
            linewidth=1.6,
        )

        ax.text(
            0.06,
            0.94,
            (
                rf"$R^2$ = {r2:.3f}"
                "\n"
                rf"MAE = {mae:.2f} pp"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13.5,
        )

        ax.set(
            xlim=(
                0.0,
                35.0,
            ),
            ylim=(
                0.0,
                35.0,
            ),
            xticks=np.arange(
                0,
                36,
                5,
            ),
            yticks=np.arange(
                0,
                36,
                5,
            ),
        )

        ax.set_box_aspect(
            1
        )

        ax.set_xlabel(
            r"Reference SLME, $\eta$ (%)"
        )

        ax.set_ylabel(
            r"Predicted SLME, $\eta$ (%)"
        )

        ax.text(
            -0.145,
            1.035,
            "a",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )

        ax = axes[
            1
        ]

        ax.plot(
            top_k,
            overlap_percent,
            marker="o",
            markersize=7,
            linewidth=2.0,
        )

        for k, count, percent in zip(
            top_k,
            overlap,
            overlap_percent,
        ):
            dx = (
                -3
                if k == 100
                else 0
            )

            ax.annotate(
                f"{count}/{k}",
                (
                    k,
                    percent,
                ),
                xytext=(
                    dx,
                    8,
                ),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12.5,
            )

        ax.set(
            xlim=(
                5,
                105,
            ),
            ylim=(
                0,
                100,
            ),
            xticks=top_k,
            yticks=np.arange(
                0,
                101,
                20,
            ),
        )

        ax.set_box_aspect(
            1
        )

        ax.set_xlabel(
            r"Top-$k$ materials"
        )

        ax.set_ylabel(
            r"Reference top-$k$ retained (%)"
        )

        ax.text(
            -0.145,
            1.035,
            "b",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )

        for axis in axes:
            axis.tick_params(
                direction="out",
                width=1.15,
                length=5,
                labelsize=12.5,
                pad=5,
            )

            for spine in axis.spines.values():
                spine.set_linewidth(
                    1.15
                )

        output_base = (
            output_dir
            / "FINAL_FIG_SLME_parity_topk_journal"
        )

        output_paths = {}

        for extension in (
            "png",
            "pdf",
            "svg",
        ):
            output_path = Path(
                f"{output_base}.{extension}"
            )

            kwargs = {
                "bbox_inches":
                    "tight",

                "facecolor":
                    "white",
            }

            if extension == "png":
                kwargs[
                    "dpi"
                ] = 600

            fig.savefig(
                output_path,
                **kwargs,
            )

            output_paths[
                extension
            ] = output_path

        if show:
            plt.show()
        else:
            plt.close(
                fig
            )

    print(
        f"N    : {len(reference)}"
    )
    print(
        f"R2   : {r2:.4f}"
    )
    print(
        f"MAE  : {mae:.4f} pp"
    )
    print(
        f"RMSE : {rmse:.4f} pp"
    )

    for k, count, percent in zip(
        top_k,
        overlap,
        overlap_percent,
    ):
        print(
            f"Top-{k}: {count}/{k} ({percent:.1f}%)"
        )

    print()
    print("Saved:")

    for extension in (
        "png",
        "pdf",
        "svg",
    ):
        print(
            output_paths[
                extension
            ]
        )

    return {
        "dataframe":
            dataframe,

        "reference":
            reference,

        "prediction":
            prediction,

        "top_k":
            top_k,

        "overlap":
            overlap,

        "overlap_percent":
            overlap_percent,

        "metrics": {
            "N":
                int(
                    len(
                        reference
                    )
                ),

            "R2":
                float(
                    r2
                ),

            "MAE_pp":
                mae,

            "RMSE_pp":
                rmse,
        },

        "figure":
            fig,

        "axes":
            axes,

        "output_png":
            output_paths[
                "png"
            ],

        "output_pdf":
            output_paths[
                "pdf"
            ],

        "output_svg":
            output_paths[
                "svg"
            ],
    }
