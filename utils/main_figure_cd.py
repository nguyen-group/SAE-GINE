"""
Final main Figure panels (c) and (d) for TASK3.
"""

from pathlib import Path
import io
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from matplotlib.lines import Line2D


Q = 1.602176634e-19
H = 6.62607015e-34
HBAR = H / (2.0 * np.pi)
C = 2.99792458e8


DEFAULT_SELECTED_IDS = (
    "mp-btzq",   # NaP5
    "mp-cqpjs",  # Rb2As2Pd
    "mp-ckxsa",  # Ba(GaP)2
    "mp-oei",    # KSbSe2
)


def formula_mathtext(
    formula,
):
    """
    Convert numeric formula subscripts to Matplotlib mathtext.
    """
    return re.sub(
        r"(\d+)",
        r"$_{\1}$",
        str(formula),
    )


def filename_material_id(
    filename,
):
    """
    Extract a Materials Project material ID from a DFT filename.
    """
    text = (
        Path(filename)
        .name
        .lower()
        .replace(
            "_",
            "-",
        )
    )

    match = re.search(
        r"mp-[a-z0-9]+",
        text,
    )

    return (
        match.group(0)
        if match
        else None
    )


def epsilon_to_alpha_cm1(
    energy_eV,
    eps1,
    eps2,
):
    """
    Convert dielectric functions to absorption coefficient in cm^-1.
    """
    energy_eV = np.asarray(
        energy_eV,
        dtype=np.float64,
    )

    eps1 = np.asarray(
        eps1,
        dtype=np.float64,
    )

    eps2 = np.asarray(
        eps2,
        dtype=np.float64,
    )

    omega = (
        energy_eV
        * Q
        / HBAR
    )

    inner = (
        np.sqrt(
            eps1**2
            + eps2**2
        )
        - eps1
    )

    inner = np.maximum(
        inner,
        0.0,
    )

    alpha_m1 = (
        np.sqrt(2.0)
        * omega
        / C
        * np.sqrt(
            inner
        )
    )

    return (
        alpha_m1
        / 100.0
    )


def run_final_main_figure_cd(
    project_root=r"D:\TB3",
    dft_zip=r"D:\DFT_TEST\collect_data_epsilon.zip",
    ml_npz=None,
    slme_csv=None,
    output_dir=None,
    selected_ids=DEFAULT_SELECTED_IDS,
    energy_min=0.0,
    energy_max=4.0,
    slme_min=29.9,
    slme_max=34.0,
    discovery_threshold=30.0,
    ml_color="#2A72B5",
    dft_color="#D62728",
    show=True,
):
    """
    Generate the final journal-ready Figure panels (c) and (d).

    Panel (c)
    ---------
    SLME-guided DFT validation using the complete external DFT batch and the
    pre-defined 30% discovery threshold.

    Panel (d)
    ---------
    Four representative RAW spectral comparisons:
    - frozen SSL-GINE ``predicted_alpha_cm1``;
    - trace-averaged DFT dielectric response converted to alpha(E);
    - 0-4 eV by default;
    - no MP-gap cleanup;
    - shared y-axis limit across all four spectra.
    """
    project_root = Path(
        project_root
    )

    root = (
        project_root
        / "processed"
        / "paired_training"
    )

    screen_dir = (
        root
        / "paper_outputs"
        / "screening_SLME_direct_alpha_500nm"
    )

    if ml_npz is None:
        ml_npz = (
            screen_dir
            / "zintl_external_1100_direct_alpha_SLME_spectra.npz"
        )
    else:
        ml_npz = Path(
            ml_npz
        )

    if slme_csv is None:
        slme_csv = (
            root
            / "paper_outputs"
            / "DFT_validation"
            / "FINAL_DFT22_SLME"
            / "FINAL_DFT22_vs_ML_SLME.csv"
        )
    else:
        slme_csv = Path(
            slme_csv
        )

    dft_zip = Path(
        dft_zip
    )

    if output_dir is None:
        output_dir = (
            root
            / "paper_outputs"
            / "DFT_validation"
            / "FINAL_panel_c_d_AM_style"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_png = (
        output_dir
        / "FINAL_panel_c_d_AM_style.png"
    )

    out_pdf = (
        output_dir
        / "FINAL_panel_c_d_AM_style.pdf"
    )

    out_svg = (
        output_dir
        / "FINAL_panel_c_d_AM_style.svg"
    )

    for path in [
        dft_zip,
        ml_npz,
        slme_csv,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    selected_ids = tuple(
        str(material_id)
        for material_id in selected_ids
    )

    if len(selected_ids) != 4:
        raise ValueError(
            "Panel (d) requires exactly four representative material IDs."
        )

    # ------------------------------------------------------------------
    # Load frozen SLME validation results for panel (c) and metadata.
    # ------------------------------------------------------------------
    slme_df = pd.read_csv(
        slme_csv
    )

    slme_df[
        "material_id"
    ] = (
        slme_df[
            "material_id"
        ]
        .astype(str)
        .str.strip()
    )

    required_cols = [
        "source_rank",
        "material_id",
        "formula_pretty",
        "ML_SLME_percent",
        "DFT_SLME_percent",
    ]

    missing = [
        column
        for column in required_cols
        if column not in slme_df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing SLME columns: {missing}"
        )

    if slme_df[
        "material_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate material_id in SLME validation table."
        )

    if "abs_error_pp" not in slme_df.columns:
        slme_df[
            "abs_error_pp"
        ] = np.abs(
            slme_df[
                "ML_SLME_percent"
            ]
            - slme_df[
                "DFT_SLME_percent"
            ]
        )

    n_dft = len(
        slme_df
    )

    n_gt30 = int(
        (
            slme_df[
                "DFT_SLME_percent"
            ]
            > float(
                discovery_threshold
            )
        ).sum()
    )

    mae_slme = float(
        slme_df[
            "abs_error_pp"
        ].mean()
    )

    # ------------------------------------------------------------------
    # Load frozen RAW SSL-GINE direct-alpha predictions.
    # ------------------------------------------------------------------
    npz = np.load(
        ml_npz,
        allow_pickle=True,
    )

    required_npz = [
        "material_id",
        "energy_eV",
        "predicted_alpha_cm1",
    ]

    missing_npz = [
        key
        for key in required_npz
        if key not in npz.files
    ]

    if missing_npz:
        raise KeyError(
            f"Missing ML NPZ keys: {missing_npz}\n"
            f"Available keys: {npz.files}"
        )

    ml_ids = np.asarray(
        npz[
            "material_id"
        ]
    ).astype(str)

    energy_ml = np.asarray(
        npz[
            "energy_eV"
        ],
        dtype=np.float64,
    )

    alpha_ml_raw_all = np.asarray(
        npz[
            "predicted_alpha_cm1"
        ],
        dtype=np.float64,
    )

    if (
        alpha_ml_raw_all.ndim != 2
        or alpha_ml_raw_all.shape[0] != len(ml_ids)
        or alpha_ml_raw_all.shape[1] != len(energy_ml)
    ):
        raise RuntimeError(
            "Unexpected ML spectral dimensions: "
            f"ids={len(ml_ids)}, "
            f"energy={len(energy_ml)}, "
            f"alpha={alpha_ml_raw_all.shape}"
        )

    id_to_ml_idx = {
        material_id: index
        for index, material_id
        in enumerate(
            ml_ids
        )
    }

    # ------------------------------------------------------------------
    # Index DFT files.
    # ------------------------------------------------------------------
    with zipfile.ZipFile(
        dft_zip,
        "r",
    ) as archive:
        dat_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(
                ".dat"
            )
        ]

    dft_file_map = {}

    for filename in dat_files:
        material_id = filename_material_id(
            filename
        )

        if material_id is not None:
            dft_file_map[
                material_id
            ] = filename

    # ------------------------------------------------------------------
    # Load the four representative RAW DFT spectra.
    # ------------------------------------------------------------------
    spectral_records = []

    with zipfile.ZipFile(
        dft_zip,
        "r",
    ) as archive:
        for material_id in selected_ids:
            if material_id not in dft_file_map:
                raise KeyError(
                    f"Missing DFT file: {material_id}"
                )

            if material_id not in id_to_ml_idx:
                raise KeyError(
                    f"Missing ML spectrum: {material_id}"
                )

            matching_rows = slme_df.loc[
                slme_df[
                    "material_id"
                ]
                == material_id
            ]

            if len(
                matching_rows
            ) != 1:
                raise KeyError(
                    f"Expected exactly one SLME row for {material_id}, "
                    f"found {len(matching_rows)}."
                )

            array = np.loadtxt(
                io.BytesIO(
                    archive.read(
                        dft_file_map[
                            material_id
                        ]
                    )
                )
            )

            array = np.asarray(
                array,
                dtype=np.float64,
            )

            if (
                array.ndim != 2
                or array.shape[1] < 7
            ):
                raise RuntimeError(
                    f"Unexpected DFT shape for "
                    f"{material_id}: {array.shape}"
                )

            energy_dft = np.asarray(
                array[
                    :,
                    0
                ],
                dtype=np.float64,
            )

            eps1_avg = np.mean(
                array[
                    :,
                    1:4
                ],
                axis=1,
            )

            eps2_avg = np.mean(
                array[
                    :,
                    4:7
                ],
                axis=1,
            )

            alpha_dft_raw = (
                epsilon_to_alpha_cm1(
                    energy_dft,
                    eps1_avg,
                    eps2_avg,
                )
            )

            alpha_ml_raw = (
                alpha_ml_raw_all[
                    id_to_ml_idx[
                        material_id
                    ]
                ]
            )

            row = matching_rows.iloc[
                0
            ]

            rank = int(
                row[
                    "source_rank"
                ]
            )

            formula = str(
                row[
                    "formula_pretty"
                ]
            )

            mask_ml = (
                (
                    energy_ml
                    >= float(
                        energy_min
                    )
                )
                &
                (
                    energy_ml
                    <= float(
                        energy_max
                    )
                )
            )

            energy_common = (
                energy_ml[
                    mask_ml
                ]
            )

            alpha_ml_window = (
                alpha_ml_raw[
                    mask_ml
                ]
            )

            alpha_dft_interp = np.interp(
                energy_common,
                energy_dft,
                alpha_dft_raw,
            )

            pearson_r = float(
                pearsonr(
                    alpha_dft_interp,
                    alpha_ml_window,
                ).statistic
            )

            mask_dft = (
                (
                    energy_dft
                    >= float(
                        energy_min
                    )
                )
                &
                (
                    energy_dft
                    <= float(
                        energy_max
                    )
                )
            )

            spectral_records.append(
                {
                    "material_id":
                        material_id,

                    "rank":
                        rank,

                    "formula":
                        formula,

                    "r":
                        pearson_r,

                    "energy_ml":
                        energy_common,

                    "alpha_ml":
                        alpha_ml_window,

                    "energy_dft":
                        energy_dft[
                            mask_dft
                        ],

                    "alpha_dft":
                        alpha_dft_raw[
                            mask_dft
                        ],
                }
            )

    # ------------------------------------------------------------------
    # Common y-scale across all four panel-(d) spectra.
    # ------------------------------------------------------------------
    all_alpha_scaled = []

    for record in spectral_records:
        all_alpha_scaled.append(
            record[
                "alpha_ml"
            ]
            / 1e6
        )

        all_alpha_scaled.append(
            record[
                "alpha_dft"
            ]
            / 1e6
        )

    global_ymax = float(
        np.nanmax(
            np.concatenate(
                all_alpha_scaled
            )
        )
    )

    common_ymax = (
        np.ceil(
            global_ymax
            * 10.0
        )
        / 10.0
    )

    common_ymax += 0.05

    print(
        "Common spectral y-limit:",
        f"0 – {common_ymax:.2f}",
    )

    # ------------------------------------------------------------------
    # Journal-style figure.
    # ------------------------------------------------------------------
    rc_params = {
        "font.family":
            "Arial",

        "font.size":
            8.0,

        "axes.linewidth":
            0.85,

        "xtick.major.width":
            0.85,

        "ytick.major.width":
            0.85,

        "xtick.major.size":
            3.6,

        "ytick.major.size":
            3.6,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    }

    with plt.rc_context(
        rc_params
    ):
        fig = plt.figure(
            figsize=(
                10.8,
                6.4,
            ),
            dpi=220,
        )

        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[
                1,
                1,
                1,
            ],
            height_ratios=[
                1,
                1,
            ],
            left=0.075,
            right=0.985,
            bottom=0.105,
            top=0.88,
            wspace=0.34,
            hspace=0.38,
        )

        ax_c = fig.add_subplot(
            grid[
                0,
                0,
            ]
        )

        ax_blank = fig.add_subplot(
            grid[
                1,
                0,
            ]
        )

        ax_blank.axis(
            "off"
        )

        spectral_axes = [
            fig.add_subplot(
                grid[
                    0,
                    1,
                ]
            ),
            fig.add_subplot(
                grid[
                    0,
                    2,
                ]
            ),
            fig.add_subplot(
                grid[
                    1,
                    1,
                ]
            ),
            fig.add_subplot(
                grid[
                    1,
                    2,
                ]
            ),
        ]

        ax_c.set_box_aspect(
            1
        )

        for ax in spectral_axes:
            ax.set_box_aspect(
                1
            )

        # --------------------------------------------------------------
        # Panel (c): SLME-guided DFT validation.
        # --------------------------------------------------------------
        x = slme_df[
            "DFT_SLME_percent"
        ].to_numpy(
            dtype=np.float64
        )

        y = slme_df[
            "ML_SLME_percent"
        ].to_numpy(
            dtype=np.float64
        )

        ax_c.scatter(
            x,
            y,
            s=8,
            color=ml_color,
            alpha=0.90,
            edgecolors="none",
            zorder=3,
        )

        ax_c.plot(
            [
                float(
                    slme_min
                ),
                float(
                    slme_max
                ),
            ],
            [
                float(
                    slme_min
                ),
                float(
                    slme_max
                ),
            ],
            linestyle="--",
            linewidth=1.05,
            color=ml_color,
            zorder=1,
        )

        ax_c.axvline(
            float(
                discovery_threshold
            ),
            linestyle=":",
            linewidth=0.90,
            color=ml_color,
        )

        ax_c.axhline(
            float(
                discovery_threshold
            ),
            linestyle=":",
            linewidth=0.90,
            color=ml_color,
        )

        ax_c.set_xlim(
            float(
                slme_min
            ),
            float(
                slme_max
            ),
        )

        ax_c.set_ylim(
            float(
                slme_min
            ),
            float(
                slme_max
            ),
        )

        ax_c.set_xlabel(
            "DFT-spectrum SLME (%)",
            fontsize=8.5,
        )

        ax_c.set_ylabel(
            "ML-predicted SLME (%)",
            fontsize=8.5,
        )

        ax_c.tick_params(
            labelsize=7.3,
        )

        ax_c.text(
            0.07,
            0.92,
            (
                f"{n_gt30}/{n_dft} DFT > "
                f"{float(discovery_threshold):g}%\n"
                f"MAE = {mae_slme:.2f} pp"
            ),
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            linespacing=1.20,
        )

        # --------------------------------------------------------------
        # Panel (d): four RAW external spectral comparisons.
        # --------------------------------------------------------------
        for index, (
            ax,
            record,
        ) in enumerate(
            zip(
                spectral_axes,
                spectral_records,
            )
        ):
            energy_ml_plot = (
                record[
                    "energy_ml"
                ]
            )

            alpha_ml_plot = (
                record[
                    "alpha_ml"
                ]
                / 1e6
            )

            energy_dft_plot = (
                record[
                    "energy_dft"
                ]
            )

            alpha_dft_plot = (
                record[
                    "alpha_dft"
                ]
                / 1e6
            )

            ax.plot(
                energy_ml_plot,
                alpha_ml_plot,
                color=ml_color,
                linewidth=1.65,
                solid_capstyle="round",
            )

            ax.plot(
                energy_dft_plot,
                alpha_dft_plot,
                color=dft_color,
                linewidth=1.65,
                linestyle=(
                    0,
                    (
                        7,
                        5,
                    )
                ),
                dash_capstyle="butt",
            )

            ax.set_xlim(
                float(
                    energy_min
                ),
                float(
                    energy_max
                ),
            )

            ax.set_ylim(
                0.0,
                common_ymax,
            )

            formula_tex = formula_mathtext(
                record[
                    "formula"
                ]
            )

            ax.set_title(
                (
                    f"Rank {record['rank']} | "
                    f"{formula_tex}"
                ),
                fontsize=8.3,
                pad=3.0,
                fontweight="normal",
            )

            ax.text(
                0.045,
                0.91,
                rf"$r$ = {record['r']:.2f}",
                transform=ax.transAxes,
                fontsize=7.4,
                ha="left",
                va="top",
            )

            ax.tick_params(
                labelsize=7.2,
            )

            if index in [
                0,
                2,
            ]:
                ax.set_ylabel(
                    r"$\alpha$ "
                    r"($\times10^6$ cm$^{-1}$)",
                    fontsize=8.3,
                )

            if index in [
                2,
                3,
            ]:
                ax.set_xlabel(
                    "Photon Energy (eV)",
                    fontsize=8.3,
                )

        # --------------------------------------------------------------
        # Panel labels and shared legend.
        # --------------------------------------------------------------
        bbox_c = (
            ax_c.get_position()
        )

        bbox_d1 = (
            spectral_axes[
                0
            ]
            .get_position()
        )

        fig.text(
            bbox_c.x0
            - 0.040,
            bbox_c.y1
            + 0.038,
            "c",
            fontsize=21,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        fig.text(
            bbox_d1.x0
            - 0.032,
            bbox_d1.y1
            + 0.038,
            "d",
            fontsize=21,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=ml_color,
                linewidth=1.8,
                label="SSL-GINE",
            ),
            Line2D(
                [0],
                [0],
                color=dft_color,
                linewidth=1.8,
                linestyle=(
                    0,
                    (
                        7,
                        5,
                    )
                ),
                label="DFT",
            ),
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(
                0.755,
                0.965,
            ),
            ncol=2,
            frameon=False,
            fontsize=9.4,
            handlelength=2.7,
            columnspacing=1.8,
        )

        fig.savefig(
            out_png,
            dpi=600,
            bbox_inches="tight",
        )

        fig.savefig(
            out_pdf,
            bbox_inches="tight",
        )

        fig.savefig(
            out_svg,
            bbox_inches="tight",
        )

        if show:
            plt.show()
        else:
            plt.close(
                fig
            )

    print()
    print("=" * 96)
    print("FINAL MAIN FIGURE — PANELS c+d")
    print("=" * 96)
    print(
        "Panel (c)"
    )
    print(
        f"  External DFT samples : "
        f"{n_dft}"
    )
    print(
        f"  DFT SLME >"
        f"{float(discovery_threshold):g}%"
        f"        : "
        f"{n_gt30}/{n_dft}"
    )
    print(
        f"  SLME MAE             : "
        f"{mae_slme:.4f} pp"
    )

    print()
    print(
        "Panel (d)"
    )
    print(
        "  ML alpha             : "
        "predicted_alpha_cm1 [RAW]"
    )
    print(
        "  DFT reference        : "
        "trace-averaged epsilon"
    )
    print(
        "  MP-gap cleanup       : NONE"
    )
    print(
        "  Spectral window      : "
        f"{float(energy_min):g}–"
        f"{float(energy_max):g} eV"
    )
    print(
        "  Shared y-limit       : "
        f"0–{common_ymax:.2f} ×10^6 cm^-1"
    )

    for record in spectral_records:
        print(
            f"  Rank {record['rank']:>2} | "
            f"{record['formula']:12s} | "
            f"r = {record['r']:.4f}"
        )

    print()
    print(
        "Saved PNG:",
        out_png,
    )
    print(
        "Saved PDF:",
        out_pdf,
    )
    print(
        "Saved SVG:",
        out_svg,
    )

    spectral_summary = pd.DataFrame(
        [
            {
                "material_id":
                    record[
                        "material_id"
                    ],

                "rank":
                    record[
                        "rank"
                    ],

                "formula":
                    record[
                        "formula"
                    ],

                "pearson_r":
                    record[
                        "r"
                    ],
            }
            for record in spectral_records
        ]
    )

    metrics = {
        "N_DFT":
            int(
                n_dft
            ),

        "N_DFT_above_threshold":
            int(
                n_gt30
            ),

        "discovery_threshold_percent":
            float(
                discovery_threshold
            ),

        "SLME_MAE_pp":
            float(
                mae_slme
            ),

        "spectral_common_ymax_x1e6_cm-1":
            float(
                common_ymax
            ),
    }

    return {
        "metrics":
            metrics,

        "slme_df":
            slme_df,

        "spectral_records":
            spectral_records,

        "spectral_summary":
            spectral_summary,

        "figure":
            fig,

        "ax_c":
            ax_c,

        "spectral_axes":
            spectral_axes,

        "output_png":
            out_png,

        "output_pdf":
            out_pdf,

        "output_svg":
            out_svg,

        "ml_npz":
            ml_npz,

        "slme_csv":
            slme_csv,

        "dft_zip":
            dft_zip,
    }
