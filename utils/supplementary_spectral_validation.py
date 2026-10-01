"""
Supplementary full-22 external DFT spectral validation for TASK3.
"""

from pathlib import Path
import io
import math
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
    Extract a Materials Project ID from a DFT filename.
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


def filename_rank(
    filename,
):
    """
    Extract the DFT shortlist rank encoded in the filename.
    """
    name = (
        Path(filename)
        .name
    )

    match = re.search(
        r"rank[_\- ]*(\d+)",
        name,
        flags=re.I,
    )

    if match:
        return int(
            match.group(1)
        )

    return 9999


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


def run_supplementary_full22_spectral_validation(
    project_root=r"D:\TB3",
    dft_zip=r"D:\DFT_TEST\collect_data_epsilon.zip",
    ml_npz=None,
    slme_csv=None,
    output_dir=None,
    energy_min=0.0,
    energy_max=4.0,
    n_cols=4,
    ml_color="#2A72B5",
    dft_color="#D62728",
    expected_samples=22,
    show=True,
):
    """
    Generate the supplementary full-22 raw spectral validation figure.

    Scientific protocol
    -------------------
    - ML uses frozen RAW ``predicted_alpha_cm1``.
    - DFT uses trace-averaged eps1/eps2 converted to alpha(E).
    - Spectral comparison is restricted to 0-4 eV by default.
    - No MP-gap cleanup is applied.
    - All panels use one common y-axis limit.

    Returns
    -------
    dict
        Records, Pearson summary, figure, axes, and output paths.
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
            / "FINAL_DFT22_SLME"
            / "Supplementary_full22_spectra"
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
        / "Supplementary_DFT22_raw_spectra_0_4eV.png"
    )

    out_pdf = (
        output_dir
        / "Supplementary_DFT22_raw_spectra_0_4eV.pdf"
    )

    out_svg = (
        output_dir
        / "Supplementary_DFT22_raw_spectra_0_4eV.svg"
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

    npz = np.load(
        ml_npz,
        allow_pickle=True,
    )

    required_keys = [
        "material_id",
        "energy_eV",
        "predicted_alpha_cm1",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in npz.files
    ]

    if missing_keys:
        raise KeyError(
            f"Missing ML NPZ keys: {missing_keys}\n"
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

    meta_df = pd.read_csv(
        slme_csv
    )

    required_meta_columns = [
        "material_id",
        "source_rank",
        "formula_pretty",
    ]

    missing_meta = [
        column
        for column in required_meta_columns
        if column not in meta_df.columns
    ]

    if missing_meta:
        raise KeyError(
            f"Missing SLME CSV columns: {missing_meta}"
        )

    meta_df[
        "material_id"
    ] = (
        meta_df[
            "material_id"
        ]
        .astype(str)
        .str.strip()
    )

    if meta_df[
        "material_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate material_id in SLME validation CSV."
        )

    meta_lookup = (
        meta_df
        .set_index(
            "material_id"
        )
    )

    records = []

    with zipfile.ZipFile(
        dft_zip,
        "r",
    ) as archive:
        dat_files = sorted(
            [
                name
                for name in archive.namelist()
                if name.lower().endswith(
                    ".dat"
                )
            ],
            key=filename_rank,
        )

        for filename in dat_files:
            material_id = filename_material_id(
                filename
            )

            if (
                material_id is None
                or material_id not in id_to_ml_idx
                or material_id not in meta_lookup.index
            ):
                print(
                    "Skip:",
                    filename,
                )
                continue

            array = np.loadtxt(
                io.BytesIO(
                    archive.read(
                        filename
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
                    f"Unexpected DFT file: "
                    f"{filename} | "
                    f"{array.shape}"
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

            alpha_ml_04 = (
                alpha_ml_raw[
                    mask_ml
                ]
            )

            dft_interp = np.interp(
                energy_common,
                energy_dft,
                alpha_dft_raw,
            )

            pearson_r = float(
                pearsonr(
                    dft_interp,
                    alpha_ml_04,
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

            row = (
                meta_lookup.loc[
                    material_id
                ]
            )

            records.append(
                {
                    "rank":
                        int(
                            row[
                                "source_rank"
                            ]
                        ),

                    "material_id":
                        material_id,

                    "formula":
                        str(
                            row[
                                "formula_pretty"
                            ]
                        ),

                    "r":
                        pearson_r,

                    "energy_ml":
                        energy_common,

                    "alpha_ml":
                        alpha_ml_04,

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

    records = sorted(
        records,
        key=lambda item: item[
            "rank"
        ],
    )

    if len(
        records
    ) != int(
        expected_samples
    ):
        print(
            f"WARNING: expected {expected_samples} samples, "
            f"found {len(records)}"
        )

    if len(
        records
    ) == 0:
        raise RuntimeError(
            "No matched DFT spectra found."
        )

    all_y = []

    for record in records:
        all_y.append(
            record[
                "alpha_ml"
            ]
            / 1e6
        )

        all_y.append(
            record[
                "alpha_dft"
            ]
            / 1e6
        )

    global_ymax = float(
        np.nanmax(
            np.concatenate(
                all_y
            )
        )
    )

    common_ymax = (
        np.ceil(
            global_ymax
            * 10.0
        )
        / 10.0
        + 0.05
    )

    print(
        "Shared y-limit:",
        f"0–{common_ymax:.2f}"
    )

    n_samples = len(
        records
    )

    n_rows = math.ceil(
        n_samples
        / int(
            n_cols
        )
    )

    rc_params = {
        "font.family":
            "Arial",

        "font.size":
            7.0,

        "axes.linewidth":
            0.75,

        "xtick.major.width":
            0.75,

        "ytick.major.width":
            0.75,

        "xtick.major.size":
            3.0,

        "ytick.major.size":
            3.0,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    }

    with plt.rc_context(
        rc_params
    ):
        fig, axes = plt.subplots(
            n_rows,
            int(
                n_cols
            ),
            figsize=(
                10.0,
                2.35
                * n_rows,
            ),
            dpi=220,
        )

        axes = np.asarray(
            axes
        ).reshape(
            -1
        )

        letters = [
            chr(
                ord("a")
                + index
            )
            for index in range(
                n_samples
            )
        ]

        for index, (
            ax,
            record,
        ) in enumerate(
            zip(
                axes,
                records,
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
                linewidth=1.25,
            )

            ax.plot(
                energy_dft_plot,
                alpha_dft_plot,
                color=dft_color,
                linewidth=1.25,
                linestyle=(
                    0,
                    (
                        6,
                        4,
                    )
                ),
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

            formula = formula_mathtext(
                record[
                    "formula"
                ]
            )

            ax.set_title(
                (
                    f"({letters[index]}) "
                    f"Rank {record['rank']} | "
                    f"{formula}"
                ),
                fontsize=7.2,
                pad=2.5,
            )

            ax.text(
                0.045,
                0.91,
                rf"$r$ = {record['r']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
            )

            ax.tick_params(
                labelsize=6.2,
            )

            row_index = (
                index
                // int(
                    n_cols
                )
            )

            col_index = (
                index
                % int(
                    n_cols
                )
            )

            if col_index == 0:
                ax.set_ylabel(
                    r"$\alpha$ "
                    r"($\times10^6$ cm$^{-1}$)",
                    fontsize=6.8,
                )

            if row_index == (
                n_rows
                - 1
            ):
                ax.set_xlabel(
                    "Photon Energy (eV)",
                    fontsize=6.8,
                )

        for ax in axes[
            n_samples:
        ]:
            ax.axis(
                "off"
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=ml_color,
                linewidth=1.4,
                label="SSL-GINE",
            ),
            Line2D(
                [0],
                [0],
                color=dft_color,
                linewidth=1.4,
                linestyle=(
                    0,
                    (
                        6,
                        4,
                    )
                ),
                label="DFT",
            ),
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                0.995,
            ),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            handlelength=2.6,
            columnspacing=1.6,
        )

        fig.suptitle(
            (
                "External DFT validation of frozen "
                r"direct-$\alpha(E)$ predictions"
            ),
            fontsize=10.0,
            fontweight="bold",
            y=1.015,
        )

        plt.tight_layout(
            rect=[
                0.03,
                0.035,
                0.99,
                0.965,
            ],
            h_pad=1.15,
            w_pad=1.00,
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

    r_values = np.asarray(
        [
            record[
                "r"
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 96)
    print("SUPPLEMENTARY FULL 22 SPECTRA")
    print("=" * 96)
    print(
        "Samples:",
        len(
            records
        ),
    )
    print(
        "ML alpha:",
        "predicted_alpha_cm1 [RAW]",
    )
    print(
        "DFT:",
        "trace-average epsilon -> alpha",
    )
    print(
        "MP-gap cleanup:",
        "NONE",
    )
    print(
        "Energy window:",
        f"{float(energy_min):g}–"
        f"{float(energy_max):g} eV",
    )
    print(
        "Shared y-limit:",
        f"0–{common_ymax:.2f} ×10^6 cm^-1",
    )
    print()
    print(
        "Pearson r range:",
        f"{np.nanmin(r_values):.4f}",
        "to",
        f"{np.nanmax(r_values):.4f}",
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

    summary_df = pd.DataFrame(
        [
            {
                "rank":
                    record[
                        "rank"
                    ],

                "material_id":
                    record[
                        "material_id"
                    ],

                "formula":
                    record[
                        "formula"
                    ],

                "pearson_r_0_4eV":
                    record[
                        "r"
                    ],
            }
            for record in records
        ]
    )

    return {
        "records":
            records,

        "summary_df":
            summary_df,

        "pearson_min":
            float(
                np.nanmin(
                    r_values
                )
            ),

        "pearson_max":
            float(
                np.nanmax(
                    r_values
                )
            ),

        "common_ymax":
            common_ymax,

        "figure":
            fig,

        "axes":
            axes,

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
