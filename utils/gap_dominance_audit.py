"""
MP-gap dominance audit for the final Database 1 external SLME ranking.

Compares:
    predicted-alpha SLME
vs
    ideal step-absorber SQ efficiency calculated from the same MP band gap.

The purpose is diagnostic: quantify whether the final external ranking is
dominated by the independent Materials Project gap cutoff rather than by
the predicted alpha(E) spectral shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import lambertw


Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
HC_EV_NM = 1239.8419843320026


def _am15_photon_flux(
    model_energy_eV,
    wavelength_nm,
    irradiance_w_m2_nm,
):
    solar_energy_eV = (
        HC_EV_NM
        / wavelength_nm
    )

    solar_energy_J = (
        solar_energy_eV
        * Q
    )

    photon_flux_lambda = (
        irradiance_w_m2_nm
        / solar_energy_J
    )

    jacobian_nm_per_eV = (
        HC_EV_NM
        / solar_energy_eV**2
    )

    photon_flux_E = (
        photon_flux_lambda
        * jacobian_nm_per_eV
    )

    order = np.argsort(
        solar_energy_eV
    )

    return np.interp(
        model_energy_eV,
        solar_energy_eV[
            order
        ],
        photon_flux_E[
            order
        ],
        left=0.0,
        right=0.0,
    )


def _blackbody_photon_radiance(
    energy_eV,
    temperature_K,
):
    energy_eV = np.asarray(
        energy_eV,
        dtype=np.float64,
    )

    energy_J = (
        energy_eV
        * Q
    )

    result = np.zeros_like(
        energy_eV,
        dtype=np.float64,
    )

    valid = (
        energy_eV
        > 0.0
    )

    exponent = np.zeros_like(
        energy_eV,
        dtype=np.float64,
    )

    exponent[
        valid
    ] = (
        energy_J[
            valid
        ]
        / (
            KB
            * float(
                temperature_K
            )
        )
    )

    safe = (
        valid
        & (
            exponent
            < 700.0
        )
    )

    spectral_per_J = (
        2.0
        * energy_J[
            safe
        ]**2
        / (
            H**3
            * C**2
        )
        / np.expm1(
            exponent[
                safe
            ]
        )
    )

    result[
        safe
    ] = (
        spectral_per_J
        * Q
    )

    return result


def _calculate_slme_batch(
    alpha_cm1,
    energy_eV,
    phi_solar,
    phi_bb,
    psolar_w_m2,
    thickness_nm=500.0,
    temperature_K=300.0,
    fr=1.0,
):
    alpha_cm1 = np.asarray(
        alpha_cm1,
        dtype=np.float64,
    )

    energy_eV = np.asarray(
        energy_eV,
        dtype=np.float64,
    )

    thickness_cm = (
        float(
            thickness_nm
        )
        * 1e-7
    )

    absorptivity = (
        1.0
        - np.exp(
            -2.0
            * alpha_cm1
            * thickness_cm
        )
    )

    absorptivity = np.clip(
        absorptivity,
        0.0,
        1.0,
    )

    jsc = (
        Q
        * np.trapezoid(
            absorptivity
            * phi_solar[
                None,
                :
            ],
            energy_eV,
            axis=1,
        )
    )

    j0 = (
        Q
        * np.pi
        / float(
            fr
        )
        * np.trapezoid(
            absorptivity
            * phi_bb[
                None,
                :
            ],
            energy_eV,
            axis=1,
        )
    )

    thermal_voltage = (
        KB
        * float(
            temperature_K
        )
        / Q
    )

    n_samples = len(
        jsc
    )

    eta = np.full(
        n_samples,
        np.nan,
    )

    voc = np.full(
        n_samples,
        np.nan,
    )

    vmp = np.full(
        n_samples,
        np.nan,
    )

    jmp = np.full(
        n_samples,
        np.nan,
    )

    pmax = np.full(
        n_samples,
        np.nan,
    )

    valid = (
        np.isfinite(
            jsc
        )
        & np.isfinite(
            j0
        )
        & (
            jsc
            > 0.0
        )
        & (
            j0
            > 0.0
        )
    )

    ratio = np.full(
        n_samples,
        np.nan,
    )

    ratio[
        valid
    ] = (
        jsc[
            valid
        ]
        / j0[
            valid
        ]
    )

    voc[
        valid
    ] = (
        thermal_voltage
        * np.log1p(
            ratio[
                valid
            ]
        )
    )

    argument = np.full(
        n_samples,
        np.nan,
    )

    argument[
        valid
    ] = (
        np.e
        * (
            1.0
            + ratio[
                valid
            ]
        )
    )

    vmp_dimensionless = np.full(
        n_samples,
        np.nan,
    )

    vmp_dimensionless[
        valid
    ] = (
        np.real(
            lambertw(
                argument[
                    valid
                ]
            )
        )
        - 1.0
    )

    vmp[
        valid
    ] = (
        thermal_voltage
        * vmp_dimensionless[
            valid
        ]
    )

    exp_vmp = np.full(
        n_samples,
        np.nan,
    )

    exp_vmp[
        valid
    ] = np.exp(
        np.clip(
            vmp_dimensionless[
                valid
            ],
            None,
            700.0,
        )
    )

    jmp[
        valid
    ] = (
        jsc[
            valid
        ]
        - j0[
            valid
        ]
        * (
            exp_vmp[
                valid
            ]
            - 1.0
        )
    )

    pmax[
        valid
    ] = (
        jmp[
            valid
        ]
        * vmp[
            valid
        ]
    )

    eta[
        valid
    ] = (
        100.0
        * pmax[
            valid
        ]
        / psolar_w_m2
    )

    return {
        "eta_percent":
            eta,

        "Jsc_A_m2":
            jsc,

        "J0_A_m2":
            j0,

        "Voc_V":
            voc,

        "Vmp_V":
            vmp,

        "Jmp_A_m2":
            jmp,

        "Pmax_W_m2":
            pmax,
    }


def run_mp_gap_dominance_audit(
    project_root=r"D:\TB3",
    all_csv=None,
    spectra_npz=None,
    output_dir=None,
    thickness_nm=None,
    temperature_K=None,
    fr=None,
    solar_lambda_min_nm=200.0,
    solar_lambda_max_nm=2500.0,
    ideal_alpha_cm1=1.0e9,
    high_slme_threshold=30.0,
    top_k_values=(10, 20, 50, 100),
    show=True,
):
    """
    Audit how strongly the MP band-gap cutoff drives the final DB1 SLME ranking.

    The function is fresh-kernel standalone. It reads the locked DB1 screening
    outputs, reconstructs the same ASTM G173 / detailed-balance engine, and
    compares the predicted-alpha ranking against an ideal step absorber that
    uses only the same MP gap.

    Returns
    -------
    dict
        Full audit table, top-k overlap table, summary table, metrics, and paths.
    """
    project_root = Path(
        project_root
    )

    screen_dir = (
        project_root
        / "processed"
        / "paired_training"
        / "paper_outputs"
        / "screening_SLME_direct_alpha_500nm"
    )

    if all_csv is None:
        all_csv = (
            screen_dir
            / "zintl_external_1100_direct_alpha_SLME_all.csv"
        )
    else:
        all_csv = Path(
            all_csv
        )

    if spectra_npz is None:
        spectra_npz = (
            screen_dir
            / "zintl_external_1100_direct_alpha_SLME_spectra.npz"
        )
    else:
        spectra_npz = Path(
            spectra_npz
        )

    if output_dir is None:
        output_dir = (
            screen_dir
            / "gap_dominance_audit"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not all_csv.exists():
        raise FileNotFoundError(
            f"Missing ranked DB1 screening CSV:\n{all_csv}"
        )

    if not spectra_npz.exists():
        raise FileNotFoundError(
            f"Missing DB1 spectra NPZ:\n{spectra_npz}"
        )

    audit_df = pd.read_csv(
        all_csv
    )

    required_columns = [
        "SLME_gap_cutoff_eV",
        "predicted_SLME_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in audit_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    if thickness_nm is None:
        if "SLME_thickness_nm" in audit_df.columns:
            thickness_nm = float(
                pd.to_numeric(
                    audit_df[
                        "SLME_thickness_nm"
                    ],
                    errors="coerce",
                ).dropna().iloc[
                    0
                ]
            )
        else:
            thickness_nm = 500.0

    if temperature_K is None:
        if "SLME_temperature_K" in audit_df.columns:
            temperature_K = float(
                pd.to_numeric(
                    audit_df[
                        "SLME_temperature_K"
                    ],
                    errors="coerce",
                ).dropna().iloc[
                    0
                ]
            )
        else:
            temperature_K = 300.0

    if fr is None:
        if "SLME_fr" in audit_df.columns:
            fr = float(
                pd.to_numeric(
                    audit_df[
                        "SLME_fr"
                    ],
                    errors="coerce",
                ).dropna().iloc[
                    0
                ]
            )
        else:
            fr = 1.0

    gap_eV = pd.to_numeric(
        audit_df[
            "SLME_gap_cutoff_eV"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    eta_ml = pd.to_numeric(
        audit_df[
            "predicted_SLME_percent"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if (
        ~np.isfinite(
            gap_eV
        )
    ).any():
        raise RuntimeError(
            "Non-finite gap values found."
        )

    if (
        ~np.isfinite(
            eta_ml
        )
    ).any():
        raise RuntimeError(
            "Non-finite ML SLME values found."
        )

    spectra = np.load(
        spectra_npz,
        allow_pickle=True,
    )

    if "energy_eV" not in spectra.files:
        raise KeyError(
            f"No energy_eV array found in:\n{spectra_npz}"
        )

    energy_eV = np.asarray(
        spectra[
            "energy_eV"
        ],
        dtype=np.float64,
    ).reshape(
        -1
    )

    if len(
        energy_eV
    ) < 2:
        raise RuntimeError(
            "Invalid energy grid."
        )

    try:
        from pvlib import spectrum

    except ImportError as exc:
        raise ImportError(
            "pvlib is required for ASTM G173 AM1.5G.\n"
            "Install once with:\n"
            "pip install pvlib"
        ) from exc

    am15 = spectrum.get_reference_spectra()

    global_column = next(
        (
            column
            for column in am15.columns
            if "global" in str(
                column
            ).lower()
        ),
        None,
    )

    if global_column is None:
        raise RuntimeError(
            "ASTM G173 global AM1.5G column was not found."
        )

    wavelength_nm = np.asarray(
        am15.index,
        dtype=np.float64,
    )

    irradiance_w_m2_nm = np.asarray(
        am15[
            global_column
        ],
        dtype=np.float64,
    )

    solar_mask = (
        np.isfinite(
            wavelength_nm
        )
        & np.isfinite(
            irradiance_w_m2_nm
        )
        & (
            wavelength_nm
            >= float(
                solar_lambda_min_nm
            )
        )
        & (
            wavelength_nm
            <= float(
                solar_lambda_max_nm
            )
        )
        & (
            irradiance_w_m2_nm
            >= 0.0
        )
    )

    wavelength_nm = (
        wavelength_nm[
            solar_mask
        ]
    )

    irradiance_w_m2_nm = (
        irradiance_w_m2_nm[
            solar_mask
        ]
    )

    phi_solar = _am15_photon_flux(
        energy_eV,
        wavelength_nm,
        irradiance_w_m2_nm,
    )

    psolar_w_m2 = np.trapezoid(
        (
            energy_eV
            * Q
        )
        * phi_solar,
        energy_eV,
    )

    if not (
        900.0
        <= psolar_w_m2
        <= 1050.0
    ):
        raise RuntimeError(
            "AM1.5G integrated power sanity check failed."
        )

    phi_bb = _blackbody_photon_radiance(
        energy_eV,
        float(
            temperature_K
        ),
    )

    alpha_ideal = np.zeros(
        (
            len(
                gap_eV
            ),
            len(
                energy_eV
            ),
        ),
        dtype=np.float64,
    )

    above_gap = (
        energy_eV[
            None,
            :
        ]
        >= gap_eV[
            :,
            None
        ]
    )

    alpha_ideal[
        above_gap
    ] = float(
        ideal_alpha_cm1
    )

    sq_result = _calculate_slme_batch(
        alpha_ideal,
        energy_eV,
        phi_solar,
        phi_bb,
        psolar_w_m2,
        thickness_nm=float(
            thickness_nm
        ),
        temperature_K=float(
            temperature_K
        ),
        fr=float(
            fr
        ),
    )

    eta_sq_gap = np.asarray(
        sq_result[
            "eta_percent"
        ],
        dtype=float,
    )

    audit_df[
        "SQ_efficiency_from_MP_gap_percent"
    ] = (
        eta_sq_gap
    )

    audit_df[
        "SLME_minus_SQ_gap_pp"
    ] = (
        eta_ml
        - eta_sq_gap
    )

    audit_df[
        "SLME_to_SQ_gap_ratio"
    ] = np.divide(
        eta_ml,
        eta_sq_gap,
        out=np.full_like(
            eta_ml,
            np.nan,
        ),
        where=eta_sq_gap > 0.0,
    )

    pearson_gap = float(
        np.corrcoef(
            eta_ml,
            eta_sq_gap,
        )[
            0,
            1
        ]
    )

    spearman_gap = float(
        pd.Series(
            eta_ml
        ).corr(
            pd.Series(
                eta_sq_gap
            ),
            method="spearman",
        )
    )

    ml_order = np.argsort(
        -eta_ml,
        kind="mergesort",
    )

    gap_order = np.argsort(
        -eta_sq_gap,
        kind="mergesort",
    )

    topk_rows = []

    for k in top_k_values:
        k = min(
            int(
                k
            ),
            len(
                audit_df
            ),
        )

        ml_top = set(
            ml_order[
                :k
            ]
        )

        gap_top = set(
            gap_order[
                :k
            ]
        )

        overlap = len(
            ml_top
            & gap_top
        )

        topk_rows.append(
            {
                "k":
                    k,

                "overlap_count":
                    overlap,

                "overlap_percent":
                    100.0
                    * overlap
                    / k,
            }
        )

    topk_gap_df = pd.DataFrame(
        topk_rows
    )

    high_mask = (
        audit_df[
            "predicted_SLME_percent"
        ]
        > float(
            high_slme_threshold
        )
    )

    high_df = audit_df[
        high_mask
    ].copy()

    mean_ratio_all = float(
        np.nanmean(
            audit_df[
                "SLME_to_SQ_gap_ratio"
            ]
        )
    )

    mean_ratio_high = float(
        np.nanmean(
            high_df[
                "SLME_to_SQ_gap_ratio"
            ]
        )
    )

    median_ratio_high = float(
        np.nanmedian(
            high_df[
                "SLME_to_SQ_gap_ratio"
            ]
        )
    )

    mean_abs_difference_high_pp = float(
        np.nanmean(
            np.abs(
                high_df[
                    "SLME_minus_SQ_gap_pp"
                ]
            )
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "n_total",

                "value":
                    len(
                        audit_df
                    ),
            },
            {
                "metric":
                    f"n_SLME_gt{high_slme_threshold:g}",

                "value":
                    len(
                        high_df
                    ),
            },
            {
                "metric":
                    "Pearson_ML_SLME_vs_gap_only_SQ",

                "value":
                    pearson_gap,
            },
            {
                "metric":
                    "Spearman_ML_SLME_vs_gap_only_SQ",

                "value":
                    spearman_gap,
            },
            {
                "metric":
                    "mean_SLME_to_SQ_ratio_all",

                "value":
                    mean_ratio_all,
            },
            {
                "metric":
                    f"mean_SLME_to_SQ_ratio_gt{high_slme_threshold:g}",

                "value":
                    mean_ratio_high,
            },
            {
                "metric":
                    f"median_SLME_to_SQ_ratio_gt{high_slme_threshold:g}",

                "value":
                    median_ratio_high,
            },
            {
                "metric":
                    f"mean_abs_difference_gt{high_slme_threshold:g}_pp",

                "value":
                    mean_abs_difference_high_pp,
            },
            {
                "metric":
                    "Psolar_W_m2",

                "value":
                    float(
                        psolar_w_m2
                    ),
            },
            {
                "metric":
                    "thickness_nm",

                "value":
                    float(
                        thickness_nm
                    ),
            },
            {
                "metric":
                    "temperature_K",

                "value":
                    float(
                        temperature_K
                    ),
            },
            {
                "metric":
                    "fr",

                "value":
                    float(
                        fr
                    ),
            },
        ]
    )

    audit_all_csv = (
        output_dir
        / "zintl_SLME_vs_MP_gap_only_SQ.csv"
    )

    audit_topk_csv = (
        output_dir
        / "zintl_SLME_vs_MP_gap_only_SQ_topk_overlap.csv"
    )

    audit_summary_csv = (
        output_dir
        / "zintl_SLME_vs_MP_gap_only_SQ_summary.csv"
    )

    audit_df.to_csv(
        audit_all_csv,
        index=False,
        encoding="utf-8-sig",
    )

    topk_gap_df.to_csv(
        audit_topk_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        audit_summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 100)
    print("MP-GAP DOMINANCE AUDIT")
    print("=" * 100)
    print(
        f"Pearson  ML-SLME vs gap-only SQ : "
        f"{pearson_gap:.4f}"
    )
    print(
        f"Spearman ML-SLME vs gap-only SQ : "
        f"{spearman_gap:.4f}"
    )
    print()
    print("Top-k overlap with gap-only SQ ranking:")

    if show:
        try:
            from IPython.display import display

            display(
                topk_gap_df
            )

        except Exception:
            print(
                topk_gap_df.to_string(
                    index=False
                )
            )
    else:
        print(
            topk_gap_df.to_string(
                index=False
            )
        )

    print(
        f"Mean eta_ML / eta_SQ for eta_ML > {high_slme_threshold:g}%:",
        f"{mean_ratio_high:.4f}",
    )

    print(
        f"Median eta_ML / eta_SQ for eta_ML > {high_slme_threshold:g}%:",
        f"{median_ratio_high:.4f}",
    )

    print(
        f"Mean |eta_ML - eta_SQ| for eta_ML > {high_slme_threshold:g}%:",
        f"{mean_abs_difference_high_pp:.4f}",
        "percentage points",
    )

    display_columns = [
        "SLME_rank",
        "material_id",
        "formula_pretty",
        "band_gap",
        "predicted_SLME_percent",
        "SQ_efficiency_from_MP_gap_percent",
        "SLME_minus_SQ_gap_pp",
        "SLME_to_SQ_gap_ratio",
        "energy_above_hull",
        "is_stable",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in audit_df.columns
    ]

    print()
    print("Top-20 FINAL candidates with SQ comparison:")

    if show:
        try:
            from IPython.display import display

            display(
                audit_df[
                    display_columns
                ].head(
                    20
                )
            )

        except Exception:
            pass

    print()
    print("Saved:")
    print(
        audit_all_csv
    )
    print(
        audit_topk_csv
    )
    print(
        audit_summary_csv
    )

    return {
        "audit_df":
            audit_df,

        "topk_gap_df":
            topk_gap_df,

        "summary_df":
            summary_df,

        "high_df":
            high_df,

        "metrics": {
            "Pearson_ML_SLME_vs_gap_only_SQ":
                pearson_gap,

            "Spearman_ML_SLME_vs_gap_only_SQ":
                spearman_gap,

            "mean_SLME_to_SQ_ratio_all":
                mean_ratio_all,

            "mean_SLME_to_SQ_ratio_high":
                mean_ratio_high,

            "median_SLME_to_SQ_ratio_high":
                median_ratio_high,

            "mean_abs_difference_high_pp":
                mean_abs_difference_high_pp,

            "Psolar_W_m2":
                float(
                    psolar_w_m2
                ),
        },

        "paths": {
            "all_csv":
                all_csv,

            "spectra_npz":
                spectra_npz,

            "output_dir":
                output_dir,

            "audit_all_csv":
                audit_all_csv,

            "audit_topk_csv":
                audit_topk_csv,

            "audit_summary_csv":
                audit_summary_csv,
        },
    }
