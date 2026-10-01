"""
Final Database 1 external Zintl-like Direct-alpha -> SLME screening.

Scientific route
----------------
1100 validated external structures
-> enhanced 19/24 graphs
-> Direct-alpha SSL-init GINE ensemble, seeds 42/123/2025
-> alpha(E)
-> Materials Project band-gap sub-gap cleanup
-> ASTM G173 AM1.5G detailed-balance SLME
-> ranking and threshold tables at L = 500 nm, T = 300 K, fr = 1

This module is designed to be called from either a notebook or the
Streamlit screening interface so that both use the same implementation.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import sys
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.special import lambertw


Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
HC_EV_NM = 1239.8419843320026


ProgressCallback = Callable[[str, float], None]


def _report(
    callback: ProgressCallback | None,
    message: str,
    fraction: float,
) -> None:
    if callback is not None:
        callback(
            str(message),
            float(
                np.clip(
                    fraction,
                    0.0,
                    1.0,
                )
            ),
        )


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
            * temperature_K
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


def run_final_db1_slme_screening(
    project_root=r"D:\TB3",
    pool_csv=None,
    structure_cache=None,
    direct_alpha_dir=None,
    output_dir=None,
    seeds=(42, 123, 2025),
    batch_size=32,
    thickness_nm=500.0,
    temperature_K=300.0,
    fr=1.0,
    solar_lambda_min_nm=200.0,
    solar_lambda_max_nm=2500.0,
    primary_threshold=30.0,
    thresholds=(30.0, 31.0, 32.0),
    top_k=50,
    expected_node_dim=19,
    expected_edge_dim=24,
    expected_pool_size=1100,
    global_efficiency_ceiling_percent=35.0,
    output_prefix="zintl_external_1100_direct_alpha_SLME",
    show=True,
    verbose=True,
    progress_callback: ProgressCallback | None = None,
):
    """
    Run the final Database 1 external Direct-alpha -> SLME screening.

    The default paths reproduce the locked paper workflow. The historical
    ``screening_srp_200nm`` directory is used only as the location of the
    frozen 1100-structure cache; no SRP-power calculation and no 200 nm
    absorber are used in this function.

    Returns
    -------
    dict
        Dataframes, spectra, metrics, output paths, and protocol metadata.
    """
    t0 = time.time()

    project_root = Path(
        project_root
    )

    if str(
        project_root
    ) not in sys.path:
        sys.path.insert(
            0,
            str(
                project_root
            ),
        )

    root = (
        project_root
        / "processed"
        / "paired_training"
    )

    if pool_csv is None:
        pool_csv = (
            root
            / "paper_outputs"
            / "external_pools"
            / "pool_zintl_like_no_element_count_filter_DFT_ready.csv"
        )
    else:
        pool_csv = Path(
            pool_csv
        )

    if structure_cache is None:
        structure_cache = (
            root
            / "paper_outputs"
            / "screening_srp_200nm"
            / "external_pool_cache"
            / "screening_zintl_like_external_fixed_1100_with_structures.pkl"
        )
    else:
        structure_cache = Path(
            structure_cache
        )

    if direct_alpha_dir is None:
        direct_alpha_dir = (
            root
            / "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed"
        )
    else:
        direct_alpha_dir = Path(
            direct_alpha_dir
        )

    if output_dir is None:
        output_dir = (
            root
            / "paper_outputs"
            / "screening_SLME_direct_alpha_500nm"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = tuple(
        int(
            seed
        )
        for seed in seeds
    )

    thresholds = tuple(
        float(
            threshold
        )
        for threshold in thresholds
    )

    if float(
        primary_threshold
    ) not in thresholds:
        thresholds = tuple(
            sorted(
                set(
                    thresholds
                    + (
                        float(
                            primary_threshold
                        ),
                    )
                )
            )
        )

    _report(
        progress_callback,
        "Validating frozen Database 1 inputs",
        0.05,
    )

    import utils_tb.external_screening as EXT

    EXT = importlib.reload(
        EXT
    )

    make_graph_dataset = getattr(
        EXT,
        "_tb3_screen_make_graph_dataset",
    )

    predict_ensemble = getattr(
        EXT,
        "_tb3_screen_predict_ensemble",
    )

    if verbose:
        print("=" * 110)
        print("FINAL EXTERNAL DIRECT-ALPHA -> SLME SCREENING")
        print("=" * 110)
        print(
            "Pool CSV          :",
            pool_csv,
        )
        print(
            "Structure cache   :",
            structure_cache,
        )
        print(
            "Direct-alpha dir  :",
            direct_alpha_dir,
        )
        print(
            "Output directory  :",
            output_dir,
        )
        print()
        print("Protocol:")
        print("  model route     : structure -> Direct-alpha GINE")
        print(
            "  graph dims      :",
            f"{expected_node_dim} / {expected_edge_dim}",
        )
        print(
            "  model seeds     :",
            seeds,
        )
        print(
            "  thickness       :",
            thickness_nm,
            "nm",
        )
        print(
            "  temperature     :",
            temperature_K,
            "K",
        )
        print(
            "  fr              :",
            fr,
        )
        print("  solar spectrum  : ASTM G173 AM1.5G")
        print(
            "  primary target  :",
            f"SLME > {primary_threshold:g}%",
        )

    for path in [
        pool_csv,
        structure_cache,
        direct_alpha_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found:\n{path}"
            )

    pool_df = pd.read_csv(
        pool_csv
    )

    pool_struct_df = pd.read_pickle(
        structure_cache
    )

    if len(
        pool_df
    ) != int(
        expected_pool_size
    ):
        warnings.warn(
            f"Expected {expected_pool_size} pool rows, "
            f"found {len(pool_df)}."
        )

    if len(
        pool_struct_df
    ) != int(
        expected_pool_size
    ):
        raise RuntimeError(
            f"Expected {expected_pool_size} cached structures, "
            f"found {len(pool_struct_df)}."
        )

    required_pool_cols = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "energy_above_hull",
        "is_stable",
    ]

    missing = [
        column
        for column in required_pool_cols
        if column not in pool_df.columns
    ]

    if missing:
        raise KeyError(
            f"Pool CSV missing columns: {missing}"
        )

    pool_df[
        "material_id"
    ] = (
        pool_df[
            "material_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    pool_struct_df[
        "material_id"
    ] = (
        pool_struct_df[
            "material_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    if pool_df[
        "material_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate material_id detected in source pool."
        )

    if pool_struct_df[
        "material_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate material_id detected in structure cache."
        )

    pool_ids = set(
        pool_df[
            "material_id"
        ]
    )

    structure_ids = set(
        pool_struct_df[
            "material_id"
        ]
    )

    if pool_ids != structure_ids:
        raise RuntimeError(
            "Pool IDs and structure-cache IDs do not match."
        )

    if verbose:
        print()
        print(
            "Source pool         :",
            len(
                pool_df
            ),
        )
        print(
            "Validated structures:",
            len(
                pool_struct_df
            ),
        )
        print("ID alignment        : PASS")

    _report(
        progress_callback,
        "Validating Direct-alpha checkpoints",
        0.12,
    )

    checkpoint_paths = []

    for seed in seeds:
        candidates = sorted(
            direct_alpha_dir.glob(
                f"best_optical_*seed{seed}.pt"
            )
        )

        if len(
            candidates
        ) != 1:
            raise RuntimeError(
                f"Expected exactly one checkpoint for seed {seed}, "
                f"found {len(candidates)}."
            )

        checkpoint_paths.append(
            candidates[
                0
            ]
        )

    if verbose:
        print()
        print("=" * 110)
        print("DIRECT-ALPHA CHECKPOINTS")
        print("=" * 110)

    checkpoint_inventory = []

    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location="cpu",
            )

        if checkpoint.get(
            "target_key"
        ) != "epsI_alpha_log10_1p":
            raise RuntimeError(
                f"Wrong target in {path.name}: "
                f"{checkpoint.get('target_key')}"
            )

        state = checkpoint[
            "model_state_dict"
        ]

        node_dim = int(
            state[
                "node_encoder.0.weight"
            ].shape[
                1
            ]
        )

        edge_dim = int(
            state[
                "convs.0.lin.weight"
            ].shape[
                1
            ]
        )

        if (
            node_dim
            != int(
                expected_node_dim
            )
            or edge_dim
            != int(
                expected_edge_dim
            )
        ):
            raise RuntimeError(
                f"Graph/checkpoint dimension mismatch in {path.name}: "
                f"{node_dim}/{edge_dim}"
            )

        checkpoint_inventory.append(
            {
                "path":
                    str(
                        path
                    ),

                "seed":
                    int(
                        checkpoint[
                            "seed"
                        ]
                    ),

                "epoch":
                    int(
                        checkpoint[
                            "epoch"
                        ]
                    ),

                "target_key":
                    checkpoint[
                        "target_key"
                    ],

                "node_dim":
                    node_dim,

                "edge_dim":
                    edge_dim,
            }
        )

        if verbose:
            print(
                f"seed {checkpoint['seed']:4d} | "
                f"epoch {checkpoint['epoch']:3d} | "
                f"target={checkpoint['target_key']} | "
                f"dims={node_dim}/{edge_dim}"
            )

    energy_eV = np.linspace(
        0.0,
        20.0,
        2001,
        dtype=np.float32,
    )

    _report(
        progress_callback,
        "Building enhanced 19/24 graphs for 1100 structures",
        0.22,
    )

    if verbose:
        print()
        print("=" * 110)
        print("BUILDING DIRECT-ALPHA 19/24 GRAPHS")
        print("=" * 110)

    graphs, meta_df = make_graph_dataset(
        pool_struct_df,
        energy_eV,
        target_key="epsI_alpha_log10_1p",
    )

    if len(
        graphs
    ) != int(
        expected_pool_size
    ):
        raise RuntimeError(
            f"Expected {expected_pool_size} graphs, "
            f"found {len(graphs)}."
        )

    first_graph = graphs[
        0
    ]

    node_dim_graph = int(
        first_graph.x.shape[
            -1
        ]
    )

    edge_dim_graph = int(
        first_graph.edge_attr.shape[
            -1
        ]
    )

    if (
        node_dim_graph
        != int(
            expected_node_dim
        )
        or edge_dim_graph
        != int(
            expected_edge_dim
        )
    ):
        raise RuntimeError(
            f"External graph dimensions are "
            f"{node_dim_graph}/{edge_dim_graph}, "
            f"expected {expected_node_dim}/{expected_edge_dim}."
        )

    if verbose:
        print(
            "Graphs built             :",
            len(
                graphs
            ),
        )
        print(
            "Graph feature dimensions:",
            f"{node_dim_graph}/{edge_dim_graph}",
        )
        print("Graph-dimension check   : PASS")

    _report(
        progress_callback,
        "Running three-seed Direct-alpha ensemble inference",
        0.45,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if verbose:
        print()
        print("=" * 110)
        print("DIRECT-ALPHA ENSEMBLE INFERENCE")
        print("=" * 110)
        print(
            "Device:",
            device,
        )

    pred_log_alpha = predict_ensemble(
        graphs,
        checkpoint_paths,
        device,
        batch_size=int(
            batch_size
        ),
    )

    pred_log_alpha = np.asarray(
        pred_log_alpha,
        dtype=np.float64,
    )

    if (
        pred_log_alpha.ndim == 3
        and pred_log_alpha.shape[
            1
        ] == 1
    ):
        pred_log_alpha = (
            pred_log_alpha[
                :,
                0,
                :
            ]
        )

    if pred_log_alpha.ndim != 2:
        raise RuntimeError(
            f"Unexpected prediction shape: "
            f"{pred_log_alpha.shape}"
        )

    if pred_log_alpha.shape != (
        len(
            graphs
        ),
        len(
            energy_eV
        ),
    ):
        raise RuntimeError(
            "Direct-alpha prediction shape mismatch: "
            f"{pred_log_alpha.shape}"
        )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        alpha_pred_cm1 = (
            np.power(
                10.0,
                pred_log_alpha,
            )
            - 1.0
        )

    alpha_pred_cm1 = np.maximum(
        np.nan_to_num(
            alpha_pred_cm1,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )

    if verbose:
        print()
        print("=" * 110)
        print("PREDICTED ALPHA")
        print("=" * 110)
        print(
            "alpha shape :",
            alpha_pred_cm1.shape,
        )
        print(
            "alpha max   :",
            f"{np.max(alpha_pred_cm1):.3e} cm^-1",
        )

    _report(
        progress_callback,
        "Aligning Materials Project metadata and applying sub-gap cleanup",
        0.62,
    )

    out_df = meta_df.copy()

    if "material_id" not in out_df.columns:
        if len(
            out_df
        ) != len(
            pool_struct_df
        ):
            raise RuntimeError(
                "Cannot recover material_id alignment."
            )

        out_df[
            "material_id"
        ] = (
            pool_struct_df[
                "material_id"
            ].to_numpy()
        )

    out_df[
        "material_id"
    ] = (
        out_df[
            "material_id"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    pool_meta = (
        pool_df[
            required_pool_cols
        ]
        .drop_duplicates(
            "material_id"
        )
    )

    out_df = out_df.merge(
        pool_meta,
        on="material_id",
        how="left",
        suffixes=(
            "",
            "_pool",
        ),
    )

    if out_df[
        "band_gap"
    ].isna().any():
        raise RuntimeError(
            "Missing Materials Project band_gap "
            "after metadata alignment."
        )

    if len(
        out_df
    ) != len(
        alpha_pred_cm1
    ):
        raise RuntimeError(
            "Metadata/prediction row mismatch."
        )

    mp_gap_eV = pd.to_numeric(
        out_df[
            "band_gap"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    if (
        ~np.isfinite(
            mp_gap_eV
        )
    ).any():
        raise RuntimeError(
            "Non-finite Materials Project band gaps found."
        )

    if (
        mp_gap_eV
        <= 0.0
    ).any():
        raise RuntimeError(
            "Non-positive Materials Project band gaps found."
        )

    alpha_clean_cm1 = (
        alpha_pred_cm1.copy()
    )

    subgap_mask = (
        energy_eV[
            None,
            :
        ]
        < mp_gap_eV[
            :,
            None
        ]
    )

    alpha_clean_cm1[
        subgap_mask
    ] = 0.0

    out_df[
        "SLME_gap_cutoff_eV"
    ] = (
        mp_gap_eV
    )

    out_df[
        "SLME_gap_cutoff_source"
    ] = (
        "Materials Project band_gap"
    )

    _report(
        progress_callback,
        "Loading ASTM G173 AM1.5G and calculating SLME",
        0.72,
    )

    try:
        from pvlib import spectrum

    except ImportError as exc:
        raise ImportError(
            "pvlib is required.\n"
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
            "ASTM G173 global spectrum column not found."
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

    if verbose:
        print()
        print("=" * 110)
        print("AM1.5G SANITY")
        print("=" * 110)
        print(
            f"Psolar = {psolar_w_m2:.3f} W/m^2"
        )

    if not (
        900.0
        <= psolar_w_m2
        <= 1050.0
    ):
        raise RuntimeError(
            "AM1.5G power sanity check failed."
        )

    phi_bb = _blackbody_photon_radiance(
        energy_eV,
        float(
            temperature_K
        ),
    )

    slme = _calculate_slme_batch(
        alpha_clean_cm1,
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

    out_df[
        "predicted_SLME_percent"
    ] = (
        slme[
            "eta_percent"
        ]
    )

    out_df[
        "predicted_Jsc_A_m2"
    ] = (
        slme[
            "Jsc_A_m2"
        ]
    )

    out_df[
        "predicted_J0_A_m2"
    ] = (
        slme[
            "J0_A_m2"
        ]
    )

    out_df[
        "predicted_Voc_V"
    ] = (
        slme[
            "Voc_V"
        ]
    )

    out_df[
        "predicted_Vmp_V"
    ] = (
        slme[
            "Vmp_V"
        ]
    )

    out_df[
        "predicted_Pmax_W_m2"
    ] = (
        slme[
            "Pmax_W_m2"
        ]
    )

    out_df[
        "predicted_alpha_mean_cm1"
    ] = np.nanmean(
        alpha_clean_cm1,
        axis=1,
    )

    out_df[
        "SLME_thickness_nm"
    ] = float(
        thickness_nm
    )

    out_df[
        "SLME_temperature_K"
    ] = float(
        temperature_K
    )

    out_df[
        "SLME_fr"
    ] = float(
        fr
    )

    out_df[
        "SLME_solar_spectrum"
    ] = (
        "ASTM G173 AM1.5G"
    )

    out_df[
        "direct_alpha_model_seeds"
    ] = (
        ",".join(
            str(
                seed
            )
            for seed in seeds
        )
    )

    out_df[
        "direct_alpha_target"
    ] = (
        "log10(1 + alpha_cm^-1)"
    )

    eta_values = out_df[
        "predicted_SLME_percent"
    ].to_numpy(
        dtype=float
    )

    if (
        ~np.isfinite(
            eta_values
        )
    ).any():
        n_bad = int(
            (
                ~np.isfinite(
                    eta_values
                )
            ).sum()
        )

        raise RuntimeError(
            f"{n_bad} candidates have invalid SLME."
        )

    eta_max = float(
        np.max(
            eta_values
        )
    )

    if verbose:
        print()
        print("=" * 110)
        print("EXTERNAL SLME SANITY CHECK")
        print("=" * 110)
        print(
            f"Predicted SLME range: "
            f"{np.min(eta_values):.3f}% "
            f"to {eta_max:.3f}%"
        )
        print(
            "Candidates > 35%:",
            int(
                (
                    eta_values
                    > float(
                        global_efficiency_ceiling_percent
                    )
                ).sum()
            ),
        )

    if eta_max > float(
        global_efficiency_ceiling_percent
    ):
        raise RuntimeError(
            "Predicted external SLME exceeds the configured physical ceiling. "
            "Do not continue ranking."
        )

    _report(
        progress_callback,
        "Ranking candidates and building threshold tables",
        0.86,
    )

    original_prediction_ids = (
        meta_df[
            "material_id"
        ]
        .astype(
            str
        )
        .str.strip()
        .to_numpy()
        if "material_id" in meta_df.columns
        else (
            pool_struct_df[
                "material_id"
            ]
            .astype(
                str
            )
            .str.strip()
            .to_numpy()
        )
    )

    out_df = (
        out_df
        .sort_values(
            "predicted_SLME_percent",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    out_df[
        "SLME_rank"
    ] = np.arange(
        1,
        len(
            out_df
        )
        + 1,
    )

    threshold_tables = {}

    for threshold in thresholds:
        threshold_tables[
            float(
                threshold
            )
        ] = (
            out_df[
                out_df[
                    "predicted_SLME_percent"
                ]
                > float(
                    threshold
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    top_df = (
        out_df
        .head(
            int(
                top_k
            )
        )
        .copy()
    )

    id_to_index = {
        material_id:
            index
        for index, material_id in enumerate(
            original_prediction_ids
        )
    }

    rank_order = np.asarray(
        [
            id_to_index[
                material_id
            ]
            for material_id in (
                out_df[
                    "material_id"
                ]
                .astype(
                    str
                )
            )
        ],
        dtype=int,
    )

    spectra_npz = (
        output_dir
        / f"{output_prefix}_spectra.npz"
    )

    np.savez_compressed(
        spectra_npz,

        material_id=
            out_df[
                "material_id"
            ]
            .astype(
                str
            )
            .to_numpy(),

        energy_eV=
            np.asarray(
                energy_eV,
                dtype=np.float32,
            ),

        pred_log10_1p_alpha=
            pred_log_alpha[
                rank_order
            ].astype(
                np.float32
            ),

        predicted_alpha_cm1=
            alpha_pred_cm1[
                rank_order
            ].astype(
                np.float32
            ),

        predicted_alpha_clean_cm1=
            alpha_clean_cm1[
                rank_order
            ].astype(
                np.float32
            ),

        gap_cutoff_eV=
            mp_gap_eV[
                rank_order
            ].astype(
                np.float32
            ),
    )

    all_csv = (
        output_dir
        / f"{output_prefix}_all.csv"
    )

    top_csv = (
        output_dir
        / f"{output_prefix}_top50.csv"
    )

    summary_csv = (
        output_dir
        / f"{output_prefix}_summary.csv"
    )

    threshold_csvs = {
        float(
            threshold
        ):
            (
                output_dir
                / f"{output_prefix}_gt{int(threshold):d}.csv"
            )
        for threshold in thresholds
    }

    out_df.to_csv(
        all_csv,
        index=False,
        encoding="utf-8-sig",
    )

    top_df.to_csv(
        top_csv,
        index=False,
        encoding="utf-8-sig",
    )

    for threshold, table in threshold_tables.items():
        table.to_csv(
            threshold_csvs[
                threshold
            ],
            index=False,
            encoding="utf-8-sig",
        )

    summary_rows = [
        {
            "stage":
                "External Zintl-like DFT-ready pool",

            "n_materials":
                len(
                    out_df
                ),

            "note":
                "Validated external structures.",
        },
    ]

    for threshold in thresholds:
        note = (
            "Primary solar-cell discovery target."
            if np.isclose(
                threshold,
                float(
                    primary_threshold
                ),
            )
            else (
                "Higher-confidence SLME subset."
                if threshold < max(
                    thresholds
                )
                else "Highest predicted SLME subset."
            )
        )

        summary_rows.append(
            {
                "stage":
                    f"Predicted SLME > {threshold:g}%",

                "n_materials":
                    len(
                        threshold_tables[
                            threshold
                        ]
                    ),

                "note":
                    note,
            }
        )

    summary_rows.append(
        {
            "stage":
                f"Top-{int(top_k)} by predicted SLME",

            "n_materials":
                len(
                    top_df
                ),

            "note":
                "Before novelty filtering.",
        }
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df[
        "SLME_max_percent"
    ] = (
        eta_max
    )

    summary_df[
        "thickness_nm"
    ] = float(
        thickness_nm
    )

    summary_df[
        "temperature_K"
    ] = float(
        temperature_K
    )

    summary_df[
        "fr"
    ] = float(
        fr
    )

    summary_df[
        "model_seeds"
    ] = (
        ",".join(
            str(
                seed
            )
            for seed in seeds
        )
    )

    summary_df[
        "gap_cutoff_source"
    ] = (
        "Materials Project band_gap"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed_min = (
        time.time()
        - t0
    ) / 60.0

    _report(
        progress_callback,
        "Database 1 screening completed",
        1.0,
    )

    if verbose:
        print()
        print("=" * 110)
        print("FINAL DIRECT-ALPHA -> SLME EXTERNAL SCREENING FINISHED")
        print("=" * 110)
        print(
            "All candidates     :",
            len(
                out_df
            ),
        )

        for threshold in thresholds:
            print(
                f"SLME > {threshold:g}%".ljust(
                    20
                )
                + ":",
                len(
                    threshold_tables[
                        threshold
                    ]
                ),
            )

        print(
            "Maximum SLME       :",
            f"{eta_max:.3f}%",
        )
        print(
            "Elapsed            :",
            f"{elapsed_min:.2f} min",
        )
        print()
        print("Saved:")
        print(
            "All candidates     :",
            all_csv,
        )

        for threshold in thresholds:
            print(
                f"SLME > {threshold:g}".ljust(
                    18
                )
                + ":",
                threshold_csvs[
                    threshold
                ],
            )

        print(
            f"Top-{int(top_k)}".ljust(
                18
            )
            + ":",
            top_csv,
        )
        print(
            "Spectra NPZ        :",
            spectra_npz,
        )
        print(
            "Summary            :",
            summary_csv,
        )

    display_columns = [
        "SLME_rank",
        "material_id",
        "formula_pretty",
        "band_gap",
        "energy_above_hull",
        "is_stable",
        "predicted_SLME_percent",
        "predicted_Voc_V",
        "predicted_Jsc_A_m2",
        "SLME_gap_cutoff_eV",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in out_df.columns
    ]

    if show:
        try:
            from IPython.display import display

            print()
            print("Top-20 by FINAL predicted SLME:")

            display(
                out_df[
                    display_columns
                ].head(
                    20
                )
            )

            print()
            print(
                f"Primary discovery pool: "
                f"predicted SLME > {primary_threshold:g}%"
            )

            display(
                threshold_tables[
                    float(
                        primary_threshold
                    )
                ][
                    display_columns
                ].head(
                    50
                )
            )

        except Exception:
            pass

    return {
        "all_df":
            out_df,

        "top_df":
            top_df,

        "threshold_tables":
            threshold_tables,

        "primary_df":
            threshold_tables[
                float(
                    primary_threshold
                )
            ],

        "summary_df":
            summary_df,

        "checkpoint_inventory":
            pd.DataFrame(
                checkpoint_inventory
            ),

        "energy_eV":
            energy_eV,

        "pred_log_alpha":
            pred_log_alpha[
                rank_order
            ],

        "predicted_alpha_cm1":
            alpha_pred_cm1[
                rank_order
            ],

        "predicted_alpha_clean_cm1":
            alpha_clean_cm1[
                rank_order
            ],

        "gap_cutoff_eV":
            mp_gap_eV[
                rank_order
            ],

        "metrics": {
            "n_all":
                int(
                    len(
                        out_df
                    )
                ),

            "n_primary":
                int(
                    len(
                        threshold_tables[
                            float(
                                primary_threshold
                            )
                        ]
                    )
                ),

            "max_SLME_percent":
                eta_max,

            "Psolar_W_m2":
                float(
                    psolar_w_m2
                ),

            "elapsed_min":
                float(
                    elapsed_min
                ),
        },

        "protocol": {
            "route":
                "structure -> Direct-alpha GINE -> SLME",

            "node_dim":
                int(
                    expected_node_dim
                ),

            "edge_dim":
                int(
                    expected_edge_dim
                ),

            "seeds":
                seeds,

            "thickness_nm":
                float(
                    thickness_nm
                ),

            "temperature_K":
                float(
                    temperature_K
                ),

            "fr":
                float(
                    fr
                ),

            "solar_spectrum":
                "ASTM G173 AM1.5G",

            "gap_cutoff_source":
                "Materials Project band_gap",

            "primary_threshold_percent":
                float(
                    primary_threshold
                ),
        },

        "paths": {
            "pool_csv":
                pool_csv,

            "structure_cache":
                structure_cache,

            "direct_alpha_dir":
                direct_alpha_dir,

            "output_dir":
                output_dir,

            "all_csv":
                all_csv,

            "top_csv":
                top_csv,

            "threshold_csvs":
                threshold_csvs,

            "spectra_npz":
                spectra_npz,

            "summary_csv":
                summary_csv,
        },
    }
