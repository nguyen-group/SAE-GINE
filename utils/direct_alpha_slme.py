"""
Held-out Direct-alpha -> SLME validation for TASK3.

This module reproduces the final 500 nm photovoltaic validation pipeline:
three-seed Direct-alpha ensemble -> physical gap cleanup -> ASTM G173
AM1.5G -> detailed-balance SLME -> SQ reference -> metrics and figure.
"""

from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import lambertw


Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
HC_EV_NM = 1239.8419843320026


def _parse_seed(path, expected_seeds):
    stem = Path(path).stem

    for seed in expected_seeds:
        if stem.endswith(f"seed{seed}"):
            return int(seed)

    return None


def _inverse_log_alpha(log_alpha):
    log_alpha = np.asarray(
        log_alpha,
        dtype=np.float64,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        alpha = (
            np.power(
                10.0,
                log_alpha,
            )
            - 1.0
        )

    alpha = np.nan_to_num(
        alpha,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.maximum(
        alpha,
        0.0,
    )


def _scalar_from_value(value):
    try:
        array = np.asarray(
            value
        ).reshape(-1)

        if len(array) == 0:
            return np.nan

        result = float(
            array[0]
        )

        if np.isfinite(
            result
        ):
            return result

    except Exception:
        pass

    return np.nan


def _find_scalar(
    obj,
    names,
    depth=0,
):
    if obj is None or depth > 6:
        return np.nan

    if isinstance(
        obj,
        dict,
    ):
        for name in names:
            if name in obj:
                result = _scalar_from_value(
                    obj[name]
                )

                if np.isfinite(
                    result
                ):
                    return result

        for value in obj.values():
            result = _find_scalar(
                value,
                names,
                depth + 1,
            )

            if np.isfinite(
                result
            ):
                return result

        return np.nan

    for name in names:
        if hasattr(
            obj,
            name,
        ):
            result = _scalar_from_value(
                getattr(
                    obj,
                    name,
                )
            )

            if np.isfinite(
                result
            ):
                return result

    if hasattr(
        obj,
        "to_dict",
    ):
        try:
            result = _find_scalar(
                obj.to_dict(),
                names,
                depth + 1,
            )

            if np.isfinite(
                result
            ):
                return result

        except Exception:
            pass

    if isinstance(
        obj,
        (list, tuple),
    ):
        for value in obj[:50]:
            result = _find_scalar(
                value,
                names,
                depth + 1,
            )

            if np.isfinite(
                result
            ):
                return result

    return np.nan


def _recover_gap(
    dataset,
    base_idx,
):
    base_idx = int(
        base_idx
    )

    candidate_objects = []

    if hasattr(
        dataset,
        "get_full_item",
    ):
        try:
            candidate_objects.append(
                dataset.get_full_item(
                    base_idx
                )
            )
        except Exception:
            pass

    if hasattr(
        dataset,
        "samples",
    ):
        try:
            candidate_objects.append(
                dataset.samples[
                    base_idx
                ]
            )
        except Exception:
            pass

    try:
        candidate_objects.append(
            dataset[
                base_idx
            ]
        )
    except Exception:
        pass

    direct = np.nan
    indirect = np.nan

    for obj in candidate_objects:
        if not np.isfinite(
            direct
        ):
            direct = _find_scalar(
                obj,
                [
                    "direct_gap",
                    "ipa_direct_gap",
                    "band_gap_direct",
                ],
            )

        if not np.isfinite(
            indirect
        ):
            indirect = _find_scalar(
                obj,
                [
                    "indirect_gap",
                    "ipa_indirect_gap",
                    "band_gap_indirect",
                ],
            )

    return (
        direct,
        indirect,
    )


def _apply_physical_gap(
    alpha_cm1,
    energy_eV,
    gap_eV,
):
    alpha = np.asarray(
        alpha_cm1,
        dtype=np.float64,
    ).copy()

    mask = (
        energy_eV[
            None,
            :
        ]
        < gap_eV[
            :,
            None
        ]
    )

    alpha[
        mask
    ] = 0.0

    return alpha


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


def _slme_from_absorptivity(
    absorptivity,
    energy_eV,
    phi_solar,
    phi_bb,
    psolar_w_m2,
    temperature_K=300.0,
    fr=1.0,
):
    A = np.asarray(
        absorptivity,
        dtype=np.float64,
    )

    if A.ndim != 2:
        raise ValueError(
            "Absorptivity must have shape [samples, energy]."
        )

    if A.shape[1] != len(
        energy_eV
    ):
        raise ValueError(
            "Absorptivity and energy axis do not match."
        )

    if fr <= 0.0:
        raise ValueError(
            "fr must be positive."
        )

    A = np.clip(
        np.nan_to_num(
            A,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    Jsc = (
        Q
        * np.trapezoid(
            A
            * phi_solar[
                None,
                :
            ],
            energy_eV,
            axis=1,
        )
    )

    J0 = (
        Q
        * np.pi
        / fr
        * np.trapezoid(
            A
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
        * temperature_K
        / Q
    )

    n_samples = A.shape[0]

    eta = np.full(
        n_samples,
        np.nan,
    )

    Voc = np.full(
        n_samples,
        np.nan,
    )

    Vmp = np.full(
        n_samples,
        np.nan,
    )

    Jmp = np.full(
        n_samples,
        np.nan,
    )

    Pmax = np.full(
        n_samples,
        np.nan,
    )

    valid = (
        np.isfinite(
            Jsc
        )
        & np.isfinite(
            J0
        )
        & (
            Jsc
            > 0.0
        )
        & (
            J0
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
        Jsc[
            valid
        ]
        / J0[
            valid
        ]
    )

    Voc[
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

    Vmp[
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

    Jmp[
        valid
    ] = (
        Jsc[
            valid
        ]
        - J0[
            valid
        ]
        * (
            exp_vmp[
                valid
            ]
            - 1.0
        )
    )

    Pmax[
        valid
    ] = (
        Jmp[
            valid
        ]
        * Vmp[
            valid
        ]
    )

    eta[
        valid
    ] = (
        100.0
        * Pmax[
            valid
        ]
        / psolar_w_m2
    )

    return {
        "eta_percent":
            eta,

        "Jsc_A_m2":
            Jsc,

        "J0_A_m2":
            J0,

        "Voc_V":
            Voc,

        "Vmp_V":
            Vmp,

        "Jmp_A_m2":
            Jmp,

        "Pmax_W_m2":
            Pmax,
    }


def _finite_thickness_absorptivity(
    alpha_cm1,
    thickness_nm,
):
    alpha_cm1 = np.maximum(
        np.nan_to_num(
            alpha_cm1,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )

    thickness_cm = (
        float(
            thickness_nm
        )
        * 1e-7
    )

    A = (
        1.0
        - np.exp(
            -2.0
            * alpha_cm1
            * thickness_cm
        )
    )

    return np.clip(
        A,
        0.0,
        1.0,
    )


def _topk_overlap(
    dataframe,
    ref_col,
    pred_col,
    k,
):
    tmp = dataframe[
        [
            ref_col,
            pred_col,
        ]
    ].dropna()

    k = min(
        int(
            k
        ),
        len(
            tmp
        ),
    )

    ref_idx = set(
        tmp.nlargest(
            k,
            ref_col,
        ).index
    )

    pred_idx = set(
        tmp.nlargest(
            k,
            pred_col,
        ).index
    )

    count = len(
        ref_idx
        & pred_idx
    )

    return (
        count,
        100.0
        * count
        / k,
    )


def run_final_direct_alpha_slme_validation(
    project_root=r"D:\TB3",
    direct_alpha_dir=None,
    data_zip=None,
    output_dir=None,
    thickness_nm=500.0,
    temperature_K=300.0,
    fr=1.0,
    solar_lambda_min_nm=200.0,
    solar_lambda_max_nm=2500.0,
    expected_model_seeds=(42, 123, 2025),
    top_k_values=(10, 20, 50, 100),
    sq_eg_min=0.20,
    sq_eg_max=6.00,
    sq_n_points=500,
    global_efficiency_ceiling_percent=35.0,
    show=True,
):
    """
    Reproduce the final held-out Direct-alpha -> SLME validation.

    Scientific protocol
    -------------------
    - Three Direct-alpha test predictions are ensembled in
      log10(1 + alpha_cm^-1) target space.
    - The ensemble is inverse transformed to physical alpha.
    - The independently available direct gap is the primary optical cutoff.
      The indirect gap is used only when the direct gap is unavailable.
    - Reference and prediction use the same gap cleanup.
    - ASTM G173 AM1.5G, 200-2500 nm.
    - L = 500 nm, T = 300 K, fr = 1 by default.
    - SLME uses the exact ideal-diode Lambert-W maximum-power solution.
    - An independent SQ step-absorber curve is calculated with the same engine.
    """
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

    processed_dir = (
        project_root
        / "processed"
    )

    training_root = (
        processed_dir
        / "paired_training"
    )

    if data_zip is None:
        data_zip = (
            project_root
            / "database_300.zip"
        )
    else:
        data_zip = Path(
            data_zip
        )

    if direct_alpha_dir is None:
        direct_alpha_dir = (
            training_root
            / "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed"
        )
    else:
        direct_alpha_dir = Path(
            direct_alpha_dir
        )

    if output_dir is None:
        output_dir = (
            training_root
            / "paper_outputs"
            / "FINAL_direct_alpha_SLME_500nm"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_zip.exists():
        raise FileNotFoundError(
            f"Dataset archive not found:\n{data_zip}"
        )

    if not direct_alpha_dir.exists():
        raise FileNotFoundError(
            f"Direct-alpha folder not found:\n{direct_alpha_dir}"
        )

    expected_model_seeds = tuple(
        int(
            seed
        )
        for seed in expected_model_seeds
    )

    expected_seed_set = set(
        expected_model_seeds
    )

    npz_files = sorted(
        direct_alpha_dir.glob(
            "test_predictions_*.npz"
        )
    )

    if len(
        npz_files
    ) != len(
        expected_model_seeds
    ):
        raise RuntimeError(
            f"Expected exactly {len(expected_model_seeds)} prediction files, "
            f"found {len(npz_files)}."
        )

    found_seeds = {
        _parse_seed(
            path,
            expected_model_seeds,
        )
        for path in npz_files
    }

    if found_seeds != expected_seed_set:
        raise RuntimeError(
            f"Expected seeds {sorted(expected_seed_set)}, "
            f"found {sorted(x for x in found_seeds if x is not None)}."
        )

    print("=" * 100)
    print("DIRECT-ALPHA SOURCE")
    print("=" * 100)
    print(
        "Folder :",
        direct_alpha_dir,
    )
    print(
        "Seeds  :",
        sorted(
            found_seeds
        ),
    )

    for path in npz_files:
        print(
            " -",
            path.name,
        )

    pred_log_list = []
    true_log_list = []

    energy_ref = None
    base_idx_ref = None

    for path in npz_files:
        z = np.load(
            path,
            allow_pickle=True,
        )

        pred_log = np.asarray(
            z[
                "pred"
            ],
            dtype=np.float64,
        )

        true_log = np.asarray(
            z[
                "true"
            ],
            dtype=np.float64,
        )

        if (
            pred_log.ndim == 3
            and pred_log.shape[1] == 1
        ):
            pred_log = (
                pred_log[
                    :,
                    0,
                    :
                ]
            )

        if (
            true_log.ndim == 3
            and true_log.shape[1] == 1
        ):
            true_log = (
                true_log[
                    :,
                    0,
                    :
                ]
            )

        if (
            pred_log.ndim != 2
            or true_log.ndim != 2
        ):
            raise RuntimeError(
                f"Unexpected spectrum shape in {path.name}: "
                f"pred={pred_log.shape}, true={true_log.shape}"
            )

        if "spectral_axis" in z.files:
            energy = np.asarray(
                z[
                    "spectral_axis"
                ],
                dtype=np.float64,
            )

        elif "axis_grid" in z.files:
            energy = np.asarray(
                z[
                    "axis_grid"
                ],
                dtype=np.float64,
            )

        else:
            raise KeyError(
                f"No spectral axis found in {path.name}."
            )

        if energy.ndim == 2:
            energy = (
                energy[
                    0
                ]
            )

        energy = energy.reshape(
            -1
        )

        if "base_idx" not in z.files:
            raise KeyError(
                f"No base_idx found in {path.name}."
            )

        base_idx = np.asarray(
            z[
                "base_idx"
            ]
        ).reshape(
            -1
        )

        if pred_log.shape != true_log.shape:
            raise RuntimeError(
                f"Prediction/reference shape mismatch in {path.name}."
            )

        if pred_log.shape[1] != len(
            energy
        ):
            raise RuntimeError(
                f"Spectrum/axis length mismatch in {path.name}."
            )

        if len(
            base_idx
        ) != pred_log.shape[0]:
            raise RuntimeError(
                f"base_idx/sample mismatch in {path.name}."
            )

        if energy_ref is None:
            energy_ref = energy.copy()
            base_idx_ref = base_idx.copy()

        else:
            if not np.allclose(
                energy_ref,
                energy,
                rtol=0.0,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"Energy-axis mismatch in {path.name}."
                )

            if not np.array_equal(
                base_idx_ref,
                base_idx,
            ):
                raise RuntimeError(
                    f"base_idx mismatch in {path.name}."
                )

            if not np.allclose(
                true_log_list[
                    0
                ],
                true_log,
                rtol=1e-6,
                atol=1e-7,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"Ground truth differs across seeds in {path.name}."
                )

        pred_log_list.append(
            pred_log
        )

        true_log_list.append(
            true_log
        )

    energy_eV = np.asarray(
        energy_ref,
        dtype=np.float64,
    )

    true_log_alpha = (
        true_log_list[
            0
        ]
    )

    pred_log_alpha = np.mean(
        np.stack(
            pred_log_list,
            axis=0,
        ),
        axis=0,
    )

    alpha_reference_cm1 = _inverse_log_alpha(
        true_log_alpha
    )

    alpha_predicted_cm1 = _inverse_log_alpha(
        pred_log_alpha
    )

    print()
    print("=" * 100)
    print("ALPHA AUDIT")
    print("=" * 100)
    print(
        "Shape               :",
        alpha_reference_cm1.shape,
    )
    print(
        "Energy range        :",
        f"{energy_eV.min():.3f} to {energy_eV.max():.3f} eV",
    )
    print(
        "Reference alpha max :",
        f"{np.max(alpha_reference_cm1):.3e} cm^-1",
    )
    print(
        "Predicted alpha max :",
        f"{np.max(alpha_predicted_cm1):.3e} cm^-1",
    )

    import utils_tb.data as DATA

    DATA = importlib.reload(
        DATA
    )

    raw_ds = DATA.CrystalOpticalDataset(
        zip_path=data_zip,
        require_any_target=True,
        require_full_spectrum=False,
        verbose=False,
    )

    direct_gap_eV = np.full(
        len(
            base_idx_ref
        ),
        np.nan,
        dtype=np.float64,
    )

    indirect_gap_eV = np.full_like(
        direct_gap_eV,
        np.nan,
    )

    for index, base_idx in enumerate(
        base_idx_ref
    ):
        direct_gap, indirect_gap = _recover_gap(
            raw_ds,
            base_idx,
        )

        direct_gap_eV[
            index
        ] = direct_gap

        indirect_gap_eV[
            index
        ] = indirect_gap

    optical_gap_eV = direct_gap_eV.copy()

    fallback_mask = (
        ~np.isfinite(
            optical_gap_eV
        )
        | (
            optical_gap_eV
            <= 0.0
        )
    )

    for index in np.where(
        fallback_mask
    )[0]:
        candidates = [
            gap
            for gap in [
                direct_gap_eV[
                    index
                ],
                indirect_gap_eV[
                    index
                ],
            ]
            if (
                np.isfinite(
                    gap
                )
                and gap > 0.0
            )
        ]

        if candidates:
            optical_gap_eV[
                index
            ] = min(
                candidates
            )

    invalid_gap = (
        ~np.isfinite(
            optical_gap_eV
        )
        | (
            optical_gap_eV
            <= 0.0
        )
    )

    if invalid_gap.any():
        raise RuntimeError(
            f"Missing positive optical gap for "
            f"{int(invalid_gap.sum())} test samples."
        )

    print()
    print("=" * 100)
    print("BAND-GAP AUDIT")
    print("=" * 100)
    print(
        "Direct-gap range   :",
        f"{np.nanmin(direct_gap_eV):.6f}",
        "to",
        f"{np.nanmax(direct_gap_eV):.6f}",
        "eV",
    )
    print(
        "Indirect-gap range :",
        f"{np.nanmin(indirect_gap_eV):.6f}",
        "to",
        f"{np.nanmax(indirect_gap_eV):.6f}",
        "eV",
    )
    print(
        "Optical cutoff     :",
        f"{np.nanmin(optical_gap_eV):.6f}",
        "to",
        f"{np.nanmax(optical_gap_eV):.6f}",
        "eV",
    )
    print(
        "Direct-gap fallback:",
        int(
            fallback_mask.sum()
        ),
    )

    alpha_reference_clean = _apply_physical_gap(
        alpha_reference_cm1,
        energy_eV,
        optical_gap_eV,
    )

    alpha_predicted_clean = _apply_physical_gap(
        alpha_predicted_cm1,
        energy_eV,
        optical_gap_eV,
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

    print()
    print("=" * 100)
    print("AM1.5G AUDIT")
    print("=" * 100)
    print(
        f"Psolar "
        f"({float(solar_lambda_min_nm):.0f}-"
        f"{float(solar_lambda_max_nm):.0f} nm) "
        f"= {psolar_w_m2:.3f} W/m^2"
    )

    if not (
        900.0
        <= psolar_w_m2
        <= 1050.0
    ):
        raise RuntimeError(
            "AM1.5G integrated power is inconsistent "
            "with expected ~1000 W/m^2."
        )

    phi_bb = _blackbody_photon_radiance(
        energy_eV,
        float(
            temperature_K
        ),
    )

    A_reference = _finite_thickness_absorptivity(
        alpha_reference_clean,
        thickness_nm,
    )

    A_predicted = _finite_thickness_absorptivity(
        alpha_predicted_clean,
        thickness_nm,
    )

    slme_reference = _slme_from_absorptivity(
        A_reference,
        energy_eV,
        phi_solar,
        phi_bb,
        psolar_w_m2,
        temperature_K=float(
            temperature_K
        ),
        fr=float(
            fr
        ),
    )

    slme_predicted = _slme_from_absorptivity(
        A_predicted,
        energy_eV,
        phi_solar,
        phi_bb,
        psolar_w_m2,
        temperature_K=float(
            temperature_K
        ),
        fr=float(
            fr
        ),
    )

    sq_eg_grid = np.linspace(
        float(
            sq_eg_min
        ),
        float(
            sq_eg_max
        ),
        int(
            sq_n_points
        ),
    )

    sq_absorptivity = (
        energy_eV[
            None,
            :
        ]
        >= sq_eg_grid[
            :,
            None
        ]
    ).astype(
        np.float64
    )

    sq_result = _slme_from_absorptivity(
        sq_absorptivity,
        energy_eV,
        phi_solar,
        phi_bb,
        psolar_w_m2,
        temperature_K=float(
            temperature_K
        ),
        fr=1.0,
    )

    sq_eta_grid = (
        sq_result[
            "eta_percent"
        ]
    )

    sq_global_index = int(
        np.nanargmax(
            sq_eta_grid
        )
    )

    sq_global_max_percent = float(
        sq_eta_grid[
            sq_global_index
        ]
    )

    sq_global_max_eg_eV = float(
        sq_eg_grid[
            sq_global_index
        ]
    )

    sample_df = pd.DataFrame(
        {
            "base_idx":
                base_idx_ref,

            "sample_id":
                base_idx_ref.astype(
                    str
                ),

            "direct_gap_eV":
                direct_gap_eV,

            "indirect_gap_eV":
                indirect_gap_eV,

            "optical_gap_cutoff_eV":
                optical_gap_eV,

            "reference_SLME_percent":
                slme_reference[
                    "eta_percent"
                ],

            "predicted_SLME_percent":
                slme_predicted[
                    "eta_percent"
                ],

            "reference_Jsc_A_m2":
                slme_reference[
                    "Jsc_A_m2"
                ],

            "predicted_Jsc_A_m2":
                slme_predicted[
                    "Jsc_A_m2"
                ],

            "reference_J0_A_m2":
                slme_reference[
                    "J0_A_m2"
                ],

            "predicted_J0_A_m2":
                slme_predicted[
                    "J0_A_m2"
                ],

            "reference_Voc_V":
                slme_reference[
                    "Voc_V"
                ],

            "predicted_Voc_V":
                slme_predicted[
                    "Voc_V"
                ],

            "reference_Vmp_V":
                slme_reference[
                    "Vmp_V"
                ],

            "predicted_Vmp_V":
                slme_predicted[
                    "Vmp_V"
                ],

            "reference_Pmax_W_m2":
                slme_reference[
                    "Pmax_W_m2"
                ],

            "predicted_Pmax_W_m2":
                slme_predicted[
                    "Pmax_W_m2"
                ],

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
        }
    )

    ref_col = (
        "reference_SLME_percent"
    )

    pred_col = (
        "predicted_SLME_percent"
    )

    valid_df = sample_df[
        [
            ref_col,
            pred_col,
        ]
    ].dropna()

    x = valid_df[
        ref_col
    ].to_numpy()

    y = valid_df[
        pred_col
    ].to_numpy()

    pearson = float(
        np.corrcoef(
            x,
            y,
        )[
            0,
            1
        ]
    )

    spearman = float(
        np.corrcoef(
            pd.Series(
                x
            ).rank().to_numpy(),
            pd.Series(
                y
            ).rank().to_numpy(),
        )[
            0,
            1
        ]
    )

    ss_res = float(
        np.sum(
            (
                x
                - y
            )**2
        )
    )

    ss_tot = float(
        np.sum(
            (
                x
                - np.mean(
                    x
                )
            )**2
        )
    )

    r2 = (
        1.0
        - ss_res
        / ss_tot
    )

    mae_pp = float(
        np.mean(
            np.abs(
                x
                - y
            )
        )
    )

    rmse_pp = float(
        np.sqrt(
            np.mean(
                (
                    x
                    - y
                )**2
            )
        )
    )

    topk_results = {}

    for k in top_k_values:
        count, percent = _topk_overlap(
            sample_df,
            ref_col,
            pred_col,
            k,
        )

        topk_results[
            int(
                k
            )
        ] = {
            "count":
                count,

            "percent":
                percent,
        }

    reference_max = float(
        np.nanmax(
            sample_df[
                ref_col
            ]
        )
    )

    predicted_max = float(
        np.nanmax(
            sample_df[
                pred_col
            ]
        )
    )

    print()
    print("=" * 100)
    print("FINAL SLME / SQ SANITY CHECK")
    print("=" * 100)
    print(
        f"SQ maximum             : "
        f"{sq_global_max_percent:.3f}% "
        f"at Eg = {sq_global_max_eg_eV:.3f} eV"
    )
    print(
        f"Reference SLME range   : "
        f"{np.nanmin(sample_df[ref_col]):.3f}% "
        f"to {reference_max:.3f}%"
    )
    print(
        f"Predicted SLME range   : "
        f"{np.nanmin(sample_df[pred_col]):.3f}% "
        f"to {predicted_max:.3f}%"
    )
    print(
        f"Reference > 35%        : "
        f"{int(np.sum(sample_df[ref_col] > 35.0))}"
    )
    print(
        f"Predicted > 35%        : "
        f"{int(np.sum(sample_df[pred_col] > 35.0))}"
    )
    print()
    print(
        f"Pearson                : "
        f"{pearson:.4f}"
    )
    print(
        f"Spearman               : "
        f"{spearman:.4f}"
    )
    print(
        f"R2                     : "
        f"{r2:.4f}"
    )
    print(
        f"MAE                    : "
        f"{mae_pp:.4f} percentage points"
    )
    print(
        f"RMSE                   : "
        f"{rmse_pp:.4f} percentage points"
    )
    print()

    for k in top_k_values:
        result = topk_results[
            int(
                k
            )
        ]

        print(
            f"Top-{int(k):<3d} overlap        : "
            f"{result['count']}/{int(k)} "
            f"({result['percent']:.1f}%)"
        )

    if not (
        30.0
        <= sq_global_max_percent
        <= float(
            global_efficiency_ceiling_percent
        )
    ):
        raise RuntimeError(
            "SQ maximum is outside the expected physical range."
        )

    if reference_max > float(
        global_efficiency_ceiling_percent
    ):
        raise RuntimeError(
            "Reference SLME exceeds the configured global ceiling. "
            "Do not use this result."
        )

    if predicted_max > float(
        global_efficiency_ceiling_percent
    ):
        raise RuntimeError(
            "Predicted SLME exceeds the configured global ceiling. "
            "Do not use this result."
        )

    eval_row = {
        "route":
            "Direct-alpha SSL-init GINE -> SLME",

        "alpha_target":
            "log10(1 + alpha_cm^-1)",

        "n_test_samples":
            len(
                sample_df
            ),

        "n_model_seeds":
            len(
                npz_files
            ),

        "model_seeds":
            ",".join(
                str(
                    seed
                )
                for seed in sorted(
                    found_seeds
                )
            ),

        "gap_cutoff":
            "independent direct gap",

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

        "Psolar_W_m2":
            psolar_w_m2,

        "SQ_max_percent":
            sq_global_max_percent,

        "SQ_max_Eg_eV":
            sq_global_max_eg_eV,

        "Pearson":
            pearson,

        "Spearman":
            spearman,

        "R2":
            r2,

        "MAE_percentage_points":
            mae_pp,

        "RMSE_percentage_points":
            rmse_pp,

        "reference_SLME_max_percent":
            reference_max,

        "predicted_SLME_max_percent":
            predicted_max,
    }

    for k in top_k_values:
        key = int(
            k
        )

        eval_row[
            f"Top{key}_overlap_count"
        ] = (
            topk_results[
                key
            ][
                "count"
            ]
        )

        eval_row[
            f"Top{key}_overlap_percent"
        ] = (
            topk_results[
                key
            ][
                "percent"
            ]
        )

    eval_df = pd.DataFrame(
        [
            eval_row
        ]
    )

    if show:
        try:
            from IPython.display import display

            display(
                eval_df
            )

        except Exception:
            print(
                eval_df.to_string(
                    index=False
                )
            )

    sample_csv = (
        output_dir
        / "FINAL_direct_alpha_SLME_test_samples.csv"
    )

    eval_csv = (
        output_dir
        / "FINAL_direct_alpha_SLME_evaluation.csv"
    )

    sq_csv = (
        output_dir
        / "FINAL_SQ_reference_curve.csv"
    )

    sample_df.to_csv(
        sample_csv,
        index=False,
        encoding="utf-8-sig",
    )

    eval_df.to_csv(
        eval_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "Eg_eV":
                sq_eg_grid,

            "SQ_efficiency_percent":
                sq_eta_grid,
        }
    ).to_csv(
        sq_csv,
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15.5,
            4.7,
        ),
        dpi=300,
        constrained_layout=True,
    )

    axis_max = 35.0

    axes[
        0
    ].scatter(
        x,
        y,
        s=13,
        alpha=0.52,
    )

    axes[
        0
    ].plot(
        [
            0.0,
            axis_max,
        ],
        [
            0.0,
            axis_max,
        ],
        "--",
        linewidth=1.4,
    )

    axes[
        0
    ].set_xlim(
        0.0,
        axis_max,
    )

    axes[
        0
    ].set_ylim(
        0.0,
        axis_max,
    )

    axes[
        0
    ].set_aspect(
        "equal",
        adjustable="box",
    )

    axes[
        0
    ].set_xlabel(
        r"Reference SLME, $\eta$ (%)",
        fontsize=11,
    )

    axes[
        0
    ].set_ylabel(
        r"Predicted SLME, $\eta$ (%)",
        fontsize=11,
    )

    axes[
        0
    ].text(
        0.045,
        0.955,
        (
            f"Pearson = {pearson:.3f}\n"
            f"Spearman = {spearman:.3f}\n"
            f"$R^2$ = {r2:.3f}"
        ),
        transform=axes[
            0
        ].transAxes,
        va="top",
        fontsize=10.5,
    )

    axes[
        0
    ].text(
        -0.12,
        1.04,
        "a",
        transform=axes[
            0
        ].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    axes[
        1
    ].plot(
        sq_eg_grid,
        sq_eta_grid,
        linewidth=2.0,
        label="SQ limit",
    )

    axes[
        1
    ].scatter(
        optical_gap_eV,
        sample_df[
            ref_col
        ],
        s=10,
        alpha=0.34,
        label="Reference",
    )

    axes[
        1
    ].scatter(
        optical_gap_eV,
        sample_df[
            pred_col
        ],
        s=10,
        alpha=0.34,
        label="Prediction",
    )

    axes[
        1
    ].set_xlim(
        0.0,
        4.5,
    )

    axes[
        1
    ].set_ylim(
        0.0,
        35.5,
    )

    axes[
        1
    ].set_xlabel(
        r"Direct gap, $E_g$ (eV)",
        fontsize=11,
    )

    axes[
        1
    ].set_ylabel(
        r"Solar conversion efficiency, $\eta$ (%)",
        fontsize=11,
    )

    axes[
        1
    ].legend(
        frameon=False,
        fontsize=9,
    )

    axes[
        1
    ].text(
        -0.12,
        1.04,
        "b",
        transform=axes[
            1
        ].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    topk_percentages = [
        topk_results[
            int(
                k
            )
        ][
            "percent"
        ]
        for k in top_k_values
    ]

    topk_counts = [
        topk_results[
            int(
                k
            )
        ][
            "count"
        ]
        for k in top_k_values
    ]

    axes[
        2
    ].plot(
        top_k_values,
        topk_percentages,
        marker="o",
        linewidth=1.8,
    )

    for k, count, percentage in zip(
        top_k_values,
        topk_counts,
        topk_percentages,
    ):
        axes[
            2
        ].annotate(
            f"{count}/{int(k)}",
            (
                k,
                percentage,
            ),
            xytext=(
                0,
                7,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
        )

    axes[
        2
    ].set_xlim(
        5,
        105,
    )

    axes[
        2
    ].set_ylim(
        0,
        100,
    )

    axes[
        2
    ].set_xticks(
        top_k_values
    )

    axes[
        2
    ].set_xlabel(
        "Top-k materials",
        fontsize=11,
    )

    axes[
        2
    ].set_ylabel(
        "Overlap with reference top-k (%)",
        fontsize=11,
    )

    axes[
        2
    ].text(
        -0.12,
        1.04,
        "c",
        transform=axes[
            2
        ].transAxes,
        fontweight="bold",
        fontsize=12,
    )

    for ax in axes:
        ax.tick_params(
            labelsize=9.5,
        )

    fig_png = (
        output_dir
        / "FINAL_direct_alpha_SLME_validation.png"
    )

    fig_pdf = (
        output_dir
        / "FINAL_direct_alpha_SLME_validation.pdf"
    )

    fig.savefig(
        fig_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        fig_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    if show:
        plt.show()
    else:
        plt.close(
            fig
        )

    print()
    print("=" * 100)
    print("FINAL OUTPUTS")
    print("=" * 100)

    for path in [
        sample_csv,
        eval_csv,
        sq_csv,
        fig_png,
        fig_pdf,
    ]:
        print(
            path
        )

    return {
        "sample_df":
            sample_df,

        "evaluation_df":
            eval_df,

        "topk_results":
            topk_results,

        "metrics": {
            "Pearson":
                pearson,

            "Spearman":
                spearman,

            "R2":
                float(
                    r2
                ),

            "MAE_pp":
                mae_pp,

            "RMSE_pp":
                rmse_pp,

            "Psolar_W_m2":
                float(
                    psolar_w_m2
                ),

            "SQ_max_percent":
                sq_global_max_percent,

            "SQ_max_Eg_eV":
                sq_global_max_eg_eV,

            "reference_SLME_max_percent":
                reference_max,

            "predicted_SLME_max_percent":
                predicted_max,
        },

        "energy_eV":
            energy_eV,

        "base_idx":
            base_idx_ref,

        "direct_gap_eV":
            direct_gap_eV,

        "indirect_gap_eV":
            indirect_gap_eV,

        "optical_gap_eV":
            optical_gap_eV,

        "alpha_reference_cm1":
            alpha_reference_cm1,

        "alpha_predicted_cm1":
            alpha_predicted_cm1,

        "alpha_reference_clean_cm1":
            alpha_reference_clean,

        "alpha_predicted_clean_cm1":
            alpha_predicted_clean,

        "slme_reference":
            slme_reference,

        "slme_predicted":
            slme_predicted,

        "sq_eg_grid":
            sq_eg_grid,

        "sq_eta_grid":
            sq_eta_grid,

        "figure":
            fig,

        "axes":
            axes,

        "sample_csv":
            sample_csv,

        "evaluation_csv":
            eval_csv,

        "sq_csv":
            sq_csv,

        "figure_png":
            fig_png,

        "figure_pdf":
            fig_pdf,

        "direct_alpha_dir":
            direct_alpha_dir,

        "data_zip":
            data_zip,

        "output_dir":
            output_dir,
    }
