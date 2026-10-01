"""
Final DFT-vs-ML SLME validation using the frozen 500 nm screening protocol.
"""

from pathlib import Path
import io
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import lambertw
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from pvlib import spectrum


THICKNESS_NM = 500.0
TEMPERATURE_K = 300.0
FR = 1.0

SOLAR_LAMBDA_MIN_NM = 200.0
SOLAR_LAMBDA_MAX_NM = 2500.0

Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
HC_EV_NM = 1239.8419843320026
HBAR = H / (2.0 * np.pi)


def am15_photon_flux(
    model_energy_eV,
    wavelength_nm,
    irradiance_w_m2_nm,
):
    """
    Convert ASTM G173 spectral irradiance to photon flux on a model energy grid.
    """
    solar_E_eV = (
        HC_EV_NM
        / wavelength_nm
    )

    solar_E_J = (
        solar_E_eV
        * Q
    )

    photon_flux_lambda = (
        irradiance_w_m2_nm
        / solar_E_J
    )

    jacobian = (
        HC_EV_NM
        / solar_E_eV**2
    )

    photon_flux_E = (
        photon_flux_lambda
        * jacobian
    )

    order = np.argsort(
        solar_E_eV
    )

    return np.interp(
        model_energy_eV,
        solar_E_eV[order],
        photon_flux_E[order],
        left=0.0,
        right=0.0,
    )


def blackbody_photon_radiance(
    energy_eV,
    temperature_K,
):
    """
    Blackbody photon radiance per eV used by the final SLME implementation.
    """
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

    exponent[valid] = (
        energy_J[valid]
        /
        (
            KB
            * temperature_K
        )
    )

    safe = (
        valid
        &
        (
            exponent
            < 700.0
        )
    )

    spectral_per_J = (
        2.0
        * energy_J[safe]**2
        /
        (
            H**3
            * C**2
        )
        /
        np.expm1(
            exponent[safe]
        )
    )

    result[safe] = (
        spectral_per_J
        * Q
    )

    return result


def calculate_slme_batch(
    alpha_cm1,
    energy_eV,
    phi_solar,
    phi_bb,
    psolar_w_m2,
    thickness_nm=500.0,
    temperature_K=300.0,
    fr=1.0,
):
    """
    Exact final SLME engine used by the frozen external screening.
    """
    alpha_cm1 = np.asarray(
        alpha_cm1,
        dtype=np.float64,
    )

    if alpha_cm1.ndim == 1:
        alpha_cm1 = (
            alpha_cm1[
                None,
                :
            ]
        )

    thickness_cm = (
        thickness_nm
        * 1e-7
    )

    absorptivity = (
        1.0
        -
        np.exp(
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

    Jsc = (
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

    J0 = (
        Q
        * np.pi
        / fr
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
        * temperature_K
        / Q
    )

    valid = (
        np.isfinite(Jsc)
        &
        np.isfinite(J0)
        &
        (Jsc > 0.0)
        &
        (J0 > 0.0)
    )

    n = len(
        Jsc
    )

    eta = np.full(
        n,
        np.nan,
    )

    Voc = np.full(
        n,
        np.nan,
    )

    Vmp = np.full(
        n,
        np.nan,
    )

    Jmp = np.full(
        n,
        np.nan,
    )

    Pmax = np.full(
        n,
        np.nan,
    )

    ratio = np.full(
        n,
        np.nan,
    )

    ratio[valid] = (
        Jsc[valid]
        / J0[valid]
    )

    Voc[valid] = (
        thermal_voltage
        * np.log1p(
            ratio[valid]
        )
    )

    argument = np.full(
        n,
        np.nan,
    )

    argument[valid] = (
        np.e
        * (
            1.0
            + ratio[valid]
        )
    )

    vmp_dimensionless = np.full(
        n,
        np.nan,
    )

    vmp_dimensionless[valid] = (
        np.real(
            lambertw(
                argument[valid]
            )
        )
        - 1.0
    )

    Vmp[valid] = (
        thermal_voltage
        * vmp_dimensionless[valid]
    )

    exp_vmp = np.full(
        n,
        np.nan,
    )

    exp_vmp[valid] = np.exp(
        np.clip(
            vmp_dimensionless[valid],
            None,
            700.0,
        )
    )

    Jmp[valid] = (
        Jsc[valid]
        -
        J0[valid]
        * (
            exp_vmp[valid]
            - 1.0
        )
    )

    Pmax[valid] = (
        Jmp[valid]
        * Vmp[valid]
    )

    eta[valid] = (
        100.0
        * Pmax[valid]
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


def _find_material_id(
    filename,
    id_to_npz,
    known_ids,
):
    text = (
        str(filename)
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

    if match:
        material_id = (
            match.group(0)
        )

        if material_id in id_to_npz:
            return material_id

    for material_id in known_ids:
        if material_id.lower() in text:
            return material_id

    return None


def _find_rank(
    filename,
):
    base = Path(
        filename
    ).name

    patterns = [
        r"rank[_\- ]*(\d+)",
        r"^(\d{3})",
        r"_(\d{3})_",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            base,
            flags=re.I,
        )

        if match:
            return int(
                match.group(1)
            )

    return np.nan


def run_final_dft22_slme_validation(
    project_root=r"D:\TB3",
    dft_zip=r"D:\DFT_TEST\collect_data_epsilon.zip",
    ml_csv=None,
    ml_npz=None,
    output_dir=None,
    thickness_nm=THICKNESS_NM,
    temperature_K=TEMPERATURE_K,
    fr=FR,
    solar_lambda_min_nm=SOLAR_LAMBDA_MIN_NM,
    solar_lambda_max_nm=SOLAR_LAMBDA_MAX_NM,
    reproduction_tolerance_pp=1e-3,
    show=True,
):
    """
    Reproduce the frozen ML SLME and compare the matched DFT batch against it.

    Scientific protocol
    -------------------
    - ASTM G173 AM1.5G.
    - Solar wavelength window 200-2500 nm.
    - Photon flux interpolated onto the ML energy grid.
    - Psolar = integral E*q*Phi_solar dE.
    - Saved CLEAN ML alpha is used directly.
    - DFT alpha is trace-averaged from eps1/eps2 and set to zero below the
      full-precision CSV band gap.
    - Thickness = 500 nm, T = 300 K, fr = 1.
    - Lambert-W maximum-power solution.
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

    if ml_csv is None:
        ml_csv = (
            screen_dir
            / "zintl_external_1100_direct_alpha_SLME_all.csv"
        )
    else:
        ml_csv = Path(
            ml_csv
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

    dft_zip = Path(
        dft_zip
    )

    if output_dir is None:
        output_dir = (
            root
            / "paper_outputs"
            / "DFT_validation"
            / "FINAL_DFT22_SLME"
        )
    else:
        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = (
        output_dir
        / "FINAL_DFT22_vs_ML_SLME.csv"
    )

    out_png = (
        output_dir
        / "FINAL_DFT22_vs_ML_SLME_parity.png"
    )

    out_pdf = (
        output_dir
        / "FINAL_DFT22_vs_ML_SLME_parity.pdf"
    )

    for path in [
        dft_zip,
        ml_csv,
        ml_npz,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    ml_df = pd.read_csv(
        ml_csv
    )

    ml_df["material_id"] = (
        ml_df["material_id"]
        .astype(str)
        .str.strip()
    )

    if ml_df[
        "material_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate material_id in ML screening CSV."
        )

    z = np.load(
        ml_npz,
        allow_pickle=True,
    )

    required_npz_keys = [
        "material_id",
        "energy_eV",
        "predicted_alpha_cm1",
        "predicted_alpha_clean_cm1",
        "gap_cutoff_eV",
    ]

    missing_npz_keys = [
        key
        for key in required_npz_keys
        if key not in z.files
    ]

    if missing_npz_keys:
        raise KeyError(
            f"Missing NPZ keys: {missing_npz_keys}\n"
            f"Available keys: {z.files}"
        )

    print("=" * 105)
    print("FROZEN ML SCREENING")
    print("=" * 105)
    print(
        "NPZ keys:",
        z.files,
    )

    ml_ids = np.asarray(
        z["material_id"]
    ).astype(str)

    energy_eV = np.asarray(
        z["energy_eV"],
        dtype=np.float64,
    )

    alpha_ml_raw = np.asarray(
        z["predicted_alpha_cm1"],
        dtype=np.float64,
    )

    alpha_ml_clean = np.asarray(
        z["predicted_alpha_clean_cm1"],
        dtype=np.float64,
    )

    gap_npz = np.asarray(
        z["gap_cutoff_eV"],
        dtype=np.float64,
    ).reshape(-1)

    id_to_npz = {
        material_id: index
        for index, material_id
        in enumerate(ml_ids)
    }

    print(
        "Materials    :",
        len(ml_ids),
    )
    print(
        "Alpha shape  :",
        alpha_ml_clean.shape,
    )
    print(
        "Energy range :",
        f"{energy_eV.min():.3f}–"
        f"{energy_eV.max():.3f} eV",
    )
    print(
        "Alpha max    :",
        f"{np.max(alpha_ml_raw):.3e} cm^-1",
    )

    if len(ml_ids) != len(ml_df):
        raise RuntimeError(
            "CSV/NPZ material count mismatch."
        )

    csv_id_set = set(
        ml_df["material_id"]
    )

    for material_id in ml_ids:
        if material_id not in csv_id_set:
            raise RuntimeError(
                f"Missing ID in ML CSV: {material_id}"
            )

    gap_csv = np.asarray(
        [
            float(
                ml_df.loc[
                    ml_df["material_id"]
                    == material_id,
                    "band_gap",
                ].iloc[0]
            )
            for material_id in ml_ids
        ],
        dtype=np.float64,
    )

    max_gap_difference = float(
        np.max(
            np.abs(
                gap_csv
                - gap_npz
            )
        )
    )

    print(
        "Max |CSV gap - NPZ gap|:",
        f"{max_gap_difference:.3e} eV",
    )

    am15 = (
        spectrum.get_reference_spectra()
    )

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
        &
        np.isfinite(
            irradiance_w_m2_nm
        )
        &
        (
            wavelength_nm
            >= float(
                solar_lambda_min_nm
            )
        )
        &
        (
            wavelength_nm
            <= float(
                solar_lambda_max_nm
            )
        )
        &
        (
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

    phi_solar = am15_photon_flux(
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
    print("=" * 105)
    print("AM1.5G SANITY")
    print("=" * 105)
    print(
        f"Psolar = "
        f"{psolar_w_m2:.3f} W/m^2"
    )

    if not (
        994.0
        <= psolar_w_m2
        <= 994.3
    ):
        raise RuntimeError(
            "Psolar does not match FINAL screening."
        )

    print(
        "Psolar check: PASS"
    )

    phi_bb = blackbody_photon_radiance(
        energy_eV,
        temperature_K,
    )

    ml_recalc = calculate_slme_batch(
        alpha_ml_clean,
        energy_eV,
        phi_solar,
        phi_bb,
        psolar_w_m2,
        thickness_nm=thickness_nm,
        temperature_K=temperature_K,
        fr=fr,
    )

    eta_ml_recalc = np.asarray(
        ml_recalc[
            "eta_percent"
        ],
        dtype=np.float64,
    )

    saved_eta_map = dict(
        zip(
            ml_df[
                "material_id"
            ],
            ml_df[
                "predicted_SLME_percent"
            ],
        )
    )

    eta_ml_saved = np.asarray(
        [
            float(
                saved_eta_map[
                    material_id
                ]
            )
            for material_id in ml_ids
        ],
        dtype=np.float64,
    )

    repro_error = np.abs(
        eta_ml_recalc
        - eta_ml_saved
    )

    max_repro_error = float(
        np.nanmax(
            repro_error
        )
    )

    mean_repro_error = float(
        np.nanmean(
            repro_error
        )
    )

    print()
    print("=" * 105)
    print("ML SLME REPRODUCIBILITY")
    print("=" * 105)
    print(
        "Max difference :",
        f"{max_repro_error:.10f} pp",
    )
    print(
        "Mean difference:",
        f"{mean_repro_error:.10f} pp",
    )

    if (
        max_repro_error
        > float(
            reproduction_tolerance_pp
        )
    ):
        raise RuntimeError(
            "ML SLME reproduction FAILED.\n"
            "Do not evaluate DFT."
        )

    print(
        "ML SLME reproduction: PASS"
    )

    known_ids = sorted(
        ml_ids,
        key=len,
        reverse=True,
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
            ]
        )

        print()
        print("=" * 105)
        print("CURRENT DFT BATCH")
        print("=" * 105)
        print(
            "DFT .dat files:",
            len(
                dat_files
            ),
        )

        for filename in dat_files:
            material_id = _find_material_id(
                filename,
                id_to_npz,
                known_ids,
            )

            if material_id is None:
                print(
                    "WARNING — cannot match:",
                    filename,
                )
                continue

            raw_bytes = archive.read(
                filename
            )

            array = np.loadtxt(
                io.BytesIO(
                    raw_bytes
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
                    f"Invalid DFT file: "
                    f"{filename} | "
                    f"{array.shape}"
                )

            energy_dft = (
                array[
                    :,
                    0
                ]
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

            alpha_dft_native = (
                epsilon_to_alpha_cm1(
                    energy_dft,
                    eps1_avg,
                    eps2_avg,
                )
            )

            if (
                len(energy_dft)
                == len(energy_eV)
                and np.allclose(
                    energy_dft,
                    energy_eV,
                    atol=1e-10,
                    rtol=0.0,
                )
            ):
                alpha_dft = (
                    alpha_dft_native.copy()
                )
            else:
                alpha_dft = np.interp(
                    energy_eV,
                    energy_dft,
                    alpha_dft_native,
                    left=0.0,
                    right=0.0,
                )

            ml_row = (
                ml_df.loc[
                    ml_df[
                        "material_id"
                    ]
                    == material_id
                ]
                .iloc[
                    0
                ]
            )

            mp_gap_eV = float(
                ml_row[
                    "band_gap"
                ]
            )

            alpha_dft_clean = (
                alpha_dft.copy()
            )

            alpha_dft_clean[
                energy_eV
                < mp_gap_eV
            ] = 0.0

            records.append(
                {
                    "rank":
                        _find_rank(
                            filename
                        ),

                    "material_id":
                        material_id,

                    "formula_pretty":
                        str(
                            ml_row[
                                "formula_pretty"
                            ]
                        ),

                    "MP_gap_eV":
                        mp_gap_eV,

                    "alpha_dft_clean":
                        alpha_dft_clean,

                    "filename":
                        filename,
                }
            )

    if len(
        records
    ) == 0:
        raise RuntimeError(
            "No DFT files matched."
        )

    ids_check = [
        record[
            "material_id"
        ]
        for record in records
    ]

    if len(
        ids_check
    ) != len(
        set(
            ids_check
        )
    ):
        raise RuntimeError(
            "Duplicate DFT material IDs."
        )

    print(
        "Successfully matched:",
        len(
            records
        ),
    )

    alpha_dft = np.stack(
        [
            record[
                "alpha_dft_clean"
            ]
            for record in records
        ],
        axis=0,
    )

    dft_slme = calculate_slme_batch(
        alpha_dft,
        energy_eV,
        phi_solar,
        phi_bb,
        psolar_w_m2,
        thickness_nm=thickness_nm,
        temperature_K=temperature_K,
        fr=fr,
    )

    eta_dft = np.asarray(
        dft_slme[
            "eta_percent"
        ],
        dtype=np.float64,
    )

    rows = []

    for index, record in enumerate(
        records
    ):
        material_id = record[
            "material_id"
        ]

        eta_ml = float(
            saved_eta_map[
                material_id
            ]
        )

        eta_dft_i = float(
            eta_dft[
                index
            ]
        )

        rows.append(
            {
                "source_rank":
                    record[
                        "rank"
                    ],

                "material_id":
                    material_id,

                "formula_pretty":
                    record[
                        "formula_pretty"
                    ],

                "MP_gap_eV":
                    record[
                        "MP_gap_eV"
                    ],

                "ML_SLME_percent":
                    eta_ml,

                "DFT_SLME_percent":
                    eta_dft_i,

                "ML_minus_DFT_pp":
                    eta_ml
                    - eta_dft_i,

                "abs_error_pp":
                    abs(
                        eta_ml
                        - eta_dft_i
                    ),

                "DFT_gt30":
                    bool(
                        eta_dft_i
                        > 30.0
                    ),

                "DFT_gt31":
                    bool(
                        eta_dft_i
                        > 31.0
                    ),

                "DFT_gt32":
                    bool(
                        eta_dft_i
                        > 32.0
                    ),

                "DFT_gt33":
                    bool(
                        eta_dft_i
                        > 33.0
                    ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df = (
        result_df
        .sort_values(
            "source_rank"
        )
        .reset_index(
            drop=True
        )
    )

    y_dft = result_df[
        "DFT_SLME_percent"
    ].to_numpy(
        dtype=np.float64
    )

    y_ml = result_df[
        "ML_SLME_percent"
    ].to_numpy(
        dtype=np.float64
    )

    valid = (
        np.isfinite(
            y_dft
        )
        &
        np.isfinite(
            y_ml
        )
    )

    yt = y_dft[
        valid
    ]

    yp = y_ml[
        valid
    ]

    mae = float(
        np.mean(
            np.abs(
                yp
                - yt
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    yp
                    - yt
                )**2
            )
        )
    )

    bias = float(
        np.mean(
            yp
            - yt
        )
    )

    r2 = float(
        r2_score(
            yt,
            yp,
        )
    )

    pearson = float(
        pearsonr(
            yt,
            yp,
        ).statistic
    )

    spearman = float(
        spearmanr(
            yt,
            yp,
        ).statistic
    )

    result_df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(
            5.3,
            5.1,
        ),
        dpi=220,
    )

    ax.scatter(
        yt,
        yp,
        s=48,
    )

    lo = float(
        min(
            yt.min(),
            yp.min(),
        )
        - 0.25
    )

    hi = float(
        max(
            yt.max(),
            yp.max(),
        )
        + 0.25
    )

    ax.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
        "--",
        linewidth=1.2,
    )

    ax.axvline(
        30.0,
        linestyle=":",
        linewidth=1.0,
    )

    ax.axhline(
        30.0,
        linestyle=":",
        linewidth=1.0,
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
        "DFT-spectrum SLME (%)"
    )

    ax.set_ylabel(
        "ML-predicted SLME (%)"
    )

    ax.text(
        0.05,
        0.95,
        (
            f"MAE = {mae:.2f} pp\n"
            f"N = {len(yt)}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(
            fig
        )

    print()
    print("=" * 105)
    print("FINAL — DFT vs ML SLME")
    print("=" * 105)
    print(
        f"Psolar             : "
        f"{psolar_w_m2:.3f} W/m^2"
    )
    print(
        "ML reproduction    : PASS"
    )
    print(
        "DFT samples         :",
        len(
            result_df
        ),
    )

    print()
    print(
        "DFT SLME >30%       :",
        int(
            (
                yt > 30.0
            ).sum()
        ),
        "/",
        len(
            yt
        ),
    )
    print(
        "DFT SLME >31%       :",
        int(
            (
                yt > 31.0
            ).sum()
        ),
        "/",
        len(
            yt
        ),
    )
    print(
        "DFT SLME >32%       :",
        int(
            (
                yt > 32.0
            ).sum()
        ),
        "/",
        len(
            yt
        ),
    )
    print(
        "DFT SLME >33%       :",
        int(
            (
                yt > 33.0
            ).sum()
        ),
        "/",
        len(
            yt
        ),
    )

    print()
    print(
        f"DFT range           : "
        f"{yt.min():.4f} – "
        f"{yt.max():.4f}%"
    )
    print(
        f"DFT mean            : "
        f"{yt.mean():.4f}%"
    )
    print(
        f"DFT median          : "
        f"{np.median(yt):.4f}%"
    )

    print()
    print(
        f"ML range            : "
        f"{yp.min():.4f} – "
        f"{yp.max():.4f}%"
    )

    print()
    print(
        f"MAE                 : "
        f"{mae:.4f} pp"
    )
    print(
        f"RMSE                : "
        f"{rmse:.4f} pp"
    )
    print(
        f"Bias ML - DFT       : "
        f"{bias:+.4f} pp"
    )
    print(
        f"R2 selected subset  : "
        f"{r2:.4f}"
    )
    print(
        f"Pearson subset      : "
        f"{pearson:.4f}"
    )
    print(
        f"Spearman subset     : "
        f"{spearman:.4f}"
    )

    print()
    print(
        "Saved CSV:",
        out_csv,
    )

    if show:
        try:
            from IPython.display import display

            display(
                result_df[
                    [
                        "source_rank",
                        "material_id",
                        "formula_pretty",
                        "MP_gap_eV",
                        "ML_SLME_percent",
                        "DFT_SLME_percent",
                        "ML_minus_DFT_pp",
                        "abs_error_pp",
                        "DFT_gt30",
                    ]
                ]
            )
        except Exception:
            print(
                result_df[
                    [
                        "source_rank",
                        "material_id",
                        "formula_pretty",
                        "MP_gap_eV",
                        "ML_SLME_percent",
                        "DFT_SLME_percent",
                        "ML_minus_DFT_pp",
                        "abs_error_pp",
                        "DFT_gt30",
                    ]
                ].to_string(
                    index=False
                )
            )

    metrics = {
        "Psolar_W_m2":
            float(
                psolar_w_m2
            ),

        "ML_reproduction_max_error_pp":
            max_repro_error,

        "ML_reproduction_mean_error_pp":
            mean_repro_error,

        "DFT_samples":
            int(
                len(
                    result_df
                )
            ),

        "DFT_gt30":
            int(
                (
                    yt > 30.0
                ).sum()
            ),

        "DFT_gt31":
            int(
                (
                    yt > 31.0
                ).sum()
            ),

        "DFT_gt32":
            int(
                (
                    yt > 32.0
                ).sum()
            ),

        "DFT_gt33":
            int(
                (
                    yt > 33.0
                ).sum()
            ),

        "DFT_min_percent":
            float(
                yt.min()
            ),

        "DFT_max_percent":
            float(
                yt.max()
            ),

        "DFT_mean_percent":
            float(
                yt.mean()
            ),

        "DFT_median_percent":
            float(
                np.median(
                    yt
                )
            ),

        "ML_min_percent":
            float(
                yp.min()
            ),

        "ML_max_percent":
            float(
                yp.max()
            ),

        "MAE_pp":
            mae,

        "RMSE_pp":
            rmse,

        "Bias_ML_minus_DFT_pp":
            bias,

        "R2_selected_subset":
            r2,

        "Pearson_selected_subset":
            pearson,

        "Spearman_selected_subset":
            spearman,
    }

    return {
        "result_df":
            result_df,

        "metrics":
            metrics,

        "records":
            records,

        "energy_eV":
            energy_eV,

        "alpha_ml_raw":
            alpha_ml_raw,

        "alpha_ml_clean":
            alpha_ml_clean,

        "gap_npz":
            gap_npz,

        "gap_csv":
            gap_csv,

        "phi_solar":
            phi_solar,

        "phi_bb":
            phi_bb,

        "psolar_w_m2":
            float(
                psolar_w_m2
            ),

        "ml_recalc":
            ml_recalc,

        "dft_slme":
            dft_slme,

        "output_csv":
            out_csv,

        "output_png":
            out_png,

        "output_pdf":
            out_pdf,

        "figure":
            fig,

        "axis":
            ax,
    }
