"""
Final screening-input audit utilities for TASK3.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def audit_final_screening_inputs(
    project_root=r"D:\TB3",
    ml_table=None,
    ml_npz=None,
    expected_candidates=1100,
    expected_energy_points=2001,
    reconstruction_tolerance=1e-8,
    show_first=10,
):
    """
    Audit the frozen external-screening inputs used for final SLME analysis.

    Checks
    ------
    1. Required NPZ keys exist.
    2. Candidate count is 1100.
    3. Raw and cleaned alpha arrays have the expected shape.
    4. Clean alpha is exactly reconstructed by zeroing raw alpha below
       ``gap_cutoff_eV``.
    5. CSV ``band_gap`` agrees with NPZ ``gap_cutoff_eV`` by material ID.

    Returns
    -------
    dict
        Loaded arrays, audit metrics, input paths, and the screening table.
    """
    project_root = Path(project_root)

    root = (
        project_root
        / "processed"
        / "paired_training"
    )

    ml_dir = (
        root
        / "paper_outputs"
        / "screening_SLME_direct_alpha_500nm"
    )

    if ml_table is None:
        ml_table = (
            ml_dir
            / "zintl_external_1100_direct_alpha_SLME_all.csv"
        )
    else:
        ml_table = Path(ml_table)

    if ml_npz is None:
        ml_npz = (
            ml_dir
            / "zintl_external_1100_direct_alpha_SLME_spectra.npz"
        )
    else:
        ml_npz = Path(ml_npz)

    if not ml_table.exists():
        raise FileNotFoundError(
            f"Screening CSV not found: {ml_table}"
        )

    if not ml_npz.exists():
        raise FileNotFoundError(
            f"Screening NPZ not found: {ml_npz}"
        )

    df = pd.read_csv(
        ml_table
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

    missing_npz = [
        key
        for key in required_npz_keys
        if key not in z.files
    ]

    if missing_npz:
        raise KeyError(
            f"Missing NPZ keys: {missing_npz}\n"
            f"Available keys: {z.files}"
        )

    required_csv_columns = [
        "material_id",
        "band_gap",
    ]

    missing_csv = [
        column
        for column in required_csv_columns
        if column not in df.columns
    ]

    if missing_csv:
        raise KeyError(
            f"Missing CSV columns: {missing_csv}"
        )

    ids = np.asarray(
        z["material_id"]
    ).astype(str)

    energy = np.asarray(
        z["energy_eV"],
        dtype=np.float64,
    )

    alpha_raw = np.asarray(
        z["predicted_alpha_cm1"],
        dtype=np.float64,
    )

    alpha_clean = np.asarray(
        z["predicted_alpha_clean_cm1"],
        dtype=np.float64,
    )

    gap_npz = np.asarray(
        z["gap_cutoff_eV"],
        dtype=np.float64,
    ).reshape(-1)

    print("=" * 100)
    print("FINAL SCREENING INPUT AUDIT")
    print("=" * 100)

    print("CSV:")
    print(ml_table)

    print("\nNPZ:")
    print(ml_npz)

    print("\nNPZ keys:")
    print(z.files)

    print()
    print(
        "Materials          :",
        len(ids),
    )
    print(
        "Energy points      :",
        len(energy),
    )
    print(
        "Raw alpha shape    :",
        alpha_raw.shape,
    )
    print(
        "Clean alpha shape  :",
        alpha_clean.shape,
    )
    print(
        "Gap-cutoff shape   :",
        gap_npz.shape,
    )

    if len(ids) != int(expected_candidates):
        raise RuntimeError(
            f"Expected {expected_candidates} screening candidates, "
            f"found {len(ids)}."
        )

    expected_shape = (
        int(expected_candidates),
        int(expected_energy_points),
    )

    if alpha_raw.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected raw alpha shape: {alpha_raw.shape}. "
            f"Expected {expected_shape}."
        )

    if alpha_clean.shape != alpha_raw.shape:
        raise RuntimeError(
            "Raw/clean alpha shape mismatch."
        )

    if len(gap_npz) != len(ids):
        raise RuntimeError(
            "gap_cutoff_eV length mismatch."
        )

    if len(energy) != int(expected_energy_points):
        raise RuntimeError(
            f"Expected {expected_energy_points} energy points, "
            f"found {len(energy)}."
        )

    if not np.all(
        np.diff(energy) > 0
    ):
        raise RuntimeError(
            "Energy axis must be strictly increasing."
        )

    reconstructed = alpha_raw.copy()

    for index in range(len(ids)):
        reconstructed[
            index,
            energy < gap_npz[index],
        ] = 0.0

    max_clean_error = float(
        np.max(
            np.abs(
                reconstructed
                - alpha_clean
            )
        )
    )

    print()
    print(
        "Max error: "
        "raw alpha + NPZ gap cutoff -> saved clean alpha =",
        f"{max_clean_error:.6e}",
    )

    df = df.copy()

    df["material_id"] = (
        df["material_id"]
        .astype(str)
        .str.strip()
    )

    ids_clean = np.asarray(
        [
            str(material_id).strip()
            for material_id in ids
        ]
    )

    if len(set(ids_clean)) != len(ids_clean):
        raise RuntimeError(
            "Duplicate material IDs found in NPZ."
        )

    if df["material_id"].duplicated().any():
        duplicate_ids = (
            df.loc[
                df["material_id"].duplicated(
                    keep=False
                ),
                "material_id",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate material IDs found in CSV: {duplicate_ids}"
        )

    id_to_row = {
        material_id: index
        for index, material_id
        in enumerate(ids_clean)
    }

    csv_ids = set(
        df["material_id"]
    )

    npz_ids = set(
        ids_clean
    )

    missing_in_npz = sorted(
        csv_ids
        - npz_ids
    )

    missing_in_csv = sorted(
        npz_ids
        - csv_ids
    )

    if missing_in_npz or missing_in_csv:
        raise RuntimeError(
            "CSV/NPZ material-ID mismatch.\n"
            f"Missing in NPZ: {missing_in_npz[:10]}\n"
            f"Missing in CSV: {missing_in_csv[:10]}"
        )

    gap_diff = []

    for _, row in df.iterrows():
        material_id = str(
            row["material_id"]
        ).strip()

        index = id_to_row[
            material_id
        ]

        gap_diff.append(
            float(row["band_gap"])
            - float(gap_npz[index])
        )

    gap_diff = np.asarray(
        gap_diff,
        dtype=np.float64,
    )

    max_gap_difference = float(
        np.max(
            np.abs(
                gap_diff
            )
        )
    )

    print(
        "Max |CSV band_gap - NPZ gap_cutoff| =",
        f"{max_gap_difference:.12e} eV",
    )

    print()
    print(
        f"First {min(int(show_first), len(ids_clean))} candidates:"
    )

    for index in range(
        min(
            int(show_first),
            len(ids_clean),
        )
    ):
        print(
            f"{index + 1:02d}",
            ids_clean[index],
            "| gap =",
            f"{gap_npz[index]:.6f}",
            "| raw max =",
            f"{alpha_raw[index].max():.3e}",
            "| clean max =",
            f"{alpha_clean[index].max():.3e}",
        )

    reconstruction_passed = (
        max_clean_error
        < float(
            reconstruction_tolerance
        )
    )

    print()
    print("=" * 100)

    if reconstruction_passed:
        print(
            "CLEAN-ALPHA RECONSTRUCTION: PASS"
        )
    else:
        print(
            "CLEAN-ALPHA RECONSTRUCTION: CHECK"
        )

    print("=" * 100)

    return {
        "dataframe":
            df,

        "material_ids":
            ids_clean,

        "energy_eV":
            energy,

        "alpha_raw":
            alpha_raw,

        "alpha_clean":
            alpha_clean,

        "gap_cutoff_eV":
            gap_npz,

        "gap_difference_eV":
            gap_diff,

        "max_clean_error":
            max_clean_error,

        "max_gap_difference_eV":
            max_gap_difference,

        "reconstruction_passed":
            reconstruction_passed,

        "ml_table":
            ml_table,

        "ml_npz":
            ml_npz,
    }



def audit_clean_alpha_forensics(
    project_root=r"D:\TB3",
    ml_npz=None,
    tolerance=1e-8,
    thresholds=(1, 10, 100, 1e3, 5e3, 1e4, 2e4, 5e4, 1e5),
    top_k=20,
):
    """
    Forensically audit how ``predicted_alpha_clean_cm1`` differs from
    ``predicted_alpha_cm1``.

    The analysis checks:
    1. Where values changed relative to the band-gap cutoff.
    2. Whether every modification is raw -> 0.
    3. Whether additional alpha-threshold rules reproduce the saved clean alpha.
    4. The largest individual raw/clean differences.

    Returns
    -------
    dict
        Summary counts, threshold-test table, top modified points, and arrays.
    """
    project_root = Path(project_root)

    root = (
        project_root
        / "processed"
        / "paired_training"
    )

    if ml_npz is None:
        ml_npz = (
            root
            / "paper_outputs"
            / "screening_SLME_direct_alpha_500nm"
            / "zintl_external_1100_direct_alpha_SLME_spectra.npz"
        )
    else:
        ml_npz = Path(ml_npz)

    if not ml_npz.exists():
        raise FileNotFoundError(
            f"Screening NPZ not found: {ml_npz}"
        )

    z = np.load(
        ml_npz,
        allow_pickle=True,
    )

    required_keys = [
        "material_id",
        "energy_eV",
        "predicted_alpha_cm1",
        "predicted_alpha_clean_cm1",
        "gap_cutoff_eV",
    ]

    missing = [
        key
        for key in required_keys
        if key not in z.files
    ]

    if missing:
        raise KeyError(
            f"Missing NPZ keys: {missing}\n"
            f"Available keys: {z.files}"
        )

    ids = np.asarray(
        z["material_id"]
    ).astype(str)

    energy = np.asarray(
        z["energy_eV"],
        dtype=np.float64,
    )

    raw = np.asarray(
        z["predicted_alpha_cm1"],
        dtype=np.float64,
    )

    clean = np.asarray(
        z["predicted_alpha_clean_cm1"],
        dtype=np.float64,
    )

    gap = np.asarray(
        z["gap_cutoff_eV"],
        dtype=np.float64,
    ).reshape(-1)

    if raw.shape != clean.shape:
        raise RuntimeError(
            "Raw/clean alpha shape mismatch."
        )

    if raw.ndim != 2:
        raise RuntimeError(
            f"Expected 2D alpha arrays, found shape {raw.shape}."
        )

    if raw.shape[0] != len(ids):
        raise RuntimeError(
            "Material-ID count does not match alpha rows."
        )

    if raw.shape[1] != len(energy):
        raise RuntimeError(
            "Energy-axis length does not match alpha columns."
        )

    if len(gap) != len(ids):
        raise RuntimeError(
            "gap_cutoff_eV length mismatch."
        )

    tol = float(tolerance)

    changed = (
        np.abs(
            clean - raw
        )
        > tol
    )

    below_gap = (
        energy[None, :]
        < gap[:, None]
    )

    above_gap = ~below_gap

    changed_below = (
        changed
        & below_gap
    )

    changed_above = (
        changed
        & above_gap
    )

    clean_zero = (
        np.abs(clean)
        <= tol
    )

    changed_to_zero = (
        changed
        & clean_zero
    )

    changed_nonzero = (
        changed
        & (~clean_zero)
    )

    total_values = int(
        raw.size
    )

    changed_values = int(
        changed.sum()
    )

    changed_below_count = int(
        changed_below.sum()
    )

    changed_above_count = int(
        changed_above.sum()
    )

    changed_to_zero_count = int(
        changed_to_zero.sum()
    )

    changed_nonzero_count = int(
        changed_nonzero.sum()
    )

    max_abs_change = (
        float(
            np.max(
                np.abs(
                    clean - raw
                )
            )
        )
        if changed_values > 0
        else 0.0
    )

    max_raw_removed = (
        float(
            np.max(
                raw[changed_to_zero]
            )
        )
        if changed_to_zero_count > 0
        else np.nan
    )

    min_raw_removed = (
        float(
            np.min(
                raw[changed_to_zero]
            )
        )
        if changed_to_zero_count > 0
        else np.nan
    )

    max_nonzero_modification = (
        float(
            np.max(
                np.abs(
                    clean[changed_nonzero]
                    - raw[changed_nonzero]
                )
            )
        )
        if changed_nonzero_count > 0
        else 0.0
    )

    print("=" * 100)
    print("CLEAN-ALPHA FORENSIC AUDIT")
    print("=" * 100)
    print(
        "Total values                 :",
        total_values,
    )
    print(
        "Changed values               :",
        changed_values,
    )
    print(
        "Changed below gap            :",
        changed_below_count,
    )
    print(
        "Changed at/above gap         :",
        changed_above_count,
    )
    print(
        "Changed -> exactly zero      :",
        changed_to_zero_count,
    )
    print(
        "Changed -> nonzero value     :",
        changed_nonzero_count,
    )

    if changed_values > 0:
        print()
        print(
            "Max |clean - raw|          :",
            f"{max_abs_change:.6e}",
        )

        print(
            "Max raw value removed      :",
            (
                f"{max_raw_removed:.6e}"
                if np.isfinite(max_raw_removed)
                else "N/A"
            ),
        )

        print(
            "Min raw value removed      :",
            (
                f"{min_raw_removed:.6e}"
                if np.isfinite(min_raw_removed)
                else "N/A"
            ),
        )

    print()

    if changed_nonzero_count == 0:
        print(
            "All modifications are RAW -> 0 only."
        )
    else:
        print(
            "Nonzero-to-nonzero modifications exist."
        )
        print(
            "Maximum such modification :",
            f"{max_nonzero_modification:.6e}",
        )

    threshold_rows = []

    print()
    print("=" * 100)
    print("THRESHOLD RULE TEST")
    print("=" * 100)

    for threshold in thresholds:
        test = raw.copy()

        test[
            energy[None, :]
            < gap[:, None]
        ] = 0.0

        test[
            test < float(threshold)
        ] = 0.0

        difference = np.abs(
            test - clean
        )

        max_error = float(
            np.max(
                difference
            )
        )

        mismatch_points = int(
            (
                difference
                > tol
            ).sum()
        )

        threshold_rows.append(
            {
                "threshold": float(threshold),
                "max_error": max_error,
                "mismatch_points": mismatch_points,
            }
        )

        print(
            f"threshold={float(threshold):9.1f} | "
            f"max error={max_error:12.6e} | "
            f"mismatch points={mismatch_points}"
        )

    threshold_table = pd.DataFrame(
        threshold_rows
    )

    flat_diff = np.abs(
        clean - raw
    ).ravel()

    n_top = min(
        int(top_k),
        flat_diff.size,
    )

    top_indices = np.argsort(
        flat_diff
    )[-n_top:][::-1]

    top_rows = []

    print()
    print("=" * 100)
    print(
        f"TOP {n_top} MODIFIED POINTS"
    )
    print("=" * 100)

    for flat_index in top_indices:
        row_index, energy_index = np.unravel_index(
            flat_index,
            raw.shape,
        )

        row = {
            "material_id":
                ids[row_index],

            "energy_eV":
                float(
                    energy[energy_index]
                ),

            "gap_eV":
                float(
                    gap[row_index]
                ),

            "raw_alpha_cm-1":
                float(
                    raw[
                        row_index,
                        energy_index,
                    ]
                ),

            "clean_alpha_cm-1":
                float(
                    clean[
                        row_index,
                        energy_index,
                    ]
                ),

            "delta_cm-1":
                float(
                    clean[
                        row_index,
                        energy_index,
                    ]
                    - raw[
                        row_index,
                        energy_index,
                    ]
                ),
        }

        top_rows.append(
            row
        )

        print(
            f"{row['material_id']:12s} | "
            f"E={row['energy_eV']:6.3f} eV | "
            f"gap={row['gap_eV']:6.3f} | "
            f"raw={row['raw_alpha_cm-1']:11.3f} | "
            f"clean={row['clean_alpha_cm-1']:11.3f} | "
            f"delta={row['delta_cm-1']:11.3f}"
        )

    top_modified_points = pd.DataFrame(
        top_rows
    )

    summary = {
        "total_values":
            total_values,

        "changed_values":
            changed_values,

        "changed_below_gap":
            changed_below_count,

        "changed_at_or_above_gap":
            changed_above_count,

        "changed_to_zero":
            changed_to_zero_count,

        "changed_to_nonzero":
            changed_nonzero_count,

        "max_abs_change":
            max_abs_change,

        "max_raw_removed":
            max_raw_removed,

        "min_raw_removed":
            min_raw_removed,

        "max_nonzero_modification":
            max_nonzero_modification,
    }

    print()
    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)

    return {
        "summary":
            summary,

        "threshold_table":
            threshold_table,

        "top_modified_points":
            top_modified_points,

        "material_ids":
            ids,

        "energy_eV":
            energy,

        "raw_alpha":
            raw,

        "clean_alpha":
            clean,

        "gap_cutoff_eV":
            gap,

        "changed_mask":
            changed,

        "below_gap_mask":
            below_gap,

        "ml_npz":
            ml_npz,
    }
