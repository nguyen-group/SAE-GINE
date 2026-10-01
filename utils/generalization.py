"""
Split-overlap and generalization audit utilities for TASK3.
"""

from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

from .data import CrystalOpticalDataset

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_MATCHER_CONFIG = {
    "ltol": 0.20,
    "stol": 0.30,
    "angle_tol": 5.0,
    "primitive_cell": True,
    "scale": True,
    "attempt_supercell": True,
}


def _clean_text_column(series):
    out = series.astype("string").str.strip()
    invalid = out.str.lower().isin(["", "nan", "none", "null", "<na>"])
    return out.mask(invalid)


def _unwrap_base_dataset(obj):
    current = obj
    visited = set()

    for _ in range(12):
        if current is None or id(current) in visited:
            return None

        visited.add(id(current))

        if hasattr(current, "samples"):
            return current

        for attr in ("base_dataset", "dataset", "spectrum_view_dataset"):
            wrapped = getattr(current, attr, None)
            if wrapped is not None:
                current = wrapped
                break
        else:
            return None

    return None


def _convert_to_structure(value):
    if isinstance(value, Structure):
        return value

    if isinstance(value, dict):
        try:
            return Structure.from_dict(value)
        except Exception:
            return None

    return None


def _composition_metadata(formula):
    try:
        composition = Composition(str(formula))
        return pd.Series(
            {
                "reduced_formula": composition.reduced_formula,
                "chemical_system": "-".join(
                    sorted(element.symbol for element in composition.elements)
                ),
                "anonymous_formula": composition.anonymized_formula.replace(" ", ""),
                "n_elements": len(composition.elements),
            }
        )
    except Exception:
        return pd.Series(
            {
                "reduced_formula": pd.NA,
                "chemical_system": pd.NA,
                "anonymous_formula": pd.NA,
                "n_elements": pd.NA,
            }
        )


def _resolve_split_csv(project_root, split_csv=None):
    if split_csv is not None:
        split_csv = Path(split_csv)
        if not split_csv.exists():
            raise FileNotFoundError(f"Fixed split CSV was not found: {split_csv}")
        return split_csv

    filename = "fixed_split_paired_epsR_epsI_seed2025_v1.csv"
    candidates = [
        project_root / "processed" / "fixed_splits" / filename,
        project_root / "processed" / "paired_training" / "fixed_splits" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Fixed split CSV was not found.\nChecked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def run_split_generalization_audit(
    project_root,
    dataset=None,
    split_csv=None,
    data_zip=None,
    output_dir=None,
    run_structure_match=True,
    run_anonymous_structure_match=False,
    run_test_vs_val_structure_match=False,
    matcher_config=None,
    save_outputs=True,
    display_tables=True,
):
    """
    Audit split overlap, duplicates, and structural generalization.

    Parameters
    ----------
    project_root : str or pathlib.Path
        TASK3 project root, typically ``D:/TB3``.
    dataset : object or None
        Loaded CrystalOpticalDataset or a wrapper that can be unwrapped to it.
        If None and structural matching is enabled, the dataset is loaded from
        ``data_zip``.
    split_csv : str or pathlib.Path or None
        Fixed split manifest. If None, the standard TASK3 locations are searched.
    data_zip : str or pathlib.Path or None
        Crystal database archive used only when ``dataset`` is None.
        Defaults to ``project_root / "database_300.zip"``.
    output_dir : str or pathlib.Path or None
        Output directory. Defaults to
        ``processed/paired_training/paper_outputs/split_generalization_audit``.
    run_structure_match : bool
        Match validation and test structures against training structures.
    run_anonymous_structure_match : bool
        Also run anonymous-structure matching. This can be expensive.
    run_test_vs_val_structure_match : bool
        Also record exact test-versus-validation structural matches.
    matcher_config : dict or None
        Keyword arguments passed to pymatgen StructureMatcher.
    save_outputs : bool
        Save audit CSV files.
    display_tables : bool
        Display the main audit tables when running in a notebook.

    Returns
    -------
    dict
        Audit DataFrames, resolved paths, and saved output paths.
    """
    project_root = Path(project_root)
    split_csv = _resolve_split_csv(project_root, split_csv)

    if data_zip is None:
        data_zip = project_root / "database_300.zip"
    else:
        data_zip = Path(data_zip)

    if output_dir is None:
        output_dir = (
            project_root
            / "processed"
            / "paired_training"
            / "paper_outputs"
            / "split_generalization_audit"
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    matcher_config = {
        **DEFAULT_MATCHER_CONFIG,
        **({} if matcher_config is None else dict(matcher_config)),
    }

    split_df = pd.read_csv(split_csv)
    split_df.columns = [str(col).strip() for col in split_df.columns]

    required_columns = {
        "split",
        "view_idx",
        "base_idx",
        "mat_id",
        "formula",
        "file_name",
    }
    missing_columns = required_columns - set(split_df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing columns in split CSV: {sorted(missing_columns)}\n"
            f"Available columns: {split_df.columns.tolist()}"
        )

    split_df["split"] = (
        split_df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "training": "train",
                "validation": "val",
                "valid": "val",
                "testing": "test",
            }
        )
    )

    unexpected_splits = set(split_df["split"]) - {"train", "val", "test"}
    if unexpected_splits:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected_splits)}")

    split_df["base_idx"] = pd.to_numeric(
        split_df["base_idx"], errors="raise"
    ).astype("Int64")
    split_df["view_idx"] = pd.to_numeric(
        split_df["view_idx"], errors="raise"
    ).astype("Int64")

    for column in ["mat_id", "formula", "file_name"]:
        split_df[column] = _clean_text_column(split_df[column])

    base_dataset = _unwrap_base_dataset(dataset)

    needs_structure_data = (
        run_structure_match
        or run_anonymous_structure_match
        or split_df["formula"].isna().any()
    )

    if needs_structure_data and base_dataset is None:
        if not data_zip.exists():
            raise FileNotFoundError(f"Crystal database archive was not found: {data_zip}")

        print("Loading CrystalOpticalDataset for structural audit...")
        base_dataset = CrystalOpticalDataset(
            zip_path=data_zip,
            require_any_target=True,
            require_full_spectrum=False,
            verbose=False,
        )

    structure_cache = {}

    def get_structure(base_idx):
        base_idx = int(base_idx)

        if base_idx in structure_cache:
            return structure_cache[base_idx]

        structure = None

        if hasattr(base_dataset, "get_full_item"):
            try:
                full_item = base_dataset.get_full_item(base_idx)
                if isinstance(full_item, dict):
                    for key in ("structure", "crystal_structure", "structure_dict"):
                        if key in full_item:
                            structure = _convert_to_structure(full_item[key])
                            if structure is not None:
                                break
            except Exception:
                pass

        if structure is None and hasattr(base_dataset, "samples"):
            sample = base_dataset.samples[base_idx]
            if isinstance(sample, dict):
                for key in ("structure", "crystal_structure", "structure_dict"):
                    if key in sample:
                        structure = _convert_to_structure(sample[key])
                        if structure is not None:
                            break

        if structure is None:
            try:
                item = base_dataset[base_idx]
                if isinstance(item, tuple) and len(item) > 0:
                    structure = _convert_to_structure(item[0])
                elif isinstance(item, dict):
                    structure = _convert_to_structure(item.get("structure"))
            except Exception:
                pass

        if structure is None:
            raise RuntimeError(
                f"Could not retrieve structure for base_idx={base_idx}"
            )

        structure_cache[base_idx] = structure
        return structure

    if base_dataset is not None and split_df["formula"].isna().any():
        for row_index in split_df.index[split_df["formula"].isna()]:
            try:
                structure = get_structure(split_df.at[row_index, "base_idx"])
                split_df.at[row_index, "formula"] = (
                    structure.composition.reduced_formula
                )
            except Exception:
                pass

    composition_df = split_df["formula"].apply(_composition_metadata)
    split_df = pd.concat([split_df, composition_df], axis=1)

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]
    audit_keys = [
        "base_idx",
        "mat_id",
        "file_name",
        "reduced_formula",
        "chemical_system",
        "anonymous_formula",
    ]

    pairwise_rows = []
    overlap_value_rows = []

    for reference_split, query_split in split_pairs:
        reference_df = split_df[split_df["split"] == reference_split]
        query_df = split_df[split_df["split"] == query_split]

        for key in audit_keys:
            reference_values = set(reference_df[key].dropna().tolist())
            query_values = set(query_df[key].dropna().tolist())
            overlap_values = reference_values & query_values

            query_valid = query_df[key].notna()
            query_seen = query_valid & query_df[key].isin(reference_values)

            pairwise_rows.append(
                {
                    "reference_split": reference_split,
                    "query_split": query_split,
                    "key": key,
                    "reference_unique": len(reference_values),
                    "query_unique": len(query_values),
                    "overlap_unique": len(overlap_values),
                    "query_samples_seen": int(query_seen.sum()),
                    "query_samples_valid": int(query_valid.sum()),
                    "query_seen_percent": (
                        100.0 * query_seen.sum() / query_valid.sum()
                        if query_valid.sum() > 0
                        else np.nan
                    ),
                }
            )

            for value in sorted(map(str, overlap_values)):
                overlap_value_rows.append(
                    {
                        "reference_split": reference_split,
                        "query_split": query_split,
                        "key": key,
                        "overlap_value": value,
                    }
                )

    pairwise_summary = pd.DataFrame(pairwise_rows)
    overlap_values_df = pd.DataFrame(
        overlap_value_rows,
        columns=[
            "reference_split",
            "query_split",
            "key",
            "overlap_value",
        ],
    )

    within_split_rows = []

    for split_name, group in split_df.groupby("split"):
        for key in audit_keys:
            valid = group[group[key].notna()]
            duplicate_mask = valid[key].duplicated(keep=False)

            within_split_rows.append(
                {
                    "split": split_name,
                    "key": key,
                    "n_samples": len(group),
                    "n_valid": len(valid),
                    "duplicate_samples": int(duplicate_mask.sum()),
                    "duplicate_unique_values": int(
                        valid.loc[duplicate_mask, key].nunique()
                    ),
                }
            )

    within_split_summary = pd.DataFrame(within_split_rows)

    structure_match_columns = [
        "match_type",
        "reference_split",
        "query_split",
        "reference_base_idx",
        "query_base_idx",
        "reference_mat_id",
        "query_mat_id",
        "reference_formula",
        "query_formula",
        "chemical_system",
        "anonymous_formula",
    ]
    structure_error_columns = [
        "query_split",
        "query_base_idx",
        "reference_base_idx",
        "error",
    ]

    structure_match_records = []
    structure_match_errors = []
    matcher = StructureMatcher(**matcher_config)

    def match_split_against_reference(
        query_split,
        reference_split="train",
        anonymous=False,
    ):
        query_df = split_df[split_df["split"] == query_split]
        reference_df = split_df[split_df["split"] == reference_split]
        grouping_key = "anonymous_formula" if anonymous else "reduced_formula"

        reference_groups = defaultdict(list)
        for reference_index, reference_row in reference_df.iterrows():
            key = reference_row[grouping_key]
            if pd.notna(key):
                reference_groups[str(key)].append(reference_index)

        match_flags = {}
        description = (
            f"{query_split} vs {reference_split} "
            f"({'anonymous' if anonymous else 'exact'} structure)"
        )

        for query_index in tqdm(
            query_df.index,
            total=len(query_df),
            desc=description,
        ):
            query_row = split_df.loc[query_index]
            group_value = query_row[grouping_key]

            if pd.isna(group_value):
                match_flags[query_index] = pd.NA
                continue

            candidate_indices = reference_groups.get(str(group_value), [])
            if not candidate_indices:
                match_flags[query_index] = False
                continue

            try:
                query_structure = get_structure(query_row["base_idx"])
            except Exception as exc:
                match_flags[query_index] = pd.NA
                structure_match_errors.append(
                    {
                        "query_split": query_split,
                        "query_base_idx": query_row["base_idx"],
                        "reference_base_idx": pd.NA,
                        "error": str(exc),
                    }
                )
                continue

            matched = False

            for reference_index in candidate_indices:
                reference_row = split_df.loc[reference_index]

                try:
                    reference_structure = get_structure(
                        reference_row["base_idx"]
                    )

                    if anonymous:
                        is_match = matcher.fit_anonymous(
                            reference_structure,
                            query_structure,
                        )
                    else:
                        is_match = matcher.fit(
                            reference_structure,
                            query_structure,
                        )

                except Exception as exc:
                    structure_match_errors.append(
                        {
                            "query_split": query_split,
                            "query_base_idx": query_row["base_idx"],
                            "reference_base_idx": reference_row["base_idx"],
                            "error": str(exc),
                        }
                    )
                    continue

                if is_match:
                    matched = True
                    structure_match_records.append(
                        {
                            "match_type": (
                                "anonymous_structure"
                                if anonymous
                                else "exact_structure"
                            ),
                            "reference_split": reference_split,
                            "query_split": query_split,
                            "reference_base_idx": reference_row["base_idx"],
                            "query_base_idx": query_row["base_idx"],
                            "reference_mat_id": reference_row["mat_id"],
                            "query_mat_id": query_row["mat_id"],
                            "reference_formula": reference_row["reduced_formula"],
                            "query_formula": query_row["reduced_formula"],
                            "chemical_system": query_row["chemical_system"],
                            "anonymous_formula": query_row["anonymous_formula"],
                        }
                    )
                    break

            match_flags[query_index] = matched

        return match_flags

    split_df["structure_match_in_train"] = pd.Series(
        pd.NA,
        index=split_df.index,
        dtype="boolean",
    )

    if run_structure_match:
        for query_split in ["val", "test"]:
            flags = match_split_against_reference(
                query_split=query_split,
                reference_split="train",
                anonymous=False,
            )

            for row_index, flag in flags.items():
                split_df.at[row_index, "structure_match_in_train"] = flag

        if run_test_vs_val_structure_match:
            match_split_against_reference(
                query_split="test",
                reference_split="val",
                anonymous=False,
            )

    split_df["anonymous_structure_match_in_train"] = pd.Series(
        pd.NA,
        index=split_df.index,
        dtype="boolean",
    )

    if run_anonymous_structure_match:
        warnings.warn(
            "Anonymous structure matching can be computationally expensive."
        )

        for query_split in ["val", "test"]:
            flags = match_split_against_reference(
                query_split=query_split,
                reference_split="train",
                anonymous=True,
            )

            for row_index, flag in flags.items():
                split_df.at[
                    row_index,
                    "anonymous_structure_match_in_train",
                ] = flag

    structure_matches_df = pd.DataFrame(
        structure_match_records,
        columns=structure_match_columns,
    )
    structure_errors_df = pd.DataFrame(
        structure_match_errors,
        columns=structure_error_columns,
    )

    train_df = split_df[split_df["split"] == "train"]
    generalization_rows = []

    generalization_keys = {
        "mat_id": "material_id",
        "reduced_formula": "reduced_formula",
        "chemical_system": "chemical_system",
        "anonymous_formula": "anonymous_formula",
    }

    for query_split in ["val", "test"]:
        query_df = split_df[split_df["split"] == query_split]
        row = {
            "split": query_split,
            "n_samples": len(query_df),
        }

        for key, label in generalization_keys.items():
            train_values = set(train_df[key].dropna().tolist())
            valid_mask = query_df[key].notna()
            novel_mask = valid_mask & ~query_df[key].isin(train_values)

            row[f"{label}_valid"] = int(valid_mask.sum())
            row[f"{label}_novel_samples"] = int(novel_mask.sum())
            row[f"{label}_novel_percent"] = (
                100.0 * novel_mask.sum() / valid_mask.sum()
                if valid_mask.sum() > 0
                else np.nan
            )

        if run_structure_match:
            valid_mask = query_df["structure_match_in_train"].notna()
            novel_mask = (
                valid_mask
                & ~query_df["structure_match_in_train"].fillna(False)
            )

            row["structure_valid"] = int(valid_mask.sum())
            row["structure_novel_samples"] = int(novel_mask.sum())
            row["structure_novel_percent"] = (
                100.0 * novel_mask.sum() / valid_mask.sum()
                if valid_mask.sum() > 0
                else np.nan
            )

        generalization_rows.append(row)

    generalization_summary = pd.DataFrame(generalization_rows)

    output_paths = {
        "split_sample_level": output_dir / "split_sample_level_overlap_audit.csv",
        "pairwise_summary": output_dir / "pairwise_split_overlap_summary.csv",
        "within_split_summary": output_dir / "within_split_duplicate_summary.csv",
        "generalization_summary": output_dir / "generalization_relative_to_train.csv",
        "overlap_values": output_dir / "cross_split_overlap_values.csv",
        "structure_matches": output_dir / "cross_split_structure_matches.csv",
        "structure_errors": output_dir / "structure_match_errors.csv",
    }

    if save_outputs:
        split_df.to_csv(
            output_paths["split_sample_level"],
            index=False,
            encoding="utf-8-sig",
        )
        pairwise_summary.to_csv(
            output_paths["pairwise_summary"],
            index=False,
            encoding="utf-8-sig",
        )
        within_split_summary.to_csv(
            output_paths["within_split_summary"],
            index=False,
            encoding="utf-8-sig",
        )
        generalization_summary.to_csv(
            output_paths["generalization_summary"],
            index=False,
            encoding="utf-8-sig",
        )
        overlap_values_df.to_csv(
            output_paths["overlap_values"],
            index=False,
            encoding="utf-8-sig",
        )
        structure_matches_df.to_csv(
            output_paths["structure_matches"],
            index=False,
            encoding="utf-8-sig",
        )
        structure_errors_df.to_csv(
            output_paths["structure_errors"],
            index=False,
            encoding="utf-8-sig",
        )

    print("=" * 100)
    print("SPLIT OVERLAP AND GENERALIZATION AUDIT")
    print("=" * 100)
    print("Split file:", split_csv)
    print("Output directory:", output_dir)

    split_sizes = (
        split_df["split"]
        .value_counts()
        .reindex(["train", "val", "test"])
        .rename_axis("split")
        .reset_index(name="n_samples")
    )

    if display_tables:
        try:
            from IPython.display import display

            print("\nSplit sizes")
            display(split_sizes)

            print("\nPairwise overlap summary")
            display(pairwise_summary)

            print("\nGeneralization relative to the training split")
            display(generalization_summary)

            if run_structure_match:
                print("\nExact cross-split structural matches")
                print("Number of matches:", len(structure_matches_df))
                if len(structure_matches_df) > 0:
                    display(structure_matches_df.head(30))
                else:
                    print("No exact structural matches were detected.")

            if len(structure_errors_df) > 0:
                print("\nStructure-matching warnings:", len(structure_errors_df))
                display(structure_errors_df.head(20))

        except Exception:
            print(split_sizes)
            print(pairwise_summary)
            print(generalization_summary)

    if save_outputs:
        print("\nSaved files:")
        for path in output_paths.values():
            print(" -", path.name)

    return {
        "split_df": split_df,
        "split_sizes": split_sizes,
        "pairwise_summary": pairwise_summary,
        "within_split_summary": within_split_summary,
        "generalization_summary": generalization_summary,
        "overlap_values": overlap_values_df,
        "structure_matches": structure_matches_df,
        "structure_errors": structure_errors_df,
        "split_csv": split_csv,
        "output_dir": output_dir,
        "output_paths": output_paths,
        "matcher_config": matcher_config,
    }
