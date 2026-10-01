"""
Data loading utilities for TASK3 optical-spectrum dataset.

This module was split from the original monolithic `utils_tb3.py`.
"""
# ============================================================
# utils_tb3.py
# Dataset utilities for DFT-IPA optical spectra
# ============================================================

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pymatgen.core import Structure


class CrystalOpticalDataset(Dataset):
    """
    Dataset for structure-to-property learning.

    Inputs
    ------
    - Crystal structure from pymatgen.

    Targets
    -------
    - Scalar targets: direct_gap, indirect_gap.
    - Spectral targets: epsI_0, epsR_0, epsI_2, epsR_2.

    The spectra are used only as prediction targets.
    """

    ENERGY_KEY_CANDIDATES = [
        "ipa_energy",
        "ipa_energies",
        "energy_grid",
        "energies",
        "energy",
        "e_grid",
    ]

    SPECTRUM_KEY_MAP = {
        "epsI_0": "ipa_epsI_0",
        "epsR_0": "ipa_epsR_0",
        "epsI_2": "ipa_epsI_2",
        "epsR_2": "ipa_epsR_2",
    }

    COMMON_ENERGY_MIN = 0.0
    COMMON_ENERGY_MAX = 20.0
    COMMON_ENERGY_N = 2001

    def __init__(
        self,
        zip_path: str,
        require_any_target: bool = True,
        require_full_spectrum: bool = False,
        max_files: Optional[int] = None,
        verbose: bool = True,
    ):
        self.zip_path = Path(zip_path)
        self.require_any_target = require_any_target
        self.require_full_spectrum = require_full_spectrum

        self.samples: List[Dict[str, Any]] = []
        self.failed_files: List[Tuple[str, str]] = []
        self.audit_rows: List[Dict[str, Any]] = []

        if verbose:
            print("=" * 100)
            print(f"Loading dataset from: {self.zip_path}")
            print("Dataset mode: structure -> DFT-IPA optical/electronic targets")
            print("=" * 100)

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".json"))

            if max_files is not None:
                names = names[:max_files]

            for name in names:
                try:
                    with zf.open(name) as f:
                        obj = json.load(f)

                    sample, audit = self._parse_one_json(obj, file_name=name)
                    self.samples.append(sample)
                    self.audit_rows.append(audit)

                except Exception as exc:
                    self.failed_files.append((name, str(exc)))
                    self.audit_rows.append(
                        {
                            "file_name": name,
                            "status": "failed",
                            "reason": str(exc),
                        }
                    )

        self.audit_df = pd.DataFrame(self.audit_rows)

        if verbose:
            self.print_summary()

    @staticmethod
    def _safe_float(value) -> float:
        try:
            out = float(value)
            return out if np.isfinite(out) else np.nan
        except Exception:
            return np.nan

    @staticmethod
    def _extract_array(data_dict: Dict[str, Any], key: str) -> Optional[np.ndarray]:
        value = data_dict.get(key, None)

        if value is None:
            return None

        if isinstance(value, dict) and "data" in value:
            arr = np.asarray(value["data"], dtype=np.float32)
        elif isinstance(value, list):
            arr = np.asarray(value, dtype=np.float32)
        else:
            return None

        if arr.ndim != 1:
            return None

        if len(arr) == 0:
            return None

        if not np.all(np.isfinite(arr)):
            return None

        return arr

    def _default_common_energy_grid(self) -> np.ndarray:
        return np.linspace(
            self.COMMON_ENERGY_MIN,
            self.COMMON_ENERGY_MAX,
            self.COMMON_ENERGY_N,
            dtype=np.float32,
        )

    def _extract_energy_grid_with_source(self, data_dict: Dict[str, Any]):
        for key in self.ENERGY_KEY_CANDIDATES:
            arr = self._extract_array(data_dict, key)
            if arr is not None:
                return arr, "original_json_axis"

        available_lengths = []

        for _, json_key in self.SPECTRUM_KEY_MAP.items():
            arr = self._extract_array(data_dict, json_key)
            if arr is not None:
                available_lengths.append(len(arr))

        has_common_length = (
            len(available_lengths) > 0
            and len(set(available_lengths)) == 1
            and available_lengths[0] == self.COMMON_ENERGY_N
        )

        if has_common_length:
            return self._default_common_energy_grid(), "common_0_20eV_axis"

        return None, "no_usable_axis"

    @staticmethod
    def _compute_spectrum_descriptors(
        arr: np.ndarray,
        prefix: str,
        energy_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        x = np.asarray(arr, dtype=np.float32)

        if energy_grid is not None and len(energy_grid) == len(x):
            axis = np.asarray(energy_grid, dtype=np.float32)
            total = float(np.sum(x))
            centroid = float(np.sum(axis * x) / total) if abs(total) > 1e-12 else np.nan
            area = float(np.trapz(x, axis))
        else:
            axis = np.arange(len(x), dtype=np.float32)
            total = float(np.sum(x))
            centroid = float(np.sum(axis * x) / total) if abs(total) > 1e-12 else np.nan
            area = float(np.sum(x))

        return {
            f"peak_{prefix}": float(np.max(x)),
            f"mean_{prefix}": float(np.mean(x)),
            f"std_{prefix}": float(np.std(x)),
            f"sum_{prefix}": float(np.sum(x)),
            f"area_{prefix}": area,
            f"centroid_{prefix}": centroid,
        }

    def _parse_one_json(self, obj: Dict[str, Any], file_name: str):
        data = obj.get("data", None)

        if data is None or not isinstance(data, dict):
            raise ValueError("Missing valid 'data' field")

        structure_dict = obj.get("structure", None)

        if structure_dict is None or not isinstance(structure_dict, dict):
            raise ValueError("Missing valid 'structure' field")

        try:
            structure = Structure.from_dict(structure_dict)
        except Exception as exc:
            raise ValueError(f"Structure parsing failed: {exc}")

        direct_gap = self._safe_float(data.get("ipa_direct_gap"))
        indirect_gap = self._safe_float(data.get("ipa_indirect_gap"))

        has_direct_gap = np.isfinite(direct_gap)
        has_indirect_gap = np.isfinite(indirect_gap)

        energy_grid, energy_axis_source = self._extract_energy_grid_with_source(data)
        has_energy_grid = energy_grid is not None
        energy_length = len(energy_grid) if has_energy_grid else np.nan

        spectra = {}
        spectrum_masks = {}

        for short_key, json_key in self.SPECTRUM_KEY_MAP.items():
            arr = self._extract_array(data, json_key)
            spectra[short_key] = arr
            spectrum_masks[short_key] = 1.0 if arr is not None else 0.0

        has_epsI_0 = spectra["epsI_0"] is not None
        has_epsR_0 = spectra["epsR_0"] is not None
        has_epsI_2 = spectra["epsI_2"] is not None
        has_epsR_2 = spectra["epsR_2"] is not None

        has_epsI0_epsR0_pair = has_epsI_0 and has_epsR_0
        is_full_spectrum = has_epsI_0 and has_epsR_0 and has_epsI_2 and has_epsR_2

        available_spectra = [arr for arr in spectra.values() if arr is not None]
        spectrum_length = len(available_spectra[0]) if available_spectra else np.nan

        if len(available_spectra) > 1:
            lengths = [len(arr) for arr in available_spectra]
            if len(set(lengths)) != 1:
                raise ValueError("Spectrum lengths do not match.")

        if has_energy_grid and available_spectra:
            if len(energy_grid) != len(available_spectra[0]):
                raise ValueError("Energy grid and spectrum length do not match.")

        if self.require_full_spectrum and not is_full_spectrum:
            raise ValueError("Full four-channel spectrum is required.")

        has_any_scalar = has_direct_gap or has_indirect_gap
        has_any_spectrum = len(available_spectra) > 0

        if self.require_any_target and not (has_any_scalar or has_any_spectrum):
            raise ValueError("No valid scalar or spectral target")

        targets = {
            "direct_gap": float(direct_gap) if has_direct_gap else np.nan,
            "indirect_gap": float(indirect_gap) if has_indirect_gap else np.nan,
        }

        masks = {
            "direct_gap": 1.0 if has_direct_gap else 0.0,
            "indirect_gap": 1.0 if has_indirect_gap else 0.0,
        }

        for spec_key, arr in spectra.items():
            descriptor_keys = [
                f"peak_{spec_key}",
                f"mean_{spec_key}",
                f"std_{spec_key}",
                f"sum_{spec_key}",
                f"area_{spec_key}",
                f"centroid_{spec_key}",
            ]

            if arr is not None:
                descriptors = self._compute_spectrum_descriptors(
                    arr,
                    spec_key,
                    energy_grid=energy_grid,
                )
                targets.update(descriptors)

                for key in descriptor_keys:
                    masks[key] = 1.0

            else:
                for key in descriptor_keys:
                    targets[key] = np.nan
                    masks[key] = 0.0

        if has_epsI_0 and has_epsI_2:
            targets["anisotropy_peak_epsI"] = float(
                abs(np.max(spectra["epsI_0"]) - np.max(spectra["epsI_2"]))
            )
            targets["anisotropy_mean_epsI"] = float(
                abs(np.mean(spectra["epsI_0"]) - np.mean(spectra["epsI_2"]))
            )
            masks["anisotropy_peak_epsI"] = 1.0
            masks["anisotropy_mean_epsI"] = 1.0

        else:
            targets["anisotropy_peak_epsI"] = np.nan
            targets["anisotropy_mean_epsI"] = np.nan
            masks["anisotropy_peak_epsI"] = 0.0
            masks["anisotropy_mean_epsI"] = 0.0

        elements = data.get("elements", None)

        meta = {
            "file_name": file_name,
            "mat_id": data.get("mat_id", None),
            "formula": data.get("formula", None),
            "elements": elements,
            "spg": data.get("spg", None),
            "nsites": data.get("nsites", None),
            "volume": float(structure.lattice.volume),
            "density": float(structure.density),
            "num_sites_from_structure": len(structure.sites),
            "num_elements": len(elements) if isinstance(elements, list) else None,
            "spectrum_length": spectrum_length,
            "has_energy_grid": has_energy_grid,
            "energy_axis_source": energy_axis_source,
            "energy_length": energy_length,
            "has_epsI_0": has_epsI_0,
            "has_epsR_0": has_epsR_0,
            "has_epsI_2": has_epsI_2,
            "has_epsR_2": has_epsR_2,
            "has_epsI0_epsR0_pair": has_epsI0_epsR0_pair,
            "is_full_spectrum": is_full_spectrum,
            "has_direct_gap": has_direct_gap,
            "has_indirect_gap": has_indirect_gap,
        }

        sample = {
            "structure": structure,
            "targets": targets,
            "masks": masks,
            "spectra": spectra,
            "spectrum_masks": spectrum_masks,
            "energy_grid": energy_grid,
            "meta": meta,
        }

        audit = {
            "file_name": file_name,
            "status": "ok",
            "reason": "",
            "mat_id": meta["mat_id"],
            "formula": meta["formula"],
            "spg": meta["spg"],
            "nsites": meta["nsites"],
            "num_elements": meta["num_elements"],
            "volume": meta["volume"],
            "density": meta["density"],
            "has_direct_gap": bool(has_direct_gap),
            "has_indirect_gap": bool(has_indirect_gap),
            "has_epsI_0": bool(has_epsI_0),
            "has_epsR_0": bool(has_epsR_0),
            "has_epsI_2": bool(has_epsI_2),
            "has_epsR_2": bool(has_epsR_2),
            "has_epsI0_epsR0_pair": bool(has_epsI0_epsR0_pair),
            "is_full_spectrum": bool(is_full_spectrum),
            "has_energy_grid": bool(has_energy_grid),
            "energy_axis_source": energy_axis_source,
            "spectrum_length": spectrum_length,
            "energy_length": energy_length,
        }

        return sample, audit

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        return item["structure"], item["targets"], item["masks"], item["meta"]

    def get_full_item(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def save_audit_csv(self, out_csv: str):
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_df.to_csv(out_path, index=False)
        print(f"Saved audit CSV: {out_path}")

    def print_summary(self):
        print(f"Total JSON in zip                   : {len(self.audit_rows)}")
        print(f"Number of valid samples             : {len(self.samples)}")
        print(f"Number of faulty samples            : {len(self.failed_files)}")

        if self.failed_files:
            print("\n--- First 10 failed files ---")
            for name, error in self.failed_files[:10]:
                print(f"{name} -> {error}")

        if not self.samples:
            return

        valid_df = self.audit_df[self.audit_df["status"] == "ok"].copy()

        print("\n--- Quick statistics ---")
        print("direct_gap labels                   :", int(valid_df["has_direct_gap"].sum()))
        print("indirect_gap labels                 :", int(valid_df["has_indirect_gap"].sum()))
        print("epsI_0 spectra                      :", int(valid_df["has_epsI_0"].sum()))
        print("epsR_0 spectra                      :", int(valid_df["has_epsR_0"].sum()))
        print("paired epsI_0 + epsR_0 spectra      :", int(valid_df["has_epsI0_epsR0_pair"].sum()))
        print("epsI_2 spectra                      :", int(valid_df["has_epsI_2"].sum()))
        print("epsR_2 spectra                      :", int(valid_df["has_epsR_2"].sum()))
        print("full four-channel spectra           :", int(valid_df["is_full_spectrum"].sum()))
        print("usable energy axis                  :", int(valid_df["has_energy_grid"].sum()))

        if "energy_axis_source" in valid_df.columns:
            print("\nEnergy axis source:")
            print(valid_df["energy_axis_source"].value_counts())

        spectrum_lengths = valid_df["spectrum_length"].dropna()

        if len(spectrum_lengths) > 0:
            print("\nUnique spectrum length:", sorted(spectrum_lengths.unique())[:10])


class CrystalSpectrumView(Dataset):
    """
    View for one selected spectral target.
    """

    def __init__(
        self,
        base_dataset: CrystalOpticalDataset,
        spectrum_key: str = "epsI_0",
        require_valid: bool = True,
        verbose: bool = True,
    ):
        self.base_dataset = base_dataset
        self.spectrum_key = spectrum_key
        self.require_valid = require_valid
        self.valid_indices = []

        for idx, sample in enumerate(self.base_dataset.samples):
            arr = sample["spectra"].get(self.spectrum_key, None)

            if arr is not None:
                self.valid_indices.append(idx)
            elif not self.require_valid:
                self.valid_indices.append(idx)

        if verbose:
            print("=" * 100)
            print(f"Spectrum view: {self.spectrum_key}")
            print(f"Usable samples: {len(self.valid_indices)} / {len(self.base_dataset)}")
            print("=" * 100)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        base_idx = self.valid_indices[idx]
        sample = self.base_dataset.get_full_item(base_idx)

        structure = sample["structure"]
        spectrum = sample["spectra"].get(self.spectrum_key, None)
        energy_grid = sample["energy_grid"]
        meta = dict(sample["meta"])

        meta["base_idx"] = base_idx
        meta["spectrum_key"] = self.spectrum_key

        if spectrum is None:
            spec_len = int(meta["spectrum_length"]) if np.isfinite(meta["spectrum_length"]) else 0
            y = torch.full((spec_len,), float("nan"), dtype=torch.float32)
            mask = torch.zeros(spec_len, dtype=torch.float32)

            if spec_len == 2001:
                spectral_axis = torch.linspace(0.0, 20.0, spec_len, dtype=torch.float32)
            else:
                spectral_axis = torch.arange(spec_len, dtype=torch.float32)

        else:
            y = torch.tensor(spectrum, dtype=torch.float32)
            mask = torch.ones_like(y, dtype=torch.float32)

            if energy_grid is None:
                if len(spectrum) == 2001:
                    spectral_axis = torch.linspace(0.0, 20.0, len(spectrum), dtype=torch.float32)
                else:
                    spectral_axis = torch.arange(len(spectrum), dtype=torch.float32)
            else:
                spectral_axis = torch.tensor(energy_grid, dtype=torch.float32)

        return structure, y, mask, spectral_axis, meta


def collate_structure_targets(batch):
    structures, targets_list, masks_list, metas = zip(*batch)

    all_target_keys = sorted(set().union(*[targets.keys() for targets in targets_list]))
    target_dict = {}
    mask_dict = {}

    for key in all_target_keys:
        values = []
        masks = []

        for targets, mask in zip(targets_list, masks_list):
            values.append(float(targets.get(key, np.nan)))
            masks.append(float(mask.get(key, 0.0)))

        target_dict[key] = torch.tensor(values, dtype=torch.float32)
        mask_dict[key] = torch.tensor(masks, dtype=torch.float32)

    return list(structures), target_dict, mask_dict, list(metas)


def collate_spectrum_batch(batch):
    structures, spectra, masks, spectral_axes, metas = zip(*batch)

    spectra = torch.stack(spectra, dim=0)
    masks = torch.stack(masks, dim=0)
    spectral_axes = torch.stack(spectral_axes, dim=0)

    shared_axis = spectral_axes[0]
    has_shared_axis = torch.allclose(
        spectral_axes,
        shared_axis.unsqueeze(0).expand_as(spectral_axes),
    )

    if has_shared_axis:
        spectral_axis_out = shared_axis
    else:
        spectral_axis_out = spectral_axes

    return list(structures), spectra, masks, spectral_axis_out, list(metas)


def build_spectrum_view(ds, spectrum_key="epsI_0", verbose=True):
    return CrystalSpectrumView(
        base_dataset=ds,
        spectrum_key=spectrum_key,
        require_valid=True,
        verbose=verbose,
    )


def make_paired_spectrum_views(ds, key_real="epsR_0", key_imag="epsI_0", verbose=True):
    """
    Build paired spectrum views for the dielectric function.

    key_real
        Real part of the dielectric function, usually epsR_0.

    key_imag
        Imaginary part of the dielectric function, usually epsI_0.
    """
    real_view = CrystalSpectrumView(
        base_dataset=ds,
        spectrum_key=key_real,
        require_valid=True,
        verbose=verbose,
    )

    imag_view = CrystalSpectrumView(
        base_dataset=ds,
        spectrum_key=key_imag,
        require_valid=True,
        verbose=verbose,
    )

    if real_view.valid_indices != imag_view.valid_indices:
        raise ValueError(
            f"{key_real} and {key_imag} do not share the same valid sample indices."
        )

    if verbose:
        print(f"\nPaired dielectric samples: {len(real_view)}")
        print(f"Real part target          : {key_real}")
        print(f"Imaginary part target     : {key_imag}")

    return real_view, imag_view


def save_dataset_summary(ds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_df = ds.audit_df.copy()
    valid_df = audit_df[audit_df["status"] == "ok"].copy()

    summary = {
        "total_json": len(audit_df),
        "valid_samples": len(valid_df),
        "failed_samples": int((audit_df["status"] != "ok").sum()),
        "direct_gap": int(valid_df["has_direct_gap"].sum()),
        "indirect_gap": int(valid_df["has_indirect_gap"].sum()),
        "epsI_0": int(valid_df["has_epsI_0"].sum()),
        "epsR_0": int(valid_df["has_epsR_0"].sum()),
        "paired_epsI0_epsR0": int(valid_df["has_epsI0_epsR0_pair"].sum()),
        "epsI_2": int(valid_df["has_epsI_2"].sum()),
        "epsR_2": int(valid_df["has_epsR_2"].sum()),
        "full_four_channel": int(valid_df["is_full_spectrum"].sum()),
        "usable_energy_axis": int(valid_df["has_energy_grid"].sum()),
    }

    summary_df = pd.DataFrame(
        [{"item": key, "count": value} for key, value in summary.items()]
    )

    audit_csv = out_dir / "dataset_audit_ipa_optical.csv"
    summary_csv = out_dir / "dataset_summary_for_paper.csv"

    audit_df.to_csv(audit_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("Saved audit CSV  :", audit_csv)
    print("Saved summary CSV:", summary_csv)

    return summary_df, audit_df


def preview_dataset(ds, spectrum_key="epsI_0", idx=0):
    structure, targets, masks, meta = ds[idx]

    print("\nExample sample")
    print("formula        :", meta["formula"])
    print("mat_id         :", meta["mat_id"])
    print("nsites         :", meta["nsites"])
    print("num_elements   :", meta["num_elements"])
    print("direct_gap     :", targets["direct_gap"])
    print("indirect_gap   :", targets["indirect_gap"])

    view = CrystalSpectrumView(
        base_dataset=ds,
        spectrum_key=spectrum_key,
        require_valid=True,
        verbose=False,
    )

    _, spectrum, _, spectral_axis, spectrum_meta = view[0]

    print(f"\nSpectrum view: {spectrum_key}")
    print("formula             :", spectrum_meta["formula"])
    print("spectrum shape      :", tuple(spectrum.shape))
    print("spectral_axis shape :", tuple(spectral_axis.shape))
    print("first 5 y           :", spectrum[:5].tolist())
    print("first 5 spectral_ax :", spectral_axis[:5].tolist())

    return view
