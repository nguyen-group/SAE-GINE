"""Legacy monolithic utils_tb3.py kept for reference only.
Prefer importing from the split modules in utils_tb.
"""

import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
import os
import time
import copy
import random
import inspect
from contextlib import nullcontext
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
# ============================================================
# Crystal graph utilities
# ============================================================

from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader


SCALAR_TARGET_KEYS = ["direct_gap", "indirect_gap"]

SPECTRUM_TARGET_TO_ID = {
    "epsI_0": 0,
    "epsR_0": 1,
    "epsI_2": 2,
    "epsR_2": 3,
}

DEFAULT_GRAPH_CONFIG = {
    "cutoff": 5.0,
    "max_neighbors": 16,
    "symmetrize": True,
    "edge_encoding": "rbf",
    "num_rbf": 16,
    "attach_graph_attr": False,
}


def graph_safe_float(value, default=np.nan):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def value_and_mask(value, fill_value=0.0):
    out = graph_safe_float(value, default=np.nan)

    if np.isfinite(out):
        return float(out), 1.0

    return float(fill_value), 0.0


def get_element_features(symbol: str):
    """
    Baseline node features with dimension 11.

    Features:
    [Z, row, group, atomic_mass, electronegativity, atomic_radius,
     ionization_energy, atomic_mass_mask, electronegativity_mask,
     atomic_radius_mask, ionization_energy_mask]
    """
    element = Element(symbol)

    atomic_number, _ = value_and_mask(getattr(element, "Z", np.nan), fill_value=0.0)
    row, _ = value_and_mask(getattr(element, "row", np.nan), fill_value=0.0)
    group, _ = value_and_mask(getattr(element, "group", np.nan), fill_value=0.0)
    atomic_mass, atomic_mass_mask = value_and_mask(getattr(element, "atomic_mass", np.nan), fill_value=0.0)

    try:
        electronegativity_raw = element.data.get("X", np.nan)
    except Exception:
        electronegativity_raw = np.nan

    electronegativity, electronegativity_mask = value_and_mask(electronegativity_raw, fill_value=0.0)
    atomic_radius, radius_mask = value_and_mask(getattr(element, "atomic_radius", np.nan), fill_value=0.0)
    ionization_energy, ionization_mask = value_and_mask(getattr(element, "ionization_energy", np.nan), fill_value=0.0)

    return [
        atomic_number,
        row,
        group,
        atomic_mass,
        electronegativity,
        atomic_radius,
        ionization_energy,
        atomic_mass_mask,
        electronegativity_mask,
        radius_mask,
        ionization_mask,
    ]


def build_graph_attr(meta, structure):
    """
    Graph-level auxiliary attributes.

    Features:
    [density, volume_per_atom, number_of_sites, number_of_elements, mean_atomic_number]
    """
    num_sites = max(len(structure.sites), 1)
    species = [site.specie for site in structure.sites]
    mean_atomic_number = float(np.mean([sp.Z for sp in species])) if species else 0.0

    return torch.tensor(
        [
            graph_safe_float(meta.get("density", structure.density), default=0.0),
            graph_safe_float(meta.get("volume", structure.lattice.volume), default=0.0) / float(num_sites),
            float(num_sites),
            graph_safe_float(meta.get("num_elements", 0.0), default=0.0),
            mean_atomic_number,
        ],
        dtype=torch.float32,
    )


def rbf_expand(distances, dmin=0.0, dmax=5.0, num_rbf=16, gamma=None):
    distances = np.asarray(distances, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(dmin, dmax, num_rbf, dtype=np.float32).reshape(1, -1)

    if gamma is None:
        step = (dmax - dmin) / max(num_rbf - 1, 1)
        gamma = 1.0 / (step**2 + 1e-12)

    return np.exp(-gamma * (distances - centers) ** 2).astype(np.float32)


def build_edge_records(structure, cutoff=5.0, max_neighbors=16, symmetrize=True):
    """
    Build periodic edge records.

    Each edge is represented as:
    (source_index, target_index, periodic_image, distance)
    """
    num_sites = len(structure.sites)
    center_idx, point_idx, offset_vecs, dists = structure.get_neighbor_list(r=cutoff)

    neighbors_by_center = {idx: [] for idx in range(num_sites)}

    for i, j, offset, distance in zip(center_idx, point_idx, offset_vecs, dists):
        i = int(i)
        j = int(j)
        distance = float(distance)
        image = tuple(int(v) for v in np.rint(offset).astype(int).tolist())

        is_true_self_loop = (i == j) and (distance < 1e-8) and (image == (0, 0, 0))

        if is_true_self_loop:
            continue

        neighbors_by_center[i].append((j, image, distance))

    edge_records = []

    for i in range(num_sites):
        neighbors = sorted(neighbors_by_center[i], key=lambda item: (item[2], item[0], item[1]))
        neighbors = neighbors[:max_neighbors]
        seen = set()

        for j, image, distance in neighbors:
            key = (i, j, image)

            if key in seen:
                continue

            seen.add(key)
            edge_records.append((i, j, image, distance))

    if symmetrize:
        present = {(i, j, image) for i, j, image, _ in edge_records}
        extra_records = []

        for i, j, image, distance in edge_records:
            reverse_key = (j, i, tuple([-v for v in image]))

            if reverse_key not in present:
                extra_records.append((j, i, tuple([-v for v in image]), distance))
                present.add(reverse_key)

        edge_records.extend(extra_records)

    return sorted(edge_records, key=lambda item: (item[0], item[1], item[2], item[3]))


def build_node_tensor(structure):
    return torch.tensor(
        [get_element_features(site.specie.symbol) for site in structure.sites],
        dtype=torch.float32,
    )


def build_edges(structure, cutoff=5.0, max_neighbors=16, symmetrize=True, edge_encoding="rbf", num_rbf=16):
    edge_records = build_edge_records(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        symmetrize=symmetrize,
    )

    if len(edge_records) == 0:
        edge_dim = num_rbf if edge_encoding == "rbf" else 1
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
        edge_dist = torch.empty((0, 1), dtype=torch.float32)
        edge_image = torch.empty((0, 3), dtype=torch.long)
        return edge_index, edge_attr, edge_dist, edge_image

    source_indices = []
    target_indices = []
    distances = []
    images = []

    for source, target, image, distance in edge_records:
        source_indices.append(source)
        target_indices.append(target)
        distances.append(distance)
        images.append(image)

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    distances_np = np.asarray(distances, dtype=np.float32)

    if edge_encoding == "rbf":
        edge_attr_np = rbf_expand(distances_np, dmin=0.0, dmax=cutoff, num_rbf=num_rbf)
    elif edge_encoding == "distance":
        edge_attr_np = distances_np.reshape(-1, 1)
    else:
        raise ValueError(f"Invalid edge encoding: {edge_encoding}")

    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    edge_dist = torch.tensor(distances_np.reshape(-1, 1), dtype=torch.float32)
    edge_image = torch.tensor(np.asarray(images, dtype=np.int64), dtype=torch.long)

    return edge_index, edge_attr, edge_dist, edge_image


def resolve_base_idx(meta, fallback_idx):
    value = meta.get("base_idx", fallback_idx)

    try:
        return int(value)
    except Exception:
        return int(fallback_idx)


def resolve_sample_uid(meta, base_idx):
    sample_uid = meta.get("sample_uid", None)

    if sample_uid is not None:
        return str(sample_uid)

    mat_id = meta.get("mat_id", "unknown")

    return f"{mat_id}__{base_idx}"


def structure_to_pyg_scalar_data(
    structure,
    targets,
    masks,
    meta,
    target_keys=None,
    cutoff=5.0,
    max_neighbors=16,
    symmetrize=True,
    edge_encoding="rbf",
    num_rbf=16,
    attach_graph_attr=False,
    base_idx=None,
    view_idx=None,
):
    if target_keys is None:
        target_keys = SCALAR_TARGET_KEYS

    x = build_node_tensor(structure)

    edge_index, edge_attr, edge_dist, edge_image = build_edges(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        symmetrize=symmetrize,
        edge_encoding=edge_encoding,
        num_rbf=num_rbf,
    )

    y_values = [graph_safe_float(targets.get(key, np.nan), default=0.0) for key in target_keys]
    y_mask_values = [float(masks.get(key, 0.0)) for key in target_keys]

    base_idx = -1 if base_idx is None else int(base_idx)
    view_idx = -1 if view_idx is None else int(view_idx)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_dist=edge_dist,
        edge_image=edge_image,
        y=torch.tensor(y_values, dtype=torch.float32).view(1, -1),
        y_mask=torch.tensor(y_mask_values, dtype=torch.float32).view(1, -1),
    )

    if attach_graph_attr:
        data.graph_attr = build_graph_attr(meta, structure).view(1, -1)

    data.base_idx = torch.tensor([base_idx], dtype=torch.long)
    data.view_idx = torch.tensor([view_idx], dtype=torch.long)
    data.sample_idx = torch.tensor([base_idx], dtype=torch.long)
    data.task_type = torch.tensor([0], dtype=torch.long)
    data.target_key_id = torch.tensor([-1], dtype=torch.long)
    data.has_energy_axis = torch.tensor([0], dtype=torch.uint8)
    data.has_physical_axis = data.has_energy_axis

    return data


def structure_to_pyg_spectrum_data(
    structure,
    spectrum,
    spectral_axis,
    meta,
    spectrum_target_key,
    cutoff=5.0,
    max_neighbors=16,
    symmetrize=True,
    edge_encoding="rbf",
    num_rbf=16,
    attach_graph_attr=False,
    base_idx=None,
    view_idx=None,
):
    x = build_node_tensor(structure)

    edge_index, edge_attr, edge_dist, edge_image = build_edges(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        symmetrize=symmetrize,
        edge_encoding=edge_encoding,
        num_rbf=num_rbf,
    )

    y = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    y_mask = np.isfinite(y).astype(np.float32)
    y_safe = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if spectral_axis is None:
        axis = np.arange(len(y_safe), dtype=np.float32)
    else:
        axis = np.asarray(spectral_axis, dtype=np.float32).reshape(-1)

    has_energy_axis = int(bool(meta.get("has_energy_grid", False)))
    base_idx = -1 if base_idx is None else int(base_idx)
    view_idx = -1 if view_idx is None else int(view_idx)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_dist=edge_dist,
        edge_image=edge_image,
        y=torch.tensor(y_safe, dtype=torch.float32).view(1, -1),
        y_mask=torch.tensor(y_mask, dtype=torch.float32).view(1, -1),
    )

    axis_tensor = torch.tensor(axis, dtype=torch.float32).view(1, -1)
    data.axis_grid = axis_tensor
    data.spectral_axis = axis_tensor

    if attach_graph_attr:
        data.graph_attr = build_graph_attr(meta, structure).view(1, -1)

    data.base_idx = torch.tensor([base_idx], dtype=torch.long)
    data.view_idx = torch.tensor([view_idx], dtype=torch.long)
    data.sample_idx = torch.tensor([base_idx], dtype=torch.long)
    data.task_type = torch.tensor([1], dtype=torch.long)
    data.target_key_id = torch.tensor([SPECTRUM_TARGET_TO_ID.get(spectrum_target_key, -1)], dtype=torch.long)
    data.has_energy_axis = torch.tensor([has_energy_axis], dtype=torch.uint8)
    data.has_physical_axis = data.has_energy_axis

    return data


class CrystalGraphScalarDataset(PyGDataset):
    """
    Crystal graph dataset for scalar targets.
    """

    def __init__(
        self,
        base_dataset,
        target_keys=None,
        cutoff=5.0,
        max_neighbors=16,
        symmetrize=True,
        edge_encoding="rbf",
        num_rbf=16,
        attach_graph_attr=False,
    ):
        super().__init__(None)

        self.base_dataset = base_dataset
        self.target_keys = target_keys if target_keys is not None else SCALAR_TARGET_KEYS
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.symmetrize = symmetrize
        self.edge_encoding = edge_encoding
        self.num_rbf = num_rbf
        self.attach_graph_attr = attach_graph_attr
        self.identity_table = []

        for idx in range(len(self.base_dataset)):
            _, _, _, meta = self.base_dataset[idx]
            base_idx = resolve_base_idx(meta, idx)

            self.identity_table.append(
                {
                    "base_idx": base_idx,
                    "view_idx": -1,
                    "sample_uid": resolve_sample_uid(meta, base_idx),
                    "mat_id": meta.get("mat_id", None),
                    "file_name": meta.get("file_name", None),
                    "formula": meta.get("formula", None),
                }
            )

    def len(self):
        return len(self.base_dataset)

    def get(self, idx):
        structure, targets, masks, meta = self.base_dataset[idx]
        base_idx = resolve_base_idx(meta, idx)

        return structure_to_pyg_scalar_data(
            structure=structure,
            targets=targets,
            masks=masks,
            meta=meta,
            target_keys=self.target_keys,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            symmetrize=self.symmetrize,
            edge_encoding=self.edge_encoding,
            num_rbf=self.num_rbf,
            attach_graph_attr=self.attach_graph_attr,
            base_idx=base_idx,
            view_idx=-1,
        )

    def get_identity(self, idx):
        return self.identity_table[idx]


class CrystalGraphSpectrumDataset(PyGDataset):
    """
    Crystal graph dataset for one spectral target.
    """

    def __init__(
        self,
        spectrum_view_dataset,
        cutoff=5.0,
        max_neighbors=16,
        symmetrize=True,
        edge_encoding="rbf",
        num_rbf=16,
        attach_graph_attr=False,
    ):
        super().__init__(None)

        self.spectrum_view_dataset = spectrum_view_dataset
        self.base_dataset = spectrum_view_dataset.base_dataset
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.symmetrize = symmetrize
        self.edge_encoding = edge_encoding
        self.num_rbf = num_rbf
        self.attach_graph_attr = attach_graph_attr
        self.spectrum_target_key = getattr(spectrum_view_dataset, "spectrum_key", "epsI_0")
        self.base_indices = list(self.spectrum_view_dataset.valid_indices)
        self.identity_table = []

        for view_idx, base_idx in enumerate(self.base_indices):
            meta = self.base_dataset.get_full_item(base_idx)["meta"]

            self.identity_table.append(
                {
                    "base_idx": int(base_idx),
                    "view_idx": int(view_idx),
                    "sample_uid": resolve_sample_uid(meta, int(base_idx)),
                    "mat_id": meta.get("mat_id", None),
                    "file_name": meta.get("file_name", None),
                    "formula": meta.get("formula", None),
                    "target_key": self.spectrum_target_key,
                }
            )

    def len(self):
        return len(self.spectrum_view_dataset)

    def get(self, idx):
        structure, spectrum, mask, spectral_axis, meta = self.spectrum_view_dataset[idx]
        base_idx = int(self.base_indices[idx])

        data = structure_to_pyg_spectrum_data(
            structure=structure,
            spectrum=spectrum.detach().cpu().numpy(),
            spectral_axis=spectral_axis.detach().cpu().numpy() if spectral_axis is not None else None,
            meta=meta,
            spectrum_target_key=self.spectrum_target_key,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            symmetrize=self.symmetrize,
            edge_encoding=self.edge_encoding,
            num_rbf=self.num_rbf,
            attach_graph_attr=self.attach_graph_attr,
            base_idx=base_idx,
            view_idx=idx,
        )

        data.y_mask = mask.view(1, -1).to(torch.float32)

        return data

    def get_identity(self, idx):
        return self.identity_table[idx]


def build_scalar_graph_dataset(ds, target_keys=None, attach_graph_attr=False, **graph_kwargs):
    config = dict(DEFAULT_GRAPH_CONFIG)
    config.update(graph_kwargs)
    config["attach_graph_attr"] = attach_graph_attr

    return CrystalGraphScalarDataset(
        base_dataset=ds,
        target_keys=target_keys if target_keys is not None else SCALAR_TARGET_KEYS,
        **config,
    )


def build_spectrum_graph_dataset(ds, spectrum_key, verbose=True, **graph_kwargs):
    config = dict(DEFAULT_GRAPH_CONFIG)
    config.update(graph_kwargs)

    spectrum_view = CrystalSpectrumView(
        base_dataset=ds,
        spectrum_key=spectrum_key,
        require_valid=True,
        verbose=verbose,
    )

    graph_dataset = CrystalGraphSpectrumDataset(
        spectrum_view_dataset=spectrum_view,
        **config,
    )

    return graph_dataset, spectrum_view


def build_paired_dielectric_graph_datasets(ds, key_real="epsR_0", key_imag="epsI_0", verbose=True, **graph_kwargs):
    real_graph_ds, real_view = build_spectrum_graph_dataset(
        ds,
        spectrum_key=key_real,
        verbose=verbose,
        **graph_kwargs,
    )

    imag_graph_ds, imag_view = build_spectrum_graph_dataset(
        ds,
        spectrum_key=key_imag,
        verbose=verbose,
        **graph_kwargs,
    )

    if real_graph_ds.base_indices != imag_graph_ds.base_indices:
        raise ValueError(f"{key_real} and {key_imag} graph datasets are not aligned by base index.")

    return real_graph_ds, imag_graph_ds, real_view, imag_view


def build_epsR_graph_dataset(ds, verbose=True, **graph_kwargs):
    return build_spectrum_graph_dataset(
        ds,
        spectrum_key="epsR_0",
        verbose=verbose,
        **graph_kwargs,
    )


def build_epsI_graph_dataset(ds, verbose=True, **graph_kwargs):
    return build_spectrum_graph_dataset(
        ds,
        spectrum_key="epsI_0",
        verbose=verbose,
        **graph_kwargs,
    )


def zero_edge_check(graph_ds, name="graph_ds", num_check=200):
    n = min(num_check, len(graph_ds))
    zero_count = 0

    for idx in range(n):
        graph = graph_ds[idx]

        if graph.edge_index.shape[1] == 0:
            zero_count += 1

    print(f"{name}: zero-edge graphs = {zero_count} / {n}")

    return zero_count


def summarize_graph_dataset(graph_ds, name="graph_ds"):
    graph = graph_ds[0]

    print("=" * 100)
    print(name)
    print("=" * 100)
    print("number of samples     :", len(graph_ds))
    print("node feature dimension:", graph.x.shape[1])
    print("edge feature dimension:", graph.edge_attr.shape[1] if graph.edge_attr.numel() > 0 else 0)
    print("number of nodes       :", int(graph.num_nodes))
    print("number of edges       :", int(graph.edge_index.shape[1]))
    print("target shape          :", tuple(graph.y.shape))

    if hasattr(graph, "axis_grid"):
        print("axis shape            :", tuple(graph.axis_grid.shape))
        print("axis range            :", float(graph.axis_grid.min()), "to", float(graph.axis_grid.max()), "eV")

    print("base_idx              :", int(graph.base_idx[0]))
    print("view_idx              :", int(graph.view_idx[0]))
    print("target_key_id         :", int(graph.target_key_id[0]))

    return graph


def batch_sanity_check(graph_ds, name="graph_ds", batch_size=4):
    loader = PyGDataLoader(graph_ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    print("=" * 100)
    print(f"Batch sanity check: {name}")
    print("=" * 100)
    print("batch.x shape          :", tuple(batch.x.shape))
    print("batch.edge_index shape :", tuple(batch.edge_index.shape))
    print("batch.edge_attr shape  :", tuple(batch.edge_attr.shape))
    print("batch.y shape          :", tuple(batch.y.shape))
    print("batch.y_mask shape     :", tuple(batch.y_mask.shape))
    print("batch.batch shape      :", tuple(batch.batch.shape))
    print("batch.base_idx shape   :", tuple(batch.base_idx.shape))

    if hasattr(batch, "axis_grid"):
        print("batch.axis_grid shape  :", tuple(batch.axis_grid.shape))

    return batch


def check_paired_graph_alignment(real_graph_ds, imag_graph_ds):
    if len(real_graph_ds) != len(imag_graph_ds):
        raise ValueError("Paired graph datasets have different lengths.")

    if real_graph_ds.base_indices != imag_graph_ds.base_indices:
        raise ValueError("Paired graph datasets do not share the same base indices.")

    real_graph = real_graph_ds[0]
    imag_graph = imag_graph_ds[0]

    if int(real_graph.base_idx[0]) != int(imag_graph.base_idx[0]):
        raise ValueError("The first paired graphs do not share the same base index.")

    if real_graph.y.shape != imag_graph.y.shape:
        raise ValueError("The paired spectral targets have different shapes.")

    if not torch.allclose(real_graph.axis_grid, imag_graph.axis_grid):
        raise ValueError("The paired spectral targets do not share the same axis grid.")

    print("=" * 100)
    print("Paired dielectric graph alignment")
    print("=" * 100)
    print("number of paired samples:", len(real_graph_ds))
    print("first base_idx          :", int(real_graph.base_idx[0]))
    print("real target shape       :", tuple(real_graph.y.shape))
    print("imag target shape       :", tuple(imag_graph.y.shape))
    print("axis shape              :", tuple(real_graph.axis_grid.shape))
    print("axis range              :", float(real_graph.axis_grid.min()), "to", float(real_graph.axis_grid.max()), "eV")
    print("status                  : passed")
# ============================================================
# Fixed paired split utilities
# ============================================================

from torch.utils.data import Subset


def build_fixed_paired_split(
    real_graph_ds,
    imag_graph_ds,
    processed_dir,
    seed=2025,
    val_ratio=0.10,
    test_ratio=0.10,
    split_subdir="fixed_splits",
    split_prefix="fixed_split_paired_epsR_epsI",
    version="v1",
    overwrite=False,
):
    """
    Build or load a fixed train/val/test split shared by epsR_0 and epsI_0.

    The split is generated over paired view indices. The same indices are used
    for both dielectric targets to keep epsR_0 and epsI_0 strictly aligned.
    """
    processed_dir = Path(processed_dir)
    split_dir = processed_dir / split_subdir
    split_dir.mkdir(parents=True, exist_ok=True)

    split_stem = f"{split_prefix}_seed{seed}_{version}"
    split_npz = split_dir / f"{split_stem}.npz"
    split_csv = split_dir / f"{split_stem}.csv"

    if len(real_graph_ds) != len(imag_graph_ds):
        raise ValueError("epsR_0 and epsI_0 graph datasets have different lengths.")

    if real_graph_ds.base_indices != imag_graph_ds.base_indices:
        raise ValueError("epsR_0 and epsI_0 graph datasets are not aligned by base_idx.")

    num_samples = len(real_graph_ds)
    paired_base_indices = np.asarray(real_graph_ds.base_indices, dtype=np.int64)

    if split_npz.exists() and not overwrite:
        split_data = np.load(split_npz)

        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
        test_idx = split_data["test_idx"]

        print("Loaded existing fixed split:", split_npz)

    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(num_samples)

        n_val = int(round(val_ratio * num_samples))
        n_test = int(round(test_ratio * num_samples))
        n_train = num_samples - n_val - n_test

        train_idx = np.sort(perm[:n_train])
        val_idx = np.sort(perm[n_train:n_train + n_val])
        test_idx = np.sort(perm[n_train + n_val:])

        np.savez_compressed(
            split_npz,
            seed=np.asarray([seed], dtype=np.int64),
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_base_idx=paired_base_indices[train_idx],
            val_base_idx=paired_base_indices[val_idx],
            test_base_idx=paired_base_indices[test_idx],
        )

        print("Saved fixed split:", split_npz)

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    if len(train_set & val_set) > 0:
        raise ValueError("Train and val splits overlap.")

    if len(train_set & test_set) > 0:
        raise ValueError("Train and test splits overlap.")

    if len(val_set & test_set) > 0:
        raise ValueError("Val and test splits overlap.")

    if len(train_idx) + len(val_idx) + len(test_idx) != num_samples:
        raise ValueError("Split sizes do not sum to the full paired dataset.")

    rows = []

    for split_name, indices in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        for view_idx in indices:
            item = real_graph_ds.get_identity(int(view_idx))

            rows.append(
                {
                    "split": split_name,
                    "view_idx": int(view_idx),
                    "base_idx": int(item["base_idx"]),
                    "mat_id": item.get("mat_id", None),
                    "formula": item.get("formula", None),
                    "file_name": item.get("file_name", None),
                }
            )

    split_df = pd.DataFrame(rows)
    split_df.to_csv(split_csv, index=False)

    epsR_train_ds = Subset(real_graph_ds, train_idx.tolist())
    epsR_val_ds = Subset(real_graph_ds, val_idx.tolist())
    epsR_test_ds = Subset(real_graph_ds, test_idx.tolist())

    epsI_train_ds = Subset(imag_graph_ds, train_idx.tolist())
    epsI_val_ds = Subset(imag_graph_ds, val_idx.tolist())
    epsI_test_ds = Subset(imag_graph_ds, test_idx.tolist())

    epsR_check_loader = PyGDataLoader(epsR_train_ds, batch_size=min(8, len(epsR_train_ds)), shuffle=False)
    epsI_check_loader = PyGDataLoader(epsI_train_ds, batch_size=min(8, len(epsI_train_ds)), shuffle=False)

    epsR_batch = next(iter(epsR_check_loader))
    epsI_batch = next(iter(epsI_check_loader))

    if not torch.equal(epsR_batch.base_idx, epsI_batch.base_idx):
        raise ValueError("epsR_0 and epsI_0 train batches are not aligned.")

    if not torch.allclose(epsR_batch.axis_grid, epsI_batch.axis_grid):
        raise ValueError("epsR_0 and epsI_0 train batches do not share the same axis grid.")

    print("=" * 100)
    print("Fixed paired split summary")
    print("=" * 100)
    print("total paired samples :", num_samples)
    print("train samples        :", len(train_idx))
    print("val samples          :", len(val_idx))
    print("test samples         :", len(test_idx))
    print("split seed           :", seed)
    print("split npz            :", split_npz)
    print("split csv            :", split_csv)
    print("epsR_0 train/val/test:", len(epsR_train_ds), len(epsR_val_ds), len(epsR_test_ds))
    print("epsI_0 train/val/test:", len(epsI_train_ds), len(epsI_val_ds), len(epsI_test_ds))
    print("first train base_idx :", int(epsR_batch.base_idx[0]))
    print("batch alignment      : passed")

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_df": split_df,
        "split_npz": split_npz,
        "split_csv": split_csv,
        "epsR_train_ds": epsR_train_ds,
        "epsR_val_ds": epsR_val_ds,
        "epsR_test_ds": epsR_test_ds,
        "epsI_train_ds": epsI_train_ds,
        "epsI_val_ds": epsI_val_ds,
        "epsI_test_ds": epsI_test_ds,
    }

# ============================================================
# Enhanced structure-only graph utilities
# ============================================================

def clip_scale(value, denom, default=0.0):
    value = graph_safe_float(value, default=np.nan)

    if not np.isfinite(value):
        return float(default), 0.0

    return float(value / denom), 1.0


def get_valence_shell_counts(element):
    s = p = d = f = 0.0

    try:
        full_config = element.full_electronic_structure

        for _, orbital, occupation in full_config:
            if orbital == "s":
                s += float(occupation)
            elif orbital == "p":
                p += float(occupation)
            elif orbital == "d":
                d += float(occupation)
            elif orbital == "f":
                f += float(occupation)

    except Exception:
        pass

    return s, p, d, f


def block_one_hot(element):
    block = getattr(element, "block", None)

    return [
        1.0 if block == "s" else 0.0,
        1.0 if block == "p" else 0.0,
        1.0 if block == "d" else 0.0,
        1.0 if block == "f" else 0.0,
    ]


def get_element_features_enhanced(symbol):
    """
    Enhanced structure-only node features with dimension 19.

    Features:
    [Z, row, group, atomic_mass, electronegativity,
     atomic_radius, atomic_radius_calculated, average_ionic_radius,
     ionization_energy,
     val_s, val_p, val_d, val_f,
     block_s, block_p, block_d, block_f,
     is_metal, is_transition_metal]
    """
    element = Element(symbol)

    atomic_number, _ = clip_scale(getattr(element, "Z", np.nan), 100.0)
    row, _ = clip_scale(getattr(element, "row", np.nan), 7.0)
    group, _ = clip_scale(getattr(element, "group", np.nan), 18.0)
    atomic_mass, _ = clip_scale(getattr(element, "atomic_mass", np.nan), 250.0)

    try:
        electronegativity_raw = element.data.get("X", np.nan)
    except Exception:
        electronegativity_raw = np.nan

    electronegativity, _ = clip_scale(electronegativity_raw, 4.0)

    atomic_radius, _ = clip_scale(getattr(element, "atomic_radius", np.nan), 3.5)
    atomic_radius_calculated, _ = clip_scale(getattr(element, "atomic_radius_calculated", np.nan), 3.5)
    average_ionic_radius, _ = clip_scale(getattr(element, "average_ionic_radius", np.nan), 3.5)
    ionization_energy, _ = clip_scale(getattr(element, "ionization_energy", np.nan), 25.0)

    val_s, val_p, val_d, val_f = get_valence_shell_counts(element)

    val_s /= 14.0
    val_p /= 14.0
    val_d /= 14.0
    val_f /= 14.0

    block_features = block_one_hot(element)

    is_metal = 1.0 if getattr(element, "is_metal", False) else 0.0
    is_transition_metal = 1.0 if getattr(element, "is_transition_metal", False) else 0.0

    return [
        atomic_number,
        row,
        group,
        atomic_mass,
        electronegativity,
        atomic_radius,
        atomic_radius_calculated,
        average_ionic_radius,
        ionization_energy,
        val_s,
        val_p,
        val_d,
        val_f,
        *block_features,
        is_metal,
        is_transition_metal,
    ]


def build_node_tensor_enhanced(structure):
    features = []

    for site in structure.sites:
        features.append(get_element_features_enhanced(site.specie.symbol))

    return torch.tensor(features, dtype=torch.float32)


def get_site_pair_chem_features(structure, edge_index):
    """
    Edge chemistry features with dimension 8.

    Features:
    [abs_delta_X, abs_delta_radius, radius_sum, abs_delta_ionization,
     abs_delta_Z, same_element, same_group, same_row]
    """
    values = []

    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        element_i = Element(structure.sites[source].specie.symbol)
        element_j = Element(structure.sites[target].specie.symbol)

        zi = graph_safe_float(getattr(element_i, "Z", np.nan), default=0.0) / 100.0
        zj = graph_safe_float(getattr(element_j, "Z", np.nan), default=0.0) / 100.0

        try:
            xi = graph_safe_float(element_i.data.get("X", np.nan), default=0.0) / 4.0
        except Exception:
            xi = 0.0

        try:
            xj = graph_safe_float(element_j.data.get("X", np.nan), default=0.0) / 4.0
        except Exception:
            xj = 0.0

        ri = graph_safe_float(getattr(element_i, "atomic_radius", np.nan), default=0.0) / 3.5
        rj = graph_safe_float(getattr(element_j, "atomic_radius", np.nan), default=0.0) / 3.5

        ii = graph_safe_float(getattr(element_i, "ionization_energy", np.nan), default=0.0) / 25.0
        ij = graph_safe_float(getattr(element_j, "ionization_energy", np.nan), default=0.0) / 25.0

        group_i = graph_safe_float(getattr(element_i, "group", np.nan), default=-1)
        group_j = graph_safe_float(getattr(element_j, "group", np.nan), default=-2)

        row_i = graph_safe_float(getattr(element_i, "row", np.nan), default=-1)
        row_j = graph_safe_float(getattr(element_j, "row", np.nan), default=-2)

        same_element = 1.0 if element_i.symbol == element_j.symbol else 0.0
        same_group = 1.0 if group_i == group_j else 0.0
        same_row = 1.0 if row_i == row_j else 0.0

        values.append(
            [
                abs(xi - xj),
                abs(ri - rj),
                ri + rj,
                abs(ii - ij),
                abs(zi - zj),
                same_element,
                same_group,
                same_row,
            ]
        )

    return torch.tensor(values, dtype=torch.float32)


def build_graph_attr_enhanced(meta, structure):
    """
    Enhanced graph-level attributes with dimension 6.

    Features:
    [density, volume_per_atom, number_of_sites, number_of_elements,
     mean_atomic_number, std_atomic_number]
    """
    num_sites = max(len(structure.sites), 1)
    atomic_numbers = [site.specie.Z for site in structure.sites]

    mean_atomic_number = float(np.mean(atomic_numbers)) / 100.0 if len(atomic_numbers) > 0 else 0.0
    std_atomic_number = float(np.std(atomic_numbers)) / 60.0 if len(atomic_numbers) > 0 else 0.0

    return torch.tensor(
        [
            graph_safe_float(meta.get("density", structure.density), default=0.0) / 25.0,
            graph_safe_float(meta.get("volume", structure.lattice.volume), default=0.0) / float(num_sites) / 100.0,
            float(num_sites) / 40.0,
            graph_safe_float(meta.get("num_elements", 0.0), default=0.0) / 10.0,
            mean_atomic_number,
            std_atomic_number,
        ],
        dtype=torch.float32,
    )


def build_edges_enhanced(
    structure,
    cutoff=5.0,
    max_neighbors=16,
    symmetrize=True,
    num_rbf=16,
):
    edge_index, edge_attr_rbf, edge_dist, edge_image = build_edges(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        symmetrize=symmetrize,
        edge_encoding="rbf",
        num_rbf=num_rbf,
    )

    if edge_index.shape[1] == 0:
        edge_chem = torch.empty((0, 8), dtype=torch.float32)
        edge_attr = torch.empty((0, num_rbf + 8), dtype=torch.float32)

        return edge_index, edge_attr, edge_dist, edge_image, edge_chem

    edge_chem = get_site_pair_chem_features(structure, edge_index)
    edge_attr = torch.cat([edge_attr_rbf, edge_chem], dim=1)

    return edge_index, edge_attr, edge_dist, edge_image, edge_chem


def structure_to_pyg_scalar_data_enhanced(
    structure,
    targets,
    masks,
    meta,
    target_keys=None,
    cutoff=5.0,
    max_neighbors=16,
    symmetrize=True,
    num_rbf=16,
    attach_graph_attr=False,
    base_idx=None,
    view_idx=None,
):
    if target_keys is None:
        target_keys = SCALAR_TARGET_KEYS

    x = build_node_tensor_enhanced(structure)

    edge_index, edge_attr, edge_dist, edge_image, edge_chem = build_edges_enhanced(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        symmetrize=symmetrize,
        num_rbf=num_rbf,
    )

    y_values = [graph_safe_float(targets.get(key, np.nan), default=0.0) for key in target_keys]
    y_mask_values = [float(masks.get(key, 0.0)) for key in target_keys]

    base_idx = -1 if base_idx is None else int(base_idx)
    view_idx = -1 if view_idx is None else int(view_idx)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_dist=edge_dist,
        edge_image=edge_image,
        edge_chem=edge_chem,
        y=torch.tensor(y_values, dtype=torch.float32).view(1, -1),
        y_mask=torch.tensor(y_mask_values, dtype=torch.float32).view(1, -1),
    )

    if attach_graph_attr:
        data.graph_attr = build_graph_attr_enhanced(meta, structure).view(1, -1)

    data.base_idx = torch.tensor([base_idx], dtype=torch.long)
    data.view_idx = torch.tensor([view_idx], dtype=torch.long)
    data.sample_idx = torch.tensor([base_idx], dtype=torch.long)
    data.task_type = torch.tensor([0], dtype=torch.long)
    data.target_key_id = torch.tensor([-1], dtype=torch.long)
    data.has_energy_axis = torch.tensor([0], dtype=torch.uint8)
    data.has_physical_axis = data.has_energy_axis

    return data


def structure_to_pyg_spectrum_data_enhanced(
    structure,
    spectrum,
    spectral_axis,
    meta,
    spectrum_target_key,
    cutoff=5.0,
    max_neighbors=16,
    symmetrize=True,
    num_rbf=16,
    attach_graph_attr=False,
    base_idx=None,
    view_idx=None,
):
    x = build_node_tensor_enhanced(structure)

    edge_index, edge_attr, edge_dist, edge_image, edge_chem = build_edges_enhanced(
        structure=structure,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        symmetrize=symmetrize,
        num_rbf=num_rbf,
    )

    y = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    y_mask = np.isfinite(y).astype(np.float32)
    y_safe = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if spectral_axis is None:
        axis = np.arange(len(y_safe), dtype=np.float32)
    else:
        axis = np.asarray(spectral_axis, dtype=np.float32).reshape(-1)

    has_energy_axis = int(bool(meta.get("has_energy_grid", False)))
    base_idx = -1 if base_idx is None else int(base_idx)
    view_idx = -1 if view_idx is None else int(view_idx)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_dist=edge_dist,
        edge_image=edge_image,
        edge_chem=edge_chem,
        y=torch.tensor(y_safe, dtype=torch.float32).view(1, -1),
        y_mask=torch.tensor(y_mask, dtype=torch.float32).view(1, -1),
    )

    axis_tensor = torch.tensor(axis, dtype=torch.float32).view(1, -1)
    data.axis_grid = axis_tensor
    data.spectral_axis = axis_tensor

    if attach_graph_attr:
        data.graph_attr = build_graph_attr_enhanced(meta, structure).view(1, -1)

    data.base_idx = torch.tensor([base_idx], dtype=torch.long)
    data.view_idx = torch.tensor([view_idx], dtype=torch.long)
    data.sample_idx = torch.tensor([base_idx], dtype=torch.long)
    data.task_type = torch.tensor([1], dtype=torch.long)
    data.target_key_id = torch.tensor(
        [SPECTRUM_TARGET_TO_ID.get(spectrum_target_key, -1)],
        dtype=torch.long,
    )
    data.has_energy_axis = torch.tensor([has_energy_axis], dtype=torch.uint8)
    data.has_physical_axis = data.has_energy_axis

    return data


class CrystalGraphScalarDatasetEnhanced(PyGDataset):
    """
    Enhanced structure-only graph dataset for scalar targets.
    """

    def __init__(
        self,
        base_dataset,
        target_keys=None,
        cutoff=5.0,
        max_neighbors=16,
        symmetrize=True,
        num_rbf=16,
        attach_graph_attr=False,
    ):
        super().__init__(None)

        self.base_dataset = base_dataset
        self.target_keys = target_keys if target_keys is not None else SCALAR_TARGET_KEYS
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.symmetrize = symmetrize
        self.num_rbf = num_rbf
        self.attach_graph_attr = attach_graph_attr
        self.identity_table = []

        for idx in range(len(self.base_dataset)):
            _, _, _, meta = self.base_dataset[idx]
            base_idx = resolve_base_idx(meta, idx)

            self.identity_table.append(
                {
                    "base_idx": base_idx,
                    "view_idx": -1,
                    "sample_uid": resolve_sample_uid(meta, base_idx),
                    "mat_id": meta.get("mat_id", None),
                    "file_name": meta.get("file_name", None),
                    "formula": meta.get("formula", None),
                }
            )

    def len(self):
        return len(self.base_dataset)

    def get(self, idx):
        structure, targets, masks, meta = self.base_dataset[idx]
        base_idx = resolve_base_idx(meta, idx)

        return structure_to_pyg_scalar_data_enhanced(
            structure=structure,
            targets=targets,
            masks=masks,
            meta=meta,
            target_keys=self.target_keys,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            symmetrize=self.symmetrize,
            num_rbf=self.num_rbf,
            attach_graph_attr=self.attach_graph_attr,
            base_idx=base_idx,
            view_idx=-1,
        )

    def get_identity(self, idx):
        return self.identity_table[idx]


class CrystalGraphSpectrumDatasetEnhanced(PyGDataset):
    """
    Enhanced structure-only graph dataset for one spectral target.
    """

    def __init__(
        self,
        spectrum_view_dataset,
        cutoff=5.0,
        max_neighbors=16,
        symmetrize=True,
        num_rbf=16,
        attach_graph_attr=False,
    ):
        super().__init__(None)

        self.spectrum_view_dataset = spectrum_view_dataset
        self.base_dataset = spectrum_view_dataset.base_dataset
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.symmetrize = symmetrize
        self.num_rbf = num_rbf
        self.attach_graph_attr = attach_graph_attr
        self.spectrum_target_key = getattr(spectrum_view_dataset, "spectrum_key", "epsI_0")
        self.base_indices = list(self.spectrum_view_dataset.valid_indices)
        self.identity_table = []

        for view_idx, base_idx in enumerate(self.base_indices):
            meta = self.base_dataset.get_full_item(base_idx)["meta"]

            self.identity_table.append(
                {
                    "base_idx": int(base_idx),
                    "view_idx": int(view_idx),
                    "sample_uid": resolve_sample_uid(meta, int(base_idx)),
                    "mat_id": meta.get("mat_id", None),
                    "file_name": meta.get("file_name", None),
                    "formula": meta.get("formula", None),
                    "target_key": self.spectrum_target_key,
                }
            )

    def len(self):
        return len(self.spectrum_view_dataset)

    def get(self, idx):
        structure, spectrum, mask, spectral_axis, meta = self.spectrum_view_dataset[idx]
        base_idx = int(self.base_indices[idx])

        data = structure_to_pyg_spectrum_data_enhanced(
            structure=structure,
            spectrum=spectrum.detach().cpu().numpy(),
            spectral_axis=spectral_axis.detach().cpu().numpy() if spectral_axis is not None else None,
            meta=meta,
            spectrum_target_key=self.spectrum_target_key,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            symmetrize=self.symmetrize,
            num_rbf=self.num_rbf,
            attach_graph_attr=self.attach_graph_attr,
            base_idx=base_idx,
            view_idx=idx,
        )

        data.y_mask = mask.view(1, -1).to(torch.float32)

        return data

    def get_identity(self, idx):
        return self.identity_table[idx]


def build_scalar_graph_dataset_enhanced(
    ds,
    target_keys=None,
    attach_graph_attr=False,
    **graph_kwargs,
):
    config = {
        "cutoff": 5.0,
        "max_neighbors": 16,
        "symmetrize": True,
        "num_rbf": 16,
        "attach_graph_attr": attach_graph_attr,
    }
    config.update(graph_kwargs)

    return CrystalGraphScalarDatasetEnhanced(
        base_dataset=ds,
        target_keys=target_keys if target_keys is not None else SCALAR_TARGET_KEYS,
        **config,
    )


def build_spectrum_graph_dataset_enhanced(
    ds,
    spectrum_key,
    verbose=True,
    **graph_kwargs,
):
    config = {
        "cutoff": 5.0,
        "max_neighbors": 16,
        "symmetrize": True,
        "num_rbf": 16,
        "attach_graph_attr": False,
    }
    config.update(graph_kwargs)

    spectrum_view = CrystalSpectrumView(
        base_dataset=ds,
        spectrum_key=spectrum_key,
        require_valid=True,
        verbose=verbose,
    )

    graph_dataset = CrystalGraphSpectrumDatasetEnhanced(
        spectrum_view_dataset=spectrum_view,
        **config,
    )

    return graph_dataset, spectrum_view


def build_epsR_graph_dataset_enhanced(ds, verbose=True, **graph_kwargs):
    return build_spectrum_graph_dataset_enhanced(
        ds,
        spectrum_key="epsR_0",
        verbose=verbose,
        **graph_kwargs,
    )


def build_epsI_graph_dataset_enhanced(ds, verbose=True, **graph_kwargs):
    return build_spectrum_graph_dataset_enhanced(
        ds,
        spectrum_key="epsI_0",
        verbose=verbose,
        **graph_kwargs,
    )


def build_paired_dielectric_graph_datasets_enhanced(
    ds,
    key_real="epsR_0",
    key_imag="epsI_0",
    verbose=True,
    **graph_kwargs,
):
    real_graph_ds, real_view = build_spectrum_graph_dataset_enhanced(
        ds,
        spectrum_key=key_real,
        verbose=verbose,
        **graph_kwargs,
    )

    imag_graph_ds, imag_view = build_spectrum_graph_dataset_enhanced(
        ds,
        spectrum_key=key_imag,
        verbose=verbose,
        **graph_kwargs,
    )

    if real_graph_ds.base_indices != imag_graph_ds.base_indices:
        raise ValueError(
            f"{key_real} and {key_imag} enhanced graph datasets are not aligned by base index."
        )

    return real_graph_ds, imag_graph_ds, real_view, imag_view

# ============================================================
# Spectrum normalization / loss / metrics utilities
# ============================================================

EPS_SPECTRUM = 1e-8


def infer_allow_negative_from_target_key(target_key):
    """
    Default convention:
    - epsI_* : negative values are not allowed
    - epsR_* : negative values are allowed
    """
    if target_key is None:
        return True

    key = str(target_key).lower()
    if key.startswith("epsi"):
        return False
    if key.startswith("epsr"):
        return True

    return True


def get_default_negative_weight(target_key, fallback=0.02):
    """
    If target is epsI_*, a small negative penalty is used.
    If target is epsR_*, the negative penalty is disabled.
    """
    allow_negative = infer_allow_negative_from_target_key(target_key)

    if allow_negative:
        return 0.0

    return float(fallback)


def ensure_2d_tensor(x):
    if x.ndim == 1:
        return x.unsqueeze(0)

    return x


def safe_mask_like(ref, mask=None):
    if mask is None:
        return torch.ones_like(ref, dtype=ref.dtype, device=ref.device)

    mask = ensure_2d_tensor(mask).to(dtype=ref.dtype, device=ref.device)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0).to(ref.dtype)

    return mask


@torch.no_grad()
def compute_spectrum_stats(dataset, device="cpu"):
    """
    Compute per-bin mean/std of spectra in a mask-aware manner.
    Use this only on the train split.
    """
    sum_y = None
    sum_y2 = None
    sum_m = None

    for i in range(len(dataset)):
        data = dataset[i]

        y = ensure_2d_tensor(data.y)[0].to(torch.float32).cpu()
        m = getattr(data, "y_mask", None)

        if m is None:
            m = torch.ones_like(y, dtype=torch.float32)
        else:
            m = ensure_2d_tensor(m)[0].to(torch.float32).cpu()

        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = (m > 0).to(torch.float32)

        if sum_y is None:
            spectrum_length = y.numel()
            sum_y = torch.zeros(spectrum_length, dtype=torch.float64)
            sum_y2 = torch.zeros(spectrum_length, dtype=torch.float64)
            sum_m = torch.zeros(spectrum_length, dtype=torch.float64)

        sum_y += (y * m).to(torch.float64)
        sum_y2 += ((y * y) * m).to(torch.float64)
        sum_m += m.to(torch.float64)

    mean = sum_y / torch.clamp(sum_m, min=1.0)
    var = (sum_y2 / torch.clamp(sum_m, min=1.0)) - mean.pow(2)
    var = torch.clamp(var, min=1e-8)

    std = torch.sqrt(var)
    std = torch.clamp(std, min=1e-4)

    mean = mean.to(torch.float32).to(device)
    std = std.to(torch.float32).to(device)

    try:
        stats_df = pd.DataFrame(
            {
                "bin_idx": np.arange(len(mean)),
                "mean_train": mean.detach().cpu().numpy(),
                "std_train": std.detach().cpu().numpy(),
            }
        )
    except Exception:
        stats_df = None

    return mean, std, stats_df


def normalize_spectrum(y, mean, std):
    return (y - mean) / (std + EPS_SPECTRUM)


def denormalize_spectrum(y_norm, mean, std):
    return y_norm * (std + EPS_SPECTRUM) + mean


def spectrum_loss(
    pred_norm,
    target_norm,
    mask=None,
    pred_denorm=None,
    target_denorm=None,
    target_key=None,
    peak_weight=2.0,
    peak_mode="denorm",
    smoothness_weight=0.05,
    smoothness_domain="norm",
    negative_weight=None,
    allow_negative=None,
):
    """
    Peak-aware spectrum loss.

    Components:
    - SmoothL1 on normalized spectra
    - peak-aware weighting
    - first-derivative consistency
    - optional negative penalty for epsI_*
    """
    pred_norm = ensure_2d_tensor(pred_norm)
    target_norm = ensure_2d_tensor(target_norm)
    mask = safe_mask_like(target_norm, mask)

    if allow_negative is None:
        allow_negative = infer_allow_negative_from_target_key(target_key)

    if negative_weight is None:
        negative_weight = get_default_negative_weight(target_key, fallback=0.02)

    if peak_mode == "denorm" and target_denorm is not None:
        peak_ref = ensure_2d_tensor(target_denorm).detach().to(
            dtype=target_norm.dtype,
            device=target_norm.device,
        )
    else:
        peak_ref = target_norm.detach()

    peak_scale = torch.abs(peak_ref)
    peak_scale = peak_scale / (peak_scale.amax(dim=1, keepdim=True) + EPS_SPECTRUM)
    weights = 1.0 + peak_weight * peak_scale

    base = F.smooth_l1_loss(pred_norm, target_norm, reduction="none")
    base = base * weights * mask
    base_loss = base.sum() / torch.clamp(mask.sum(), min=1.0)

    smooth_loss = pred_norm.new_tensor(0.0)

    if pred_norm.shape[1] >= 2 and smoothness_weight > 0:
        if smoothness_domain == "denorm" and pred_denorm is not None and target_denorm is not None:
            pred_ref = ensure_2d_tensor(pred_denorm)
            true_ref = ensure_2d_tensor(target_denorm)
        else:
            pred_ref = pred_norm
            true_ref = target_norm

        dp = pred_ref[:, 1:] - pred_ref[:, :-1]
        dt = true_ref[:, 1:] - true_ref[:, :-1]
        dm = mask[:, 1:] * mask[:, :-1]

        smooth_loss = torch.abs(dp - dt)
        smooth_loss = (smooth_loss * dm).sum() / torch.clamp(dm.sum(), min=1.0)

    total = base_loss + smoothness_weight * smooth_loss

    neg_loss = pred_norm.new_tensor(0.0)

    if (not allow_negative) and negative_weight > 0 and pred_denorm is not None:
        pred_denorm = ensure_2d_tensor(pred_denorm)
        neg_part = torch.relu(-pred_denorm)
        neg_loss = (neg_part * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        total = total + negative_weight * neg_loss

    info = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_base": float(base_loss.detach().cpu().item()),
        "loss_smooth": float(smooth_loss.detach().cpu().item()),
        "loss_negative": float(neg_loss.detach().cpu().item()),
        "allow_negative": bool(allow_negative),
        "negative_weight": float(negative_weight),
        "peak_mode": str(peak_mode),
        "smoothness_domain": str(smoothness_domain),
    }

    return total, info


@torch.no_grad()
def compute_spectrum_metrics(
    pred_denorm,
    true_denorm,
    mask=None,
    axis_grid=None,
):
    """
    Evaluation metrics on denormalized spectra.
    """
    pred_denorm = ensure_2d_tensor(pred_denorm)
    true_denorm = ensure_2d_tensor(true_denorm)
    mask = safe_mask_like(true_denorm, mask)

    valid = torch.clamp(mask.sum(dim=1), min=1.0)

    diff = (pred_denorm - true_denorm) * mask

    mae = diff.abs().sum(dim=1) / valid
    mse = diff.pow(2).sum(dim=1) / valid
    rmse = torch.sqrt(torch.clamp(mse, min=0.0))

    pred_m = pred_denorm * mask
    true_m = true_denorm * mask

    dot = (pred_m * true_m).sum(dim=1)
    pred_norm_v = torch.sqrt(pred_m.pow(2).sum(dim=1).clamp_min(EPS_SPECTRUM))
    true_norm_v = torch.sqrt(true_m.pow(2).sum(dim=1).clamp_min(EPS_SPECTRUM))
    cosine = dot / (pred_norm_v * true_norm_v + EPS_SPECTRUM)

    pred_peak_search = pred_denorm.masked_fill(mask <= 0, float("-inf"))
    true_peak_search = true_denorm.masked_fill(mask <= 0, float("-inf"))

    pred_peak = pred_peak_search.max(dim=1).values
    true_peak = true_peak_search.max(dim=1).values
    peak_mae = (pred_peak - true_peak).abs()

    pred_peak_idx = pred_peak_search.argmax(dim=1)
    true_peak_idx = true_peak_search.argmax(dim=1)
    peak_bin_shift = (pred_peak_idx - true_peak_idx).abs().to(torch.float32)

    neg_rate = ((pred_denorm < 0).to(torch.float32) * mask).sum(dim=1) / valid

    metrics = {
        "MAE_mean": float(mae.mean().detach().cpu().item()),
        "RMSE_mean": float(rmse.mean().detach().cpu().item()),
        "CosineSim_mean": float(cosine.mean().detach().cpu().item()),
        "PeakMAE_mean": float(peak_mae.mean().detach().cpu().item()),
        "PeakBinShift_mean": float(peak_bin_shift.mean().detach().cpu().item()),
        "PeakIndexError_mean": float(peak_bin_shift.mean().detach().cpu().item()),
        "NegativeBinRate_mean": float(neg_rate.mean().detach().cpu().item()),
    }

    if axis_grid is not None:
        axis_grid = ensure_2d_tensor(axis_grid).to(
            dtype=pred_denorm.dtype,
            device=pred_denorm.device,
        )

        if axis_grid.shape[0] == 1 and pred_denorm.shape[0] > 1:
            axis_grid = axis_grid.expand(pred_denorm.shape[0], -1)

        pred_peak_axis = torch.gather(axis_grid, 1, pred_peak_idx.view(-1, 1)).squeeze(1)
        true_peak_axis = torch.gather(axis_grid, 1, true_peak_idx.view(-1, 1)).squeeze(1)
        peak_axis_err = (pred_peak_axis - true_peak_axis).abs()

        metrics["PeakAxisError_mean"] = float(peak_axis_err.mean().detach().cpu().item())

    return metrics


@torch.no_grad()
def run_paired_spectrum_stats_smoke_test(
    epsR_train_ds,
    epsI_train_ds,
    device="cpu",
):
    """
    Compact smoke test for paired epsR_0 / epsI_0 train datasets.
    """
    outputs = {}

    for target_key, train_ds in [
        ("epsR_0", epsR_train_ds),
        ("epsI_0", epsI_train_ds),
    ]:
        mean_train, std_train, stats_df = compute_spectrum_stats(train_ds, device=device)

        y0 = ensure_2d_tensor(train_ds[0].y.to(torch.float32))
        m0 = ensure_2d_tensor(train_ds[0].y_mask.to(torch.float32))
        axis0 = ensure_2d_tensor(train_ds[0].axis_grid.to(torch.float32))

        metrics_debug = compute_spectrum_metrics(
            pred_denorm=y0.clone(),
            true_denorm=y0,
            mask=m0,
            axis_grid=axis0,
        )

        outputs[target_key] = {
            "mean_train": mean_train,
            "std_train": std_train,
            "stats_df": stats_df,
            "metrics_debug": metrics_debug,
            "allow_negative": infer_allow_negative_from_target_key(target_key),
            "negative_weight": get_default_negative_weight(target_key, fallback=0.02),
        }

    return outputs

# ============================================================
# Compact optical-response GINE training utilities
# ============================================================

def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    os.environ["PYTHONHASHSEED"] = str(int(seed))

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def flat2d_tensor(x):
    x = x.to(torch.float32)

    if x.dim() == 1:
        return x.view(1, -1)

    return x.view(x.shape[0], -1)


def get_axis_from_graph_batch(batch):
    axis = getattr(batch, "axis_grid", None)

    if axis is None:
        axis = getattr(batch, "spectral_axis", None)

    if axis is None:
        return None

    return flat2d_tensor(axis)


def maybe_shared_axis_array(axis_2d):
    if axis_2d is None:
        return None

    if torch.is_tensor(axis_2d):
        if axis_2d.ndim != 2:
            return axis_2d.detach().cpu().numpy()

        if axis_2d.shape[0] == 1:
            return axis_2d[0].detach().cpu().numpy()

        first = axis_2d[0:1].expand_as(axis_2d)
        if torch.allclose(axis_2d, first):
            return axis_2d[0].detach().cpu().numpy()

        return axis_2d.detach().cpu().numpy()

    return axis_2d


def autocast_context_for_device(device, use_amp):
    if not use_amp or device.type != "cuda":
        return nullcontext()

    return torch.amp.autocast(device_type="cuda", enabled=True)


def make_grad_scaler_for_device(device, use_amp):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=use_amp)

    return None


def build_model_from_signature(
    model_class,
    node_in_dim,
    edge_in_dim,
    out_dim,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.10,
    graph_attr_dim=0,
    use_graph_attr=False,
):
    init_sig = inspect.signature(model_class.__init__)
    pset = set(init_sig.parameters.keys())

    name_map = {
        "node_in_dim": node_in_dim,
        "in_dim": node_in_dim,
        "num_node_features": node_in_dim,
        "x_dim": node_in_dim,

        "edge_in_dim": edge_in_dim,
        "edge_dim": edge_in_dim,
        "num_edge_features": edge_in_dim,

        "out_dim": out_dim,
        "output_dim": out_dim,
        "num_outputs": out_dim,
        "spectrum_dim": out_dim,
        "target_dim": out_dim,

        "hidden_dim": hidden_dim,
        "hidden_channels": hidden_dim,
        "channels": hidden_dim,
        "emb_dim": hidden_dim,
        "embed_dim": hidden_dim,

        "latent_dim": latent_dim,

        "num_layers": num_layers,
        "n_layers": num_layers,
        "gnn_layers": num_layers,
        "num_convs": num_layers,

        "dropout": dropout,
        "dropout_rate": dropout,

        "graph_attr_dim": graph_attr_dim,
        "use_graph_attr": use_graph_attr,
        "attach_graph_attr": use_graph_attr,
    }

    kwargs = {k: v for k, v in name_map.items() if k in pset}

    try:
        return model_class(**kwargs)
    except Exception:
        pass

    fallback_args = [
        (node_in_dim, edge_in_dim, out_dim),
        (node_in_dim, edge_in_dim, hidden_dim, out_dim),
        (node_in_dim, edge_in_dim, hidden_dim, latent_dim, out_dim),
    ]

    for args in fallback_args:
        try:
            return model_class(*args)
        except Exception:
            pass

    raise RuntimeError("Could not initialize the optical model from the provided model_class.")


def extract_prediction_from_output(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ["pred", "prediction", "y_hat", "out", "output", "spectrum", "logits"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        for value in output.values():
            if torch.is_tensor(value):
                return value

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

            if isinstance(item, dict):
                pred = extract_prediction_from_output(item)

                if torch.is_tensor(pred):
                    return pred

    raise RuntimeError("Could not extract a prediction tensor from model output.")


def call_model_safely(model, batch):
    try:
        pnames = list(inspect.signature(model.forward).parameters.keys())
    except Exception:
        pnames = []

    if len(pnames) == 1 and pnames[0] in ["data", "batch", "graph", "g"]:
        return model(batch)

    available = {
        "x": batch.x,
        "edge_index": batch.edge_index,
        "edge_attr": getattr(batch, "edge_attr", None),
        "batch": batch.batch,
        "graph_attr": getattr(batch, "graph_attr", None),
        "spectral_axis": getattr(batch, "spectral_axis", None),
        "axis_grid": getattr(batch, "axis_grid", None),
    }

    kwargs = {k: v for k, v in available.items() if k in pnames and v is not None}

    trials = [
        lambda: model(**kwargs) if len(kwargs) > 0 else (_ for _ in ()).throw(RuntimeError()),
        lambda: model(batch),
        lambda: model(batch.x, batch.edge_index, batch.edge_attr, batch.batch),
        lambda: model(batch.x, batch.edge_index, batch.batch),
    ]

    for trial in trials:
        try:
            return trial()
        except Exception:
            pass

    raise RuntimeError("Calling model.forward failed.")


def load_partial_checkpoint(model, checkpoint_path, device="cpu", verbose=True):
    """
    Load matching weights only. Useful for SSL-init checkpoints with partially
    different heads.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict", None)
            or ckpt.get("state_dict", None)
            or ckpt.get("model", None)
            or ckpt
        )
    else:
        state = ckpt

    model_state = model.state_dict()
    matched = {}
    skipped = []

    for k, v in state.items():
        kk = k.replace("module.", "")

        if kk in model_state and tuple(model_state[kk].shape) == tuple(v.shape):
            matched[kk] = v
        else:
            skipped.append(k)

    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)

    if verbose:
        print(f"Loaded partial checkpoint: {checkpoint_path}")
        print(f"Matched tensors          : {len(matched)}")
        print(f"Skipped tensors          : {len(skipped)}")

    return model


def train_optical_gine_single_target(
    target_key,
    variant_name,
    train_ds,
    val_ds,
    test_ds,
    model_class,
    output_dir,
    device=None,
    seeds=(42, 123, 2025),
    batch_size=16,
    num_epochs=100,
    early_stop_patience=15,
    lr=5e-4,
    weight_decay=1e-4,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.10,
    peak_weight=2.0,
    smoothness_weight=0.03,
    sched_factor=0.5,
    sched_patience=4,
    min_lr=1e-6,
    clip_grad_norm=5.0,
    num_workers=0,
    pin_memory=None,
    use_amp=None,
    pretrained_ckpt_path=None,
    run_name=None,
):
    """
    Train one optical-response model for one target channel.

    This function is intentionally explicit:
    - pass epsR_train_ds / epsI_train_ds directly
    - do not rely on old global names such as opt_train_graph_ds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    if use_amp is None:
        use_amp = bool(torch.cuda.is_available())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"{target_key}_{variant_name}".replace("/", "_")

    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    sample0 = train_ds[0]

    node_in_dim = int(sample0.x.shape[-1])
    edge_in_dim = int(sample0.edge_attr.shape[-1]) if getattr(sample0, "edge_attr", None) is not None else 0
    out_dim = int(sample0.y.view(-1).shape[0])

    has_graph_attr = getattr(sample0, "graph_attr", None) is not None
    graph_attr_dim = int(sample0.graph_attr.view(-1).shape[0]) if has_graph_attr else 0

    sample_axis = get_axis_from_graph_batch(sample0)
    shared_spectral_axis = maybe_shared_axis_array(sample_axis) if sample_axis is not None else None

    spec_mean, spec_std, spec_stats_df = compute_spectrum_stats(train_ds, device="cpu")

    if spec_mean.numel() != out_dim or spec_std.numel() != out_dim:
        raise RuntimeError(
            f"spec_mean/spec_std do not match out_dim: {spec_mean.numel()}, {spec_std.numel()} vs {out_dim}"
        )

    spec_stats_path = run_output_dir / f"spectrum_train_stats_{target_key}_{variant_name}.csv"

    if spec_stats_df is not None:
        spec_stats_df.to_csv(spec_stats_path, index=False)

    spec_mean_dev = spec_mean.to(device)
    spec_std_dev = spec_std.to(device)

    print("=" * 100)
    print(f"TRAIN OPTICAL GINE | target={target_key} | variant={variant_name}")
    print("=" * 100)
    print("device              :", device)
    print("run_output_dir      :", run_output_dir)
    print("train/val/test      :", len(train_ds), len(val_ds), len(test_ds))
    print("node_in_dim         :", node_in_dim)
    print("edge_in_dim         :", edge_in_dim)
    print("out_dim             :", out_dim)
    print("graph_attr_dim      :", graph_attr_dim if has_graph_attr else "<none>")
    print("seeds               :", list(seeds))
    print("batch_size          :", batch_size)
    print("num_epochs          :", num_epochs)
    print("AMP                 :", use_amp)

    if sample_axis is not None:
        axis_np = sample_axis.view(-1).detach().cpu().numpy()
        print("axis_min            :", float(axis_np.min()))
        print("axis_max            :", float(axis_np.max()))
        if len(axis_np) > 1:
            print("axis_step           :", float(axis_np[1] - axis_np[0]))

    print("Mean(mean_train)    :", float(spec_mean.mean()))
    print("Mean(std_train)     :", float(spec_std.mean()))
    print("allow_negative      :", infer_allow_negative_from_target_key(target_key))
    print("negative_weight     :", get_default_negative_weight(target_key, fallback=0.02))

    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = PyGDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = PyGDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    def normalize_local(y):
        return (y - spec_mean_dev.view(1, -1)) / (spec_std_dev.view(1, -1) + EPS_SPECTRUM)

    def compute_loss_local(pred_denorm, true_denorm, mask):
        pred_denorm = flat2d_tensor(pred_denorm)
        true_denorm = flat2d_tensor(true_denorm)
        mask = flat2d_tensor(mask)

        pred_norm = normalize_local(pred_denorm)
        true_norm = normalize_local(true_denorm)

        return spectrum_loss(
            pred_norm=pred_norm,
            target_norm=true_norm,
            mask=mask,
            pred_denorm=pred_denorm,
            target_denorm=true_denorm,
            target_key=target_key,
            peak_weight=peak_weight,
            peak_mode="denorm",
            smoothness_weight=smoothness_weight,
            smoothness_domain="norm",
            negative_weight=None,
            allow_negative=None,
        )

    def compute_metrics_local(pred_denorm, true_denorm, mask, axis_grid=None):
        return compute_spectrum_metrics(
            pred_denorm=flat2d_tensor(pred_denorm),
            true_denorm=flat2d_tensor(true_denorm),
            mask=flat2d_tensor(mask),
            axis_grid=axis_grid,
        )

    with torch.no_grad():
        y0 = flat2d_tensor(sample0.y).to(device)
        m0 = flat2d_tensor(sample0.y_mask).to(device)
        a0 = get_axis_from_graph_batch(sample0)
        if a0 is not None:
            a0 = a0.to(device)
        smoke_metrics = compute_metrics_local(y0, y0, m0, axis_grid=a0)

    print("\nSmoke-test metrics on perfect prediction:")
    for k, v in smoke_metrics.items():
        print(f"{k}: {v:.6f}")

    def train_one_epoch(model, loader, optimizer, scaler=None):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            target = flat2d_tensor(batch.y)
            mask = flat2d_tensor(batch.y_mask)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context_for_device(device, use_amp):
                pred = flat2d_tensor(extract_prediction_from_output(call_model_safely(model, batch)))

                if pred.shape != target.shape:
                    raise RuntimeError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

                loss, _ = compute_loss_local(pred, target, mask)

            if scaler is not None and use_amp and device.type == "cuda":
                scaler.scale(loss).backward()

                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate_epoch(model, loader, return_predictions=False):
        model.eval()

        total_loss = 0.0
        n_batches = 0

        preds_all = []
        tgts_all = []
        masks_all = []
        axis_all = []

        base_idx_all = []
        sample_idx_all = []
        view_idx_all = []

        for batch in loader:
            batch = batch.to(device)

            target = flat2d_tensor(batch.y)
            mask = flat2d_tensor(batch.y_mask)

            axis_grid = get_axis_from_graph_batch(batch)
            if axis_grid is not None:
                axis_grid = axis_grid.to(device)

            pred = flat2d_tensor(extract_prediction_from_output(call_model_safely(model, batch)))

            if pred.shape != target.shape:
                raise RuntimeError(f"[EVAL] pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

            loss, _ = compute_loss_local(pred, target, mask)
            total_loss += float(loss.detach().item())
            n_batches += 1

            preds_all.append(pred.detach().cpu())
            tgts_all.append(target.detach().cpu())
            masks_all.append(mask.detach().cpu())

            if axis_grid is not None:
                axis_all.append(flat2d_tensor(axis_grid).detach().cpu())

            if hasattr(batch, "base_idx"):
                base_idx_all.append(batch.base_idx.detach().cpu())
            if hasattr(batch, "sample_idx"):
                sample_idx_all.append(batch.sample_idx.detach().cpu())
            if hasattr(batch, "view_idx"):
                view_idx_all.append(batch.view_idx.detach().cpu())

        preds_all = torch.cat(preds_all, dim=0)
        tgts_all = torch.cat(tgts_all, dim=0)
        masks_all = torch.cat(masks_all, dim=0)

        axis_cat = torch.cat(axis_all, dim=0) if len(axis_all) > 0 else None

        metrics = compute_metrics_local(preds_all, tgts_all, masks_all, axis_grid=axis_cat)
        metrics["loss"] = total_loss / max(n_batches, 1)

        if return_predictions:
            pack = {
                "pred": preds_all.numpy(),
                "true": tgts_all.numpy(),
                "mask": masks_all.numpy(),
                "spectral_axis": maybe_shared_axis_array(axis_cat) if axis_cat is not None else shared_spectral_axis,
                "axis_grid": maybe_shared_axis_array(axis_cat) if axis_cat is not None else shared_spectral_axis,
            }

            if len(base_idx_all) > 0:
                pack["base_idx"] = torch.cat(base_idx_all, dim=0).view(-1).numpy()
            if len(sample_idx_all) > 0:
                pack["sample_idx"] = torch.cat(sample_idx_all, dim=0).view(-1).numpy()
            if len(view_idx_all) > 0:
                pack["view_idx"] = torch.cat(view_idx_all, dim=0).view(-1).numpy()

            return metrics, pack

        return metrics

    summary_rows = []

    for seed in seeds:
        print("\n" + "=" * 118)
        print(f"RUN | target={target_key} | variant={variant_name} | seed={seed}")
        print("=" * 118)

        set_all_seeds(seed)

        model = build_model_from_signature(
            model_class=model_class,
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            graph_attr_dim=graph_attr_dim,
            use_graph_attr=has_graph_attr,
        ).to(device)

        if pretrained_ckpt_path is not None:
            model = load_partial_checkpoint(
                model=model,
                checkpoint_path=pretrained_ckpt_path,
                device=device,
                verbose=True,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_factor,
            patience=sched_patience,
            min_lr=min_lr,
        )

        scaler = make_grad_scaler_for_device(device, use_amp)

        best_state = None
        best_epoch = -1
        best_val = float("inf")
        patience_counter = 0
        history = []

        ckpt_path = run_output_dir / f"best_optical_{target_key}_{variant_name}_seed{seed}.pt"
        val_pred_path = run_output_dir / f"val_predictions_{target_key}_{variant_name}_seed{seed}.npz"
        test_pred_path = run_output_dir / f"test_predictions_{target_key}_{variant_name}_seed{seed}.npz"
        hist_path = run_output_dir / f"history_optical_{target_key}_{variant_name}_seed{seed}.csv"

        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_info = train_one_epoch(model, train_loader, optimizer, scaler)
            val_metrics = evaluate_epoch(model, val_loader, return_predictions=False)

            scheduler.step(val_metrics["MAE_mean"])

            row = {
                "target_key": target_key,
                "variant": variant_name,
                "seed": seed,
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **train_info,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)

            if val_metrics["MAE_mean"] < best_val:
                best_val = val_metrics["MAE_mean"]
                best_epoch = epoch
                patience_counter = 0

                best_state = {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "seed": seed,
                    "target_key": target_key,
                    "variant": variant_name,
                    "val_metrics": val_metrics,
                    "config": {
                        "target_key": target_key,
                        "variant": variant_name,
                        "node_in_dim": node_in_dim,
                        "edge_in_dim": edge_in_dim,
                        "out_dim": out_dim,
                        "graph_attr_dim": graph_attr_dim,
                        "hidden_dim": hidden_dim,
                        "latent_dim": latent_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "peak_weight": peak_weight,
                        "smoothness_weight": smoothness_weight,
                        "clip_grad_norm": clip_grad_norm,
                        "pretrained_ckpt_path": str(pretrained_ckpt_path) if pretrained_ckpt_path is not None else None,
                    },
                }

                torch.save(best_state, ckpt_path)

            else:
                patience_counter += 1

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"[target={target_key} | variant={variant_name} | seed={seed}] "
                    f"Epoch {epoch:03d} | train_loss={train_info['train_loss']:.5f} | "
                    f"val_MAE={val_metrics['MAE_mean']:.5f} | "
                    f"val_RMSE={val_metrics['RMSE_mean']:.5f} | "
                    f"val_Cos={val_metrics['CosineSim_mean']:.5f} | "
                    f"best_epoch={best_epoch:03d}"
                )

            if patience_counter >= early_stop_patience:
                print(f"[target={target_key} | variant={variant_name} | seed={seed}] Early stopping at epoch {epoch}.")
                break

        if best_state is None:
            raise RuntimeError("best_state is None. Training failed.")

        model.load_state_dict(best_state["model_state_dict"])

        val_best_metrics, val_pack = evaluate_epoch(model, val_loader, return_predictions=True)
        test_best_metrics, test_pack = evaluate_epoch(model, test_loader, return_predictions=True)

        np.savez_compressed(val_pred_path, **val_pack)
        np.savez_compressed(test_pred_path, **test_pack)
        pd.DataFrame(history).to_csv(hist_path, index=False)

        elapsed = time.time() - t0

        row = {
            "target_key": target_key,
            "variant": variant_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,

            "val_loss": val_best_metrics["loss"],
            "val_MAE_mean": val_best_metrics["MAE_mean"],
            "val_RMSE_mean": val_best_metrics["RMSE_mean"],
            "val_CosineSim_mean": val_best_metrics["CosineSim_mean"],
            "val_PeakMAE_mean": val_best_metrics["PeakMAE_mean"],
            "val_PeakBinShift_mean": val_best_metrics["PeakBinShift_mean"],
            "val_PeakAxisError_mean": val_best_metrics.get("PeakAxisError_mean", np.nan),
            "val_PeakIndexError_mean": val_best_metrics.get("PeakIndexError_mean", np.nan),
            "val_NegativeBinRate_mean": val_best_metrics["NegativeBinRate_mean"],

            "test_loss": test_best_metrics["loss"],
            "test_MAE_mean": test_best_metrics["MAE_mean"],
            "test_RMSE_mean": test_best_metrics["RMSE_mean"],
            "test_CosineSim_mean": test_best_metrics["CosineSim_mean"],
            "test_PeakMAE_mean": test_best_metrics["PeakMAE_mean"],
            "test_PeakBinShift_mean": test_best_metrics["PeakBinShift_mean"],
            "test_PeakAxisError_mean": test_best_metrics.get("PeakAxisError_mean", np.nan),
            "test_PeakIndexError_mean": test_best_metrics.get("PeakIndexError_mean", np.nan),
            "test_NegativeBinRate_mean": test_best_metrics["NegativeBinRate_mean"],

            "ckpt_path": str(ckpt_path),
            "val_pred_path": str(val_pred_path),
            "test_pred_path": str(test_pred_path),
            "history_path": str(hist_path),
        }

        summary_rows.append(row)

        print("-" * 118)
        print(f"[target={target_key} | variant={variant_name} | seed={seed}] BEST EPOCH = {best_epoch}")
        print("VAL :", {k: round(v, 6) if isinstance(v, float) else v for k, v in val_best_metrics.items()})
        print("TEST:", {k: round(v, 6) if isinstance(v, float) else v for k, v in test_best_metrics.items()})

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = run_output_dir / f"summary_optical_{target_key}_{variant_name}.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg_cols = [
        "val_MAE_mean", "val_RMSE_mean", "val_CosineSim_mean", "val_PeakMAE_mean",
        "val_PeakBinShift_mean", "val_PeakAxisError_mean", "val_PeakIndexError_mean",
        "val_NegativeBinRate_mean",
        "test_MAE_mean", "test_RMSE_mean", "test_CosineSim_mean", "test_PeakMAE_mean",
        "test_PeakBinShift_mean", "test_PeakAxisError_mean", "test_PeakIndexError_mean",
        "test_NegativeBinRate_mean",
    ]

    agg_summary = {}

    for col in agg_cols:
        if col in summary_df.columns:
            agg_summary[col + "_mean"] = float(summary_df[col].mean())
            agg_summary[col + "_std"] = float(summary_df[col].std(ddof=0))

    agg_json = run_output_dir / f"aggregate_optical_{target_key}_{variant_name}.json"

    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_key": target_key,
                "variant": variant_name,
                "n_seeds": len(list(seeds)),
                "seeds": list(seeds),
                "aggregate": agg_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_config = {
        "target_key": target_key,
        "variant": variant_name,
        "run_name": run_name,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "out_dim": out_dim,
        "graph_attr_dim": graph_attr_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "early_stop_patience": early_stop_patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "peak_weight": peak_weight,
        "smoothness_weight": smoothness_weight,
        "clip_grad_norm": clip_grad_norm,
        "use_amp": use_amp,
        "device": str(device),
        "pretrained_ckpt_path": str(pretrained_ckpt_path) if pretrained_ckpt_path is not None else None,
        "axis_available": sample_axis is not None,
        "axis_range_eV": [
            float(sample_axis.view(-1).detach().cpu().numpy().min()),
            float(sample_axis.view(-1).detach().cpu().numpy().max()),
        ] if sample_axis is not None else None,
        "axis_points": int(out_dim),
        "allow_negative": infer_allow_negative_from_target_key(target_key),
        "negative_weight": get_default_negative_weight(target_key, fallback=0.02),
    }

    with open(run_output_dir / f"train_config_{target_key}_{variant_name}.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    print("\nFINAL SUMMARY")
    summary_cols = [
        "target_key", "variant", "seed", "best_epoch",
        "val_MAE_mean", "val_RMSE_mean", "val_CosineSim_mean",
        "val_PeakMAE_mean", "val_PeakBinShift_mean", "val_PeakAxisError_mean",
        "test_MAE_mean", "test_RMSE_mean", "test_CosineSim_mean",
        "test_PeakMAE_mean", "test_PeakBinShift_mean", "test_PeakAxisError_mean",
    ]
    summary_cols = [c for c in summary_cols if c in summary_df.columns]

    print(summary_df[summary_cols].to_string(index=False))

    artifacts = {
        "run_output_dir": str(run_output_dir),
        "summary_csv": str(summary_csv),
        "aggregate_json": str(agg_json),
        "spectrum_stats_csv": str(spec_stats_path),
        "aggregate": agg_summary,
    }

    return summary_df, artifacts

# ============================================================
# OpticalResponseGINE model
# ============================================================

class OpticalResponseGINE(nn.Module):
    """
    GINE model for optical-response spectrum prediction.

    Supports:
    - baseline graph: node dim 11, edge dim 16
    - enhanced graph: node dim 19, edge dim 24
    - optional graph_attr
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        out_dim,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.10,
        graph_attr_dim=0,
        use_graph_attr=False,
    ):
        super().__init__()

        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.graph_attr_dim = int(graph_attr_dim)
        self.use_graph_attr = bool(use_graph_attr and self.graph_attr_dim > 0)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            conv = GINEConv(
                nn=mlp,
                edge_dim=self.edge_in_dim,
                train_eps=True,
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        head_in_dim = self.hidden_dim

        if self.use_graph_attr:
            head_in_dim += self.graph_attr_dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.out_dim),
        )

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
        graph_attr=None,
        **kwargs,
    ):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch
            graph_attr = getattr(data, "graph_attr", None)

        if x is None or edge_index is None or edge_attr is None:
            raise RuntimeError("OpticalResponseGINE requires x, edge_index, and edge_attr.")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res

        g = global_mean_pool(h, batch)

        if self.use_graph_attr and graph_attr is not None:
            graph_attr = graph_attr.view(g.size(0), -1).to(g.device)
            g = torch.cat([g, graph_attr], dim=1)

        out = self.head(g)
        return out


def build_optical_response_gine(
    node_in_dim,
    edge_in_dim,
    out_dim,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.10,
    graph_attr_dim=0,
    use_graph_attr=False,
):
    return OpticalResponseGINE(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        graph_attr_dim=graph_attr_dim,
        use_graph_attr=use_graph_attr,
    )

# ============================================================
# AB2C2 SRP-POWER 200 nm SCREENING UTILITIES
# Added for Q1_PAPER3 external screening
# Database: AB2C2 with C = P or N + band_gap > 0
# ============================================================

def _tb3_screen_safe_float(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _tb3_screen_doc_to_dict(doc):
    if hasattr(doc, "model_dump"):
        return doc.model_dump()
    if hasattr(doc, "dict"):
        return doc.dict()
    return dict(doc)


def _tb3_screen_to_pmg_structure(structure_obj):
    """
    Convert Materials Project structure output to pymatgen Structure.
    MP API may return pymatgen Structure, dict, or stringified dict.
    """
    import json
    from pymatgen.core import Structure

    if isinstance(structure_obj, Structure):
        return structure_obj

    if isinstance(structure_obj, dict):
        return Structure.from_dict(structure_obj)

    if isinstance(structure_obj, str):
        return Structure.from_dict(json.loads(structure_obj))

    raise TypeError(f"Unsupported structure type: {type(structure_obj)}")


def _tb3_screen_get_mpr():
    """
    Materials Project client with Python 3.10 typing patch.
    Requires MP_API_KEY or PMG_MAPI_KEY in environment.
    """
    import os
    import sys
    import typing

    try:
        from typing_extensions import NotRequired, Required
        typing.NotRequired = NotRequired
        typing.Required = Required
    except Exception as exc:
        raise ImportError(
            "Please install typing_extensions first:\n"
            "pip install typing_extensions"
        ) from exc

    for name in list(sys.modules.keys()):
        if name.startswith("mp_api") or name.startswith("emmet"):
            del sys.modules[name]

    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if key is None:
        raise RuntimeError(
            "Missing Materials Project API key.\n"
            "Run this before screening:\n"
            "import os\n"
            "os.environ['MP_API_KEY'] = 'PASTE_YOUR_NEW_API_KEY_HERE'"
        )

    return MPRester(key)


def _tb3_screen_is_ab2c2_C_PN(formula):
    """
    Check reduced composition AB2C2 where C-site element is P or N.
    Conditions:
    - exactly 3 elements
    - reduced amounts are [1, 2, 2]
    - P or N has amount 2, but not both
    """
    from pymatgen.core import Composition

    try:
        comp = Composition(formula).reduced_composition
        d = {str(el): float(v) for el, v in comp.get_el_amt_dict().items()}

        if len(d) != 3:
            return False

        amounts = sorted(d.values())
        if amounts != [1.0, 2.0, 2.0]:
            return False

        has_P2 = np.isclose(d.get("P", 0.0), 2.0)
        has_N2 = np.isclose(d.get("N", 0.0), 2.0)

        return bool(has_P2 ^ has_N2)

    except Exception:
        return False


def _tb3_screen_build_ab2c2_C_PN_pool(
    pool_csv,
    band_gap_min=0.001,
    energy_above_hull_max=None,
):
    """
    Build AB2C2 C=P/N candidate pool from Materials Project.
    """
    rows = []

    with _tb3_screen_get_mpr() as mpr:
        print("Querying ternary band-gap materials from Materials Project...")

        docs = mpr.materials.summary.search(
            num_elements=3,
            band_gap=(float(band_gap_min), None),
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull",
                "is_stable",
                "structure",
            ],
        )

        for doc in docs:
            d = _tb3_screen_doc_to_dict(doc)

            formula = d.get("formula_pretty", None)
            if formula is None:
                continue

            if not _tb3_screen_is_ab2c2_C_PN(formula):
                continue

            bg = _tb3_screen_safe_float(d.get("band_gap", np.nan))
            e_hull = _tb3_screen_safe_float(d.get("energy_above_hull", np.nan))

            if not np.isfinite(bg) or bg <= 0:
                continue

            if energy_above_hull_max is not None:
                if not np.isfinite(e_hull) or e_hull > float(energy_above_hull_max):
                    continue

            structure_obj = d.get("structure", None)
            if structure_obj is None:
                continue

            rows.append({
                "material_id": str(d.get("material_id")),
                "formula_pretty": formula,
                "band_gap": bg,
                "energy_above_hull": e_hull,
                "is_stable": d.get("is_stable", None),
                "structure": structure_obj,
            })

    pool_df = pd.DataFrame(rows).drop_duplicates("material_id").reset_index(drop=True)

    pool_csv = Path(pool_csv)
    pool_csv.parent.mkdir(parents=True, exist_ok=True)

    save_df = pool_df.drop(columns=["structure"]).copy()
    save_df.to_csv(pool_csv, index=False, encoding="utf-8-sig")

    print("Final AB2C2 C=P/N pool, band_gap > 0:", len(pool_df))
    print("Saved pool:", pool_csv)

    return pool_df


def _tb3_screen_load_training_summary(root, preferred_summary=None):
    root = Path(root)

    if preferred_summary is not None:
        preferred_summary = Path(preferred_summary)
        if preferred_summary.exists():
            print("Loaded training summary:", preferred_summary)
            return pd.read_csv(preferred_summary)
        raise FileNotFoundError(f"Cannot find preferred summary: {preferred_summary}")

    candidates = [
        root / "lowdata_ssl_init_ue_gine_raw.csv",
        root / "lowdata_ue_gine_vs_ue_ssl_gine_summary.csv",
        root / "lowdata_ssl_vs_scratch_summary.csv",
    ]

    for p in candidates:
        if p.exists():
            print("Loaded training summary:", p)
            return pd.read_csv(p)

    raise FileNotFoundError("Cannot find training summary CSV with ckpt_path.")


def _tb3_screen_is_ssl_100pct_row(row):
    variant = str(row.get("variant", "")).lower()
    model_name = str(row.get("model_name", "")).lower()

    fraction = _tb3_screen_safe_float(
        row.get("fraction", row.get("label_fraction", 1.0)),
        default=1.0,
    )

    has_ssl = ("ssl" in variant) or ("ssl" in model_name)
    has_100 = np.isclose(fraction, 1.0) or ("100pct" in variant)

    return has_ssl and has_100


def _tb3_screen_get_ssl_100pct_ckpt_paths(
    raw_summary,
    target_key,
    seeds=(42, 123, 2025),
):
    target_col = "target_key" if "target_key" in raw_summary.columns else "target"

    sub = raw_summary[raw_summary[target_col] == target_key].copy()
    sub = sub[sub.apply(_tb3_screen_is_ssl_100pct_row, axis=1)]

    if "seed" in sub.columns:
        sub["seed"] = pd.to_numeric(sub["seed"], errors="coerce")
        sub = sub[sub["seed"].isin(list(seeds))]
        sub = sub.sort_values("seed").drop_duplicates("seed", keep="last")

    if len(sub) == 0:
        raise RuntimeError(f"No SSL-init 100% checkpoint found for {target_key}.")

    if "ckpt_path" not in sub.columns:
        raise RuntimeError("Training summary does not contain ckpt_path column.")

    paths = [Path(p) for p in sub["ckpt_path"].tolist()]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {missing[:3]}")

    print(f"{target_key} SSL-init 100% checkpoints:", len(paths))
    return paths


def _tb3_screen_make_graph_dataset(pool_df, energy_ev, target_key="epsR_0"):
    """
    Convert external structures into enhanced 19/24 PyG graphs.
    """
    graphs = []
    meta_rows = []
    failed_rows = []

    y_dummy = np.zeros_like(energy_ev, dtype=np.float32)

    for i, row in pool_df.reset_index(drop=True).iterrows():
        try:
            structure = _tb3_screen_to_pmg_structure(row["structure"])

            meta = {
                "base_idx": i,
                "material_id": row["material_id"],
                "mat_id": row["material_id"],
                "formula": row.get("formula_pretty", None),
                "formula_pretty": row.get("formula_pretty", None),
                "has_energy_grid": True,
                "spectrum_key": target_key,
            }

            graph = structure_to_pyg_spectrum_data_enhanced(
                structure=structure,
                spectrum=y_dummy,
                spectral_axis=energy_ev,
                meta=meta,
                spectrum_target_key=target_key,
                cutoff=5.0,
                max_neighbors=16,
                symmetrize=True,
                num_rbf=16,
                attach_graph_attr=False,
                base_idx=i,
                view_idx=i,
            )

            graphs.append(graph)

            meta_rows.append({
                "screen_idx": len(graphs) - 1,
                "material_id": row["material_id"],
                "formula_pretty": row.get("formula_pretty", None),
                "band_gap": row.get("band_gap", np.nan),
                "energy_above_hull": row.get("energy_above_hull", np.nan),
                "is_stable": row.get("is_stable", None),
            })

        except Exception as exc:
            failed_rows.append({
                "row_idx": i,
                "material_id": row.get("material_id", None),
                "formula_pretty": row.get("formula_pretty", None),
                "reason": str(exc),
            })

    meta_df = pd.DataFrame(meta_rows)
    failed_df = pd.DataFrame(failed_rows)

    print(f"Valid graphs for {target_key}: {len(graphs)} / {len(pool_df)}")
    print(f"Failed structures for {target_key}: {len(failed_df)}")

    if len(failed_df) > 0:
        try:
            from IPython.display import display
            display(failed_df.head(10))
        except Exception:
            print(failed_df.head(10))

    return graphs, meta_df


def _tb3_screen_load_model_from_ckpt(ckpt_path, device):
    ckpt_path = Path(ckpt_path)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    config = ckpt.get("config", {})

    model = OpticalResponseGINE(
        node_in_dim=int(config.get("node_in_dim", 19)),
        edge_in_dim=int(config.get("edge_in_dim", 24)),
        out_dim=int(config.get("out_dim", 2001)),
        hidden_dim=int(config.get("hidden_dim", 192)),
        latent_dim=int(config.get("latent_dim", 256)),
        num_layers=int(config.get("num_layers", 4)),
        dropout=float(config.get("dropout", 0.05)),
        graph_attr_dim=int(config.get("graph_attr_dim", 0) or 0),
        use_graph_attr=False,
    ).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


@torch.no_grad()
def _tb3_screen_predict_ensemble(graphs, ckpt_paths, device, batch_size=32):
    from torch_geometric.loader import DataLoader as PyGDataLoader

    loader = PyGDataLoader(graphs, batch_size=int(batch_size), shuffle=False)
    all_seed_preds = []

    for ckpt_path in ckpt_paths:
        print("Predicting with:", ckpt_path)

        model = _tb3_screen_load_model_from_ckpt(ckpt_path, device=device)
        preds = []

        for batch in loader:
            batch = batch.to(device)

            out = extract_prediction_from_output(
                call_model_safely(model, batch)
            )

            preds.append(out.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        all_seed_preds.append(preds)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ens = np.mean(np.stack(all_seed_preds, axis=0), axis=0)
    return ens


def _tb3_screen_dielectric_to_optical(eps1, eps2, energy_ev):
    eps1 = np.asarray(eps1, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    energy_ev = np.asarray(energy_ev, dtype=float)

    eps_abs = np.sqrt(eps1**2 + eps2**2)

    n = np.sqrt(np.clip((eps_abs + eps1) / 2.0, 0.0, None))
    k = np.sqrt(np.clip((eps_abs - eps1) / 2.0, 0.0, None))

    alpha_cm1 = np.zeros_like(k)
    valid = energy_ev > 0

    alpha_cm1[:, valid] = (
        4.0 * np.pi * k[:, valid] * energy_ev[valid]
        / (1239.841984 * 1e-7)
    )

    R = ((n - 1.0)**2 + k**2) / (((n + 1.0)**2 + k**2) + 1e-12)
    R = np.clip(R, 0.0, 1.0)

    return n, k, alpha_cm1, R


def _tb3_screen_make_solar_proxy(
    energy_ev,
    pin_w_m2=1000.0,
    min_energy_ev=0.05,
    max_solar_ev=4.5,
    t_sun=5778.0,
):
    kb_ev_k = 8.617333262145e-5

    E = np.asarray(energy_ev, dtype=float)
    valid = E > min_energy_ev

    shape = np.zeros_like(E)
    x = E[valid] / (kb_ev_k * t_sun)

    shape[valid] = E[valid]**3 / np.expm1(x)
    shape[E > max_solar_ev] = 0.0

    area = np.trapz(shape, E)
    irradiance = shape * (pin_w_m2 / area)

    return irradiance


def _tb3_screen_compute_srp_power(
    alpha_cm1,
    R,
    energy_ev,
    thickness_nm=200.0,
):
    solar_irradiance_e = _tb3_screen_make_solar_proxy(energy_ev)
    thickness_cm = float(thickness_nm) * 1e-7

    alpha = np.clip(
        np.nan_to_num(alpha_cm1, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )

    R = np.clip(
        np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )

    A = (1.0 - R) * (1.0 - np.exp(-alpha * thickness_cm))
    A = np.clip(A, 0.0, 1.0)

    absorbed_power = np.trapz(
        solar_irradiance_e.reshape(1, -1) * A,
        energy_ev,
        axis=1,
    )

    total_power = np.trapz(solar_irradiance_e, energy_ev)

    return 100.0 * absorbed_power / total_power


def screen_ab2c2_C_PN_srp_power_200nm(
    root,
    top_k=50,
    thickness_nm=200.0,
    band_gap_min=0.001,
    energy_above_hull_max=None,
    seeds=(42, 123, 2025),
    batch_size=32,
    preferred_summary=None,
    show=True,
):
    """
    Screen AB2C2 materials with C = P or N using SSL-init UE-GINE.

    This function:
    1) queries Materials Project for ternary band-gap materials,
    2) keeps AB2C2 with P2 or N2,
    3) predicts epsR_0 and epsI_0 using 100% SSL-init checkpoints,
    4) derives n, k, alpha, R,
    5) ranks by SRP-power at 200 nm.

    Notes
    -----
    This function intentionally does NOT use Zintl Description filtering.
    """
    root = Path(root)

    output_dir = root / "paper_outputs"
    screen_dir = output_dir / "screening_srp_200nm"
    pool_dir = output_dir / "external_pools"

    screen_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    thickness_tag = f"{int(round(float(thickness_nm)))}nm"

    out_pool = pool_dir / "pool_ab2c2_C_PN_bandgap_gt0.csv"
    out_all = screen_dir / f"screening_ab2c2_C_PN_srp_power_{thickness_tag}_all.csv"
    out_top = screen_dir / f"top{int(top_k)}_ab2c2_C_PN_srp_power_{thickness_tag}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    energy_ev = np.linspace(0.0, 20.0, 2001).astype(np.float32)

    raw_summary = _tb3_screen_load_training_summary(
        root=root,
        preferred_summary=preferred_summary,
    )

    eps1_ckpts = _tb3_screen_get_ssl_100pct_ckpt_paths(
        raw_summary=raw_summary,
        target_key="epsR_0",
        seeds=seeds,
    )

    eps2_ckpts = _tb3_screen_get_ssl_100pct_ckpt_paths(
        raw_summary=raw_summary,
        target_key="epsI_0",
        seeds=seeds,
    )

    pool_df = _tb3_screen_build_ab2c2_C_PN_pool(
        pool_csv=out_pool,
        band_gap_min=band_gap_min,
        energy_above_hull_max=energy_above_hull_max,
    )

    print("\n" + "=" * 100)
    print("SCREENING POOL: AB2C2 with C=P/N + band_gap > 0")
    print("=" * 100)

    eps1_graphs, meta_df = _tb3_screen_make_graph_dataset(
        pool_df=pool_df,
        energy_ev=energy_ev,
        target_key="epsR_0",
    )

    eps2_graphs, meta_df2 = _tb3_screen_make_graph_dataset(
        pool_df=pool_df,
        energy_ev=energy_ev,
        target_key="epsI_0",
    )

    if len(eps1_graphs) == 0:
        raise RuntimeError("No valid structures in AB2C2 pool.")

    if len(eps1_graphs) != len(eps2_graphs):
        raise RuntimeError("epsR_0 and epsI_0 graph counts do not match.")

    print("Valid structures:", len(eps1_graphs))

    eps1_pred = _tb3_screen_predict_ensemble(
        graphs=eps1_graphs,
        ckpt_paths=eps1_ckpts,
        device=device,
        batch_size=batch_size,
    )

    eps2_pred = _tb3_screen_predict_ensemble(
        graphs=eps2_graphs,
        ckpt_paths=eps2_ckpts,
        device=device,
        batch_size=batch_size,
    )

    n_pred, k_pred, alpha_pred, R_pred = _tb3_screen_dielectric_to_optical(
        eps1=eps1_pred,
        eps2=eps2_pred,
        energy_ev=energy_ev,
    )

    srp_power = _tb3_screen_compute_srp_power(
        alpha_cm1=alpha_pred,
        R=R_pred,
        energy_ev=energy_ev,
        thickness_nm=thickness_nm,
    )

    out_df = meta_df.copy()
    out_df["predicted_SRP_power_200nm_percent"] = srp_power
    out_df["predicted_eps1_mean"] = np.nanmean(eps1_pred, axis=1)
    out_df["predicted_eps2_mean"] = np.nanmean(eps2_pred, axis=1)
    out_df["predicted_alpha_mean_cm1"] = np.nanmean(alpha_pred, axis=1)
    out_df["predicted_R_mean"] = np.nanmean(R_pred, axis=1)
    out_df["screening_score"] = out_df["predicted_SRP_power_200nm_percent"]

    out_df = out_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    out_df["rank"] = np.arange(1, len(out_df) + 1)

    top_df = out_df.head(int(top_k)).copy()

    out_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    top_df.to_csv(out_top, index=False, encoding="utf-8-sig")

    print("\nSaved all:", out_all)
    print(f"Saved top{int(top_k)}:", out_top)

    display_cols = [
        "rank",
        "material_id",
        "formula_pretty",
        "band_gap",
        "energy_above_hull",
        "is_stable",
        "predicted_SRP_power_200nm_percent",
    ]

    if show:
        try:
            from IPython.display import display
            display(top_df[display_cols].head(20))
        except Exception:
            print(top_df[display_cols].head(20))

    return {
        "pool_df": pool_df,
        "all_df": out_df,
        "top_df": top_df,
        "pool_csv": str(out_pool),
        "all_csv": str(out_all),
        "top_csv": str(out_top),
        "energy_ev": energy_ev,
        "eps1_pred": eps1_pred,
        "eps2_pred": eps2_pred,
        "n_pred": n_pred,
        "k_pred": k_pred,
        "alpha_pred_cm1": alpha_pred,
        "R_pred": R_pred,
    }
# ============================================================
# ZINTL-LIKE CHEMICAL-RULE SRP-POWER 200 nm SCREENING UTILITIES
# Added for Q1_PAPER3 external screening
# Database: Zintl-like candidate pool from chemical rules, not MP Description
# ============================================================


def _tb3_zintl_default_sets(strict_main_group_only=True):
    donor_elements = {
        "Li", "Na", "K", "Rb", "Cs",
        "Mg", "Ca", "Sr", "Ba",
        "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    }

    framework_elements = {
        "B", "Al", "Ga", "In", "Tl",
        "Si", "Ge", "Sn", "Pb",
        "P", "As", "Sb", "Bi",
    }

    excluded_elements = {
        "H", "C", "N", "O", "F", "Cl", "Br", "I",
        "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    }

    relaxed_extra_elements = {"Zn", "Cd", "Hg", "Cu", "Ag", "Au"}

    if strict_main_group_only:
        allowed_elements = donor_elements | framework_elements
    else:
        allowed_elements = donor_elements | framework_elements | relaxed_extra_elements

    return donor_elements, framework_elements, excluded_elements, allowed_elements


def _tb3_is_zintl_like_formula(
    formula,
    strict_main_group_only=True,
    num_elements_min=2,
    num_elements_max=5,
):
    """
    Conservative chemical-rule filter for Zintl-like candidates.

    This does not claim confirmed Zintl phase identity. It only builds a
    Zintl-like candidate pool for ML screening.
    """
    from pymatgen.core import Composition

    donor_elements, framework_elements, excluded_elements, allowed_elements = _tb3_zintl_default_sets(
        strict_main_group_only=strict_main_group_only
    )

    try:
        comp = Composition(formula).reduced_composition
        amt_dict = comp.get_el_amt_dict()
        elements = set(str(el) for el in amt_dict.keys())

        if len(elements) < int(num_elements_min) or len(elements) > int(num_elements_max):
            return False, {}

        if len(elements & excluded_elements) > 0:
            return False, {}

        donor_set = elements & donor_elements
        framework_set = elements & framework_elements

        if len(donor_set) < 1 or len(framework_set) < 1:
            return False, {}

        if strict_main_group_only and not elements.issubset(allowed_elements):
            return False, {}

        contains_pnictogen = len(elements & {"P", "As", "Sb", "Bi"}) > 0
        contains_group14 = len(elements & {"Si", "Ge", "Sn", "Pb"}) > 0
        contains_group13 = len(elements & {"B", "Al", "Ga", "In", "Tl"}) > 0
        contains_alkali = len(elements & {"Li", "Na", "K", "Rb", "Cs"}) > 0
        contains_alkaline_earth = len(elements & {"Mg", "Ca", "Sr", "Ba"}) > 0
        rare_earth_set = donor_elements - {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}
        contains_rare_earth = len(elements & rare_earth_set) > 0

        rule_score = 0
        rule_score += 1 if len(donor_set) >= 1 else 0
        rule_score += 1 if len(framework_set) >= 1 else 0
        rule_score += 1 if contains_pnictogen else 0
        rule_score += 1 if contains_group14 else 0
        rule_score += 1 if contains_alkali or contains_alkaline_earth else 0

        info = {
            "elements": ",".join(sorted(elements)),
            "donor_elements": ",".join(sorted(donor_set)),
            "framework_elements": ",".join(sorted(framework_set)),
            "n_elements": len(elements),
            "n_donor_elements": len(donor_set),
            "n_framework_elements": len(framework_set),
            "contains_pnictogen": contains_pnictogen,
            "contains_group14": contains_group14,
            "contains_group13": contains_group13,
            "contains_alkali": contains_alkali,
            "contains_alkaline_earth": contains_alkaline_earth,
            "contains_rare_earth": contains_rare_earth,
            "zintl_like_rule_score": rule_score,
        }
        return True, info

    except Exception:
        return False, {}


def _tb3_screen_build_zintl_like_pool(
    pool_csv,
    pool_pkl=None,
    band_gap_min=0.001,
    strict_main_group_only=True,
    num_elements_min=2,
    num_elements_max=5,
    use_cache=True,
):
    """
    Query Materials Project and build a Zintl-like chemical-rule pool.
    Structures are cached in a pickle for fast reruns.
    """
    pool_csv = Path(pool_csv)
    pool_csv.parent.mkdir(parents=True, exist_ok=True)

    if pool_pkl is not None:
        pool_pkl = Path(pool_pkl)
        pool_pkl.parent.mkdir(parents=True, exist_ok=True)
        if use_cache and pool_pkl.exists():
            print("Loading cached Zintl-like pool:", pool_pkl)
            pool_df = pd.read_pickle(pool_pkl)
            print("Cached Zintl-like pool:", len(pool_df))
            return pool_df

    rows = []

    with _tb3_screen_get_mpr() as mpr:
        for ne in range(int(num_elements_min), int(num_elements_max) + 1):
            print("=" * 100)
            print(f"Querying Materials Project: num_elements = {ne}, band_gap > {band_gap_min}")
            print("=" * 100)

            docs = mpr.materials.summary.search(
                num_elements=ne,
                band_gap=(float(band_gap_min), None),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "energy_above_hull",
                    "is_stable",
                    "structure",
                ],
            )

            for doc in docs:
                d = _tb3_screen_doc_to_dict(doc)
                formula = d.get("formula_pretty", None)
                if formula is None:
                    continue

                keep, rule_info = _tb3_is_zintl_like_formula(
                    formula=formula,
                    strict_main_group_only=strict_main_group_only,
                    num_elements_min=num_elements_min,
                    num_elements_max=num_elements_max,
                )
                if not keep:
                    continue

                structure_obj = d.get("structure", None)
                if structure_obj is None:
                    continue

                rows.append({
                    "material_id": str(d.get("material_id")),
                    "formula_pretty": formula,
                    "band_gap": _tb3_screen_safe_float(d.get("band_gap", np.nan)),
                    "energy_above_hull": _tb3_screen_safe_float(d.get("energy_above_hull", np.nan)),
                    "is_stable": d.get("is_stable", None),
                    "structure": structure_obj,
                    "pool_name": "Zintl-like chemical-rule pool",
                    **rule_info,
                })

    pool_df = pd.DataFrame(rows).drop_duplicates("material_id").reset_index(drop=True)

    if len(pool_df) == 0:
        raise RuntimeError(
            "No Zintl-like candidates found. Try strict_main_group_only=False "
            "or relaxing the chemical-rule criteria."
        )

    pool_df = pool_df.sort_values(
        ["is_stable", "energy_above_hull", "band_gap"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    save_df = pool_df.drop(columns=["structure"], errors="ignore").copy()
    save_df.to_csv(pool_csv, index=False, encoding="utf-8-sig")

    if pool_pkl is not None:
        pool_df.to_pickle(pool_pkl)
        print("Saved pool PKL:", pool_pkl)

    print("Final Zintl-like chemical-rule pool:", len(pool_df))
    print("Saved pool CSV:", pool_csv)

    return pool_df


def _tb3_merge_screening_meta_with_pool(meta_df, pool_df):
    pool_meta = pool_df.drop(columns=["structure"], errors="ignore").copy()
    out = meta_df.merge(pool_meta, on="material_id", how="left", suffixes=("", "_pool"))

    for col in ["formula_pretty", "band_gap", "energy_above_hull", "is_stable"]:
        pool_col = f"{col}_pool"
        if pool_col in out.columns:
            out[col] = out[col].combine_first(out[pool_col])
            out = out.drop(columns=[pool_col])

    return out


def _tb3_save_filtered_screening_tables(
    out_df,
    out_dir,
    prefix,
    top_k=50,
    dft_ehull_max=0.10,
    dft_bg_min=0.30,
    dft_bg_max=4.50,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_top_all = out_dir / f"top{int(top_k)}_{prefix}_all.csv"
    out_top_stable = out_dir / f"top{int(top_k)}_{prefix}_stable_only.csv"
    out_top_ehull010 = out_dir / f"top{int(top_k)}_{prefix}_ehull_le_010.csv"
    out_top_ehull020 = out_dir / f"top{int(top_k)}_{prefix}_ehull_le_020.csv"
    out_top_dft = out_dir / f"top{int(top_k)}_{prefix}_for_dft_validation.csv"

    top_all = out_df.head(int(top_k)).copy()

    stable_df = out_df[out_df["is_stable"].astype(str).str.lower() == "true"].copy()
    stable_df = stable_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    stable_df["filtered_rank"] = np.arange(1, len(stable_df) + 1)
    top_stable = stable_df.head(int(top_k)).copy()

    ehull010_df = out_df[out_df["energy_above_hull"] <= 0.10].copy()
    ehull010_df = ehull010_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    ehull010_df["filtered_rank"] = np.arange(1, len(ehull010_df) + 1)
    top_ehull010 = ehull010_df.head(int(top_k)).copy()

    ehull020_df = out_df[out_df["energy_above_hull"] <= 0.20].copy()
    ehull020_df = ehull020_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    ehull020_df["filtered_rank"] = np.arange(1, len(ehull020_df) + 1)
    top_ehull020 = ehull020_df.head(int(top_k)).copy()

    dft_df = out_df[
        (out_df["energy_above_hull"] <= float(dft_ehull_max))
        & (out_df["band_gap"] >= float(dft_bg_min))
        & (out_df["band_gap"] <= float(dft_bg_max))
    ].copy()
    dft_df = dft_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    dft_df["filtered_rank"] = np.arange(1, len(dft_df) + 1)

    if len(dft_df) < int(top_k):
        print(
            f"DFT shortlist has only {len(dft_df)} candidates with "
            f"ehull <= {dft_ehull_max} and {dft_bg_min} <= band_gap <= {dft_bg_max}."
        )
        print("Fallback: using ehull <= 0.10 without band-gap window.")
        dft_df = ehull010_df.copy()

    top_dft = dft_df.head(int(top_k)).copy()

    top_all.to_csv(out_top_all, index=False, encoding="utf-8-sig")
    top_stable.to_csv(out_top_stable, index=False, encoding="utf-8-sig")
    top_ehull010.to_csv(out_top_ehull010, index=False, encoding="utf-8-sig")
    top_ehull020.to_csv(out_top_ehull020, index=False, encoding="utf-8-sig")
    top_dft.to_csv(out_top_dft, index=False, encoding="utf-8-sig")

    return {
        "top_all": top_all,
        "top_stable": top_stable,
        "top_ehull010": top_ehull010,
        "top_ehull020": top_ehull020,
        "top_dft": top_dft,
        "stable_df": stable_df,
        "ehull010_df": ehull010_df,
        "ehull020_df": ehull020_df,
        "dft_df": dft_df,
        "top_all_csv": str(out_top_all),
        "top_stable_csv": str(out_top_stable),
        "top_ehull010_csv": str(out_top_ehull010),
        "top_ehull020_csv": str(out_top_ehull020),
        "top_dft_csv": str(out_top_dft),
    }


def screen_zintl_like_srp_power_200nm(
    root,
    top_k=50,
    thickness_nm=200.0,
    band_gap_min=0.001,
    strict_main_group_only=True,
    num_elements_min=2,
    num_elements_max=5,
    dft_ehull_max=0.10,
    dft_bg_min=0.30,
    dft_bg_max=4.50,
    seeds=(42, 123, 2025),
    batch_size=32,
    preferred_summary=None,
    use_cache=True,
    show=True,
):
    """
    Screen a Zintl-like candidate pool using SSL-init UE-GINE.

    This function:
    1) queries Materials Project for band-gap materials,
    2) keeps Zintl-like formulas using chemical rules,
    3) predicts epsR_0 and epsI_0 using 100% SSL-init checkpoints,
    4) derives n, k, alpha, R,
    5) ranks by SRP-power at 200 nm,
    6) exports top-k all/stable/ehull-filtered/DFT-validation lists.
    """
    root = Path(root)

    output_dir = root / "paper_outputs"
    screen_dir = output_dir / "screening_srp_200nm"
    pool_dir = output_dir / "external_pools"
    screen_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    thickness_tag = f"{int(round(float(thickness_nm)))}nm"
    pool_csv = pool_dir / "pool_zintl_like_chemical_rules_bandgap_gt0.csv"
    pool_pkl = pool_dir / "pool_zintl_like_chemical_rules_bandgap_gt0_with_structures.pkl"
    out_all = screen_dir / f"screening_zintl_like_srp_power_{thickness_tag}_all.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    energy_ev = np.linspace(0.0, 20.0, 2001).astype(np.float32)

    raw_summary = _tb3_screen_load_training_summary(root=root, preferred_summary=preferred_summary)
    eps1_ckpts = _tb3_screen_get_ssl_100pct_ckpt_paths(raw_summary, "epsR_0", seeds=seeds)
    eps2_ckpts = _tb3_screen_get_ssl_100pct_ckpt_paths(raw_summary, "epsI_0", seeds=seeds)

    pool_df = _tb3_screen_build_zintl_like_pool(
        pool_csv=pool_csv,
        pool_pkl=pool_pkl,
        band_gap_min=band_gap_min,
        strict_main_group_only=strict_main_group_only,
        num_elements_min=num_elements_min,
        num_elements_max=num_elements_max,
        use_cache=use_cache,
    )

    print("\n" + "=" * 100)
    print("SCREENING POOL: Zintl-like chemical-rule pool")
    print("=" * 100)
    print("Total candidates:", len(pool_df))
    print("Stable candidates:", int(pool_df["is_stable"].astype(str).str.lower().eq("true").sum()))
    print("Ehull <= 0.10:", int((pool_df["energy_above_hull"] <= 0.10).sum()))
    print("Ehull <= 0.20:", int((pool_df["energy_above_hull"] <= 0.20).sum()))

    eps1_graphs, meta_df = _tb3_screen_make_graph_dataset(pool_df, energy_ev, target_key="epsR_0")
    eps2_graphs, meta_df2 = _tb3_screen_make_graph_dataset(pool_df, energy_ev, target_key="epsI_0")

    if len(eps1_graphs) == 0:
        raise RuntimeError("No valid structures in Zintl-like pool.")

    if len(eps1_graphs) != len(eps2_graphs):
        raise RuntimeError("epsR_0 and epsI_0 graph counts do not match.")

    print("Valid structures:", len(eps1_graphs))

    eps1_pred = _tb3_screen_predict_ensemble(eps1_graphs, eps1_ckpts, device=device, batch_size=batch_size)
    eps2_pred = _tb3_screen_predict_ensemble(eps2_graphs, eps2_ckpts, device=device, batch_size=batch_size)

    n_pred, k_pred, alpha_pred, R_pred = _tb3_screen_dielectric_to_optical(eps1_pred, eps2_pred, energy_ev)
    srp_power = _tb3_screen_compute_srp_power(alpha_pred, R_pred, energy_ev, thickness_nm=thickness_nm)

    out_df = _tb3_merge_screening_meta_with_pool(meta_df, pool_df)
    out_df["predicted_SRP_power_200nm_percent"] = srp_power
    out_df["predicted_eps1_mean"] = np.nanmean(eps1_pred, axis=1)
    out_df["predicted_eps2_mean"] = np.nanmean(eps2_pred, axis=1)
    out_df["predicted_alpha_mean_cm1"] = np.nanmean(alpha_pred, axis=1)
    out_df["predicted_R_mean"] = np.nanmean(R_pred, axis=1)
    out_df["screening_score"] = out_df["predicted_SRP_power_200nm_percent"]
    out_df = out_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    out_df["rank"] = np.arange(1, len(out_df) + 1)
    out_df.to_csv(out_all, index=False, encoding="utf-8-sig")

    filtered = _tb3_save_filtered_screening_tables(
        out_df=out_df,
        out_dir=screen_dir,
        prefix=f"zintl_like_srp_power_{thickness_tag}",
        top_k=top_k,
        dft_ehull_max=dft_ehull_max,
        dft_bg_min=dft_bg_min,
        dft_bg_max=dft_bg_max,
    )

    print("\nSaved all:", out_all)
    print("Saved top-k DFT validation:", filtered["top_dft_csv"])
    print("\nScreening summary:")
    print("Total Zintl-like candidates:", len(out_df))
    print("Stable-only candidates:", len(filtered["stable_df"]))
    print("Ehull <= 0.10 candidates:", len(filtered["ehull010_df"]))
    print("Ehull <= 0.20 candidates:", len(filtered["ehull020_df"]))
    print("DFT shortlist candidates:", len(filtered["dft_df"]))

    if show:
        display_cols = [
            "filtered_rank", "rank", "material_id", "formula_pretty",
            "band_gap", "energy_above_hull", "is_stable",
            "donor_elements", "framework_elements", "zintl_like_rule_score",
            "predicted_SRP_power_200nm_percent",
        ]
        try:
            from IPython.display import display
            display(filtered["top_dft"][[c for c in display_cols if c in filtered["top_dft"].columns]].head(20))
        except Exception:
            print(filtered["top_dft"].head(20))

    return {
        "pool_df": pool_df,
        "all_df": out_df,
        "top_df": filtered["top_dft"],
        "top_dft": filtered["top_dft"],
        "top_stable": filtered["top_stable"],
        "top_ehull010": filtered["top_ehull010"],
        "top_ehull020": filtered["top_ehull020"],
        "pool_csv": str(pool_csv),
        "pool_pkl": str(pool_pkl),
        "all_csv": str(out_all),
        "top_csv": filtered["top_dft_csv"],
        "filtered_outputs": filtered,
        "energy_ev": energy_ev,
        "eps1_pred": eps1_pred,
        "eps2_pred": eps2_pred,
        "n_pred": n_pred,
        "k_pred": k_pred,
        "alpha_pred_cm1": alpha_pred,
        "R_pred": R_pred,
    }

# ============================================================
# Paper-style model configuration table utility
# Added for Q1_PAPER3 benchmark figures
# ============================================================

def plot_model_config_table(out_dir):
    """
    Plot and save a paper-style table summarizing graph-model configurations.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Output directory for PNG and PDF files.

    Returns
    -------
    dict
        Paths to saved PNG/PDF files and the underlying DataFrame.
    """
    import textwrap
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "Model": [
            "GCN",
            "GraphSAGE",
            "GINE",
            "UE-GINE",
            "SSL-init UE-GINE",
        ],
        "Graph features": [
            "Basic",
            "Basic",
            "Basic",
            "Enhanced",
            "Enhanced",
        ],
        "Node/edge": [
            "11 / 16",
            "11 / 16",
            "11 / 16",
            "19 / 24",
            "19 / 24",
        ],
        "Operator": [
            "GCNConv",
            "SAGEConv",
            "GINEConv",
            "GINEConv",
            "GINEConv",
        ],
        "Initialization": [
            "Scratch",
            "Scratch",
            "Scratch",
            "Scratch",
            "SSL-init",
        ],
    })

    fig, ax = plt.subplots(figsize=(13.5, 3.6), dpi=300)
    ax.axis("off")

    fig.text(
        0.02,
        0.94,
        "Table 1.",
        fontsize=11,
        fontweight="bold",
        family="serif",
    )
    fig.text(
        0.105,
        0.94,
        "Summary of graph-model configurations used in this work",
        fontsize=11,
        family="serif",
    )

    bbox = [0.02, 0.26, 0.96, 0.56]

    tab = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
        colWidths=[0.23, 0.20, 0.14, 0.20, 0.20],
    )

    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    tab.scale(1, 1.35)

    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.PAD = 0.08

        if r == 0:
            cell.set_text_props(weight="bold", family="serif", fontsize=10.5)
        else:
            cell.set_text_props(family="serif", fontsize=10.5)

    for r in range(1, len(df) + 1):
        tab[(r, 0)].set_text_props(weight="bold", family="serif", fontsize=10.5)

    x0, y0, w, h = bbox
    row_h = h / (len(df) + 1)
    rule_color = "#1f77b4"

    for y, lw in [
        (y0 + h + 0.012, 1.4),
        (y0 + h - row_h - 0.010, 0.9),
        (y0 - 0.012, 1.4),
    ]:
        ax.add_line(
            plt.Line2D(
                [x0, x0 + w],
                [y, y],
                lw=lw,
                color=rule_color,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    note = (
        "Note: Basic graphs contain 11 node features and 16 edge features. "
        "Enhanced graphs contain 19 node features and 24 edge features. "
        "SSL-init UE-GINE initializes the GINE backbone using self-supervised pretraining."
    )

    fig.text(
        0.02,
        0.13,
        "\n".join(textwrap.wrap(note, 145)),
        fontsize=9.5,
        family="serif",
        va="top",
    )

    png = out_dir / "table_model_configurations_q1_style.png"
    pdf = out_dir / "table_model_configurations_q1_style.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.show()

    return {"png": png, "pdf": pdf, "data": df}


# ============================================================
# Low-data SSL-init vs scratch experiment utility
# Added for Q1_PAPER3 low-data ablation
# ============================================================

def run_lowdata_ssl_vs_scratch(
    targets,
    root,
    ssl_ckpt,
    label_fraction=0.05,
    fraction_tag=None,
    split_seed=2025,
    seeds=(42, 123, 2025),
    model_class=None,
    train_kwargs=None,
    metric_cols=None,
    save_outputs=True,
    show=True,
):
    """
    Run low-data UE-GINE scratch vs SSL-init UE-GINE experiments.

    Parameters
    ----------
    targets : dict
        Mapping from target key to (train_ds, val_ds, test_ds), for example:
        {
            "epsI_0": (epsI_enhanced_train_ds, epsI_enhanced_val_ds, epsI_enhanced_test_ds),
            "epsR_0": (epsR_enhanced_train_ds, epsR_enhanced_val_ds, epsR_enhanced_test_ds),
        }
    root : str or pathlib.Path
        Output root, usually D:/TB3/processed/paired_training.
    ssl_ckpt : str or pathlib.Path
        SSL-init backbone checkpoint for enhanced 19/24 graphs.
    label_fraction : float
        Fraction of labeled training data to keep.
    fraction_tag : str or None
        Tag used in output names. If None, generated from label_fraction.
    split_seed : int
        Random seed for choosing the low-data subset.
    seeds : tuple/list
        Training seeds for each scratch/SSL-init run.
    model_class : class or None
        Model class. Defaults to OpticalResponseGINE.
    train_kwargs : dict or None
        Extra keyword arguments for train_optical_gine_single_target.
    metric_cols : list or None
        Metrics to aggregate. Defaults to common validation/test metrics.
    save_outputs : bool
        Save summary/meanstd/gain CSV files.
    show : bool
        Display tables in notebook if IPython is available.

    Returns
    -------
    dict
        summary, meanstd, gain DataFrames and output paths.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Subset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ssl_ckpt = Path(ssl_ckpt)
    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_ckpt}")

    if model_class is None:
        model_class = OpticalResponseGINE

    if fraction_tag is None:
        fraction_tag = f"{int(round(float(label_fraction) * 100))}pct_enhanced19_24"

    if train_kwargs is None:
        train_kwargs = {}

    device = train_kwargs.pop("device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    default_train_kwargs = dict(
        model_class=model_class,
        output_dir=root,
        device=device,
        seeds=list(seeds),
        batch_size=16,
        num_epochs=100,
        early_stop_patience=15,
        lr=5e-4,
        weight_decay=5e-5,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.05,
        peak_weight=2.0,
        smoothness_weight=0.03,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        use_amp=bool(torch.cuda.is_available()),
    )
    default_train_kwargs.update(train_kwargs)

    if metric_cols is None:
        metric_cols = [
            "test_MAE_mean",
            "test_RMSE_mean",
            "test_CosineSim_mean",
            "test_PeakMAE_mean",
            "test_PeakBinShift_mean",
            "test_PeakAxisError_mean",
            "val_MAE_mean",
            "val_RMSE_mean",
            "val_CosineSim_mean",
        ]

    def _check_enhanced(ds, name):
        g = ds[0]
        node_dim = int(g.x.shape[1])
        edge_dim = int(g.edge_attr.shape[1]) if getattr(g, "edge_attr", None) is not None else None

        print(f"{name}: n={len(ds)} | node={node_dim} | edge={edge_dim}")

        if node_dim != 19 or edge_dim != 24:
            raise RuntimeError(
                f"{name} is not enhanced 19/24. Found node={node_dim}, edge={edge_dim}."
            )

    def _make_subset(ds, frac, seed):
        if float(frac) >= 1.0:
            idx = np.arange(len(ds), dtype=np.int64)
            return ds, idx

        n_keep = max(1, int(round(len(ds) * float(frac))))
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(len(ds), size=n_keep, replace=False)).astype(np.int64)
        return Subset(ds, idx.tolist()), idx

    print("=" * 100)
    print("LOW-DATA UE-GINE vs SSL-init UE-GINE")
    print("=" * 100)
    print("root           :", root)
    print("ssl_ckpt       :", ssl_ckpt)
    print("label_fraction :", label_fraction)
    print("fraction_tag   :", fraction_tag)
    print("split_seed     :", split_seed)
    print("seeds          :", list(seeds))
    print("device         :", device)

    rows = []

    for target_key, datasets in targets.items():
        if len(datasets) != 3:
            raise ValueError(f"targets[{target_key!r}] must be (train_ds, val_ds, test_ds).")

        train_full, val_ds, test_ds = datasets

        print("\n" + "=" * 100)
        print("Target:", target_key)
        print("=" * 100)

        _check_enhanced(train_full, f"{target_key} train")
        _check_enhanced(val_ds, f"{target_key} val")
        _check_enhanced(test_ds, f"{target_key} test")

        train_low, selected_idx = _make_subset(train_full, label_fraction, split_seed)
        print(f"Low-data subset: {len(train_low)} / {len(train_full)} = {100 * len(train_low) / len(train_full):.2f}%")

        idx_path = root / f"lowdata_{fraction_tag}_{target_key}_indices_seed{split_seed}.npy"
        if save_outputs:
            np.save(idx_path, selected_idx)
            print("Saved low-data indices:", idx_path)

        for init_type, ckpt in [("scratch", None), ("ssl_init", ssl_ckpt)]:
            variant = f"lowdata_{fraction_tag}_{init_type}"
            run_name = f"{target_key}_{variant}"

            print("\n" + "-" * 100)
            print(f"RUN | target={target_key} | init={init_type} | variant={variant}")
            print("-" * 100)

            df, artifacts = train_optical_gine_single_target(
                target_key=target_key,
                variant_name=variant,
                train_ds=train_low,
                val_ds=val_ds,
                test_ds=test_ds,
                pretrained_ckpt_path=ckpt,
                run_name=run_name,
                **default_train_kwargs,
            )

            df["label_fraction"] = float(label_fraction)
            df["label_fraction_tag"] = fraction_tag
            df["init_type"] = init_type
            df["lowdata_split_seed"] = int(split_seed)
            df["n_train_lowdata"] = int(len(train_low))
            df["n_train_full"] = int(len(train_full))
            df["node_dim"] = 19
            df["edge_dim"] = 24
            df["lowdata_indices_path"] = str(idx_path)
            df["run_output_dir"] = artifacts.get("run_output_dir", "") if isinstance(artifacts, dict) else ""

            rows.append(df)

    summary = pd.concat(rows, ignore_index=True)

    out_summary = root / f"lowdata_{fraction_tag}_ue_gine_vs_ssl_init_summary.csv"
    out_meanstd = root / f"lowdata_{fraction_tag}_ue_gine_vs_ssl_init_meanstd.csv"
    out_gain = root / f"lowdata_{fraction_tag}_ssl_init_gain.csv"

    available_metric_cols = [c for c in metric_cols if c in summary.columns]

    mean_aggs = {f"{c}_avg": (c, "mean") for c in available_metric_cols}
    std_aggs = {f"{c}_std": (c, lambda s: float(s.std(ddof=0))) for c in available_metric_cols}

    meanstd = (
        summary
        .groupby(["target_key", "init_type", "variant", "label_fraction_tag"], dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            label_fraction=("label_fraction", "first"),
            n_train_lowdata=("n_train_lowdata", "first"),
            n_train_full=("n_train_full", "first"),
            node_dim=("node_dim", "first"),
            edge_dim=("edge_dim", "first"),
            **mean_aggs,
            **std_aggs,
        )
        .reset_index()
    )

    gain_rows = []
    for target_key in targets.keys():
        sub = meanstd[meanstd["target_key"] == target_key].copy()
        scratch = sub[sub["init_type"] == "scratch"]
        ssl = sub[sub["init_type"] == "ssl_init"]

        if len(scratch) == 1 and len(ssl) == 1 and "test_MAE_mean_avg" in sub.columns:
            scratch = scratch.iloc[0]
            ssl = ssl.iloc[0]

            scratch_mae = float(scratch["test_MAE_mean_avg"])
            ssl_mae = float(ssl["test_MAE_mean_avg"])
            scratch_rmse = float(scratch.get("test_RMSE_mean_avg", np.nan))
            ssl_rmse = float(ssl.get("test_RMSE_mean_avg", np.nan))

            gain_rows.append({
                "target_key": target_key,
                "label_fraction": float(label_fraction),
                "label_fraction_tag": fraction_tag,
                "n_train_lowdata": int(scratch["n_train_lowdata"]),
                "n_train_full": int(scratch["n_train_full"]),
                "scratch_test_MAE": scratch_mae,
                "ssl_init_test_MAE": ssl_mae,
                "MAE_reduction_abs": scratch_mae - ssl_mae,
                "MAE_reduction_percent": 100.0 * (scratch_mae - ssl_mae) / scratch_mae if scratch_mae > 0 else np.nan,
                "scratch_test_RMSE": scratch_rmse,
                "ssl_init_test_RMSE": ssl_rmse,
                "RMSE_reduction_abs": scratch_rmse - ssl_rmse,
            })

    gain = pd.DataFrame(gain_rows)

    if save_outputs:
        summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
        meanstd.to_csv(out_meanstd, index=False, encoding="utf-8-sig")
        gain.to_csv(out_gain, index=False, encoding="utf-8-sig")

        print("\nSaved:")
        print(out_summary)
        print(out_meanstd)
        print(out_gain)

    if show:
        try:
            from IPython.display import display
            display(summary)
            display(meanstd)
            display(gain)
        except Exception:
            print(summary)
            print(meanstd)
            print(gain)

    print("\nIMPORTANT CHECK:")
    print("The training log must show node_in_dim = 19 and edge_in_dim = 24.")
    print("If it shows 11/16, stop and rebuild enhanced datasets.")

    return {
        "summary": summary,
        "meanstd": meanstd,
        "gain": gain,
        "summary_csv": str(out_summary),
        "meanstd_csv": str(out_meanstd),
        "gain_csv": str(out_gain),
    }


# ============================================================
# Eta / SRP test-output utility
# Added for Q1_PAPER3 dielectric-first screening validation
# ============================================================

def compute_eta_srp_test_outputs(
    root,
    thickness_nm=200.0,
    derived_npz=None,
    output_subdir="paper_outputs/eta_srp_ranking",
    show=False,
):
    """
    Compute SRP-power and SRP-photon scores on the held-out test set.

    This utility uses the already saved dielectric-first derived optical
    properties from SSL-init UE-GINE:

        paper_outputs/derived_optical_properties/
        derived_optical_test_ensemble_ssl_init_ue_gine_100pct.npz

    Required arrays in the NPZ file:
    - energy_eV
    - alpha_true_cm1, alpha_pred_cm1
    - R_true, R_pred
    - sample_ids, optional

    Parameters
    ----------
    root : str or pathlib.Path
        Root folder, usually D:/TB3/processed/paired_training.
    thickness_nm : float
        Film thickness used in the absorption model.
    derived_npz : str or pathlib.Path or None
        Optional custom path to the derived optical-property NPZ file.
    output_subdir : str
        Output subdirectory under root for CSV files.
    show : bool
        Display the first rows in notebook.

    Returns
    -------
    dict
        score_df and output CSV path.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    root = Path(root)

    if derived_npz is None:
        derived_npz = (
            root
            / "paper_outputs"
            / "derived_optical_properties"
            / "derived_optical_test_ensemble_ssl_init_ue_gine_100pct.npz"
        )
    else:
        derived_npz = Path(derived_npz)

    if not derived_npz.exists():
        raise FileNotFoundError(f"Cannot find derived optical NPZ: {derived_npz}")

    z = np.load(derived_npz, allow_pickle=True)

    required = ["energy_eV", "alpha_true_cm1", "alpha_pred_cm1", "R_true", "R_pred"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(f"Derived NPZ is missing required arrays: {missing}")

    energy_eV = z["energy_eV"].astype(float)
    alpha_true = z["alpha_true_cm1"].astype(float)
    alpha_pred = z["alpha_pred_cm1"].astype(float)
    R_true = z["R_true"].astype(float)
    R_pred = z["R_pred"].astype(float)

    if "sample_ids" in z.files:
        sample_ids = z["sample_ids"].astype(str)
    elif "base_idx" in z.files:
        sample_ids = z["base_idx"].astype(str)
    else:
        sample_ids = np.arange(alpha_true.shape[0]).astype(str)

    def _solar_proxy(E, pin_w_m2=1000.0, min_energy_ev=0.05, max_solar_ev=4.5):
        kb_ev_k = 8.617333262145e-5
        t_sun = 5778.0

        E = np.asarray(E, dtype=float)
        shape = np.zeros_like(E, dtype=float)
        valid = E > float(min_energy_ev)

        x = E[valid] / (kb_ev_k * t_sun)
        shape[valid] = E[valid] ** 3 / np.expm1(x)
        shape[E > float(max_solar_ev)] = 0.0

        area = np.trapz(shape, E)
        if not np.isfinite(area) or area <= 0:
            raise RuntimeError("Solar proxy normalization failed.")

        return shape * (float(pin_w_m2) / area)

    def _srp_score(alpha_cm1, R, photon=False):
        I = _solar_proxy(energy_eV)
        d_cm = float(thickness_nm) * 1e-7

        alpha = np.clip(
            np.nan_to_num(alpha_cm1, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            None,
        )
        R = np.clip(
            np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        )

        A = (1.0 - R) * (1.0 - np.exp(-alpha * d_cm))
        A = np.clip(A, 0.0, 1.0)

        if photon:
            weight = np.zeros_like(energy_eV, dtype=float)
            valid = energy_eV > 0.05
            weight[valid] = I[valid] / energy_eV[valid]
        else:
            weight = I

        absorbed = np.trapz(A * weight.reshape(1, -1), energy_eV, axis=1)
        total = np.trapz(weight, energy_eV)

        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("SRP normalization failed.")

        return 100.0 * absorbed / total

    score_df = pd.DataFrame({
        "sample_id": sample_ids,
        "reference_SRP_abs_power_percent": _srp_score(alpha_true, R_true, photon=False),
        "predicted_SRP_abs_power_percent": _srp_score(alpha_pred, R_pred, photon=False),
        "reference_SRP_abs_photon_percent": _srp_score(alpha_true, R_true, photon=True),
        "predicted_SRP_abs_photon_percent": _srp_score(alpha_pred, R_pred, photon=True),
        "thickness_nm": float(thickness_nm),
    })

    out_dir = root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"eta_srp_test_outputs_{int(round(float(thickness_nm)))}nm.csv"
    score_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if show:
        try:
            from IPython.display import display
            display(score_df.head())
        except Exception:
            print(score_df.head())

    return {
        "score_df": score_df,
        "csv": str(out_csv),
        "derived_npz": str(derived_npz),
        "thickness_nm": float(thickness_nm),
    }
# ============================================================
# Gated contribution weight export / plotting utility
# ============================================================

def export_gated_contribution_weights(
    epsI_df,
    epsR_df,
    epsI_ds,
    epsR_ds,
    model_class,
    root,
    device,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.05,
    out_subdir="gated_contribution_weights",
    show=True,
):
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    root = Path(root)
    out_dir = root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "node_basic": "Basic\natomic",
        "node_electronegativity": "Electro-\nnegativity",
        "node_radius": "Atomic /\nionic radius",
        "node_ionization_valence": "Ionization /\nvalence",
        "node_chemical_flags": "Chemical\nflags",
        "edge_basic_distance_rbf": "Distance\nRBF",
        "edge_pair_chemistry": "Pair\nchemistry",
        "edge_relational_flags": "Relational\nflags",
    }

    target_label = {
        "epsI_0": r"$\varepsilon_2(E)$ / epsI$_0$",
        "epsR_0": r"$\varepsilon_1(E)$ / epsR$_0$",
    }

    node_order = [
        "node_basic",
        "node_electronegativity",
        "node_radius",
        "node_ionization_valence",
        "node_chemical_flags",
    ]

    edge_order = [
        "edge_basic_distance_rbf",
        "edge_pair_chemistry",
        "edge_relational_flags",
    ]

    def _build_model(ds):
        g0 = ds[0]
        model = model_class(
            node_in_dim=g0.x.shape[1],
            edge_in_dim=g0.edge_attr.shape[1],
            out_dim=g0.y.view(-1).shape[0],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        return model

    def _load_one(ckpt_path, ds, target_name, seed):
        model = _build_model(ds)

        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt

        model.load_state_dict(state, strict=False)
        model.eval()

        node_w, edge_w = model.get_contribution_weights()

        node_df = pd.DataFrame({
            "target": target_name,
            "seed": int(seed),
            "type": "node",
            "group": list(node_w.keys()),
            "weight": list(node_w.values()),
        })

        edge_df = pd.DataFrame({
            "target": target_name,
            "seed": int(seed),
            "type": "edge",
            "group": list(edge_w.keys()),
            "weight": list(edge_w.values()),
        })

        return pd.concat([node_df, edge_df], ignore_index=True)

    rows = []

    for _, r in epsI_df.iterrows():
        rows.append(_load_one(r["ckpt_path"], epsI_ds, "epsI_0", r["seed"]))

    for _, r in epsR_df.iterrows():
        rows.append(_load_one(r["ckpt_path"], epsR_ds, "epsR_0", r["seed"]))

    gate_df = pd.concat(rows, ignore_index=True)

    gate_mean_df = (
        gate_df
        .groupby(["target", "type", "group"], as_index=False)
        .agg(
            weight_mean=("weight", "mean"),
            weight_std=("weight", "std"),
            n=("weight", "count"),
        )
    )

    csv_all = out_dir / "gated_contribution_weights_all_seeds.csv"
    csv_mean = out_dir / "gated_contribution_weights_mean.csv"

    gate_df.to_csv(csv_all, index=False)
    gate_mean_df.to_csv(csv_mean, index=False)

    def _plot(feature_type, order, baseline, title, save_name):
        d = gate_mean_df[gate_mean_df["type"] == feature_type].copy()

        targets = ["epsI_0", "epsR_0"]
        x = np.arange(len(order))
        width = 0.36

        plt.figure(figsize=(7.2, 4.4))

        for i, target in enumerate(targets):
            sub = (
                d[d["target"] == target]
                .set_index("group")
                .loc[order]
                .reset_index()
            )

            xpos = x + (i - 0.5) * width

            plt.bar(
                xpos,
                sub["weight_mean"],
                width=width,
                yerr=sub["weight_std"].fillna(0),
                capsize=3,
                label=target_label[target],
            )

        plt.xticks(x, [label_map[g] for g in order])
        plt.ylabel("Learned contribution weight")
        plt.title(title)

        y_max = max(
            0.42,
            float(d["weight_mean"].max() + d["weight_std"].fillna(0).max() + 0.05),
        )
        plt.ylim(0, y_max)

        plt.legend(frameon=False, fontsize=9)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        png_path = out_dir / f"{save_name}.png"
        svg_path = out_dir / f"{save_name}.svg"

        plt.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return png_path, svg_path

    node_png, node_svg = _plot(
        feature_type="node",
        order=node_order,
        baseline=1 / 5,
        title="Node descriptor contribution",
        save_name="figure_gated_node_descriptor_contribution",
    )

    edge_png, edge_svg = _plot(
        feature_type="edge",
        order=edge_order,
        baseline=1 / 3,
        title="Edge descriptor contribution",
        save_name="figure_gated_edge_descriptor_contribution",
    )

    print("=" * 100)
    print("GATED CONTRIBUTION WEIGHTS EXPORTED")
    print("=" * 100)
    print("CSV all :", csv_all)
    print("CSV mean:", csv_mean)
    print("Node fig:", node_png)
    print("Edge fig:", edge_png)

    return {
        "all": gate_df,
        "mean": gate_mean_df,
        "csv_all": csv_all,
        "csv_mean": csv_mean,
        "node_png": node_png,
        "node_svg": node_svg,
        "edge_png": edge_png,
        "edge_svg": edge_svg,
    }
# ============================================================
# Final benchmark table utilities for TASK3
# ============================================================

def build_task3_final_benchmark_tables(
    root=r"D:\TB3\processed\paired_training",
    out_subdir="final_benchmark_tables",
    r2_filename="weighted_r2_all_main_models_average.csv",
    display_tables=True,
):
    """
    Build Q1-style benchmark tables for TASK3, including SSL-init Gated UE-GINE.

    Returns
    -------
    dict with raw/display tables and saved CSV paths.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    try:
        from IPython.display import display
    except Exception:
        display = None

    ROOT = Path(root)
    OUT = ROOT / out_subdir
    OUT.mkdir(parents=True, exist_ok=True)

    r2_path = OUT / r2_filename
    if r2_path.exists():
        r2 = pd.read_csv(r2_path, header=[0, 1], index_col=0)
        r2.columns = [f"{a}_{b}" for a, b in r2.columns]
    else:
        print("Missing R2 table:", r2_path)
        r2 = pd.DataFrame()

    models = [
        # epsI_0
        ("epsI_0 | GCN", "ε₂(E) [epsI₀]", "GCN", "Basic", "Scratch", "epsI_0_gcn"),
        ("epsI_0 | GraphSAGE", "ε₂(E) [epsI₀]", "GraphSAGE", "Basic", "Scratch", "epsI_0_graphsage"),
        ("epsI_0 | GINE scratch 11/16", "ε₂(E) [epsI₀]", "GINE scratch", "Basic", "Scratch", "epsI_0_baseline"),
        ("epsI_0 | GINE enhanced 19/24", "ε₂(E) [epsI₀]", "UE-GINE", "Enhanced", "Scratch", "epsI_0_enhanced"),
        ("epsI_0 | SSL-init baseline 11/16", "ε₂(E) [epsI₀]", "SSL-init GINE", "Basic", "SSL-pretrained", "epsI_0_ssl_init_baseline"),
        ("epsI_0 | Optimized SSL-init enhanced 19/24", "ε₂(E) [epsI₀]", "SSL-init UE-GINE", "Enhanced", "SSL-pretrained", "epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"),
        ("epsI_0 | SSL-init Gated UE-GINE", "ε₂(E) [epsI₀]", "SSL-init Gated UE-GINE", "Enhanced + Gated", "SSL-pretrained", "epsI_0_SSL_init_gated_UE_GINE_epsI_0_3seed"),

        # epsR_0
        ("epsR_0 | GCN", "ε₁(E) [epsR₀]", "GCN", "Basic", "Scratch", "epsR_0_gcn"),
        ("epsR_0 | GraphSAGE", "ε₁(E) [epsR₀]", "GraphSAGE", "Basic", "Scratch", "epsR_0_graphsage"),
        ("epsR_0 | GINE scratch 11/16", "ε₁(E) [epsR₀]", "GINE scratch", "Basic", "Scratch", "epsR_0_baseline"),
        ("epsR_0 | GINE enhanced 19/24", "ε₁(E) [epsR₀]", "UE-GINE", "Enhanced", "Scratch", "epsR_0_enhanced"),
        ("epsR_0 | SSL-init baseline 11/16", "ε₁(E) [epsR₀]", "SSL-init GINE", "Basic", "SSL-pretrained", "epsR_0_ssl_init_baseline"),
        ("epsR_0 | Optimized SSL-init enhanced 19/24", "ε₁(E) [epsR₀]", "SSL-init UE-GINE", "Enhanced", "SSL-pretrained", "epsR_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed"),
        ("epsR_0 | SSL-init Gated UE-GINE", "ε₁(E) [epsR₀]", "SSL-init Gated UE-GINE", "Enhanced + Gated", "SSL-pretrained", "epsR_0_SSL_init_gated_UE_GINE_epsR_0_3seed"),

        # direct-alpha ablation
        ("alpha_log | Direct-alpha GINE scratch", "log-scaled α(E)", "Direct-alpha GINE", "Enhanced", "Scratch", "epsI_alpha_log10_1p_direct_alpha_enhanced_scratch_3seed"),
        ("alpha_log | Direct-alpha SSL-init", "log-scaled α(E)", "Direct-alpha SSL-init GINE", "Enhanced", "SSL-pretrained", "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed"),
    ]

    rows = []
    for key, target, model, graph, init, folder in models:
        files = list((ROOT / folder).glob("summary_optical_*.csv"))
        if not files:
            print("Missing summary:", folder)
            continue

        s = pd.read_csv(files[0])
        mae_values = s["test_MAE_mean"]

        rows.append({
            "R2_key": key,
            "Target": target,
            "Model": model,
            "Graph": graph,
            "Initialization": init,
            "MAE_m": mae_values.mean(),
            "MAE_s": mae_values.std(),
            "WR2_m": r2.loc[key, "weighted_R2_mean"] if key in r2.index else np.nan,
            "WR2_s": r2.loc[key, "weighted_R2_std"] if key in r2.index else np.nan,
            "MR2_m": r2.loc[key, "mean_R2_mean"] if key in r2.index else np.nan,
            "MR2_s": r2.loc[key, "mean_R2_std"] if key in r2.index else np.nan,
        })

    df = pd.DataFrame(rows)

    def _metric(mean, std, digits=3):
        if pd.isna(mean):
            return "N/A"
        if pd.isna(std):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    show = df[["Target", "Model", "Graph", "Initialization"]].copy()
    show["MAE ↓"] = df.apply(lambda r: _metric(r.MAE_m, r.MAE_s), axis=1)
    show["Weighted R² ↑"] = df.apply(lambda r: _metric(r.WR2_m, r.WR2_s), axis=1)
    show["Mean R² ↑"] = df.apply(lambda r: _metric(r.MR2_m, r.MR2_s), axis=1)

    main_targets = ["ε₂(E) [epsI₀]", "ε₁(E) [epsR₀]"]
    main_show = show[show["Target"].isin(main_targets)].reset_index(drop=True)
    main_raw = df[df["Target"].isin(main_targets)].reset_index(drop=True)
    alpha_show = show[show["Target"].eq("log-scaled α(E)")].reset_index(drop=True)
    alpha_raw = df[df["Target"].eq("log-scaled α(E)")].reset_index(drop=True)

    def _style(show_df, raw_df, caption):
        def highlight(row):
            css = [""] * len(row)
            raw = raw_df.loc[row.name]
            sub = raw_df[raw_df["Target"] == raw["Target"]]

            if raw["Model"] in ["UE-GINE", "SSL-init UE-GINE", "SSL-init Gated UE-GINE"]:
                css[1] = "font-weight: bold;"
            if np.isclose(raw["MAE_m"], sub["MAE_m"].min(), equal_nan=False):
                css[4] = "font-weight: bold; background-color: #fff2cc;"
            if not pd.isna(raw["WR2_m"]) and np.isclose(raw["WR2_m"], sub["WR2_m"].max(skipna=True)):
                css[5] = "font-weight: bold; background-color: #d9ead3;"
            if not pd.isna(raw["MR2_m"]) and np.isclose(raw["MR2_m"], sub["MR2_m"].max(skipna=True)):
                css[6] = "font-weight: bold; background-color: #d9ead3;"
            if raw["Model"] == "SSL-init Gated UE-GINE":
                css = [c + " border-top: 2px solid #666;" for c in css]
            return css

        return (
            show_df.style
            .apply(highlight, axis=1)
            .hide(axis="index")
            .set_caption(caption)
            .set_table_styles([
                {"selector": "caption", "props": "caption-side: top; font-weight: bold; font-size: 13pt;"},
                {"selector": "th", "props": "font-weight: bold; text-align: center; border-bottom: 1px solid black;"},
                {"selector": "td", "props": "text-align: center; padding: 5px 8px;"},
                {"selector": "td:nth-child(2)", "props": "text-align: left;"},
            ])
        )

    main_csv = OUT / "main_dielectric_first_benchmark_Q1_style_with_gated.csv"
    alpha_csv = OUT / "direct_alpha_ablation_Q1_style.csv"
    main_show.to_csv(main_csv, index=False, encoding="utf-8-sig")
    alpha_show.to_csv(alpha_csv, index=False, encoding="utf-8-sig")

    main_style = _style(main_show, main_raw, "Performance comparison of graph neural networks for dielectric-spectrum prediction")
    alpha_style = _style(alpha_show, alpha_raw, "Direct-alpha prediction ablation study")

    if display_tables and display is not None:
        display(main_style)
        display(alpha_style)

    missing_r2 = df[df[["WR2_m", "MR2_m"]].isna().any(axis=1)][["Target", "Model", "R2_key"]]
    if len(missing_r2) > 0:
        print("\nModels missing R2 values:")
        if display is not None:
            display(missing_r2)
        else:
            print(missing_r2)

    print("Saved:")
    print(main_csv)
    print(alpha_csv)

    return {
        "raw": df,
        "main_show": main_show,
        "alpha_show": alpha_show,
        "main_style": main_style,
        "alpha_style": alpha_style,
        "missing_r2": missing_r2,
        "main_csv": main_csv,
        "alpha_csv": alpha_csv,
    }


# ============================================================
# TASK3 benchmark R2 utilities
# ============================================================

def parse_task3_seed_from_name(name):
    """Extract seed number from a prediction filename such as test_predictions_seed2025.npz."""
    import re
    m = re.search(r"seed(\d+)", str(name))
    return int(m.group(1)) if m else None


def masked_weighted_r2_task3(y_true, y_pred, mask=None):
    """Compute weighted R2 and mean bin-wise R2 for full-spectrum prediction."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)

    if mask is None:
        mask = np.ones_like(y_true, dtype=np.float64)
    else:
        mask = np.asarray(mask, dtype=np.float64).reshape(len(y_true), -1)

    wsum = mask.sum(axis=0)
    valid_mean = wsum > 0

    mean_true = np.zeros(y_true.shape[1], dtype=np.float64)
    mean_true[valid_mean] = (
        y_true[:, valid_mean] * mask[:, valid_mean]
    ).sum(axis=0) / wsum[valid_mean]

    sse = (((y_true - y_pred) ** 2) * mask).sum(axis=0)
    sst = (((y_true - mean_true[None, :]) ** 2) * mask).sum(axis=0)
    valid = sst > 1e-12

    weighted_r2 = 1.0 - sse[valid].sum() / sst[valid].sum()

    r2_bins = np.full_like(sst, np.nan, dtype=np.float64)
    r2_bins[valid] = 1.0 - sse[valid] / sst[valid]
    mean_r2 = np.nanmean(r2_bins)

    return float(weighted_r2), float(mean_r2)


def get_task3_r2_model_folders(include_gated=True):
    """Folder map used to compute weighted R2 for all main TASK3 benchmark models."""
    folders = {
        # epsI_0
        "epsI_0 | GCN": "epsI_0_gcn",
        "epsI_0 | GraphSAGE": "epsI_0_graphsage",
        "epsI_0 | GINE scratch 11/16": "epsI_0_baseline",
        "epsI_0 | GINE enhanced 19/24": "epsI_0_enhanced",
        "epsI_0 | SSL-init baseline 11/16": "epsI_0_ssl_init_baseline",
        "epsI_0 | Optimized SSL-init enhanced 19/24": "epsI_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed",

        # epsR_0
        "epsR_0 | GCN": "epsR_0_gcn",
        "epsR_0 | GraphSAGE": "epsR_0_graphsage",
        "epsR_0 | GINE scratch 11/16": "epsR_0_baseline",
        "epsR_0 | GINE enhanced 19/24": "epsR_0_enhanced",
        "epsR_0 | SSL-init baseline 11/16": "epsR_0_ssl_init_baseline",
        "epsR_0 | Optimized SSL-init enhanced 19/24": "epsR_0_ssl_init_enhanced_19_24_d005_wd5e5_3seed",

        # direct-alpha
        "alpha_log | Direct-alpha GINE scratch": "epsI_alpha_log10_1p_direct_alpha_enhanced_scratch_3seed",
        "alpha_log | Direct-alpha SSL-init": "epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed",
    }

    if include_gated:
        folders.update({
            "epsI_0 | SSL-init Gated UE-GINE": "epsI_0_SSL_init_gated_UE_GINE_epsI_0_3seed",
            "epsR_0 | SSL-init Gated UE-GINE": "epsR_0_SSL_init_gated_UE_GINE_epsR_0_3seed",
        })

    return folders


def compute_task3_weighted_r2_tables(
    root,
    out_dir=None,
    include_gated=True,
    display_tables=True,
):
    """
    Compute and save weighted R2 tables for TASK3 main benchmark models.

    Outputs
    -------
    weighted_r2_all_main_models_per_seed.csv
    weighted_r2_all_main_models_average.csv
    """
    from pathlib import Path

    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "final_benchmark_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    folders = get_task3_r2_model_folders(include_gated=include_gated)

    for label, folder_name in folders.items():
        folder = root / folder_name

        if not folder.exists():
            print("SKIP missing:", folder_name)
            continue

        pred_files = sorted(folder.glob("test_predictions*.npz"))
        if len(pred_files) == 0:
            print("SKIP no test prediction:", folder_name)
            continue

        for p in pred_files:
            z = np.load(p, allow_pickle=True)

            if not {"pred", "true"}.issubset(set(z.keys())):
                print("SKIP bad keys:", p.name, list(z.keys()))
                continue

            weighted_r2, mean_r2 = masked_weighted_r2_task3(
                y_true=z["true"],
                y_pred=z["pred"],
                mask=z["mask"] if "mask" in z.keys() else None,
            )

            rows.append({
                "label": label,
                "folder": folder_name,
                "seed": parse_task3_seed_from_name(p.name),
                "weighted_R2": weighted_r2,
                "mean_R2": mean_r2,
                "prediction_file": p.name,
            })

    r2_all_df = pd.DataFrame(rows)

    if len(r2_all_df) == 0:
        print("No valid prediction files found.")
        return {"per_seed": r2_all_df, "average": pd.DataFrame()}

    r2_all_df = r2_all_df.sort_values(["label", "seed"]).reset_index(drop=True)
    r2_avg_df = (
        r2_all_df
        .groupby("label")[["weighted_R2", "mean_R2"]]
        .agg(["mean", "std"])
        .round(6)
    )

    out_seed = out_dir / "weighted_r2_all_main_models_per_seed.csv"
    out_avg = out_dir / "weighted_r2_all_main_models_average.csv"

    r2_all_df.to_csv(out_seed, index=False)
    r2_avg_df.to_csv(out_avg)

    print("\n================ Weighted mean R2 | per seed ================")
    if display_tables and "display" in globals():
        display(r2_all_df[["label", "seed", "weighted_R2", "mean_R2"]].round(6))
    else:
        print(r2_all_df[["label", "seed", "weighted_R2", "mean_R2"]].round(6))

    print("\n================ Weighted mean R2 | average ================")
    if display_tables and "display" in globals():
        display(r2_avg_df)
    else:
        print(r2_avg_df)

    print("Saved per-seed:", out_seed)
    print("Saved average :", out_avg)

    return {
        "per_seed": r2_all_df,
        "average": r2_avg_df,
        "per_seed_csv": out_seed,
        "average_csv": out_avg,
    }


#contribution analysis utilities
# Enhanced graph 20/24 with covalent_radius_cordero
# Added by ChatGPT for TASK3 contribution-weight workflow
# ============================================================

# Canonical feature names for the contribution-analysis graph.
NODE_FEATURE_NAMES_19 = [
    "atomic_number",
    "row",
    "group",
    "atomic_mass",
    "electronegativity",
    "atomic_radius",
    "atomic_radius_calculated",
    "average_ionic_radius",
    "ionization_energy",
    "val_s",
    "val_p",
    "val_d",
    "val_f",
    "block_s",
    "block_p",
    "block_d",
    "block_f",
    "is_metal",
    "is_transition_metal",
]

NODE_FEATURE_NAMES_20 = [
    "atomic_number",
    "row",
    "group",
    "atomic_mass",
    "electronegativity",
    "atomic_radius",
    "atomic_radius_calculated",
    "average_ionic_radius",
    "covalent_radius_cordero",
    "ionization_energy",
    "val_s",
    "val_p",
    "val_d",
    "val_f",
    "block_s",
    "block_p",
    "block_d",
    "block_f",
    "is_metal",
    "is_transition_metal",
]

EDGE_FEATURE_NAMES_24 = (
    [f"distance_rbf_{i+1}" for i in range(16)]
    + [
        "abs_delta_X",
        "abs_delta_radius",
        "radius_sum",
        "abs_delta_ionization",
        "abs_delta_Z",
        "same_element",
        "same_group",
        "same_row",
    ]
)


def _tb3_safe_float_value(value, default=np.nan):
    """Convert pymatgen FloatWithUnit / normal number / None to float."""
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def get_covalent_radius_cordero(element):
    """
    Robust extraction of a Cordero-like covalent radius from pymatgen Element.

    Different pymatgen versions expose this value under different names.
    The function tries direct attributes, common data keys, then a broad
    fallback search over keys containing both "covalent" and "radius".
    """
    for attr in [
        "covalent_radius_cordero",
        "covalent_radius",
        "covalent_radius_bragg",
    ]:
        value = _tb3_safe_float_value(getattr(element, attr, None))
        if np.isfinite(value) and value > 0:
            return value

    data = getattr(element, "data", {})

    for key in [
        "Covalent radius Cordero",
        "Cordero covalent radius",
        "Covalent radius",
        "CovalentRadius",
        "covalent_radius_cordero",
        "covalent_radius",
    ]:
        value = _tb3_safe_float_value(data.get(key, None))
        if np.isfinite(value) and value > 0:
            return value

    try:
        for key, raw_value in data.items():
            key_lower = str(key).lower()
            if "covalent" in key_lower and "radius" in key_lower:
                value = _tb3_safe_float_value(raw_value)
                if np.isfinite(value) and value > 0:
                    return value
    except Exception:
        pass

    return np.nan


def get_element_features_enhanced_20(symbol):
    """
    Descriptor-enriched structure-only node features with dimension 20.

    Features:
    [Z, row, group, atomic_mass,
     electronegativity,
     atomic_radius, atomic_radius_calculated, average_ionic_radius,
     covalent_radius_cordero,
     ionization_energy,
     val_s, val_p, val_d, val_f,
     block_s, block_p, block_d, block_f,
     is_metal, is_transition_metal]
    """
    element = Element(symbol)

    atomic_number, _ = clip_scale(getattr(element, "Z", np.nan), 100.0)
    row, _ = clip_scale(getattr(element, "row", np.nan), 7.0)
    group, _ = clip_scale(getattr(element, "group", np.nan), 18.0)
    atomic_mass, _ = clip_scale(getattr(element, "atomic_mass", np.nan), 250.0)

    try:
        electronegativity_raw = element.data.get("X", np.nan)
    except Exception:
        electronegativity_raw = np.nan

    electronegativity, _ = clip_scale(electronegativity_raw, 4.0)

    atomic_radius, _ = clip_scale(getattr(element, "atomic_radius", np.nan), 3.5)
    atomic_radius_calculated, _ = clip_scale(getattr(element, "atomic_radius_calculated", np.nan), 3.5)
    average_ionic_radius, _ = clip_scale(getattr(element, "average_ionic_radius", np.nan), 3.5)

    covalent_radius_cordero_raw = get_covalent_radius_cordero(element)
    covalent_radius_cordero, _ = clip_scale(covalent_radius_cordero_raw, 3.5)

    ionization_energy, _ = clip_scale(getattr(element, "ionization_energy", np.nan), 25.0)

    val_s, val_p, val_d, val_f = get_valence_shell_counts(element)
    val_s /= 14.0
    val_p /= 14.0
    val_d /= 14.0
    val_f /= 14.0

    block_s, block_p, block_d, block_f = block_one_hot(element)

    is_metal = 1.0 if getattr(element, "is_metal", False) else 0.0
    is_transition_metal = 1.0 if getattr(element, "is_transition_metal", False) else 0.0

    features = [
        atomic_number,
        row,
        group,
        atomic_mass,
        electronegativity,
        atomic_radius,
        atomic_radius_calculated,
        average_ionic_radius,
        covalent_radius_cordero,
        ionization_energy,
        val_s,
        val_p,
        val_d,
        val_f,
        block_s,
        block_p,
        block_d,
        block_f,
        is_metal,
        is_transition_metal,
    ]

    if len(features) != 20:
        raise RuntimeError(f"Expected 20 node features, got {len(features)}")

    return features


def build_node_tensor_enhanced_20(structure):
    """Build 20-dimensional enhanced node tensor with covalent_radius_cordero."""
    return torch.tensor(
        [get_element_features_enhanced_20(site.specie.symbol) for site in structure.sites],
        dtype=torch.float32,
    )


def patch_enhanced_graph_builder_to_20_24(verbose=True):
    """
    Monkey-patch the enhanced graph builder to use the 20-node-feature version.

    After calling this function, build_epsR_graph_dataset_enhanced() and
    build_epsI_graph_dataset_enhanced() will produce 20/24 graphs.
    """
    global get_element_features_enhanced
    global build_node_tensor_enhanced

    get_element_features_enhanced = get_element_features_enhanced_20
    build_node_tensor_enhanced = build_node_tensor_enhanced_20

    if verbose:
        print("=" * 100)
        print("PATCHED ENHANCED GRAPH BUILDER TO 20/24")
        print("=" * 100)
        print("Node dim expected:", len(NODE_FEATURE_NAMES_20))
        print("Edge dim expected:", len(EDGE_FEATURE_NAMES_24))
        print("Added node feature: covalent_radius_cordero")


def _unwrap_to_crystal_optical_dataset(obj):
    """Return the underlying CrystalOpticalDataset-like object with `.samples`.

    In notebooks, `ds` is sometimes accidentally reassigned to a
    torch.utils.data.Subset. The graph builders require the original
    dataset object because CrystalSpectrumView iterates over `.samples`.
    """
    from torch.utils.data import Subset

    cur = obj
    seen = set()

    for _ in range(10):
        if hasattr(cur, "samples"):
            return cur

        if isinstance(cur, Subset):
            cur = cur.dataset
            continue

        # CrystalGraphSpectrumDatasetEnhanced / CrystalGraphSpectrumDataset
        # stores the original CrystalOpticalDataset in `.base_dataset`.
        base = getattr(cur, "base_dataset", None)
        if base is not None and id(base) not in seen:
            seen.add(id(base))
            cur = base
            continue

        # CrystalSpectrumView also stores the original dataset in `.base_dataset`.
        view = getattr(cur, "spectrum_view_dataset", None)
        if view is not None:
            base = getattr(view, "base_dataset", None)
            if base is not None:
                cur = base
                continue

        break

    raise AttributeError(
        "Could not find an underlying dataset with `.samples`. "
        "Pass the original CrystalOpticalDataset, or pass existing graph datasets "
        "so the function can recover `.base_dataset`."
    )


def _as_index_list(idx):
    if hasattr(idx, "tolist"):
        return idx.tolist()
    return list(idx)


def build_enhanced_20_24_graph_splits(
    ds,
    optical_graph_epsR_ds,
    optical_graph_epsI_ds,
    train_idx,
    val_idx,
    test_idx,
    verbose=True,
):
    """
    Build epsR_0 / epsI_0 enhanced graph datasets with 20 node and 24 edge features,
    then recreate train/val/test subsets using an existing fixed split.

    This version is robust if `ds` is accidentally a Subset: it recovers the
    original CrystalOpticalDataset before calling CrystalSpectrumView.

    Returns a dictionary that can be pushed to notebook globals:
        globals().update(outputs)
    """
    from torch.utils.data import Subset

    base_ds = _unwrap_to_crystal_optical_dataset(ds)

    if verbose and base_ds is not ds:
        print("Input ds was not the raw CrystalOpticalDataset; recovered base dataset with .samples.")

    patch_enhanced_graph_builder_to_20_24(verbose=verbose)

    optical_graph_epsR_enhanced_ds, epsR_enhanced_spectrum_ds = build_epsR_graph_dataset_enhanced(
        base_ds,
        verbose=verbose,
    )

    optical_graph_epsI_enhanced_ds, epsI_enhanced_spectrum_ds = build_epsI_graph_dataset_enhanced(
        base_ds,
        verbose=verbose,
    )

    if optical_graph_epsR_enhanced_ds.base_indices != optical_graph_epsR_ds.base_indices:
        raise RuntimeError("epsR enhanced 20/24 dataset is not aligned with baseline epsR graph dataset.")

    if optical_graph_epsI_enhanced_ds.base_indices != optical_graph_epsI_ds.base_indices:
        raise RuntimeError("epsI enhanced 20/24 dataset is not aligned with baseline epsI graph dataset.")

    check_paired_graph_alignment(
        optical_graph_epsR_enhanced_ds,
        optical_graph_epsI_enhanced_ds,
    )

    epsR_enhanced_train_ds = Subset(optical_graph_epsR_enhanced_ds, _as_index_list(train_idx))
    epsR_enhanced_val_ds = Subset(optical_graph_epsR_enhanced_ds, _as_index_list(val_idx))
    epsR_enhanced_test_ds = Subset(optical_graph_epsR_enhanced_ds, _as_index_list(test_idx))

    epsI_enhanced_train_ds = Subset(optical_graph_epsI_enhanced_ds, _as_index_list(train_idx))
    epsI_enhanced_val_ds = Subset(optical_graph_epsI_enhanced_ds, _as_index_list(val_idx))
    epsI_enhanced_test_ds = Subset(optical_graph_epsI_enhanced_ds, _as_index_list(test_idx))

    outputs = {
        "optical_graph_epsR_enhanced_ds": optical_graph_epsR_enhanced_ds,
        "optical_graph_epsI_enhanced_ds": optical_graph_epsI_enhanced_ds,
        "epsR_enhanced_spectrum_ds": epsR_enhanced_spectrum_ds,
        "epsI_enhanced_spectrum_ds": epsI_enhanced_spectrum_ds,
        "epsR_enhanced_train_ds": epsR_enhanced_train_ds,
        "epsR_enhanced_val_ds": epsR_enhanced_val_ds,
        "epsR_enhanced_test_ds": epsR_enhanced_test_ds,
        "epsI_enhanced_train_ds": epsI_enhanced_train_ds,
        "epsI_enhanced_val_ds": epsI_enhanced_val_ds,
        "epsI_enhanced_test_ds": epsI_enhanced_test_ds,
    }

    if verbose:
        print("\n" + "=" * 100)
        print("ENHANCED GRAPH 20/24 SPLITS READY")
        print("=" * 100)

    for name in [
        "epsI_enhanced_train_ds",
        "epsI_enhanced_val_ds",
        "epsI_enhanced_test_ds",
        "epsR_enhanced_train_ds",
        "epsR_enhanced_val_ds",
        "epsR_enhanced_test_ds",
    ]:
        g = outputs[name][0]
        node_dim = int(g.x.shape[1])
        edge_dim = int(g.edge_attr.shape[1])

        if verbose:
            print(
                name,
                "| node_dim =", node_dim,
                "| edge_dim =", edge_dim,
                "| y =", tuple(g.y.shape),
            )

        if node_dim != 20 or edge_dim != 24:
            raise RuntimeError(f"{name}: expected 20/24, got {node_dim}/{edge_dim}")

    return outputs


class LinearEnsembleEmbedding(nn.Module):
    """
    GNNOpt-style linear ensemble embedding.

    Each descriptor has its own branch:
        descriptor_i -> Linear_i(1 -> out_dim) -> activation

    Then descriptor embeddings are mixed by learnable probability p_i:
        output = sum_i p_i * embedding_i

    where:
        p_i = softmax(mix_logits_i)
        sum_i p_i = 1
    """

    def __init__(
        self,
        num_features,
        out_dim,
        feature_names=None,
        activation="silu",
        init_logits="zeros",
    ):
        super().__init__()

        self.num_features = int(num_features)
        self.out_dim = int(out_dim)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.num_features)]

        if len(self.feature_names) != self.num_features:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} does not match num_features {self.num_features}"
            )

        activation = str(activation).lower()

        if activation == "silu":
            act_layer = nn.SiLU
        elif activation == "gelu":
            act_layer = nn.GELU
        elif activation == "relu":
            act_layer = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.feature_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.out_dim),
                act_layer(),
            )
            for _ in range(self.num_features)
        ])

        self.mix_logits = nn.Parameter(torch.zeros(self.num_features))

        if init_logits == "small_noise":
            nn.init.normal_(self.mix_logits, mean=0.0, std=1e-3)

    def mixing_weights(self):
        return F.softmax(self.mix_logits, dim=0)

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(f"Expected input shape [N, F], got {tuple(x.shape)}")

        if x.shape[1] != self.num_features:
            raise RuntimeError(
                f"Feature dimension mismatch: got {x.shape[1]}, expected {self.num_features}"
            )

        embedded_list = []

        for i in range(self.num_features):
            xi = x[:, i:i + 1]
            ei = self.feature_branches[i](xi)
            embedded_list.append(ei)

        embedded = torch.stack(embedded_list, dim=1)
        p = self.mixing_weights()

        return (embedded * p.view(1, -1, 1)).sum(dim=1)

    @torch.no_grad()
    def contribution_dataframe(self, feature_type):
        weights = self.mixing_weights().detach().cpu().numpy()

        return (
            pd.DataFrame(
                {
                    "type": feature_type,
                    "feature": self.feature_names,
                    "contribution_weight": weights,
                }
            )
            .sort_values("contribution_weight", ascending=False)
            .reset_index(drop=True)
        )

    @torch.no_grad()
    def contribution_dict(self):
        weights = self.mixing_weights().detach().cpu().numpy()
        return {name: float(w) for name, w in zip(self.feature_names, weights)}


class OpticalResponseGINE_LinearEnsembleUE(nn.Module):
    """
    GNNOpt-style Linear Ensemble-Embedding UE-GINE.

    This model is intended for feature-level contribution analysis.

    Difference from old raw Feature-Gated UE-GINE:
    - Old: x' = x * softmax_gate
    - New: descriptor_i -> Linear_i + activation -> embedding_i,
           then weighted mixture by learnable p_i.

    Contribution weights are the learned p_i.
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        out_dim=2001,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.05,
        graph_attr_dim=0,
        use_graph_attr=False,
        node_feature_names=None,
        edge_feature_names=None,
        activation="silu",
        **kwargs,
    ):
        super().__init__()

        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.graph_attr_dim = int(graph_attr_dim)
        self.use_graph_attr = bool(use_graph_attr and self.graph_attr_dim > 0)

        self.node_feature_names = node_feature_names or (
            NODE_FEATURE_NAMES_20
            if self.node_in_dim == len(NODE_FEATURE_NAMES_20)
            else NODE_FEATURE_NAMES_19
            if self.node_in_dim == len(NODE_FEATURE_NAMES_19)
            else [f"node_feature_{i}" for i in range(self.node_in_dim)]
        )

        self.edge_feature_names = edge_feature_names or (
            EDGE_FEATURE_NAMES_24
            if self.edge_in_dim == len(EDGE_FEATURE_NAMES_24)
            else [f"edge_feature_{i}" for i in range(self.edge_in_dim)]
        )

        if len(self.node_feature_names) != self.node_in_dim:
            raise ValueError("node_feature_names length does not match node_in_dim.")

        if len(self.edge_feature_names) != self.edge_in_dim:
            raise ValueError("edge_feature_names length does not match edge_in_dim.")

        self.node_ensemble = LinearEnsembleEmbedding(
            num_features=self.node_in_dim,
            out_dim=self.hidden_dim,
            feature_names=self.node_feature_names,
            activation=activation,
        )

        # Keep edge_attr dimension equal to edge_in_dim so GINEConv(edge_dim=edge_in_dim) remains compatible.
        self.edge_ensemble = LinearEnsembleEmbedding(
            num_features=self.edge_in_dim,
            out_dim=self.edge_in_dim,
            feature_names=self.edge_feature_names,
            activation=activation,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            conv = GINEConv(
                nn=mlp,
                edge_dim=self.edge_in_dim,
                train_eps=True,
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        head_in_dim = self.hidden_dim

        if self.use_graph_attr:
            head_in_dim += self.graph_attr_dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.out_dim),
        )

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
        graph_attr=None,
        **kwargs,
    ):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = getattr(data, "batch", None)
            graph_attr = getattr(data, "graph_attr", None)

        if x is None or edge_index is None or edge_attr is None:
            raise RuntimeError("OpticalResponseGINE_LinearEnsembleUE requires x, edge_index, and edge_attr.")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_ensemble(x)
        edge_h = self.edge_ensemble(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_h)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res

        g = global_mean_pool(h, batch)

        if self.use_graph_attr and graph_attr is not None:
            graph_attr = graph_attr.view(g.size(0), -1).to(g.device)
            g = torch.cat([g, graph_attr], dim=1)

        return self.head(g)

    @torch.no_grad()
    def get_contribution_weights(self, top_k=None):
        node_df = self.node_ensemble.contribution_dataframe(feature_type="node")
        edge_df = self.edge_ensemble.contribution_dataframe(feature_type="edge")

        if top_k is not None:
            node_df = node_df.head(top_k).reset_index(drop=True)
            edge_df = edge_df.head(top_k).reset_index(drop=True)

        return node_df, edge_df

    @torch.no_grad()
    def get_contribution_dicts(self):
        return {
            "node": self.node_ensemble.contribution_dict(),
            "edge": self.edge_ensemble.contribution_dict(),
        }


@torch.no_grad()
def sanity_check_linear_ensemble_ue_gine(
    train_ds,
    device=None,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.05,
    top_k=10,
):
    """
    Compact sanity check for OpticalResponseGINE_LinearEnsembleUE.

    Returns:
        model, node_df, edge_df
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    g0 = train_ds[0]
    node_dim = int(g0.x.shape[1])
    edge_dim = int(g0.edge_attr.shape[1])
    out_dim = int(g0.y.view(-1).shape[0])

    if node_dim == 20:
        graph_tag = "20/24"
        node_feature_names = NODE_FEATURE_NAMES_20
    elif node_dim == 19:
        graph_tag = "19/24"
        node_feature_names = NODE_FEATURE_NAMES_19
    else:
        graph_tag = f"{node_dim}/{edge_dim}"
        node_feature_names = [f"node_feature_{i}" for i in range(node_dim)]

    edge_feature_names = (
        EDGE_FEATURE_NAMES_24
        if edge_dim == len(EDGE_FEATURE_NAMES_24)
        else [f"edge_feature_{i}" for i in range(edge_dim)]
    )

    model = OpticalResponseGINE_LinearEnsembleUE(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        node_feature_names=node_feature_names,
        edge_feature_names=edge_feature_names,
    ).to(device)

    model.eval()
    y_hat = model(g0.to(device))

    node_df, edge_df = model.get_contribution_weights(top_k=top_k)

    print("=" * 100)
    print("Linear Ensemble UE-GINE sanity check")
    print("=" * 100)
    print("Detected graph tag:", graph_tag)
    print("Node dim:", node_dim)
    print("Edge dim:", edge_dim)
    print("Output dim:", out_dim)
    print("Prediction shape:", tuple(y_hat.shape))
    print("Sum node p:", float(model.node_ensemble.mixing_weights().sum().detach().cpu()))
    print("Sum edge p:", float(model.edge_ensemble.mixing_weights().sum().detach().cpu()))

    return model, node_df, edge_df

# ============================================================
# Linear Ensemble contribution export utility
# Added for TASK3 GNNOpt-style Linear Ensemble UE-GINE 20/24
# ============================================================

def export_linear_ensemble_contribution_weights(
    epsI_df=None,
    epsR_df=None,
    epsI_ds=None,
    epsR_ds=None,
    root=r"D:\\TB3\\processed\\paired_training",
    device=None,
    model_class=None,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.05,
    out_subdir="linear_ensemble_contribution_20_24",
    epsI_folder="epsI_0_SSL_init_linear_ensemble_UE_GINE_20_24_epsI_0_3seed",
    epsR_folder="epsR_0_SSL_init_linear_ensemble_UE_GINE_20_24_epsR_0_3seed",
    top_k=12,
    show=True,
):
    """
    Export descriptor-level contribution weights from Linear Ensemble UE-GINE.

    Expected model:
        OpticalResponseGINE_LinearEnsembleUE

    Outputs:
        - node raw CSV by seed
        - edge raw CSV by seed
        - node mean/std CSV
        - edge mean/std CSV
        - top-k node/edge contribution figures for epsI_0 and epsR_0

    Notes:
        Contribution weights are learned mixing probabilities p_i from the
        linear ensemble embedding layers. They are relative usage weights,
        not causal physical proof.
    """
    import re
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    try:
        from IPython.display import display
    except Exception:
        display = None

    root = Path(root)
    out_dir = root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if model_class is None:
        model_class = OpticalResponseGINE_LinearEnsembleUE

    def _load_summary(df, folder_name):
        if df is not None:
            return df.copy()

        folder = root / folder_name
        files = sorted(folder.glob("summary_optical_*.csv"))

        if len(files) == 0:
            raise FileNotFoundError(f"Cannot find summary CSV in: {folder}")

        print("Loaded summary from:", files[0])
        return pd.read_csv(files[0])

    def _parse_seed_from_ckpt(path):
        match = re.search(r"seed(\d+)", str(path))
        return int(match.group(1)) if match else None

    def _feature_names_for_dims(node_dim, edge_dim):
        if int(node_dim) == 20:
            node_names = NODE_FEATURE_NAMES_20
        elif int(node_dim) == 19:
            node_names = NODE_FEATURE_NAMES_19
        else:
            node_names = [f"node_feature_{i}" for i in range(int(node_dim))]

        if int(edge_dim) == len(EDGE_FEATURE_NAMES_24):
            edge_names = EDGE_FEATURE_NAMES_24
        else:
            edge_names = [f"edge_feature_{i}" for i in range(int(edge_dim))]

        return node_names, edge_names

    def _build_model(ds):
        g0 = ds[0]
        node_dim = int(g0.x.shape[1])
        edge_dim = int(g0.edge_attr.shape[1])
        out_dim = int(g0.y.view(-1).shape[0])

        node_names, edge_names = _feature_names_for_dims(node_dim, edge_dim)

        model = model_class(
            node_in_dim=node_dim,
            edge_in_dim=edge_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            node_feature_names=node_names,
            edge_feature_names=edge_names,
        ).to(device)

        return model

    def _load_one_and_extract(ckpt_path, ds, target_key):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = _build_model(ds)

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt

        model.load_state_dict(state, strict=False)
        model.eval()

        node_df, edge_df = model.get_contribution_weights(top_k=None)
        seed = _parse_seed_from_ckpt(ckpt_path)

        node_df = node_df.copy()
        edge_df = edge_df.copy()

        node_df["target_key"] = target_key
        node_df["seed"] = seed
        node_df["ckpt_path"] = str(ckpt_path)

        edge_df["target_key"] = target_key
        edge_df["seed"] = seed
        edge_df["ckpt_path"] = str(ckpt_path)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return node_df, edge_df

    def _meanstd(raw_df, baseline):
        out = (
            raw_df
            .groupby(["target_key", "type", "feature"], as_index=False)
            .agg(
                weight_mean=("contribution_weight", "mean"),
                weight_std=("contribution_weight", "std"),
                n_seeds=("contribution_weight", "count"),
            )
            .sort_values(["target_key", "weight_mean"], ascending=[True, False])
            .reset_index(drop=True)
        )
        out["uniform_baseline"] = float(baseline)
        out["delta_from_uniform"] = out["weight_mean"] - float(baseline)
        return out

    def _plot_top(mean_df, target_key, feature_type):
        d = mean_df[mean_df["target_key"] == target_key].copy()
        d = d.sort_values("weight_mean", ascending=False).head(int(top_k))
        d = d.sort_values("weight_mean", ascending=True)

        baseline = 1.0 / 20.0 if feature_type == "node" else 1.0 / 24.0
        target_label = (
            r"$\varepsilon_2(E)$ / epsI$_0$"
            if target_key == "epsI_0"
            else r"$\varepsilon_1(E)$ / epsR$_0$"
        )

        fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=300)
        ax.barh(
            d["feature"],
            d["weight_mean"],
            xerr=d["weight_std"].fillna(0),
            capsize=3,
        )
        ax.axvline(
            baseline,
            linestyle="--",
            linewidth=1.2,
            label=f"Uniform baseline = {baseline:.4f}",
        )
        ax.set_xlabel("Learned contribution probability")
        ax.set_ylabel("Descriptor")
        ax.set_title(f"{feature_type.capitalize()} descriptor contribution — {target_label}")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        png = out_dir / f"top_{int(top_k)}_{feature_type}_contribution_{target_key}_linear_ensemble_20_24.png"
        svg = out_dir / f"top_{int(top_k)}_{feature_type}_contribution_{target_key}_linear_ensemble_20_24.svg"
        fig.savefig(png, dpi=600, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return png, svg

    if epsI_ds is None or epsR_ds is None:
        raise ValueError("epsI_ds and epsR_ds must be provided.")

    epsI_df = _load_summary(epsI_df, epsI_folder)
    epsR_df = _load_summary(epsR_df, epsR_folder)

    if "ckpt_path" not in epsI_df.columns:
        raise RuntimeError("epsI_df does not contain ckpt_path column.")
    if "ckpt_path" not in epsR_df.columns:
        raise RuntimeError("epsR_df does not contain ckpt_path column.")

    print("=" * 100)
    print("EXPORT LINEAR ENSEMBLE CONTRIBUTION WEIGHTS")
    print("=" * 100)
    print("Output directory:", out_dir)
    print("epsI_0 seeds:", epsI_df["seed"].tolist() if "seed" in epsI_df.columns else "N/A")
    print("epsR_0 seeds:", epsR_df["seed"].tolist() if "seed" in epsR_df.columns else "N/A")

    node_rows = []
    edge_rows = []

    for _, row in epsI_df.iterrows():
        ndf, edf = _load_one_and_extract(row["ckpt_path"], epsI_ds, "epsI_0")
        node_rows.append(ndf)
        edge_rows.append(edf)

    for _, row in epsR_df.iterrows():
        ndf, edf = _load_one_and_extract(row["ckpt_path"], epsR_ds, "epsR_0")
        node_rows.append(ndf)
        edge_rows.append(edf)

    node_raw = pd.concat(node_rows, ignore_index=True)
    edge_raw = pd.concat(edge_rows, ignore_index=True)

    node_dim = int(epsI_ds[0].x.shape[1])
    edge_dim = int(epsI_ds[0].edge_attr.shape[1])

    node_meanstd = _meanstd(node_raw, baseline=1.0 / max(node_dim, 1))
    edge_meanstd = _meanstd(edge_raw, baseline=1.0 / max(edge_dim, 1))

    node_raw_csv = out_dir / "node_feature_contribution_raw_by_seed_linear_ensemble_20_24.csv"
    edge_raw_csv = out_dir / "edge_feature_contribution_raw_by_seed_linear_ensemble_20_24.csv"
    node_mean_csv = out_dir / "node_feature_contribution_mean_std_linear_ensemble_20_24.csv"
    edge_mean_csv = out_dir / "edge_feature_contribution_mean_std_linear_ensemble_20_24.csv"

    node_raw.to_csv(node_raw_csv, index=False, encoding="utf-8-sig")
    edge_raw.to_csv(edge_raw_csv, index=False, encoding="utf-8-sig")
    node_meanstd.to_csv(node_mean_csv, index=False, encoding="utf-8-sig")
    edge_meanstd.to_csv(edge_mean_csv, index=False, encoding="utf-8-sig")

    fig_paths = {}
    for target_key in ["epsI_0", "epsR_0"]:
        fig_paths[(target_key, "node")] = _plot_top(node_meanstd, target_key, "node")
        fig_paths[(target_key, "edge")] = _plot_top(edge_meanstd, target_key, "edge")

    print("\nSaved CSV files:")
    print("Node raw     :", node_raw_csv)
    print("Edge raw     :", edge_raw_csv)
    print("Node mean/std:", node_mean_csv)
    print("Edge mean/std:", edge_mean_csv)

    print("\nSaved figures:")
    for key, paths in fig_paths.items():
        print(key, ":", paths[0])

    if display is not None and show:
        print("\nTop node descriptors:")
        display(node_meanstd.groupby("target_key").head(10).reset_index(drop=True))
        print("\nTop edge descriptors:")
        display(edge_meanstd.groupby("target_key").head(10).reset_index(drop=True))

    print("\n" + "=" * 100)
    print("DONE: Linear Ensemble UE-GINE contribution weights exported")
    print("=" * 100)

    return {
        "node_raw": node_raw,
        "edge_raw": edge_raw,
        "node_meanstd": node_meanstd,
        "edge_meanstd": edge_meanstd,
        "node_raw_csv": node_raw_csv,
        "edge_raw_csv": edge_raw_csv,
        "node_mean_csv": node_mean_csv,
        "edge_mean_csv": edge_mean_csv,
        "fig_paths": fig_paths,
        "out_dir": out_dir,
    }


# ============================================================
# Clean SSL-init gain dot plot for slide | Linear Ensemble UE-GINE 20/24
# Added for Cell 26H clean workflow
# ============================================================

def plot_linear_ensemble_ssl_gain_dotplot_final(
    root=r"D:\\TB3\\processed\\paired_training",
    contrib_subdir="linear_ensemble_contribution_20_24",
    gain_filename="linear_ensemble_20_24_ssl_gain_vs_scratch.csv",
    save_prefix="fig_linear_ensemble_ssl_gain_dotplot_final_no_note",
    show=True,
):
    """
    Plot a clean lollipop/dot plot showing SSL-init MAE gain over scratch.

    Input CSV must contain:
    - target_key
    - MAE_reduction_percent

    Output:
    - PNG/SVG/PDF figure
    - gain dataframe and plot dataframe

    Interpretation:
    - Positive MAE reduction means SSL-init has lower test MAE than scratch.
    - Near-zero MAE reduction is labeled as comparable.
    """
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    root = Path(root)
    out_dir = root / contrib_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_csv = out_dir / gain_filename
    if not gain_csv.exists():
        raise FileNotFoundError(f"Cannot find gain CSV: {gain_csv}")

    gain_df = pd.read_csv(gain_csv)

    if "target_key" not in gain_df.columns:
        raise RuntimeError("gain CSV must contain target_key column.")
    if "MAE_reduction_percent" not in gain_df.columns:
        raise RuntimeError("gain CSV must contain MAE_reduction_percent column.")

    plot_df = gain_df.copy()

    label_map = {
        "epsI_0": r"$\varepsilon_2(E)$",
        "epsR_0": r"$\varepsilon_1(E)$",
    }

    # Put epsI_0 below and epsR_0 above in the final plot.
    order_map = {"epsI_0": 0, "epsR_0": 1}

    plot_df = plot_df[plot_df["target_key"].isin(order_map.keys())].copy()
    plot_df["target_label"] = plot_df["target_key"].map(label_map)
    plot_df["gain_percent"] = plot_df["MAE_reduction_percent"].astype(float)
    plot_df["order"] = plot_df["target_key"].map(order_map)
    plot_df = plot_df.sort_values("order").reset_index(drop=True)

    if len(plot_df) == 0:
        raise RuntimeError("No epsI_0 / epsR_0 rows found in gain CSV.")

    # Font settings for PowerPoint-friendly output.
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=300)

    y_pos = range(len(plot_df))

    # Zero reference line.
    ax.axvline(0, color="black", linewidth=1.3)

    # Lollipop guide lines.
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        gain = float(row["gain_percent"])
        ax.hlines(
            y=y,
            xmin=0,
            xmax=gain,
            linewidth=2.3,
            alpha=0.65,
        )

    # Points.
    ax.scatter(
        plot_df["gain_percent"],
        list(y_pos),
        s=135,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    # Value labels.
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        gain = float(row["gain_percent"])

        if gain >= 1.0:
            status = "better"
            x_offset = 0.10
        elif gain <= -1.0:
            status = "worse"
            x_offset = -0.10
        else:
            status = "comparable"
            x_offset = 0.16

        ha = "left" if gain >= 0 else "right"

        ax.text(
            gain + x_offset,
            y,
            f"{gain:.2f}% ({status})",
            va="center",
            ha=ha,
            fontsize=10.8,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["target_label"], fontsize=14)

    ax.set_xlabel("Test MAE reduction by SSL-init (%)", fontsize=12)
    ax.set_title("SSL-init Gain over Scratch", fontsize=15, fontweight="bold", pad=8)

    xmin = min(-0.25, float(plot_df["gain_percent"].min()) - 0.25)
    xmax = max(4.35, float(plot_df["gain_percent"].max()) + 0.55)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.45, len(plot_df) - 0.55)

    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = out_dir / f"{save_prefix}.png"
    out_svg = out_dir / f"{save_prefix}.svg"
    out_pdf = out_dir / f"{save_prefix}.pdf"

    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("Saved SSL-init gain dot plot:")
    print(out_png)
    print(out_svg)
    print(out_pdf)

    return {
        "gain": gain_df,
        "plot_df": plot_df,
        "png": out_png,
        "svg": out_svg,
        "pdf": out_pdf,
    }
