"""
Crystal-graph construction utilities.

Contains:
- baseline graph 11/16
- enhanced graph 19/24
- fixed paired train/val/test split
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .data import CrystalSpectrumView

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
