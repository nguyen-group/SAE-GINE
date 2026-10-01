"""
Feature-level contribution analysis utilities.

This module contains:
- canonical 19/20 node feature names and 24 edge feature names
- enhanced 20/24 graph builder with covalent_radius_cordero
- Linear Ensemble UE-GINE contribution model
- contribution-weight export utilities
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymatgen.core.periodic_table import Element
from torch_geometric.nn import GINEConv, global_mean_pool

from .features import *
import utils_tb.features as _features_module

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


# ------------------------------------------------------------
# Split-package patch override
# ------------------------------------------------------------
def patch_enhanced_graph_builder_to_20_24(verbose=True):
    """
    Patch the actual `utils_tb.features` module so that enhanced graph builders
    produce 20/24 graphs with covalent_radius_cordero.

    This replaces the original monolithic monkey-patch behavior safely after
    splitting `utils_tb3.py` into a package.
    """
    _features_module.get_element_features_enhanced = get_element_features_enhanced_20
    _features_module.build_node_tensor_enhanced = build_node_tensor_enhanced_20

    # also update this module's imported aliases
    globals()["get_element_features_enhanced"] = get_element_features_enhanced_20
    globals()["build_node_tensor_enhanced"] = build_node_tensor_enhanced_20

    if verbose:
        print("=" * 100)
        print("PATCHED ENHANCED GRAPH BUILDER TO 20/24")
        print("=" * 100)
        print("Node dim expected:", len(NODE_FEATURE_NAMES_20))
        print("Edge dim expected:", len(EDGE_FEATURE_NAMES_24))
        print("Added node feature: covalent_radius_cordero")

# ============================================================
# OVERRIDE: 4-panel-only clean Linear Ensemble contribution figure
# Added to replace the earlier slide-style figure with header/text.
# ============================================================

def plot_linear_ensemble_contribution_slidefig(
    node_meanstd,
    edge_meanstd,
    out_dir,
    top_k=12,
    save_prefix="fig_linear_ensemble_contribution_20_24_four_panels",
    show=True,
    include_header=False,
    include_summary=False,
):
    """
    Draw a clean 4-panel contribution figure only.

    This override intentionally removes:
    - the green title header
    - the right-side text summary block

    It keeps only:
    a) epsI_0 / epsilon_2 node descriptors
    b) epsI_0 / epsilon_2 edge descriptors
    c) epsR_0 / epsilon_1 node descriptors
    d) epsR_0 / epsilon_1 edge descriptors

    Parameters include `include_header` and `include_summary` only for
    backward compatibility; they are ignored so the output remains 4-panel only.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    required = {"target_key", "feature", "weight_mean", "weight_std"}
    for name, df in [("node_meanstd", node_meanstd), ("edge_meanstd", edge_meanstd)]:
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{name} is missing required columns: {sorted(missing)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _get_top(df, target_key, label_func):
        d = df[df["target_key"].astype(str) == str(target_key)].copy()
        if len(d) == 0:
            raise ValueError(f"No contribution rows found for target_key={target_key}")

        d["weight_mean"] = pd.to_numeric(d["weight_mean"], errors="coerce")
        d["weight_std"] = pd.to_numeric(d["weight_std"], errors="coerce").fillna(0)
        d = d.sort_values("weight_mean", ascending=False).head(int(top_k))
        d = d.iloc[::-1].copy()

        labels = d["feature"].apply(label_func).tolist()
        values = d["weight_mean"].to_numpy(dtype=float)
        stds = d["weight_std"].to_numpy(dtype=float)
        return labels, values, stds

    eps2_node_labels, eps2_node_values, eps2_node_stds = _get_top(
        node_meanstd, "epsI_0", _tb3_pretty_linear_node_label
    )
    eps2_edge_labels, eps2_edge_values, eps2_edge_stds = _get_top(
        edge_meanstd, "epsI_0", _tb3_pretty_linear_edge_label
    )
    eps1_node_labels, eps1_node_values, eps1_node_stds = _get_top(
        node_meanstd, "epsR_0", _tb3_pretty_linear_node_label
    )
    eps1_edge_labels, eps1_edge_values, eps1_edge_stds = _get_top(
        edge_meanstd, "epsR_0", _tb3_pretty_linear_edge_label
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 7.6),
        dpi=240,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    def _panel_barh(ax, labels, values, stds, title):
        y = np.arange(len(labels))

        ax.barh(
            y,
            values,
            xerr=np.nan_to_num(stds, nan=0.0),
            height=0.72,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.35,
            error_kw=dict(ecolor="black", lw=1.0, capsize=3, capthick=1.0),
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Contribution probability")
        ax.set_title(title, loc="left", pad=5)

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        finite_values = values[np.isfinite(values)]
        finite_stds = np.nan_to_num(stds[np.isfinite(stds)], nan=0.0) if np.any(np.isfinite(stds)) else np.zeros_like(finite_values)
        if finite_values.size:
            xmax = float(np.max(finite_values + np.resize(finite_stds, finite_values.shape))) * 1.10
        else:
            xmax = 1.0
        ax.set_xlim(0, xmax)

        ax.tick_params(axis="both", direction="out", length=3)

    _panel_barh(
        axes[0, 0],
        eps2_node_labels,
        eps2_node_values,
        eps2_node_stds,
        r"a) $\varepsilon_2(E)$ node descriptors",
    )
    _panel_barh(
        axes[0, 1],
        eps2_edge_labels,
        eps2_edge_values,
        eps2_edge_stds,
        r"b) $\varepsilon_2(E)$ edge descriptors",
    )
    _panel_barh(
        axes[1, 0],
        eps1_node_labels,
        eps1_node_values,
        eps1_node_stds,
        r"c) $\varepsilon_1(E)$ node descriptors",
    )
    _panel_barh(
        axes[1, 1],
        eps1_edge_labels,
        eps1_edge_values,
        eps1_edge_stds,
        r"d) $\varepsilon_1(E)$ edge descriptors",
    )

    eps2_top_node = eps2_node_labels[-1]
    eps2_top_edge = eps2_edge_labels[-1]
    eps1_top_node = eps1_node_labels[-1]
    eps1_top_edge = eps1_edge_labels[-1]

    eps2_top_node_value = eps2_node_values[-1]
    eps2_top_edge_value = eps2_edge_values[-1]
    eps1_top_node_value = eps1_node_values[-1]
    eps1_top_edge_value = eps1_edge_values[-1]

    png = out_dir / f"{save_prefix}.png"
    pdf = out_dir / f"{save_prefix}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)

    summary = {
        "epsI_0_node_top": eps2_top_node,
        "epsI_0_node_top_value": float(eps2_top_node_value),
        "epsI_0_edge_top": eps2_top_edge,
        "epsI_0_edge_top_value": float(eps2_top_edge_value),
        "epsR_0_node_top": eps1_top_node,
        "epsR_0_node_top_value": float(eps1_top_node_value),
        "epsR_0_edge_top": eps1_top_edge,
        "epsR_0_edge_top_value": float(eps1_top_edge_value),
    }

    return {"png": png, "pdf": pdf, "summary": summary}


def export_linear_ensemble_contribution_weights_cleanfig(
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
    save_prefix="fig_linear_ensemble_contribution_20_24_four_panels",
    show=True,
    include_header=False,
    include_summary=False,
):
    """
    One-call clean export for LinearEnsembleUE contribution analysis.

    This override suppresses default standalone figures and creates exactly
    one clean 4-panel figure. It does not include a title header or right-side
    text block.
    """
    outputs = export_linear_ensemble_contribution_weights(
        epsI_df=epsI_df,
        epsR_df=epsR_df,
        epsI_ds=epsI_ds,
        epsR_ds=epsR_ds,
        root=root,
        device=device,
        model_class=model_class,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        out_subdir=out_subdir,
        epsI_folder=epsI_folder,
        epsR_folder=epsR_folder,
        top_k=top_k,
        show=False,
    )

    fig_paths = plot_linear_ensemble_contribution_slidefig(
        node_meanstd=outputs["node_meanstd"],
        edge_meanstd=outputs["edge_meanstd"],
        out_dir=outputs["out_dir"],
        top_k=top_k,
        save_prefix=save_prefix,
        show=show,
        include_header=False,
        include_summary=False,
    )

    outputs["slide_fig_paths"] = fig_paths
    outputs["slide_summary"] = fig_paths["summary"]

    print("\nSaved 4-panel-only contribution figure:")
    print("PNG:", fig_paths["png"])
    print("PDF:", fig_paths["pdf"])

    return outputs
