"""
External screening utilities:
- AB2C2 (C=P/N)
- Zintl-like Materials Project pool
- SRP-power at 200 nm
- test-set SRP output generation
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .features import structure_to_pyg_spectrum_data_enhanced
from .models import OpticalResponseGINE
from .train import extract_prediction_from_output, call_model_safely

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
