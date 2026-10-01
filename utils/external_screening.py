# ============================================================
# External fixed Zintl-like pool screening
# Uses existing helper functions from utils_tb.screening
# ============================================================

from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
import torch

from .screening import (
    _tb3_screen_get_mpr,
    _tb3_screen_load_training_summary,
    _tb3_screen_get_ssl_100pct_ckpt_paths,
    _tb3_screen_make_graph_dataset,
    _tb3_screen_predict_ensemble,
    _tb3_screen_dielectric_to_optical,
    _tb3_screen_compute_srp_power,
    _tb3_merge_screening_meta_with_pool,
    _tb3_save_filtered_screening_tables,
)


def _fetch_structures_for_fixed_pool(pool_df, cache_pkl, failed_csv, use_cache=True, chunk_size=80):
    """Fetch Materials Project structures for a fixed pool and cache them."""

    cache_pkl = Path(cache_pkl)
    failed_csv = Path(failed_csv)
    cache_pkl.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_pkl.exists():
        pool_with_struct = pd.read_pickle(cache_pkl)
        failed_df = pd.read_csv(failed_csv) if failed_csv.exists() else pd.DataFrame()
        print(f"Loaded structure cache: {cache_pkl}")
        print("Valid structures:", len(pool_with_struct))
        print("Failed structures:", len(failed_df))
        return pool_with_struct, failed_df

    if "material_id" not in pool_df.columns:
        raise KeyError("pool_df must contain material_id column.")

    material_ids = (
        pool_df["material_id"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    structure_map = {}
    failed_rows = []

    n_total = len(material_ids)
    n_chunks = int(np.ceil(n_total / chunk_size))
    t0 = time.time()

    print("=" * 80)
    print("Fetching structures from Materials Project")
    print("=" * 80)
    print("Total materials:", n_total)
    print("Chunk size:", chunk_size)

    with _tb3_screen_get_mpr() as mpr:
        for i, start in enumerate(range(0, n_total, chunk_size), start=1):
            chunk = material_ids[start:start + chunk_size]
            before = len(structure_map)

            print(f"[{i:03d}/{n_chunks:03d}] {start + 1}-{start + len(chunk)} / {n_total}")

            try:
                try:
                    docs = mpr.materials.summary.search(
                        material_ids=chunk,
                        fields=["material_id", "structure"],
                        all_fields=False,
                    )
                except TypeError:
                    docs = mpr.materials.summary.search(
                        material_ids=chunk,
                        fields=["material_id", "structure"],
                    )

                for doc in docs:
                    d = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
                    mid = str(d.get("material_id", getattr(doc, "material_id", ""))).strip()
                    structure = d.get("structure", getattr(doc, "structure", None))

                    if mid and structure is not None:
                        structure_map[mid] = structure

            except Exception as exc:
                print("Chunk failed, fallback one-by-one:", exc)

                for mid in chunk:
                    try:
                        structure_map[mid] = mpr.get_structure_by_material_id(mid)
                    except Exception as one_exc:
                        failed_rows.append({"material_id": mid, "error": str(one_exc)})

            elapsed = (time.time() - t0) / 60
            print(
                f"    got this chunk: {len(structure_map) - before} | "
                f"total: {len(structure_map)}/{n_total} | "
                f"elapsed: {elapsed:.1f} min"
            )

    out_df = pool_df.copy()
    out_df["material_id"] = out_df["material_id"].astype(str).str.strip()
    out_df["structure"] = out_df["material_id"].map(structure_map)

    failed_by_missing = out_df[out_df["structure"].isna()][["material_id"]].copy()
    if "formula_pretty" in out_df.columns:
        failed_by_missing["formula_pretty"] = out_df.loc[out_df["structure"].isna(), "formula_pretty"].values
    failed_by_missing["error"] = "Structure not returned by Materials Project"

    valid_df = out_df[out_df["structure"].notna()].reset_index(drop=True)

    failed_df = pd.concat(
        [pd.DataFrame(failed_rows), failed_by_missing],
        ignore_index=True,
    ).drop_duplicates()

    valid_df.to_pickle(cache_pkl)
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Structure fetch finished")
    print("=" * 80)
    print("Input:", len(pool_df))
    print("Valid structures:", len(valid_df))
    print("Failed:", len(failed_df))
    print("Cache:", cache_pkl)

    return valid_df, failed_df


def screen_external_zintl_fixed_pool_srp_power_200nm(
    pool_csv,
    root,
    mp_api_key=None,
    output_dir=None,
    thickness_nm=200.0,
    top_k=50,
    dft_ehull_max=0.10,
    dft_bg_min=0.30,
    dft_bg_max=4.50,
    seeds=(42, 123, 2025),
    batch_size=32,
    preferred_summary=None,
    use_structure_cache=True,
    chunk_size=80,
    output_prefix="screening_zintl_like_external_fixed_1100",
):
    """
    Screen a fixed external Zintl-like DFT-ready pool using SSL-init UE-GINE.
    """

    root = Path(root)
    pool_csv = Path(pool_csv)

    if output_dir is None:
        output_dir = root / "paper_outputs" / "screening_srp_200nm"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if mp_api_key:
        os.environ["MP_API_KEY"] = str(mp_api_key).strip()

    t0 = time.time()

    pool_df = pd.read_csv(pool_csv).drop_duplicates("material_id").reset_index(drop=True)

    required = ["material_id", "formula_pretty", "band_gap", "energy_above_hull", "is_stable"]
    missing = [c for c in required if c not in pool_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in fixed pool CSV: {missing}")

    pool_df["source_pool_csv"] = str(pool_csv)
    pool_df["external_fixed_zintl_pool"] = True

    cache_dir = output_dir / "external_pool_cache"
    cache_pkl = cache_dir / f"{output_prefix}_with_structures.pkl"
    failed_csv = cache_dir / f"{output_prefix}_failed_structures.csv"

    pool_struct_df, failed_df = _fetch_structures_for_fixed_pool(
        pool_df=pool_df,
        cache_pkl=cache_pkl,
        failed_csv=failed_csv,
        use_cache=use_structure_cache,
        chunk_size=chunk_size,
    )

    if len(pool_struct_df) == 0:
        raise RuntimeError("No valid structures were found for external fixed pool.")

    print("=" * 80)
    print("Loading SSL-init UE-GINE checkpoints")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_ev = np.linspace(0.0, 20.0, 2001).astype(np.float32)

    raw_summary = _tb3_screen_load_training_summary(root, preferred_summary=preferred_summary)

    eps1_ckpts = _tb3_screen_get_ssl_100pct_ckpt_paths(
        raw_summary,
        target_key="epsR_0",
        seeds=seeds,
    )

    eps2_ckpts = _tb3_screen_get_ssl_100pct_ckpt_paths(
        raw_summary,
        target_key="epsI_0",
        seeds=seeds,
    )

    print("Device:", device)
    print("epsR_0 checkpoints:", len(eps1_ckpts))
    print("epsI_0 checkpoints:", len(eps2_ckpts))

    print("=" * 80)
    print("Building graph datasets")
    print("=" * 80)

    eps1_graphs, meta_df = _tb3_screen_make_graph_dataset(
        pool_struct_df,
        energy_ev,
        target_key="epsR_0",
    )

    eps2_graphs, _ = _tb3_screen_make_graph_dataset(
        pool_struct_df,
        energy_ev,
        target_key="epsI_0",
    )

    print("Graphs:", len(eps1_graphs))

    print("=" * 80)
    print("Predicting dielectric spectra")
    print("=" * 80)

    eps1_pred = _tb3_screen_predict_ensemble(
        eps1_graphs,
        eps1_ckpts,
        device,
        batch_size=batch_size,
    )

    eps2_pred = _tb3_screen_predict_ensemble(
        eps2_graphs,
        eps2_ckpts,
        device,
        batch_size=batch_size,
    )

    print("=" * 80)
    print("Computing optical properties and SRP power")
    print("=" * 80)

    _, _, alpha_cm1, R = _tb3_screen_dielectric_to_optical(
        eps1_pred,
        eps2_pred,
        energy_ev,
    )

    srp_power = _tb3_screen_compute_srp_power(
        alpha_cm1=alpha_cm1,
        R=R,
        energy_ev=energy_ev,
        thickness_nm=thickness_nm,
    )

    out_df = _tb3_merge_screening_meta_with_pool(meta_df, pool_struct_df)

    srp_col = f"predicted_SRP_power_{int(round(thickness_nm))}nm_percent"
    out_df[srp_col] = srp_power
    out_df["predicted_SRP_power_200nm_percent"] = srp_power
    out_df["predicted_eps1_mean"] = np.nanmean(eps1_pred, axis=1)
    out_df["predicted_eps2_mean"] = np.nanmean(eps2_pred, axis=1)
    out_df["predicted_alpha_mean_cm1"] = np.nanmean(alpha_cm1, axis=1)
    out_df["predicted_R_mean"] = np.nanmean(R, axis=1)
    out_df["screening_score"] = out_df[srp_col]

    out_df["source_pool_csv"] = str(pool_csv)
    out_df["screening_input_pool_size"] = len(pool_df)
    out_df["screening_valid_structure_count"] = len(pool_struct_df)
    out_df["screening_failed_structure_count"] = len(failed_df)

    out_df = out_df.sort_values("screening_score", ascending=False).reset_index(drop=True)
    out_df["rank"] = np.arange(1, len(out_df) + 1)

    all_csv = output_dir / f"{output_prefix}_srp_power_200nm_all.csv"
    top50_csv = output_dir / f"{output_prefix}_srp_power_200nm_top50.csv"
    summary_csv = output_dir / f"{output_prefix}_summary.csv"

    out_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    filtered = _tb3_save_filtered_screening_tables(
        out_df=out_df,
        out_dir=output_dir,
        prefix=f"{output_prefix}_srp_power_200nm",
        top_k=top_k,
        dft_ehull_max=dft_ehull_max,
        dft_bg_min=dft_bg_min,
        dft_bg_max=dft_bg_max,
    )

    if isinstance(filtered, dict) and "top_dft" in filtered:
        top_df = filtered["top_dft"].copy()
    else:
        tmp = out_df.copy()
        tmp["energy_above_hull"] = pd.to_numeric(tmp["energy_above_hull"], errors="coerce")
        tmp["band_gap"] = pd.to_numeric(tmp["band_gap"], errors="coerce")
        tmp = tmp[
            (tmp["energy_above_hull"] <= dft_ehull_max)
            & (tmp["band_gap"] >= dft_bg_min)
            & (tmp["band_gap"] <= dft_bg_max)
        ]
        top_df = tmp.head(top_k).copy()

    top_df.to_csv(top50_csv, index=False, encoding="utf-8-sig")

    elapsed_min = (time.time() - t0) / 60

    summary_df = pd.DataFrame(
        [
            {
                "stage": "Fixed external Zintl-like DFT-ready pool",
                "n_materials": len(pool_df),
                "output_file": str(pool_csv),
            },
            {
                "stage": "Valid MP structures used for screening",
                "n_materials": len(pool_struct_df),
                "output_file": str(cache_pkl),
            },
            {
                "stage": "Failed MP structures",
                "n_materials": len(failed_df),
                "output_file": str(failed_csv),
            },
            {
                "stage": "All SSL-UE-GINE screened candidates",
                "n_materials": len(out_df),
                "output_file": str(all_csv),
            },
            {
                "stage": f"Top-{top_k} DFT-ready candidates",
                "n_materials": len(top_df),
                "output_file": str(top50_csv),
            },
        ]
    )

    summary_df["thickness_nm"] = thickness_nm
    summary_df["seeds"] = ",".join(map(str, seeds))
    summary_df["elapsed_min"] = elapsed_min
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("External fixed Zintl-like screening finished")
    print("=" * 80)
    print(f"Elapsed: {elapsed_min:.1f} min")
    print("All CSV:", all_csv)
    print("Top50 CSV:", top50_csv)
    print("Summary CSV:", summary_csv)

    return {
        "all_csv": str(all_csv),
        "top50_csv": str(top50_csv),
        "summary_csv": str(summary_csv),
        "failed_structure_csv": str(failed_csv),
        "structure_cache_pkl": str(cache_pkl),
        "source_pool_csv": str(pool_csv),
        "source_count": len(pool_df),
        "valid_structure_count": len(pool_struct_df),
        "failed_structure_count": len(failed_df),
    }