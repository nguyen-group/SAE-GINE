"""
Repeated-subset robustness utilities for TASK3.
"""

from pathlib import Path
import gc
import importlib

import pandas as pd
import torch

from .lowdata import run_lowdata_ssl_vs_scratch


def _fraction_label(fraction):
    return f"{int(round(100 * float(fraction)))}pct"


def _infer_init_type(row):
    if "init_type" in row and pd.notna(row["init_type"]):
        return str(row["init_type"])

    variant = str(row.get("variant", "")).lower()

    if "ssl_init" in variant:
        return "ssl_init"

    if "scratch" in variant:
        return "scratch"

    return pd.NA


def _validate_enhanced_targets(
    targets,
    expected_node_dim=19,
    expected_edge_dim=24,
    expected_split_sizes=(7931, 992, 992),
):
    required_targets = ("epsR_0", "epsI_0")

    if not isinstance(targets, dict):
        raise TypeError("targets must be a dictionary.")

    missing = [key for key in required_targets if key not in targets]
    if missing:
        raise KeyError(f"Missing targets: {missing}")

    rows = []

    for target_key in required_targets:
        train_ds, val_ds, test_ds = targets[target_key]
        sample = train_ds[0]

        node_dim = int(sample.x.shape[-1])
        edge_dim = int(sample.edge_attr.shape[-1])
        split_sizes = (len(train_ds), len(val_ds), len(test_ds))

        if (node_dim, edge_dim) != (
            int(expected_node_dim),
            int(expected_edge_dim),
        ):
            raise RuntimeError(
                f"{target_key} must use graph "
                f"{expected_node_dim}/{expected_edge_dim}, "
                f"received {node_dim}/{edge_dim}."
            )

        if (
            expected_split_sizes is not None
            and split_sizes != tuple(expected_split_sizes)
        ):
            raise RuntimeError(
                f"{target_key} split sizes must be "
                f"{tuple(expected_split_sizes)}, received {split_sizes}."
            )

        rows.append(
            {
                "target_key": target_key,
                "train_samples": split_sizes[0],
                "val_samples": split_sizes[1],
                "test_samples": split_sizes[2],
                "node_dim": node_dim,
                "edge_dim": edge_dim,
            }
        )

    return pd.DataFrame(rows)



def _build_default_enhanced_targets(project_root, dataset=None):
    """
    Build the canonical enhanced 19/24 paired datasets and fixed seed-2025 split.
    """
    project_root = Path(project_root)
    processed_dir = project_root / "processed"
    data_zip = project_root / "database_300.zip"

    from . import data as DATA
    from . import features as FEAT

    # Reload the canonical graph builder so earlier notebook-side patches do not
    # alter the enhanced 19/24 representation used by this experiment.
    FEAT = importlib.reload(FEAT)

    if dataset is None:
        if not data_zip.exists():
            raise FileNotFoundError(f"Dataset archive not found: {data_zip}")

        dataset = DATA.CrystalOpticalDataset(
            zip_path=data_zip,
            require_any_target=True,
            require_full_spectrum=False,
            verbose=True,
        )

    (
        epsR_graph,
        epsI_graph,
        _,
        _,
    ) = FEAT.build_paired_dielectric_graph_datasets_enhanced(
        dataset,
        key_real="epsR_0",
        key_imag="epsI_0",
        verbose=True,
    )

    split_pack = FEAT.build_fixed_paired_split(
        real_graph_ds=epsR_graph,
        imag_graph_ds=epsI_graph,
        processed_dir=processed_dir,
        seed=2025,
        val_ratio=0.10,
        test_ratio=0.10,
        split_subdir="fixed_splits",
        split_prefix="fixed_split_paired_epsR_epsI",
        version="v1",
        overwrite=False,
    )

    targets = {
        "epsR_0": (
            split_pack["epsR_train_ds"],
            split_pack["epsR_val_ds"],
            split_pack["epsR_test_ds"],
        ),
        "epsI_0": (
            split_pack["epsI_train_ds"],
            split_pack["epsI_val_ds"],
            split_pack["epsI_test_ds"],
        ),
    }

    return targets, dataset


def run_repeated_subset_robustness(
    targets=None,
    root=None,
    ssl_ckpt=None,
    project_root=None,
    dataset=None,
    label_fractions=(0.25, 0.50),
    subset_seeds=(777, 2026, 3407),
    model_seeds=(42, 123, 2025),
    output_dir=None,
    train_kwargs=None,
    force_rerun=True,
    expected_split_sizes=(7931, 992, 992),
    save_outputs=True,
    show=True,
):
    """
    Compare scratch and SSL initialization across repeated labeled subsets.

    The experiment first averages over model seeds within each independently
    sampled labeled subset, then reports mean and sample standard deviation
    across subset-level means.

    Parameters
    ----------
    targets : dict or None
        Mapping from target key to (train_ds, val_ds, test_ds). If None,
        canonical enhanced 19/24 datasets are reconstructed from project_root.
    root : str or pathlib.Path or None
        Paired-training root directory. Derived from project_root when omitted.
    ssl_ckpt : str or pathlib.Path or None
        SSL backbone checkpoint. Derived from project_root when omitted.
    project_root : str or pathlib.Path or None
        TASK3 project root, typically ``D:/TB3``.
    dataset : object or None
        Optional loaded CrystalOpticalDataset used when targets must be rebuilt.
    label_fractions : sequence of float
        Labeled fractions to evaluate.
    subset_seeds : sequence of int
        Independent subset-selection seeds.
    model_seeds : sequence of int
        Model-training seeds used within each subset.
    output_dir : str or pathlib.Path or None
        Directory for robustness CSV outputs.
    train_kwargs : dict or None
        Optional overrides to the matched optimization protocol.
    force_rerun : bool
        Retrain even when a completed per-subset summary CSV exists.
    expected_split_sizes : tuple or None
        Expected train/validation/test sizes.
    save_outputs : bool
        Save combined robustness outputs.
    show : bool
        Display the comparison and robustness-summary tables.

    Returns
    -------
    dict
        Per-seed runs, subset-level statistics, paired comparisons,
        robustness summary, run manifest, and output paths.
    """
    if project_root is not None:
        project_root = Path(project_root)

        if root is None:
            root = project_root / "processed" / "paired_training"

        if ssl_ckpt is None:
            ssl_ckpt = (
                project_root
                / "processed"
                / "ssl_enhanced_19_24"
                / "ssl_enhanced_19_24_backbone_only_for_finetune.pt"
            )

    if root is None:
        raise ValueError("Provide either root or project_root.")

    if ssl_ckpt is None:
        raise ValueError("Provide either ssl_ckpt or project_root.")

    root = Path(root)
    ssl_ckpt = Path(ssl_ckpt)

    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_ckpt}")

    if targets is None:
        if project_root is None:
            raise ValueError(
                "targets is None, so project_root is required to reconstruct "
                "the enhanced datasets and fixed split."
            )

        print("Enhanced targets are not available in memory.")
        print("Reconstructing the canonical 19/24 datasets and fixed split...")
        targets, dataset = _build_default_enhanced_targets(
            project_root=project_root,
            dataset=dataset,
        )

    if output_dir is None:
        output_dir = root / "paper_outputs" / "repeated_subset_robustness"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_verification = _validate_enhanced_targets(
        targets,
        expected_node_dim=19,
        expected_edge_dim=24,
        expected_split_sizes=expected_split_sizes,
    )

    matched_hparams = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 16,
        "num_epochs": 100,
        "early_stop_patience": 15,
        "lr": 5e-4,
        "weight_decay": 5e-5,
        "hidden_dim": 192,
        "latent_dim": 256,
        "num_layers": 4,
        "dropout": 0.05,
        "peak_weight": 2.0,
        "smoothness_weight": 0.03,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
        "use_amp": torch.cuda.is_available(),
    }

    if train_kwargs:
        matched_hparams.update(dict(train_kwargs))

    print("\n" + "=" * 100)
    print("DATASET VERIFICATION")
    print("=" * 100)

    for row in dataset_verification.itertuples(index=False):
        print(
            f"{row.target_key}: "
            f"train={row.train_samples}, "
            f"val={row.val_samples}, "
            f"test={row.test_samples}, "
            f"node={row.node_dim}, "
            f"edge={row.edge_dim}"
        )

    print("\n" + "=" * 100)
    print("REPEATED-SUBSET EXPERIMENT")
    print("=" * 100)
    print("Fractions    :", list(label_fractions))
    print("Subset seeds :", list(subset_seeds))
    print("Model seeds  :", list(model_seeds))
    print("Device       :", matched_hparams["device"])
    print("Learning rate:", matched_hparams["lr"])
    print("Weight decay :", matched_hparams["weight_decay"])
    print("Dropout      :", matched_hparams["dropout"])
    print("Output       :", output_dir)

    all_run_tables = []
    run_manifest = []

    for fraction in label_fractions:
        for subset_seed in subset_seeds:
            fraction_tag = (
                f"repeated_subset_"
                f"{_fraction_label(fraction)}_"
                f"subset{subset_seed}_"
                f"enhanced19_24"
            )

            cached_summary_csv = (
                root
                / f"lowdata_{fraction_tag}_ue_gine_vs_ssl_init_summary.csv"
            )

            print("\n" + "#" * 100)
            print(
                f"FRACTION={_fraction_label(fraction)} | "
                f"SUBSET SEED={subset_seed}"
            )
            print("#" * 100)

            if cached_summary_csv.exists() and not force_rerun:
                print("Loading completed result:")
                print(cached_summary_csv)
                run_df = pd.read_csv(cached_summary_csv)
                status = "loaded_existing"
            else:
                result_pack = run_lowdata_ssl_vs_scratch(
                    targets=targets,
                    root=root,
                    ssl_ckpt=ssl_ckpt,
                    label_fraction=float(fraction),
                    fraction_tag=fraction_tag,
                    split_seed=int(subset_seed),
                    seeds=tuple(model_seeds),
                    train_kwargs=matched_hparams,
                    save_outputs=True,
                    show=False,
                )

                run_df = result_pack["summary"].copy()
                status = "trained"

            run_df["label_fraction"] = float(fraction)
            run_df["label_percent"] = int(round(100 * float(fraction)))
            run_df["subset_seed"] = int(subset_seed)

            if "seed" in run_df.columns:
                run_df["model_seed"] = pd.to_numeric(
                    run_df["seed"],
                    errors="coerce",
                ).astype("Int64")
            elif "model_seed" not in run_df.columns:
                raise KeyError(
                    "Neither 'seed' nor 'model_seed' exists in the summary."
                )

            run_df["init_type"] = run_df.apply(
                _infer_init_type,
                axis=1,
            )

            run_df = run_df[
                run_df["init_type"].isin(["scratch", "ssl_init"])
            ].copy()

            run_df = run_df[
                run_df["test_MAE_mean"].notna()
            ].copy()

            expected_rows = len(targets) * 2 * len(model_seeds)

            if len(run_df) != expected_rows:
                raise RuntimeError(
                    f"Expected {expected_rows} per-seed rows for "
                    f"fraction={fraction}, subset_seed={subset_seed}, "
                    f"but found {len(run_df)}."
                )

            all_run_tables.append(run_df)

            run_manifest.append(
                {
                    "label_fraction": float(fraction),
                    "label_percent": int(round(100 * float(fraction))),
                    "subset_seed": int(subset_seed),
                    "fraction_tag": fraction_tag,
                    "status": status,
                    "summary_csv": str(cached_summary_csv),
                    "n_rows": len(run_df),
                }
            )

            del run_df

            if "result_pack" in locals():
                del result_pack

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_runs = pd.concat(
        all_run_tables,
        ignore_index=True,
    )

    all_runs = all_runs.sort_values(
        [
            "label_percent",
            "target_key",
            "subset_seed",
            "init_type",
            "model_seed",
        ]
    ).reset_index(drop=True)

    manifest_df = pd.DataFrame(run_manifest)

    aggregation_dict = {
        "n_model_seeds": ("test_MAE_mean", "count"),
        "test_MAE_mean": ("test_MAE_mean", "mean"),
        "test_MAE_model_sd": (
            "test_MAE_mean",
            lambda x: x.std(ddof=1),
        ),
    }

    if "test_RMSE_mean" in all_runs.columns:
        aggregation_dict.update(
            {
                "test_RMSE_mean": ("test_RMSE_mean", "mean"),
                "test_RMSE_model_sd": (
                    "test_RMSE_mean",
                    lambda x: x.std(ddof=1),
                ),
            }
        )

    if "test_CosineSim_mean" in all_runs.columns:
        aggregation_dict.update(
            {
                "test_Cosine_mean": (
                    "test_CosineSim_mean",
                    "mean",
                ),
                "test_Cosine_model_sd": (
                    "test_CosineSim_mean",
                    lambda x: x.std(ddof=1),
                ),
            }
        )

    subset_level = (
        all_runs
        .groupby(
            [
                "label_fraction",
                "label_percent",
                "target_key",
                "subset_seed",
                "init_type",
            ],
            sort=True,
        )
        .agg(**aggregation_dict)
        .reset_index()
    )

    paired_subset = (
        subset_level
        .pivot_table(
            index=[
                "label_fraction",
                "label_percent",
                "target_key",
                "subset_seed",
            ],
            columns="init_type",
            values="test_MAE_mean",
            aggfunc="first",
        )
        .reset_index()
    )

    if not {"scratch", "ssl_init"}.issubset(paired_subset.columns):
        raise RuntimeError(
            "Both scratch and ssl_init subset-level results are required."
        )

    paired_subset["MAE_reduction_abs"] = (
        paired_subset["scratch"]
        - paired_subset["ssl_init"]
    )

    paired_subset["MAE_reduction_percent"] = (
        100.0
        * paired_subset["MAE_reduction_abs"]
        / paired_subset["scratch"]
    )

    paired_subset["SSL_better"] = (
        paired_subset["MAE_reduction_abs"] > 0
    )

    scratch_subset = (
        subset_level[
            subset_level["init_type"] == "scratch"
        ]
        .rename(
            columns={
                "test_MAE_mean": "scratch_MAE",
                "test_MAE_model_sd": "scratch_model_seed_sd",
            }
        )
    )

    ssl_subset = (
        subset_level[
            subset_level["init_type"] == "ssl_init"
        ]
        .rename(
            columns={
                "test_MAE_mean": "ssl_init_MAE",
                "test_MAE_model_sd": "ssl_model_seed_sd",
            }
        )
    )

    subset_comparison = scratch_subset.merge(
        ssl_subset,
        on=[
            "label_fraction",
            "label_percent",
            "target_key",
            "subset_seed",
        ],
        suffixes=("_scratch", "_ssl"),
    )

    subset_comparison["MAE_reduction_abs"] = (
        subset_comparison["scratch_MAE"]
        - subset_comparison["ssl_init_MAE"]
    )

    subset_comparison["MAE_reduction_percent"] = (
        100.0
        * subset_comparison["MAE_reduction_abs"]
        / subset_comparison["scratch_MAE"]
    )

    subset_comparison["SSL_better"] = (
        subset_comparison["MAE_reduction_abs"] > 0
    )

    robustness_summary = (
        subset_comparison
        .groupby(
            [
                "label_fraction",
                "label_percent",
                "target_key",
            ],
            sort=True,
        )
        .agg(
            n_subset_seeds=("subset_seed", "count"),
            scratch_MAE_mean=("scratch_MAE", "mean"),
            scratch_MAE_subset_sd=(
                "scratch_MAE",
                lambda x: x.std(ddof=1),
            ),
            ssl_init_MAE_mean=("ssl_init_MAE", "mean"),
            ssl_init_MAE_subset_sd=(
                "ssl_init_MAE",
                lambda x: x.std(ddof=1),
            ),
            MAE_reduction_abs_mean=(
                "MAE_reduction_abs",
                "mean",
            ),
            MAE_reduction_abs_subset_sd=(
                "MAE_reduction_abs",
                lambda x: x.std(ddof=1),
            ),
            MAE_reduction_percent_mean=(
                "MAE_reduction_percent",
                "mean",
            ),
            MAE_reduction_percent_subset_sd=(
                "MAE_reduction_percent",
                lambda x: x.std(ddof=1),
            ),
            SSL_better_subsets=("SSL_better", "sum"),
            mean_scratch_model_seed_sd=(
                "scratch_model_seed_sd",
                "mean",
            ),
            mean_ssl_model_seed_sd=(
                "ssl_model_seed_sd",
                "mean",
            ),
        )
        .reset_index()
    )

    target_labels = {
        "epsR_0": "ε₁(E)",
        "epsI_0": "ε₂(E)",
    }

    for frame in [
        all_runs,
        subset_level,
        paired_subset,
        subset_comparison,
        robustness_summary,
    ]:
        frame["Target"] = frame["target_key"].map(target_labels)

    robustness_summary["Scratch_MAE_paper"] = (
        robustness_summary.apply(
            lambda row:
            f"{row['scratch_MAE_mean']:.4f} ± "
            f"{row['scratch_MAE_subset_sd']:.4f}",
            axis=1,
        )
    )

    robustness_summary["SSL_MAE_paper"] = (
        robustness_summary.apply(
            lambda row:
            f"{row['ssl_init_MAE_mean']:.4f} ± "
            f"{row['ssl_init_MAE_subset_sd']:.4f}",
            axis=1,
        )
    )

    robustness_summary["Relative_reduction_paper"] = (
        robustness_summary.apply(
            lambda row:
            f"{row['MAE_reduction_percent_mean']:.2f} ± "
            f"{row['MAE_reduction_percent_subset_sd']:.2f}%",
            axis=1,
        )
    )

    output_paths = {
        "all_runs": output_dir / "repeated_subset_all_per_seed_runs.csv",
        "subset_level": output_dir / "repeated_subset_model_seed_meanstd.csv",
        "paired_subset": output_dir / "repeated_subset_paired_gain_by_subset.csv",
        "subset_comparison": output_dir / "repeated_subset_comparison.csv",
        "robustness_summary": output_dir / "repeated_subset_robustness_summary.csv",
        "manifest": output_dir / "repeated_subset_run_manifest.csv",
    }

    if save_outputs:
        all_runs.to_csv(
            output_paths["all_runs"],
            index=False,
            encoding="utf-8-sig",
        )
        subset_level.to_csv(
            output_paths["subset_level"],
            index=False,
            encoding="utf-8-sig",
        )
        paired_subset.to_csv(
            output_paths["paired_subset"],
            index=False,
            encoding="utf-8-sig",
        )
        subset_comparison.to_csv(
            output_paths["subset_comparison"],
            index=False,
            encoding="utf-8-sig",
        )
        robustness_summary.to_csv(
            output_paths["robustness_summary"],
            index=False,
            encoding="utf-8-sig",
        )
        manifest_df.to_csv(
            output_paths["manifest"],
            index=False,
            encoding="utf-8-sig",
        )

    if show:
        try:
            from IPython.display import display

            print("\n" + "=" * 100)
            print("REPEATED-SUBSET COMPARISON")
            print("Each row is averaged over three model seeds.")
            print("=" * 100)

            display(
                subset_comparison[
                    [
                        "label_percent",
                        "Target",
                        "subset_seed",
                        "scratch_MAE",
                        "scratch_model_seed_sd",
                        "ssl_init_MAE",
                        "ssl_model_seed_sd",
                        "MAE_reduction_abs",
                        "MAE_reduction_percent",
                        "SSL_better",
                    ]
                ]
            )

            print("\n" + "=" * 100)
            print("ROBUSTNESS SUMMARY ACROSS THREE LABELED SUBSETS")
            print("Mean ± sample SD is calculated across subset-level means.")
            print("=" * 100)

            display(
                robustness_summary[
                    [
                        "label_percent",
                        "Target",
                        "n_subset_seeds",
                        "Scratch_MAE_paper",
                        "SSL_MAE_paper",
                        "Relative_reduction_paper",
                        "SSL_better_subsets",
                        "mean_scratch_model_seed_sd",
                        "mean_ssl_model_seed_sd",
                    ]
                ]
            )

        except Exception:
            print(subset_comparison)
            print(robustness_summary)

    if save_outputs:
        print("\nSaved outputs:")
        for path in output_paths.values():
            print(" -", path)

    return {
        "targets": targets,
        "dataset": dataset,
        "dataset_verification": dataset_verification,
        "all_runs": all_runs,
        "subset_level": subset_level,
        "paired_subset": paired_subset,
        "subset_comparison": subset_comparison,
        "robustness_summary": robustness_summary,
        "manifest": manifest_df,
        "output_dir": output_dir,
        "output_paths": output_paths,
        "train_kwargs": matched_hparams,
    }
