"""
Self-contained DFT validation utilities for TASK3 optical-screening figures.
"""

from pathlib import Path
import io
import re
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


DEFAULT_VALIDATION_MATERIALS = (
    {
        "material_id": "mp-ekf",
        "formula": r"CsSbSe$_2$",
    },
    {
        "material_id": "mp-bxhaf",
        "formula": r"K$_3$Bi(AsSe$_2$)$_6$",
    },
)


def _filename_material_id(filename):
    text = (
        Path(filename)
        .name
        .lower()
        .replace("_", "-")
    )

    match = re.search(
        r"mp-[a-z0-9]+",
        text,
    )

    return match.group(0) if match else None


def _epsilon_to_alpha_cm1(
    energy_eV,
    eps1,
    eps2,
):
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

    elementary_charge = 1.602176634e-19
    planck = 6.62607015e-34
    hbar = planck / (2.0 * np.pi)
    speed_of_light = 2.99792458e8

    omega = (
        energy_eV
        * elementary_charge
        / hbar
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
        / speed_of_light
        * np.sqrt(inner)
    )

    return alpha_m1 / 100.0


def _normalized_filename_text(value):
    return (
        str(value)
        .lower()
        .replace("_", "-")
        .replace("\\", "/")
    )


def _candidate_dft_sources(primary_source):
    """
    Return the requested DFT source plus nearby archives and loose .dat files.
    """
    primary_source = Path(primary_source)

    if primary_source.is_dir():
        search_root = primary_source
        candidates = []
    else:
        search_root = primary_source.parent
        candidates = [primary_source]

    if search_root.exists():
        candidates.extend(
            sorted(search_root.rglob("*.zip"))
        )
        candidates.extend(
            sorted(search_root.rglob("*.dat"))
        )

    unique = []
    seen = set()

    for candidate in candidates:
        key = str(candidate.resolve()).lower()

        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    return unique


def _discover_target_dft_files(
    primary_source,
    material_ids,
):
    """
    Search the requested source and nearby DFT_TEST files for target material IDs.
    """
    material_ids = tuple(
        str(material_id).lower()
        for material_id in material_ids
    )

    found = {}
    searched = []

    for source in _candidate_dft_sources(
        primary_source
    ):
        searched.append(source)

        if source.suffix.lower() == ".dat":
            normalized = _normalized_filename_text(
                source.name
            )

            for material_id in material_ids:
                if material_id in normalized:
                    found.setdefault(
                        material_id,
                        {
                            "kind": "dat",
                            "source": source,
                            "member": None,
                        },
                    )

        elif source.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(
                    source,
                    "r",
                ) as archive:
                    for member in archive.namelist():
                        if not member.lower().endswith(".dat"):
                            continue

                        normalized = _normalized_filename_text(
                            member
                        )

                        for material_id in material_ids:
                            if material_id in normalized:
                                found.setdefault(
                                    material_id,
                                    {
                                        "kind": "zip",
                                        "source": source,
                                        "member": member,
                                    },
                                )

            except zipfile.BadZipFile:
                continue

        if len(found) == len(material_ids):
            break

    return found, searched


def _load_dft_array(entry):
    if entry["kind"] == "dat":
        return np.loadtxt(
            entry["source"]
        )

    with zipfile.ZipFile(
        entry["source"],
        "r",
    ) as archive:
        return np.loadtxt(
            io.BytesIO(
                archive.read(
                    entry["member"]
                )
            )
        )


def build_dft_validation_inputs(
    project_root=r"D:\TB3",
    dft_source=r"D:\DFT_TEST\collect_data_epsilon.zip",
    materials=DEFAULT_VALIDATION_MATERIALS,
    ml_npz=None,
    output_dir=None,
):
    """
    Reconstruct the inputs used by the standalone 0-4 eV DFT-validation figure.

    The function does not silently substitute other materials. If the requested
    DFT archive does not contain the target material IDs, it searches nearby
    ZIP archives and loose DAT files under the same DFT_TEST directory.

    The ML spectrum is the raw direct-alpha prediction stored in
    ``predicted_alpha_cm1``. The DFT reference is calculated from the
    trace-averaged dielectric tensor and interpolated to the ML energy grid.
    """
    project_root = Path(project_root)

    training_root = (
        project_root
        / "processed"
        / "paired_training"
    )

    if ml_npz is None:
        ml_npz = (
            training_root
            / "paper_outputs"
            / "screening_SLME_direct_alpha_500nm"
            / "zintl_external_1100_direct_alpha_SLME_spectra.npz"
        )
    else:
        ml_npz = Path(ml_npz)

    dft_source = Path(dft_source)

    if output_dir is None:
        output_dir = (
            training_root
            / "paper_outputs"
            / "DFT_validation"
            / "two_overlap_FINAL_SLME"
        )
    else:
        output_dir = Path(output_dir)

    if not ml_npz.exists():
        raise FileNotFoundError(
            f"ML spectra file not found: {ml_npz}"
        )

    if not dft_source.exists():
        raise FileNotFoundError(
            f"DFT source not found: {dft_source}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    npz = np.load(
        ml_npz,
        allow_pickle=True,
    )

    required_keys = (
        "material_id",
        "energy_eV",
        "predicted_alpha_cm1",
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in npz.files
    ]

    if missing_keys:
        raise KeyError(
            f"Missing ML NPZ keys: {missing_keys}. "
            f"Available keys: {npz.files}"
        )

    material_ids = np.asarray(
        npz["material_id"]
    ).astype(str)

    energy_ml = np.asarray(
        npz["energy_eV"],
        dtype=np.float64,
    )

    alpha_ml_all = np.asarray(
        npz["predicted_alpha_cm1"],
        dtype=np.float64,
    )

    if (
        alpha_ml_all.ndim != 2
        or alpha_ml_all.shape[0] != len(material_ids)
        or alpha_ml_all.shape[1] != len(energy_ml)
    ):
        raise RuntimeError(
            "Unexpected ML spectra dimensions: "
            f"ids={len(material_ids)}, "
            f"energy={len(energy_ml)}, "
            f"alpha={alpha_ml_all.shape}"
        )

    id_to_ml_index = {
        material_id: index
        for index, material_id
        in enumerate(material_ids)
    }

    requested_ids = [
        item["material_id"]
        for item in materials
    ]

    missing_ml = [
        material_id
        for material_id in requested_ids
        if material_id not in id_to_ml_index
    ]

    if missing_ml:
        raise KeyError(
            f"Missing ML spectra for: {missing_ml}"
        )

    dft_entries, searched_sources = (
        _discover_target_dft_files(
            primary_source=dft_source,
            material_ids=requested_ids,
        )
    )

    missing_dft = [
        material_id
        for material_id in requested_ids
        if material_id.lower() not in dft_entries
    ]

    if missing_dft:
        print("=" * 90)
        print("DFT SOURCE DIAGNOSTIC")
        print("=" * 90)
        print("Requested materials:", requested_ids)
        print("Primary DFT source :", dft_source)
        print("Found DFT materials:", sorted(dft_entries))
        print("Searched sources   :")

        for source in searched_sources:
            print(" -", source)

        raise FileNotFoundError(
            "The exact DFT spectra required by the historical figure "
            f"were not found for: {missing_dft}. "
            "No replacement materials were used."
        )

    records = {}

    for item in materials:
        material_id = item["material_id"]
        entry = dft_entries[
            material_id.lower()
        ]

        array = np.asarray(
            _load_dft_array(
                entry
            ),
            dtype=np.float64,
        )

        if (
            array.ndim != 2
            or array.shape[1] < 7
        ):
            raise RuntimeError(
                f"Unexpected DFT shape for "
                f"{material_id}: {array.shape}"
            )

        energy_dft = np.asarray(
            array[:, 0],
            dtype=np.float64,
        )

        eps1_avg = np.mean(
            array[:, 1:4],
            axis=1,
        )

        eps2_avg = np.mean(
            array[:, 4:7],
            axis=1,
        )

        alpha_dft_native = (
            _epsilon_to_alpha_cm1(
                energy_dft,
                eps1_avg,
                eps2_avg,
            )
        )

        alpha_ml_raw = np.asarray(
            alpha_ml_all[
                id_to_ml_index[
                    material_id
                ]
            ],
            dtype=np.float64,
        )

        alpha_dft_raw = np.interp(
            energy_ml,
            energy_dft,
            alpha_dft_native,
        )

        records[material_id] = {
            "formula": item["formula"],
            "alpha_ml_raw": alpha_ml_raw,
            "alpha_dft_raw": alpha_dft_raw,
            "energy_dft_native": energy_dft,
            "alpha_dft_native": alpha_dft_native,
            "dft_source": str(entry["source"]),
            "dft_member": entry["member"],
        }

    print("=" * 90)
    print("DFT VALIDATION INPUTS RECONSTRUCTED")
    print("=" * 90)
    print("ML NPZ     :", ml_npz)
    print(
        "Energy     :",
        f"{energy_ml.min():.3f}-{energy_ml.max():.3f} eV",
    )
    print("Energy pts :", len(energy_ml))

    for material_id in requested_ids:
        entry = dft_entries[
            material_id.lower()
        ]

        if entry["kind"] == "zip":
            print(
                f"{material_id:10s}: "
                f"{entry['source']} :: {entry['member']}"
            )
        else:
            print(
                f"{material_id:10s}: "
                f"{entry['source']}"
            )

    print("Output     :", output_dir)

    return {
        "energy_axis": energy_ml,
        "records": records,
        "output_dir": output_dir,
        "ml_npz": ml_npz,
        "dft_source": dft_source,
        "dft_entries": dft_entries,
    }


def plot_dft_validation_0_4ev(
    energy_axis,
    records,
    output_dir,
    materials=DEFAULT_VALIDATION_MATERIALS,
    energy_min=0.0,
    energy_max=4.0,
    model_label="SSL-GINE",
    dft_label="DFT",
    model_color="#3B73B9",
    dft_color="#D23B35",
    model_linewidth=2.4,
    dft_linewidth=2.4,
    save_stem="Fig_d_DFT_vs_SSL_GINE_0_4eV_FINAL",
    show=True,
):
    """
    Plot SSL-GINE versus DFT absorption spectra in the exact 0-4 eV window.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    energy_axis = np.asarray(
        energy_axis,
        dtype=np.float64,
    )

    mask = (
        (energy_axis >= float(energy_min))
        & (energy_axis <= float(energy_max))
    )

    energy_window = np.asarray(
        energy_axis[mask],
        dtype=np.float64,
    )

    if len(energy_window) == 0:
        raise RuntimeError(
            f"No spectral points found in "
            f"{energy_min:g}-{energy_max:g} eV."
        )

    print("=" * 90)
    print(
        f"FINAL DFT VALIDATION — EXACT "
        f"{energy_min:g}-{energy_max:g} eV"
    )
    print("=" * 90)
    print(
        f"Energy window : "
        f"{energy_window.min():.3f}-"
        f"{energy_window.max():.3f} eV"
    )
    print(
        f"Energy points : "
        f"{len(energy_window)}"
    )

    plot_data = {}
    metric_rows = []

    for item in materials:
        material_id = item["material_id"]

        if material_id not in records:
            raise KeyError(
                f"{material_id} not found in records."
            )

        alpha_ml = np.asarray(
            records[
                material_id
            ]["alpha_ml_raw"][mask],
            dtype=np.float64,
        )

        alpha_dft = np.asarray(
            records[
                material_id
            ]["alpha_dft_raw"][mask],
            dtype=np.float64,
        )

        valid = (
            np.isfinite(alpha_ml)
            & np.isfinite(alpha_dft)
        )

        reference = alpha_dft[valid]
        prediction = alpha_ml[valid]

        if len(reference) == 0:
            raise RuntimeError(
                f"No valid spectral points found for "
                f"{material_id}."
            )

        ss_res = float(
            np.sum(
                (prediction - reference) ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (reference - np.mean(reference)) ** 2
            )
        )

        r2 = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )

        pearson = (
            float(
                pearsonr(
                    reference,
                    prediction,
                ).statistic
            )
            if len(reference) > 1
            else np.nan
        )

        mae = float(
            np.mean(
                np.abs(
                    prediction - reference
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (prediction - reference) ** 2
                )
            )
        )

        plot_data[material_id] = {
            "formula": item["formula"],
            "ml": alpha_ml,
            "dft": alpha_dft,
        }

        metric_rows.append(
            {
                "material_id": material_id,
                "formula": item["formula"],
                "R2_0_4eV": r2,
                "Pearson_0_4eV": pearson,
                "MAE_x1e6_cm-1": mae / 1e6,
                "RMSE_x1e6_cm-1": rmse / 1e6,
            }
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    print("\n" + "=" * 90)
    print("0-4 eV METRICS")
    print("=" * 90)

    for _, row in metrics.iterrows():
        print()
        print(
            row["material_id"],
            row["formula"],
        )
        print(
            f"  R2      : "
            f"{row['R2_0_4eV']:.4f}"
        )
        print(
            f"  Pearson : "
            f"{row['Pearson_0_4eV']:.4f}"
        )
        print(
            f"  MAE     : "
            f"{row['MAE_x1e6_cm-1']:.4f} "
            f"x10^6 cm^-1"
        )
        print(
            f"  RMSE    : "
            f"{row['RMSE_x1e6_cm-1']:.4f} "
            f"x10^6 cm^-1"
        )

    global_max = max(
        max(
            float(
                np.nanmax(
                    plot_data[
                        item["material_id"]
                    ]["ml"]
                )
            ),
            float(
                np.nanmax(
                    plot_data[
                        item["material_id"]
                    ]["dft"]
                )
            ),
        )
        for item in materials
    )

    ymax = global_max / 1e6

    ymax = (
        np.ceil(
            (ymax * 1.15)
            / 0.1
        )
        * 0.1
    )

    ymax = max(
        ymax,
        0.8,
    )

    rc_params = {
        "font.family": "Arial",
        "font.size": 11,
        "axes.linewidth": 1.15,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
    }

    with plt.rc_context(
        rc_params
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(4.8, 6.3),
            sharex=True,
            dpi=300,
        )

        axes = np.atleast_1d(
            axes
        )

        for ax, item in zip(
            axes,
            materials,
        ):
            material_id = item[
                "material_id"
            ]
            data = plot_data[
                material_id
            ]

            ax.plot(
                energy_window,
                data["ml"] / 1e6,
                color=model_color,
                linewidth=model_linewidth,
                solid_capstyle="round",
                label=model_label,
                zorder=3,
            )

            ax.plot(
                energy_window,
                data["dft"] / 1e6,
                color=dft_color,
                linewidth=dft_linewidth,
                linestyle=(0, (5, 3)),
                dash_capstyle="butt",
                label=dft_label,
                zorder=4,
            )

            ax.text(
                0.96,
                0.81,
                item["formula"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=15,
            )

            ax.set_xlim(
                energy_min,
                energy_max,
            )
            ax.set_ylim(
                0.0,
                ymax,
            )
            ax.set_xticks(
                [0, 1, 2, 3, 4]
            )

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=11,
                width=1.1,
                length=4.5,
            )

            for spine in ax.spines.values():
                spine.set_linewidth(
                    1.15
                )

        axes[-1].set_xlabel(
            "Photon Energy (eV)",
            fontsize=15,
            labelpad=7,
        )

        fig.text(
            0.018,
            0.50,
            r"Absorption Coefficient $\alpha$ "
            r"($\times10^6$ cm$^{-1}$)",
            rotation=90,
            va="center",
            ha="center",
            fontsize=15,
        )

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=model_color,
                linewidth=model_linewidth,
                label=model_label,
            ),
            plt.Line2D(
                [0],
                [0],
                color=dft_color,
                linewidth=dft_linewidth,
                linestyle=(0, (5, 3)),
                label=dft_label,
            ),
        ]

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.59, 0.995),
            ncol=2,
            frameon=False,
            fontsize=15,
            handlelength=2.7,
            handletextpad=0.7,
            columnspacing=1.5,
        )

        fig.text(
            0.015,
            0.985,
            "d",
            fontsize=27,
            fontweight="normal",
            va="top",
            ha="left",
        )

        fig.subplots_adjust(
            left=0.225,
            right=0.975,
            bottom=0.105,
            top=0.895,
            hspace=0.16,
        )

        output_paths = {
            "png":
                output_dir
                / f"{save_stem}.png",
            "pdf":
                output_dir
                / f"{save_stem}.pdf",
            "svg":
                output_dir
                / f"{save_stem}.svg",
            "metrics":
                output_dir
                / f"{save_stem}_metrics.csv",
        }

        metrics.to_csv(
            output_paths["metrics"],
            index=False,
            encoding="utf-8-sig",
        )

        fig.savefig(
            output_paths["png"],
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            output_paths["pdf"],
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            output_paths["svg"],
            bbox_inches="tight",
            facecolor="white",
        )

        if show:
            plt.show()

    print("\n" + "=" * 90)
    print("FINAL FIGURE SAVED")
    print("=" * 90)
    print(
        "PNG     :",
        output_paths["png"],
    )
    print(
        "PDF     :",
        output_paths["pdf"],
    )
    print(
        "SVG     :",
        output_paths["svg"],
    )
    print(
        "Metrics :",
        output_paths["metrics"],
    )

    return {
        "metrics": metrics,
        "plot_data": plot_data,
        "energy_window": energy_window,
        "figure": fig,
        "axes": axes,
        "output_paths": output_paths,
    }


def run_dft_validation_figure(
    project_root=r"D:\TB3",
    dft_source=r"D:\DFT_TEST\collect_data_epsilon.zip",
    materials=DEFAULT_VALIDATION_MATERIALS,
    ml_npz=None,
    output_dir=None,
    show=True,
):
    """
    Reconstruct validation inputs and generate the final 0-4 eV figure.
    """
    inputs = build_dft_validation_inputs(
        project_root=project_root,
        dft_source=dft_source,
        materials=materials,
        ml_npz=ml_npz,
        output_dir=output_dir,
    )

    result = plot_dft_validation_0_4ev(
        energy_axis=inputs["energy_axis"],
        records=inputs["records"],
        output_dir=inputs["output_dir"],
        materials=materials,
        show=show,
    )

    result["inputs"] = inputs

    return result
