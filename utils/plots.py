"""
Plotting and visualization utilities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Representative spectrum plots extracted from TASK3 notebook
# ------------------------------------------------------------
def plot_pred_true_cases(
    npz_path,
    main_title,
    y_label,
    case_mode="paper",   # "paper", "quantile", "best_median_worst"
    use_mask=True,
    save_path=None,
):
    data = np.load(npz_path)

    pred = data["pred"]
    true = data["true"]
    mask = data["mask"] if "mask" in data else np.ones_like(true)

    axis = data["spectral_axis"] if "spectral_axis" in data else data["axis_grid"]
    axis = axis[0] if axis.ndim > 1 else axis

    if use_mask:
        mae_each = (np.abs(pred - true) * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-8, None)
    else:
        mae_each = np.mean(np.abs(pred - true), axis=1)

    order = np.argsort(mae_each)

    if case_mode == "paper":
        qs = [10, 50, 90]
        titles = ["Good", "Median", "Challenging"]
        idxs = [order[int(q / 100 * (len(order) - 1))] for q in qs]

    elif case_mode == "best_median_worst":
        idxs = [order[0], order[len(order) // 2], order[-1]]
        titles = ["Best", "Median", "Worst"]

    else:
        qs = [5, 25, 50, 75, 90]
        idxs = [order[int(q / 100 * (len(order) - 1))] for q in qs]
        titles = [f"Q{q}" for q in qs]

    fig, axes = plt.subplots(
        1,
        len(idxs),
        figsize=(3.8 * len(idxs), 3.2),
        dpi=300,
        sharex=True,
    )

    if len(idxs) == 1:
        axes = [axes]

    for ax, idx, title in zip(axes, idxs, titles):
        valid = mask[idx].astype(bool) if use_mask else np.ones_like(true[idx], dtype=bool)

        ax.plot(axis[valid], true[idx][valid], lw=2.0, label="Reference")
        ax.plot(axis[valid], pred[idx][valid], lw=2.0, ls="--", label="Prediction")

        ax.set_title(f"{title}\nMAE = {mae_each[idx]:.3f}", fontsize=11)
        ax.set_xlabel("Energy (eV)", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel(y_label, fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

        print("Saved:", save_path)
        print("Saved:", save_path.with_suffix(".pdf"))

    plt.show()
    return mae_each, idxs


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
