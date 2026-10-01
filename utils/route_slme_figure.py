"""Held-out SLME route-comparison figure for the paper SI."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_route_slme_comparison(project_root='D:\\TB3', *, input_csv=None, output_dir=None, show=True):
    """Plot held-out Route 1/2 from the existing common-valid SLME CSV.

Preserves the original filtering, statistics, figure design and filenames.
Does not recalculate SLME or modify the input data. By default, saves under
project_root/figures rather than the notebook process working directory.
Existing files with the same names are overwritten, as in the original cell.
"""
    if output_dir is None:
        output_dir = Path(project_root) / 'figures'
    INPUT_CSV = Path(input_csv) if input_csv is not None else Path(project_root) / 'processed/paired_training/paper_outputs/FINAL_alpha_route_SLME_comparison_500nm/FINAL_SLME_route1_vs_route2_samples.csv'
    OUTPUT_PDF = Path(output_dir) / 'S5_route1_vs_route2_SLME.pdf'
    OUTPUT_PNG = Path(output_dir) / 'S5_route1_vs_route2_SLME.png'
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    def resolve_column(df, accepted_names, description):
        cols_lower = {c.lower(): c for c in df.columns}
        found = []
        for name in accepted_names:
            if name.lower() in cols_lower:
                found.append(cols_lower[name.lower()])
        found = list(dict.fromkeys(found))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise RuntimeError(f'Có nhiều cột phù hợp cho {description}: {found}\nCần chọn thủ công để tránh nhầm dữ liệu.')
        raise KeyError(f'Không tìm thấy cột {description}.\nĐã thử: {accepted_names}\n\nColumns thực tế:\n{list(df.columns)}')

    def r2_score_manual(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return np.nan
        return 1.0 - ss_res / ss_tot

    def mae_manual(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean(np.abs(y_true - y_pred))

    def pearson_r_manual(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) < 2:
            return np.nan
        return np.corrcoef(y_true, y_pred)[0, 1]
    df = pd.read_csv(INPUT_CSV)
    REF_COL = resolve_column(df, ['reference_SLME_percent', 'Reference_SLME_percent', 'reference_slme_percent', 'reference_slme'], 'Reference SLME')
    ROUTE1_COL = resolve_column(df, ['route1_derived_alpha_SLME_percent', 'route1_SLME_percent', 'Route1_SLME_percent', 'derived_alpha_SLME_percent'], 'Route 1 SLME')
    ROUTE2_COL = resolve_column(df, ['route2_direct_alpha_SLME_percent', 'route2_SLME_percent', 'Route2_SLME_percent', 'direct_alpha_SLME_percent'], 'Route 2 SLME')
    if 'common_valid' in df.columns:
        mask = df['common_valid'].astype(bool)
        df = df.loc[mask].copy()
    df = df[[REF_COL, ROUTE1_COL, ROUTE2_COL]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y_ref = df[REF_COL].to_numpy()
    y_r1 = df[ROUTE1_COL].to_numpy()
    y_r2 = df[ROUTE2_COL].to_numpy()
    r2_r1 = r2_score_manual(y_ref, y_r1)
    mae_r1 = mae_manual(y_ref, y_r1)
    pr_r1 = pearson_r_manual(y_ref, y_r1)
    r2_r2 = r2_score_manual(y_ref, y_r2)
    mae_r2 = mae_manual(y_ref, y_r2)
    pr_r2 = pearson_r_manual(y_ref, y_r2)
    n = len(df)
    all_vals = np.concatenate([y_ref, y_r1, y_r2])
    xmin = max(0.0, np.floor(all_vals.min()))
    xmax = np.ceil(all_vals.max())
    pad = 0.5
    xmin -= pad
    xmax += pad
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(y_ref, y_r1, s=30, alpha=0.8)
    ax.plot([xmin, xmax], [xmin, xmax], '--', linewidth=1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel('Reference SLME, $\\eta$ (%)')
    ax.set_ylabel('Predicted SLME, $\\eta$ (%)')
    ax.set_title('(a) Route 1: $\\varepsilon_1,\\varepsilon_2 \\rightarrow \\alpha \\rightarrow \\mathrm{SLME}$')
    ax.text(0.06, 0.95, f'$R^2 = {r2_r1:.3f}$\nMAE = {mae_r1:.2f} pp\n$r = {pr_r1:.3f}$\n$n = {n}$', transform=ax.transAxes, ha='left', va='top')
    ax.text(0.95, 0.05, '$L = 500$ nm', transform=ax.transAxes, ha='right', va='bottom')
    ax = axes[1]
    ax.scatter(y_ref, y_r2, s=30, alpha=0.8)
    ax.plot([xmin, xmax], [xmin, xmax], '--', linewidth=1.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_xlabel('Reference SLME, $\\eta$ (%)')
    ax.set_title('(b) Route 2: direct $\\alpha \\rightarrow \\mathrm{SLME}$')
    ax.text(0.06, 0.95, f'$R^2 = {r2_r2:.3f}$\nMAE = {mae_r2:.2f} pp\n$r = {pr_r2:.3f}$\n$n = {n}$', transform=ax.transAxes, ha='left', va='top')
    ax.text(0.95, 0.05, '$L = 500$ nm', transform=ax.transAxes, ha='right', va='bottom')
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches='tight')
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    print('Saved PDF:', OUTPUT_PDF.resolve())
    print('Saved PNG:', OUTPUT_PNG.resolve())
    print()
    print('Route 1:', f'R^2={r2_r1:.6f}, MAE={mae_r1:.6f} pp, r={pr_r1:.6f}, n={n}')
    print('Route 2:', f'R^2={r2_r2:.6f}, MAE={mae_r2:.6f} pp, r={pr_r2:.6f}, n={n}')
    return {'figure': fig, 'paths': {'pdf': OUTPUT_PDF.resolve(), 'png': OUTPUT_PNG.resolve()}, 'samples': df, 'metrics': pd.DataFrame([{'route': 'Route 1', 'n': n, 'R2': r2_r1, 'MAE_pp': mae_r1, 'Pearson_r': pr_r1}, {'route': 'Route 2', 'n': n, 'R2': r2_r2, 'MAE_pp': mae_r2, 'Pearson_r': pr_r2}])}
