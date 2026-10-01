"""Single-panel direct-alpha SLME figure."""
import builtins
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_direct_slme_parity(project_root='D:\\TB3', *, input_csv=None, output_dir=None, show=True, verbose=True):
    """Plot only the direct-alpha held-out SLME parity panel at 90 x 90 mm.

Reads the existing SLME table, preserving its finite-pair filter and metrics.
Does not recompute SLME or change the cohort. Top-k remains in the exported
CSV for reproducibility but is not drawn. Each run creates a new folder.
"""

    def print(*args, **kwargs):
        if verbose:
            builtins.print(*args, **kwargs)
    PROJECT_ROOT = Path(project_root)
    RESULT_DIR = PROJECT_ROOT / 'processed' / 'paired_training' / 'paper_outputs' / 'FINAL_direct_alpha_SLME_500nm'
    CSV_PATH = Path(input_csv) if input_csv is not None else RESULT_DIR / 'FINAL_direct_alpha_SLME_test_samples.csv'
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f'Không tìm thấy bảng dữ liệu:\n{CSV_PATH}')
    df_all = pd.read_csv(CSV_PATH)
    required_columns = ['sample_id', 'reference_SLME_percent', 'predicted_SLME_percent', 'thickness_nm']
    missing = [column for column in required_columns if column not in df_all]
    if missing:
        raise KeyError(f'Thiếu các cột: {missing}')
    if len(df_all) != 992:
        raise ValueError(f'Bảng held-out dự kiến có 992 dòng, hiện có {len(df_all)}.')
    if df_all['sample_id'].isna().any():
        raise ValueError('Có sample_id bị thiếu.')
    if df_all['sample_id'].duplicated().any():
        raise ValueError('Có sample_id bị trùng.')
    if not np.allclose(df_all['thickness_nm'].to_numpy(dtype=float), 500.0):
        raise ValueError('Dữ liệu không đồng nhất ở độ dày 500 nm.')
    slme_columns = ['reference_SLME_percent', 'predicted_SLME_percent']
    all_values = df_all[slme_columns].to_numpy(dtype=float)
    valid_mask = np.isfinite(all_values).all(axis=1)
    excluded_df = df_all.loc[~valid_mask].copy()
    plot_df = df_all.loc[valid_mask].copy()
    n_total = len(df_all)
    n_valid = len(plot_df)
    n_excluded = len(excluded_df)
    if n_valid < 100:
        raise ValueError('Không đủ 100 cặp SLME hữu hạn để đánh giá Top-100.')
    reference = plot_df['reference_SLME_percent'].to_numpy(dtype=float)
    prediction = plot_df['predicted_SLME_percent'].to_numpy(dtype=float)
    if min(reference.min(), prediction.min()) < 0 or max(reference.max(), prediction.max()) > 35:
        raise ValueError('Có SLME ngoài khoảng 0–35%. Cần mở rộng trục trước khi vẽ.')
    error = prediction - reference
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((reference - reference.mean()) ** 2))
    if ss_tot <= 0:
        raise ValueError('Không thể tính R² vì SLME tham chiếu không có phương sai.')
    r2 = 1.0 - ss_res / ss_tot
    top_k = np.array([10, 20, 50, 100], dtype=int)
    reference_order = np.argsort(-reference, kind='mergesort')
    prediction_order = np.argsort(-prediction, kind='mergesort')
    overlap_count = np.array([len(set(reference_order[:k]) & set(prediction_order[:k])) for k in top_k], dtype=int)
    overlap_percent = 100.0 * overlap_count / top_k
    topk_df = pd.DataFrame({'k': top_k, 'overlap_count': overlap_count, 'overlap_percent': overlap_percent})
    metrics_df = pd.DataFrame([{'n_heldout_total': n_total, 'n_finite_pairs': n_valid, 'n_excluded_nonfinite': n_excluded, 'thickness_nm': 500, 'R2': r2, 'MAE_pp': mae, 'RMSE_pp': rmse}])
    BLUE = '#2878B5'
    GRAY = '#666666'
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 1.0, 'axes.labelpad': 5, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.direction': 'out', 'ytick.direction': 'out', 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'text.color': 'black', 'axes.labelcolor': 'black', 'figure.autolayout': False, 'figure.constrained_layout.use': False, 'savefig.bbox': None}
    with plt.rc_context(style):
        fig, ax_a = plt.subplots(1, 1, figsize=(90 / 25.4, 90 / 25.4), dpi=160)
        fig.subplots_adjust(left=0.2, right=0.97, bottom=0.18, top=0.95)
        ax_a.plot([0, 35], [0, 35], color=GRAY, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax_a.scatter(reference, prediction, s=12, color=BLUE, alpha=0.75, edgecolors='none', zorder=2)
        ax_a.set(xlim=(0, 35), ylim=(0, 35), xticks=np.arange(0, 36, 5), yticks=np.arange(0, 36, 5), xlabel='Reference SLME, $\\eta$ (%)', ylabel='Predicted SLME, $\\eta$ (%)')
        ax_a.set_box_aspect(1)
        ax_a.text(0.06, 0.95, f'$R^2$ = {r2:.3f}\nMAE = {mae:.2f} pp\n$n$ = {n_valid}', transform=ax_a.transAxes, ha='left', va='top', fontsize=9, linespacing=1.45, zorder=3)
        ax_a.text(0.95, 0.055, '$L$ = 500 nm', transform=ax_a.transAxes, ha='right', va='bottom', fontsize=9, color='#333333')
        ax_a.grid(False)
        ax_a.tick_params(axis='both', which='major', top=False, right=False, pad=3)
        ax_a.minorticks_off()
        for spine in ax_a.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        OUTPUT_DIR = Path(output_dir) if output_dir is not None else RESULT_DIR / f'figure_parity_only_{datetime.now():%Y%m%d_%H%M%S_%f}'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        figure_stem = OUTPUT_DIR / 'Heldout_SLME_500nm_parity'
        output_paths = {}
        for extension in ('pdf', 'svg', 'png'):
            path = figure_stem.with_suffix(f'.{extension}')
            fig.savefig(path, dpi=600, bbox_inches=None, facecolor='white', transparent=False)
            output_paths[extension] = path
        plot_df.to_csv(OUTPUT_DIR / 'plotted_finite_pairs.csv', index=False)
        excluded_df.to_csv(OUTPUT_DIR / 'excluded_nonfinite_pairs.csv', index=False)
        metrics_df.to_csv(OUTPUT_DIR / 'figure_metrics.csv', index=False)
        topk_df.to_csv(OUTPUT_DIR / 'topk_overlap.csv', index=False)
        if show:
            plt.show()
        plt.close(fig)
    print(f'Nguồn: {CSV_PATH}')
    print(f'Tổng số mẫu held-out: {n_total}')
    print(f'Cặp SLME hữu hạn dùng cho hình và metrics: {n_valid}')
    print(f'Mẫu không đưa vào hình và metrics: {n_excluded}')
    print(f'R² = {r2:.6f}')
    print(f'MAE = {mae:.6f} pp | RMSE = {rmse:.6f} pp')
    print('\nTop-k overlap:')
    print(topk_df.to_string(index=False))
    print('\nHình đã lưu:')
    for extension, path in output_paths.items():
        print(f'{extension.upper()}: {path}')
    print('Hình parity duy nhất, khổ 90 x 90 mm; ưu tiên PDF vector.')
    return {'metrics': metrics_df, 'topk': topk_df, 'samples': plot_df, 'excluded': excluded_df, 'paths': output_paths, 'output_dir': OUTPUT_DIR}
