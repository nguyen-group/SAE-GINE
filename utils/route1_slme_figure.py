"""Route-1 held-out SLME figure extracted from the paper notebook."""
import builtins
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_heldout_route1_slme(project_root='D:\\TB3', *, result_dir=None, output_dir=None, show=True, verbose=True):
    """Render the existing Route-1 held-out SLME parity figure and export metrics.

Preserves the original CSV selection, common_valid mask, ID checks, stable
Top-k calculation, visual style and output names. SLME is read from CSV,
not recalculated. The supplied cell draws a single 90 mm parity panel;
Top-k results are reported and saved as a table, not drawn as a second panel.
Each run creates a new timestamped output directory unless output_dir is given.
An explicit output_dir must not already exist.
"""

    def print(*args, **kwargs):
        if verbose:
            builtins.print(*args, **kwargs)
    PROJECT_ROOT = Path(project_root)
    RESULT_DIR = Path(result_dir) if result_dir is not None else PROJECT_ROOT / 'processed/paired_training/paper_outputs/FINAL_alpha_route_SLME_comparison_500nm'
    if not RESULT_DIR.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục:\n{RESULT_DIR}')
    preferred_names = ['FINAL_SLME_route1_vs_route2_samples.csv', 'FINAL_route1_vs_route2_SLME_samples.csv', 'route1_vs_route2_samples.csv']
    CSV_PATH = None
    for name in preferred_names:
        p = RESULT_DIR / name
        if p.is_file():
            CSV_PATH = p
            break
    if CSV_PATH is None:
        candidates = sorted([p for p in RESULT_DIR.glob('*.csv') if 'sample' in p.name.lower()])
        preferred = [p for p in candidates if 'route1' in p.name.lower() and 'route2' in p.name.lower()]
        if len(preferred) == 1:
            CSV_PATH = preferred[0]
        elif len(candidates) == 1:
            CSV_PATH = candidates[0]
    if CSV_PATH is None:
        print('Các CSV hiện có:')
        for p in sorted(RESULT_DIR.glob('*.csv')):
            print(' -', p.name)
        raise FileNotFoundError('Không xác định được per-sample CSV.')
    print('=' * 100)
    print('FINAL HELD-OUT ROUTE-1 SLME')
    print('=' * 100)
    print('Source CSV:')
    print(CSV_PATH)
    df_all = pd.read_csv(CSV_PATH)
    print()
    print('Rows:', len(df_all))
    print('Columns:')
    for col in df_all.columns:
        print(' -', col)
    ID_COL = 'base_idx'
    REF_COL = 'reference_SLME_percent'
    ROUTE1_COL = 'route1_derived_alpha_SLME_percent'
    ROUTE2_COL = 'route2_direct_alpha_SLME_percent'
    COMMON_VALID_COL = 'common_valid'
    required_columns = [ID_COL, REF_COL, ROUTE1_COL, COMMON_VALID_COL]
    missing = [col for col in required_columns if col not in df_all.columns]
    if missing:
        raise KeyError(f'Thiếu các cột bắt buộc: {missing}\nColumns hiện có:\n{list(df_all.columns)}')
    print()
    print('Column audit:')
    print(' ID        :', ID_COL)
    print(' Reference :', REF_COL)
    print(' Route 1   :', ROUTE1_COL)
    print(' Validity  :', COMMON_VALID_COL)
    if ROUTE2_COL in df_all.columns:
        print(' Route 2   :', ROUTE2_COL, '(present but NOT used)')
    if 'route2' in ROUTE1_COL.lower() or 'direct' in ROUTE1_COL.lower():
        raise RuntimeError('STOP: Route-1 column points to Direct-alpha / Route 2.')
    EXPECTED_TOTAL = 992
    THICKNESS_NM = 500.0
    if len(df_all) != EXPECTED_TOTAL:
        raise ValueError(f'Expected {EXPECTED_TOTAL} held-out rows, found {len(df_all)}.')
    if df_all[ID_COL].isna().any():
        raise ValueError(f'{ID_COL} contains missing values.')
    if df_all[ID_COL].duplicated().any():
        raise ValueError(f'{ID_COL} contains duplicates.')

    def parse_bool_series(series):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).astype(bool).to_numpy()
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.isna().any():
                raise ValueError('common_valid contains invalid numeric values.')
            unique = set(numeric.unique())
            if not unique.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f'Unexpected common_valid values: {unique}')
            return numeric.astype(int).astype(bool).to_numpy()
        mapping = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False}
        parsed = series.astype(str).str.strip().str.lower().map(mapping)
        if parsed.isna().any():
            bad = series[parsed.isna()].astype(str).unique()
            raise ValueError(f'Không nhận diện được common_valid: {bad[:10]}')
        return parsed.astype(bool).to_numpy()
    common_valid = parse_bool_series(df_all[COMMON_VALID_COL])
    reference_all = pd.to_numeric(df_all[REF_COL], errors='coerce').to_numpy(dtype=float)
    route1_all = pd.to_numeric(df_all[ROUTE1_COL], errors='coerce').to_numpy(dtype=float)
    finite_mask = np.isfinite(reference_all) & np.isfinite(route1_all)
    valid_mask = common_valid & finite_mask
    plot_df = df_all.loc[valid_mask].copy()
    excluded_df = df_all.loc[~valid_mask].copy()
    reference = reference_all[valid_mask]
    prediction = route1_all[valid_mask]
    n_total = len(df_all)
    n_valid = len(plot_df)
    n_excluded = len(excluded_df)
    print()
    print('=' * 100)
    print('FROZEN COHORT AUDIT')
    print('=' * 100)
    print('Held-out total       :', n_total)
    print('common_valid=True    :', int(common_valid.sum()))
    print('Finite ref + Route1  :', int(finite_mask.sum()))
    print('Final evaluation n   :', n_valid)
    print('Excluded             :', n_excluded)
    if n_valid < 100:
        raise ValueError('Không đủ 100 valid pairs để tính Top-100.')
    all_values = np.r_[reference, prediction]
    if not np.isfinite(all_values).all():
        raise RuntimeError('Non-finite values remain after filtering.')
    if all_values.min() < 0:
        raise ValueError('Có SLME âm.')
    if all_values.max() > 35:
        raise ValueError('Có SLME > 35%; cần mở rộng figure axis.')
    error = prediction - reference
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((reference - reference.mean()) ** 2))
    if ss_tot <= 0:
        raise RuntimeError('Cannot calculate R²: zero reference variance.')
    r2 = float(1.0 - ss_res / ss_tot)
    pearson = float(np.corrcoef(reference, prediction)[0, 1])
    top_k = np.array([10, 20, 50, 100], dtype=int)
    reference_order = np.argsort(-reference, kind='mergesort')
    prediction_order = np.argsort(-prediction, kind='mergesort')
    overlap_count = np.array([len(set(reference_order[:k]) & set(prediction_order[:k])) for k in top_k], dtype=int)
    overlap_percent = 100.0 * overlap_count / top_k
    topk_df = pd.DataFrame({'k': top_k, 'overlap_count': overlap_count, 'overlap_percent': overlap_percent})
    metrics_df = pd.DataFrame([{'route': 'SAE-eps1 + SAE-eps2 -> alpha_derived -> SLME', 'n_heldout_total': n_total, 'n_common_valid': int(common_valid.sum()), 'n_finite_pairs': n_valid, 'n_excluded': n_excluded, 'thickness_nm': THICKNESS_NM, 'R2': r2, 'Pearson_r': pearson, 'MAE_pp': mae, 'RMSE_pp': rmse, 'Bias_pp': bias}])
    print()
    print('=' * 100)
    print('ROUTE-1 HELD-OUT METRICS')
    print('=' * 100)
    print(f'R²      = {r2:.9f}')
    print(f'Pearson = {pearson:.9f}')
    print(f'MAE     = {mae:.9f} pp')
    print(f'RMSE    = {rmse:.9f} pp')
    print(f'Bias    = {bias:+.9f} pp')
    print()
    print('Top-k overlap:')
    print(topk_df.to_string(index=False))
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
        OUTPUT_DIR = Path(output_dir) if output_dir is not None else RESULT_DIR / ('figure_NC_ROUTE1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        figure_stem = OUTPUT_DIR / 'Heldout_ROUTE1_SLME_500nm_parity_topk'
        output_paths = {}
        for extension in ('pdf', 'svg', 'png'):
            path = figure_stem.with_suffix(f'.{extension}')
            fig.savefig(path, dpi=600, bbox_inches=None, facecolor='white', transparent=False)
            output_paths[extension] = path
        plotted_df = pd.DataFrame({'base_idx': plot_df[ID_COL].to_numpy(), 'reference_SLME_percent': reference, 'route1_derived_alpha_SLME_percent': prediction, 'route1_error_pp': error, 'route1_abs_error_pp': abs_error})
        plotted_df.to_csv(OUTPUT_DIR / 'plotted_ROUTE1_finite_pairs.csv', index=False, encoding='utf-8-sig')
        excluded_df.to_csv(OUTPUT_DIR / 'excluded_ROUTE1_pairs.csv', index=False, encoding='utf-8-sig')
        metrics_df.to_csv(OUTPUT_DIR / 'figure_ROUTE1_metrics.csv', index=False, encoding='utf-8-sig')
        topk_df.to_csv(OUTPUT_DIR / 'route1_topk_overlap.csv', index=False, encoding='utf-8-sig')
        if show:
            plt.show()
        plt.close(fig)
    print()
    print('=' * 100)
    print('FINAL HELD-OUT ROUTE-1 SLME FIGURE')
    print('=' * 100)
    print('Route       : SAE-eps1 + SAE-eps2 -> alpha_derived -> SLME')
    print('Route1 col  :', ROUTE1_COL)
    print('Route2 used : NO')
    print(f'Source      : {CSV_PATH}')
    print(f'Total       : {n_total}')
    print(f'Finite      : {n_valid}')
    print(f'Excluded    : {n_excluded}')
    print(f'R²          : {r2:.9f}')
    print(f'Pearson     : {pearson:.9f}')
    print(f'MAE         : {mae:.9f} pp')
    print(f'RMSE        : {rmse:.9f} pp')
    print(f'Bias        : {bias:+.9f} pp')
    print()
    print('Top-k overlap:')
    print(topk_df.to_string(index=False))
    print()
    print('Hình đã lưu:')
    for extension, path in output_paths.items():
        print(f'{extension.upper():4s}: {path}')
    print()
    print('PASS: held-out figure uses ONLY reference_SLME_percent + route1_derived_alpha_SLME_percent on the frozen common_valid cohort.')
    return {'metrics': metrics_df, 'topk': topk_df, 'samples': plotted_df, 'excluded': excluded_df, 'output_dir': OUTPUT_DIR, 'paths': output_paths, 'input_csv': CSV_PATH}
