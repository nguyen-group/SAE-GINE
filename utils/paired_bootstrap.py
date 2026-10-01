"""Paired SLME bootstrap extracted from the final paper notebook."""
import builtins
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

def run_paired_slme_bootstrap(project_root='D:\\TB3', *, input_csv=None, output_dir=None, n_boot=20000, seed=2025, batch_size=250, top_k=(10, 20, 50, 100), show=True):
    """Compare frozen held-out Route 1/2 SLME on the CSV common_valid cohort.

Defaults preserve the original cell: 20,000 paired draws, seed 2025,
batch size 250, stable ranking, percentile intervals and exact paired tests.
Top-k bootstrap conditions on the existing reference Top-k and fixed rankings;
it does not rerank a bootstrapped full candidate pool. Bootstrap win fractions
are not p-values, and a confidence interval crossing zero is not equivalence.
This function does not recompute SLME or change the input validity mask.
By default, the same three output filenames are overwritten as in the cell.
The return dictionary exposes the tables, distributions and output paths.
"""
    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError('n_boot must be a positive integer')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if any((not isinstance(k, int) or k <= 0 for k in top_k)):
        raise ValueError('top_k must contain positive integers')

    def print(*args, **kwargs):
        if show:
            builtins.print(*args, **kwargs)
    ROOT = Path(project_root) / 'processed' / 'paired_training'
    INPUT_CSV = Path(input_csv) if input_csv is not None else ROOT / 'paper_outputs' / 'FINAL_alpha_route_SLME_comparison_500nm' / 'FINAL_SLME_route1_vs_route2_samples.csv'
    OUT_DIR = Path(output_dir) if output_dir is not None else INPUT_CSV.parent / 'paired_bootstrap'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    N_BOOT = n_boot
    BOOTSTRAP_SEED = seed
    BATCH_SIZE = batch_size
    TOP_K_VALUES = list(top_k)
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f'Không tìm thấy:\n{INPUT_CSV}')
    df = pd.read_csv(INPUT_CSV)
    required_columns = ['base_idx', 'reference_SLME_percent', 'route1_derived_alpha_SLME_percent', 'route2_direct_alpha_SLME_percent', 'common_valid']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f'Thiếu columns: {missing}\nAvailable: {list(df.columns)}')
    if df['common_valid'].dtype == bool:
        common_mask = df['common_valid'].to_numpy()
    else:
        common_mask = df['common_valid'].astype(str).str.lower().isin(['true', '1', 'yes']).to_numpy()
    ref_all = df['reference_SLME_percent'].to_numpy(dtype=np.float64)
    r1_all = df['route1_derived_alpha_SLME_percent'].to_numpy(dtype=np.float64)
    r2_all = df['route2_direct_alpha_SLME_percent'].to_numpy(dtype=np.float64)
    common_mask = common_mask & np.isfinite(ref_all) & np.isfinite(r1_all) & np.isfinite(r2_all)
    work = df.loc[common_mask].copy().reset_index(drop=True)
    REF = work['reference_SLME_percent'].to_numpy(dtype=np.float64)
    R1 = work['route1_derived_alpha_SLME_percent'].to_numpy(dtype=np.float64)
    R2 = work['route2_direct_alpha_SLME_percent'].to_numpy(dtype=np.float64)
    BASE_IDX = work['base_idx'].to_numpy()
    N = len(REF)
    print('=' * 105)
    print('PAIRED BOOTSTRAP INPUT')
    print('=' * 105)
    print(f'Input CSV           : {INPUT_CSV}')
    print(f'Held-out total      : {len(df)}')
    print(f'Common finite pairs : {N}')
    print(f'Bootstrap replicates: {N_BOOT:,}')
    print(f'Bootstrap seed      : {BOOTSTRAP_SEED}')
    if N != 853:
        print(f'WARNING: expected previous common set = 853, current = {N}')

    def r2_score_manual(reference, prediction):
        reference = np.asarray(reference, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        ss_res = np.sum((prediction - reference) ** 2)
        ss_tot = np.sum((reference - np.mean(reference)) ** 2)
        if ss_tot <= 0.0:
            return np.nan
        return float(1.0 - ss_res / ss_tot)

    def full_metrics(reference, prediction):
        error = prediction - reference
        return {'R2': r2_score_manual(reference, prediction), 'MAE_pp': float(np.mean(np.abs(error))), 'RMSE_pp': float(np.sqrt(np.mean(error ** 2)))}
    FULL_R1 = full_metrics(REF, R1)
    FULL_R2 = full_metrics(REF, R2)
    DELTA_R2_OBS = FULL_R2['R2'] - FULL_R1['R2']
    DELTA_MAE_OBS = FULL_R2['MAE_pp'] - FULL_R1['MAE_pp']
    DELTA_RMSE_OBS = FULL_R2['RMSE_pp'] - FULL_R1['RMSE_pp']
    print()
    print('=' * 105)
    print('OBSERVED FULL-SAMPLE METRICS')
    print('=' * 105)
    print(f"Route 1 R²   = {FULL_R1['R2']:.9f}")
    print(f"Route 2 R²   = {FULL_R2['R2']:.9f}")
    print(f'ΔR² (2 - 1) = {DELTA_R2_OBS:+.9f}')
    print()
    print(f"Route 1 MAE  = {FULL_R1['MAE_pp']:.9f} pp")
    print(f"Route 2 MAE  = {FULL_R2['MAE_pp']:.9f} pp")
    print(f'ΔMAE (2 - 1)= {DELTA_MAE_OBS:+.9f} pp')
    print()
    print(f"Route 1 RMSE = {FULL_R1['RMSE_pp']:.9f} pp")
    print(f"Route 2 RMSE = {FULL_R2['RMSE_pp']:.9f} pp")
    print(f'ΔRMSE (2-1) = {DELTA_RMSE_OBS:+.9f} pp')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_delta_r2 = np.full(N_BOOT, np.nan, dtype=np.float64)
    boot_delta_mae = np.full(N_BOOT, np.nan, dtype=np.float64)
    boot_delta_rmse = np.full(N_BOOT, np.nan, dtype=np.float64)
    position = 0
    while position < N_BOOT:
        m = min(BATCH_SIZE, N_BOOT - position)
        idx = rng.integers(low=0, high=N, size=(m, N), dtype=np.int32)
        ref_b = REF[idx]
        r1_b = R1[idx]
        r2_b = R2[idx]
        e1 = r1_b - ref_b
        e2 = r2_b - ref_b
        mae1 = np.mean(np.abs(e1), axis=1)
        mae2 = np.mean(np.abs(e2), axis=1)
        rmse1 = np.sqrt(np.mean(e1 ** 2, axis=1))
        rmse2 = np.sqrt(np.mean(e2 ** 2, axis=1))
        ref_mean = np.mean(ref_b, axis=1, keepdims=True)
        ss_tot = np.sum((ref_b - ref_mean) ** 2, axis=1)
        ss_res1 = np.sum(e1 ** 2, axis=1)
        ss_res2 = np.sum(e2 ** 2, axis=1)
        r2_1 = 1.0 - ss_res1 / ss_tot
        r2_2 = 1.0 - ss_res2 / ss_tot
        sl = slice(position, position + m)
        boot_delta_r2[sl] = r2_2 - r2_1
        boot_delta_mae[sl] = mae2 - mae1
        boot_delta_rmse[sl] = rmse2 - rmse1
        position += m

    def percentile_ci(values, confidence=0.95):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        alpha = 1.0 - confidence
        lower = np.quantile(values, alpha / 2.0)
        median = np.quantile(values, 0.5)
        upper = np.quantile(values, 1.0 - alpha / 2.0)
        return (float(lower), float(median), float(upper))
    R2_CI = percentile_ci(boot_delta_r2)
    MAE_CI = percentile_ci(boot_delta_mae)
    RMSE_CI = percentile_ci(boot_delta_rmse)
    P_R2_ROUTE2_BETTER = float(np.mean(boot_delta_r2 > 0.0))
    P_MAE_ROUTE2_BETTER = float(np.mean(boot_delta_mae < 0.0))
    P_RMSE_ROUTE2_BETTER = float(np.mean(boot_delta_rmse < 0.0))

    def interpret_delta(ci, metric_type):
        low, _, high = ci
        if metric_type == 'higher':
            if low > 0.0:
                return 'Route 2 higher; CI excludes 0'
            elif high < 0.0:
                return 'Route 1 higher; CI excludes 0'
            else:
                return 'No clear difference; 95% CI crosses 0'
        elif metric_type == 'lower':
            if high < 0.0:
                return 'Route 2 lower error; CI excludes 0'
            elif low > 0.0:
                return 'Route 1 lower error; CI excludes 0'
            else:
                return 'No clear difference; 95% CI crosses 0'
        raise ValueError(metric_type)
    bootstrap_metric_df = pd.DataFrame([{'Metric': 'R2', 'Route1': FULL_R1['R2'], 'Route2': FULL_R2['R2'], 'Delta_Route2_minus_Route1': DELTA_R2_OBS, 'Bootstrap_median_delta': R2_CI[1], 'CI95_lower': R2_CI[0], 'CI95_upper': R2_CI[2], 'P_bootstrap_Route2_better': P_R2_ROUTE2_BETTER, 'Interpretation': interpret_delta(R2_CI, 'higher')}, {'Metric': 'MAE_pp', 'Route1': FULL_R1['MAE_pp'], 'Route2': FULL_R2['MAE_pp'], 'Delta_Route2_minus_Route1': DELTA_MAE_OBS, 'Bootstrap_median_delta': MAE_CI[1], 'CI95_lower': MAE_CI[0], 'CI95_upper': MAE_CI[2], 'P_bootstrap_Route2_better': P_MAE_ROUTE2_BETTER, 'Interpretation': interpret_delta(MAE_CI, 'lower')}, {'Metric': 'RMSE_pp', 'Route1': FULL_R1['RMSE_pp'], 'Route2': FULL_R2['RMSE_pp'], 'Delta_Route2_minus_Route1': DELTA_RMSE_OBS, 'Bootstrap_median_delta': RMSE_CI[1], 'CI95_lower': RMSE_CI[0], 'CI95_upper': RMSE_CI[2], 'P_bootstrap_Route2_better': P_RMSE_ROUTE2_BETTER, 'Interpretation': interpret_delta(RMSE_CI, 'lower')}])
    reference_order = np.argsort(-REF, kind='mergesort')
    route1_order = np.argsort(-R1, kind='mergesort')
    route2_order = np.argsort(-R2, kind='mergesort')
    topk_rows = []
    topk_boot_arrays = {}
    for k in TOP_K_VALUES:
        ref_top = reference_order[:k]
        route1_top = set(route1_order[:k])
        route2_top = set(route2_order[:k])
        retained1 = np.array([int(idx in route1_top) for idx in ref_top], dtype=np.int8)
        retained2 = np.array([int(idx in route2_top) for idx in ref_top], dtype=np.int8)
        count1 = int(retained1.sum())
        count2 = int(retained2.sum())
        percent1 = 100.0 * count1 / k
        percent2 = 100.0 * count2 / k
        observed_delta = percent2 - percent1
        delta_item = retained2.astype(np.float64) - retained1.astype(np.float64)
        rng_k = np.random.default_rng(BOOTSTRAP_SEED + 1000 + k)
        boot_topk_delta = np.empty(N_BOOT, dtype=np.float64)
        position = 0
        while position < N_BOOT:
            m = min(1000, N_BOOT - position)
            idx_boot = rng_k.integers(low=0, high=k, size=(m, k), dtype=np.int16)
            boot_topk_delta[position:position + m] = 100.0 * np.mean(delta_item[idx_boot], axis=1)
            position += m
        topk_boot_arrays[f'delta_top{k}_percent'] = boot_topk_delta
        ci = percentile_ci(boot_topk_delta)
        p_route2_better = float(np.mean(boot_topk_delta > 0.0))
        only1 = int(np.sum((retained1 == 1) & (retained2 == 0)))
        only2 = int(np.sum((retained1 == 0) & (retained2 == 1)))
        n_discordant = only1 + only2
        if n_discordant > 0:
            exact_p = float(binomtest(k=only2, n=n_discordant, p=0.5, alternative='two-sided').pvalue)
        else:
            exact_p = 1.0
        topk_rows.append({'Top_k': k, 'Route1_overlap_count': count1, 'Route1_retention_percent': percent1, 'Route2_overlap_count': count2, 'Route2_retention_percent': percent2, 'Delta_Route2_minus_Route1_pp': observed_delta, 'Bootstrap_median_delta_pp': ci[1], 'CI95_lower_pp': ci[0], 'CI95_upper_pp': ci[2], 'P_bootstrap_Route2_better': p_route2_better, 'Route1_only_retained': only1, 'Route2_only_retained': only2, 'Exact_paired_p': exact_p, 'Interpretation': interpret_delta(ci, 'higher')})
    topk_bootstrap_df = pd.DataFrame(topk_rows)
    print()
    print('=' * 105)
    print('PAIRED BOOTSTRAP — SLME METRIC DIFFERENCES')
    print('=' * 105)
    print()
    print(bootstrap_metric_df.to_string(index=False, formatters={'Route1': lambda x: f'{x:.6f}', 'Route2': lambda x: f'{x:.6f}', 'Delta_Route2_minus_Route1': lambda x: f'{x:+.6f}', 'Bootstrap_median_delta': lambda x: f'{x:+.6f}', 'CI95_lower': lambda x: f'{x:+.6f}', 'CI95_upper': lambda x: f'{x:+.6f}', 'P_bootstrap_Route2_better': lambda x: f'{x:.4f}'}))
    print()
    print('=' * 105)
    print('PAIRED BOOTSTRAP — TOP-k RETENTION')
    print('=' * 105)
    print()
    print(topk_bootstrap_df.to_string(index=False, formatters={'Route1_retention_percent': lambda x: f'{x:.1f}%', 'Route2_retention_percent': lambda x: f'{x:.1f}%', 'Delta_Route2_minus_Route1_pp': lambda x: f'{x:+.1f}', 'Bootstrap_median_delta_pp': lambda x: f'{x:+.2f}', 'CI95_lower_pp': lambda x: f'{x:+.2f}', 'CI95_upper_pp': lambda x: f'{x:+.2f}', 'P_bootstrap_Route2_better': lambda x: f'{x:.4f}', 'Exact_paired_p': lambda x: f'{x:.6f}'}))
    continuous_clear_route2 = R2_CI[0] > 0.0 and MAE_CI[2] < 0.0 and (RMSE_CI[2] < 0.0)
    continuous_clear_route1 = R2_CI[2] < 0.0 and MAE_CI[0] > 0.0 and (RMSE_CI[0] > 0.0)
    print()
    print('=' * 105)
    print('OVERALL SCIENTIFIC INTERPRETATION')
    print('=' * 105)
    if continuous_clear_route2:
        print('The paired bootstrap consistently favors Route 2 across R², MAE, and RMSE.')
    elif continuous_clear_route1:
        print('The paired bootstrap consistently favors Route 1 across R², MAE, and RMSE.')
    else:
        print('The paired bootstrap does NOT establish a clear overall superiority of either route.')
        print('Recommended wording: the two routes provide comparable application-level SLME performance.')
    metric_csv = OUT_DIR / 'FINAL_paired_bootstrap_SLME_metrics.csv'
    topk_csv = OUT_DIR / 'FINAL_paired_bootstrap_TopK.csv'
    npz_path = OUT_DIR / 'FINAL_paired_bootstrap_distributions.npz'
    bootstrap_metric_df.to_csv(metric_csv, index=False)
    topk_bootstrap_df.to_csv(topk_csv, index=False)
    np.savez_compressed(npz_path, bootstrap_seed=np.array([BOOTSTRAP_SEED]), n_bootstrap=np.array([N_BOOT]), base_idx=BASE_IDX, reference_SLME_percent=REF, route1_SLME_percent=R1, route2_SLME_percent=R2, delta_R2_route2_minus_route1=boot_delta_r2, delta_MAE_route2_minus_route1_pp=boot_delta_mae, delta_RMSE_route2_minus_route1_pp=boot_delta_rmse, **topk_boot_arrays)
    print()
    print('=' * 105)
    print('SAVED')
    print('=' * 105)
    print('Metrics:', metric_csv)
    print('Top-k  :', topk_csv)
    print('NPZ    :', npz_path)
    return {'metrics': bootstrap_metric_df, 'topk': topk_bootstrap_df, 'n_samples': N, 'output_dir': OUT_DIR, 'paths': {'metrics_csv': metric_csv, 'topk_csv': topk_csv, 'distributions_npz': npz_path}, 'distributions': {'delta_R2': boot_delta_r2, 'delta_MAE': boot_delta_mae, 'delta_RMSE': boot_delta_rmse, **topk_boot_arrays}}
