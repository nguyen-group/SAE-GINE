"""Matched held-out comparison of frozen absorption routes."""
import builtins
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.stats import pearsonr, spearmanr

def run_fair_slme_comparison(project_root='D:\\TB3', *, audit_npz=None, reference_csv=None, output_dir=None, verbose=True):
    """Recompute held-out SLME for both frozen alpha routes and save comparisons.

Preserves the original optical-gap cutoffs, Lambert-W solver, finite-cohort
rule, metrics, stable Top-k ranking, and reproduction checks. No retraining.
Inputs include an upstream alpha-route audit NPZ and the direct-SLME CSV.
The NPZ is produced outside this notebook; it must exist before this call.
The original solver's Jsc > 0 rule remains unchanged in this code refactor.
Default output filenames overwrite previous comparison tables as before.
"""

    def print(*args, **kwargs):
        if verbose:
            builtins.print(*args, **kwargs)
    try:
        from pvlib import spectrum
    except ImportError as exc:
        raise ImportError('Thiếu pvlib. Cài một lần bằng:\npip install pvlib') from exc
    ROOT = Path(project_root) / 'processed/paired_training'
    AUDIT_NPZ = Path(audit_npz) if audit_npz is not None else ROOT / 'paper_outputs/final_alpha_route_audit/FINAL_alpha_route_comparison_arrays.npz'
    CURRENT_SLME_CSV = Path(reference_csv) if reference_csv is not None else ROOT / 'paper_outputs/FINAL_direct_alpha_SLME_500nm/FINAL_direct_alpha_SLME_test_samples.csv'
    DIRECT_DIR = ROOT / 'epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed'
    DIRECT_SEED42 = DIRECT_DIR / 'test_predictions_epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed_seed42.npz'
    OUT_DIR = Path(output_dir) if output_dir is not None else ROOT / 'paper_outputs/FINAL_alpha_route_SLME_comparison_500nm'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THICKNESS_NM = 500.0
    TEMPERATURE_K = 300.0
    FR = 1.0
    SOLAR_LAMBDA_MIN_NM = 200.0
    SOLAR_LAMBDA_MAX_NM = 2500.0
    TOP_K = [10, 20, 50, 100]
    Q = 1.602176634e-19
    H = 6.62607015e-34
    C = 299792458.0
    KB = 1.380649e-23
    HC_EV_NM = 1239.8419843320025
    for p in [AUDIT_NPZ, CURRENT_SLME_CSV, DIRECT_SEED42]:
        if not p.is_file():
            raise FileNotFoundError(f'Không tìm thấy:\n{p}')
    audit = np.load(AUDIT_NPZ, allow_pickle=True)
    required_audit_keys = ['energy_eV', 'alpha_true_cm1', 'alpha_derived_SAE_eps1_eps2_cm1', 'alpha_direct_SAE_cm1']
    missing = [k for k in required_audit_keys if k not in audit.files]
    if missing:
        raise KeyError(f'AUDIT NPZ thiếu keys: {missing}\nAvailable: {audit.files}')
    E = np.asarray(audit['energy_eV'], dtype=np.float64).reshape(-1)
    ALPHA_DERIVED = np.asarray(audit['alpha_derived_SAE_eps1_eps2_cm1'], dtype=np.float64)
    ALPHA_DIRECT = np.asarray(audit['alpha_direct_SAE_cm1'], dtype=np.float64)
    if ALPHA_DERIVED.shape != ALPHA_DIRECT.shape:
        raise RuntimeError('Derived/direct alpha shape mismatch.')
    if ALPHA_DERIVED.shape != (992, 2001):
        raise RuntimeError(f'Expected (992, 2001), got {ALPHA_DERIVED.shape}')
    z_direct = np.load(DIRECT_SEED42, allow_pickle=True)
    if 'base_idx' not in z_direct.files:
        raise KeyError('Direct prediction NPZ không có base_idx.')
    BASE_IDX = np.asarray(z_direct['base_idx']).reshape(-1)
    if len(BASE_IDX) != 992:
        raise RuntimeError(f'Expected 992 base_idx, got {len(BASE_IDX)}')
    if len(np.unique(BASE_IDX)) != 992:
        raise RuntimeError('Duplicate base_idx trong test prediction.')
    old_df = pd.read_csv(CURRENT_SLME_CSV)
    required_csv = ['base_idx', 'optical_gap_cutoff_eV', 'reference_SLME_percent', 'predicted_SLME_percent']
    missing = [c for c in required_csv if c not in old_df.columns]
    if missing:
        raise KeyError(f'Current SLME CSV thiếu columns: {missing}')
    if len(old_df) != 992:
        raise RuntimeError(f'Expected 992 rows in current SLME CSV, got {len(old_df)}')
    if old_df['base_idx'].duplicated().any():
        raise RuntimeError('Duplicate base_idx trong current SLME CSV.')
    meta = old_df.set_index('base_idx').reindex(BASE_IDX)
    if meta.index.isna().any():
        raise RuntimeError('Không align được base_idx.')
    if meta['optical_gap_cutoff_eV'].isna().any():
        raise RuntimeError('Có optical_gap_cutoff_eV bị thiếu.')
    GAP_EV = meta['optical_gap_cutoff_eV'].to_numpy(dtype=np.float64)
    OLD_REF_SLME = meta['reference_SLME_percent'].to_numpy(dtype=np.float64)
    OLD_DIRECT_SLME = meta['predicted_SLME_percent'].to_numpy(dtype=np.float64)
    direct_files = []
    for seed in [42, 123, 2025]:
        p = DIRECT_DIR / f'test_predictions_epsI_alpha_log10_1p_direct_alpha_ssl_init_enhanced_19_24_d005_wd5e5_3seed_seed{seed}.npz'
        if not p.is_file():
            raise FileNotFoundError(p)
        direct_files.append(p)
    true_log_ref = None
    base_idx_ref = None
    for p in direct_files:
        z = np.load(p, allow_pickle=True)
        true_log = np.asarray(z['true'], dtype=np.float64)
        if true_log.ndim == 3 and true_log.shape[1] == 1:
            true_log = true_log[:, 0, :]
        b = np.asarray(z['base_idx']).reshape(-1)
        if true_log_ref is None:
            true_log_ref = true_log
            base_idx_ref = b
        else:
            if not np.array_equal(base_idx_ref, b):
                raise RuntimeError(f'base_idx mismatch: {p.name}')
            if not np.allclose(true_log_ref, true_log, rtol=1e-06, atol=1e-07, equal_nan=True):
                raise RuntimeError(f'TRUE alpha target mismatch: {p.name}')
    if not np.array_equal(BASE_IDX, base_idx_ref):
        raise RuntimeError('Audit arrays and Direct-alpha ordering do not match.')
    with np.errstate(over='ignore', invalid='ignore'):
        ALPHA_REF = np.power(10.0, true_log_ref) - 1.0
    ALPHA_REF = np.maximum(np.nan_to_num(ALPHA_REF, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    ALPHA_TRUE_AUDIT = np.asarray(audit['alpha_true_cm1'], dtype=np.float64)
    reference_delta = np.abs(ALPHA_REF - ALPHA_TRUE_AUDIT)
    print('=' * 110)
    print('REFERENCE ALPHA CONSISTENCY')
    print('=' * 110)
    print(f'MAE(direct-target reference vs dielectric-derived reference) = {np.mean(reference_delta):.6e} cm^-1')
    print(f'Max difference = {np.max(reference_delta):.6e} cm^-1')

    def apply_gap(alpha_cm1, energy_eV, gap_eV):
        alpha = np.asarray(alpha_cm1, dtype=np.float64).copy()
        alpha = np.maximum(np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        below_gap = energy_eV[None, :] < gap_eV[:, None]
        alpha[below_gap] = 0.0
        return alpha
    ALPHA_REF_CLEAN = apply_gap(ALPHA_REF, E, GAP_EV)
    ALPHA_1_CLEAN = apply_gap(ALPHA_DERIVED, E, GAP_EV)
    ALPHA_2_CLEAN = apply_gap(ALPHA_DIRECT, E, GAP_EV)
    am15 = spectrum.get_reference_spectra()
    global_column = next((column for column in am15.columns if 'global' in str(column).lower()), None)
    if global_column is None:
        raise RuntimeError('Không tìm thấy Global AM1.5G column.')
    wavelength_nm = np.asarray(am15.index, dtype=np.float64)
    irradiance_w_m2_nm = np.asarray(am15[global_column], dtype=np.float64)
    solar_mask = np.isfinite(wavelength_nm) & np.isfinite(irradiance_w_m2_nm) & (wavelength_nm >= SOLAR_LAMBDA_MIN_NM) & (wavelength_nm <= SOLAR_LAMBDA_MAX_NM) & (irradiance_w_m2_nm >= 0.0)
    wavelength_nm = wavelength_nm[solar_mask]
    irradiance_w_m2_nm = irradiance_w_m2_nm[solar_mask]

    def am15_photon_flux(model_energy_eV, wavelength_nm, irradiance_w_m2_nm):
        solar_E_eV = HC_EV_NM / wavelength_nm
        solar_E_J = solar_E_eV * Q
        photon_flux_lambda = irradiance_w_m2_nm / solar_E_J
        jacobian = HC_EV_NM / solar_E_eV ** 2
        photon_flux_E = photon_flux_lambda * jacobian
        order = np.argsort(solar_E_eV)
        return np.interp(model_energy_eV, solar_E_eV[order], photon_flux_E[order], left=0.0, right=0.0)
    PHI_SOLAR = am15_photon_flux(E, wavelength_nm, irradiance_w_m2_nm)
    P_SOLAR = np.trapezoid(E * Q * PHI_SOLAR, E)
    print()
    print('=' * 110)
    print('AM1.5G')
    print('=' * 110)
    print(f'Psolar = {P_SOLAR:.6f} W/m^2')
    if not 900.0 <= P_SOLAR <= 1050.0:
        raise RuntimeError('Psolar không hợp lý.')

    def blackbody_photon_radiance(energy_eV, temperature_K):
        energy_eV = np.asarray(energy_eV, dtype=np.float64)
        energy_J = energy_eV * Q
        result = np.zeros_like(energy_eV, dtype=np.float64)
        valid = energy_eV > 0.0
        exponent = np.zeros_like(energy_eV, dtype=np.float64)
        exponent[valid] = energy_J[valid] / (KB * temperature_K)
        safe = valid & (exponent < 700.0)
        spectral_per_J = 2.0 * energy_J[safe] ** 2 / (H ** 3 * C ** 2) / np.expm1(exponent[safe])
        result[safe] = spectral_per_J * Q
        return result
    PHI_BB = blackbody_photon_radiance(E, TEMPERATURE_K)

    def calculate_slme_batch(alpha_cm1, energy_eV, phi_solar, phi_bb, psolar_w_m2, thickness_nm=500.0, temperature_K=300.0, fr=1.0):
        alpha_cm1 = np.asarray(alpha_cm1, dtype=np.float64)
        if alpha_cm1.ndim == 1:
            alpha_cm1 = alpha_cm1[None, :]
        alpha_cm1 = np.maximum(np.nan_to_num(alpha_cm1, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        thickness_cm = thickness_nm * 1e-07
        A = 1.0 - np.exp(-2.0 * alpha_cm1 * thickness_cm)
        A = np.clip(A, 0.0, 1.0)
        Jsc = Q * np.trapezoid(A * phi_solar[None, :], energy_eV, axis=1)
        J0 = Q * np.pi / fr * np.trapezoid(A * phi_bb[None, :], energy_eV, axis=1)
        thermal_voltage = KB * temperature_K / Q
        valid = np.isfinite(Jsc) & np.isfinite(J0) & (Jsc > 0.0) & (J0 > 0.0)
        n = len(Jsc)
        eta = np.full(n, np.nan, dtype=np.float64)
        Voc = np.full(n, np.nan, dtype=np.float64)
        Vmp = np.full(n, np.nan, dtype=np.float64)
        Jmp = np.full(n, np.nan, dtype=np.float64)
        Pmax = np.full(n, np.nan, dtype=np.float64)
        ratio = np.full(n, np.nan, dtype=np.float64)
        ratio[valid] = Jsc[valid] / J0[valid]
        Voc[valid] = thermal_voltage * np.log1p(ratio[valid])
        argument = np.full(n, np.nan, dtype=np.float64)
        argument[valid] = np.e * (1.0 + ratio[valid])
        vmp_dimensionless = np.full(n, np.nan, dtype=np.float64)
        vmp_dimensionless[valid] = np.real(lambertw(argument[valid])) - 1.0
        Vmp[valid] = thermal_voltage * vmp_dimensionless[valid]
        exp_vmp = np.full(n, np.nan, dtype=np.float64)
        exp_vmp[valid] = np.exp(np.clip(vmp_dimensionless[valid], None, 700.0))
        Jmp[valid] = Jsc[valid] - J0[valid] * (exp_vmp[valid] - 1.0)
        Pmax[valid] = Jmp[valid] * Vmp[valid]
        eta[valid] = 100.0 * Pmax[valid] / psolar_w_m2
        return {'eta_percent': eta, 'Jsc_A_m2': Jsc, 'J0_A_m2': J0, 'Voc_V': Voc, 'Vmp_V': Vmp, 'Jmp_A_m2': Jmp, 'Pmax_W_m2': Pmax}
    SLME_REF_RESULT = calculate_slme_batch(ALPHA_REF_CLEAN, E, PHI_SOLAR, PHI_BB, P_SOLAR, thickness_nm=THICKNESS_NM, temperature_K=TEMPERATURE_K, fr=FR)
    SLME_1_RESULT = calculate_slme_batch(ALPHA_1_CLEAN, E, PHI_SOLAR, PHI_BB, P_SOLAR, thickness_nm=THICKNESS_NM, temperature_K=TEMPERATURE_K, fr=FR)
    SLME_2_RESULT = calculate_slme_batch(ALPHA_2_CLEAN, E, PHI_SOLAR, PHI_BB, P_SOLAR, thickness_nm=THICKNESS_NM, temperature_K=TEMPERATURE_K, fr=FR)
    SLME_REF = SLME_REF_RESULT['eta_percent']
    SLME_1 = SLME_1_RESULT['eta_percent']
    SLME_2 = SLME_2_RESULT['eta_percent']
    mask_ref_repro = np.isfinite(SLME_REF) & np.isfinite(OLD_REF_SLME)
    mask_direct_repro = np.isfinite(SLME_2) & np.isfinite(OLD_DIRECT_SLME)
    ref_repro_mae = np.mean(np.abs(SLME_REF[mask_ref_repro] - OLD_REF_SLME[mask_ref_repro]))
    direct_repro_mae = np.mean(np.abs(SLME_2[mask_direct_repro] - OLD_DIRECT_SLME[mask_direct_repro]))
    print()
    print('=' * 110)
    print('FIGURE-5 REPRODUCTION CHECK')
    print('=' * 110)
    print(f'Reference SLME reproduction MAE = {ref_repro_mae:.10f} pp')
    print(f'Direct SLME reproduction MAE    = {direct_repro_mae:.10f} pp')
    print('Existing Figure-5 finite pairs =', int(np.sum(np.isfinite(OLD_REF_SLME) & np.isfinite(OLD_DIRECT_SLME))))
    if ref_repro_mae > 1e-05:
        raise RuntimeError('Reference SLME does NOT reproduce current Figure 5.')
    if direct_repro_mae > 1e-05:
        raise RuntimeError('Direct-alpha SLME does NOT reproduce current Figure 5.')
    print('REPRODUCTION CHECK: PASS')

    def ordinary_r2(y_true, y_pred):
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def route_metrics(reference, prediction):
        error = prediction - reference
        return {'R2': ordinary_r2(reference, prediction), 'Pearson': float(pearsonr(reference, prediction).statistic), 'Spearman': float(spearmanr(reference, prediction).statistic), 'MAE_pp': float(np.mean(np.abs(error))), 'RMSE_pp': float(np.sqrt(np.mean(error ** 2))), 'Max_predicted_SLME_percent': float(np.max(prediction))}

    def topk_overlap(reference, prediction, k):
        ref_order = np.argsort(-reference, kind='mergesort')
        pred_order = np.argsort(-prediction, kind='mergesort')
        count = len(set(ref_order[:k]) & set(pred_order[:k]))
        return (count, 100.0 * count / k)
    COMMON_VALID = np.isfinite(SLME_REF) & np.isfinite(SLME_1) & np.isfinite(SLME_2)
    N_COMMON = int(COMMON_VALID.sum())
    REF = SLME_REF[COMMON_VALID]
    PRED_1 = SLME_1[COMMON_VALID]
    PRED_2 = SLME_2[COMMON_VALID]
    print()
    print('=' * 110)
    print('FAIR HEAD-TO-HEAD SAMPLE AUDIT')
    print('=' * 110)
    print('Held-out total          :', len(SLME_REF))
    print('Reference finite       :', int(np.isfinite(SLME_REF).sum()))
    print('SLME_1 finite          :', int(np.isfinite(SLME_1).sum()))
    print('SLME_2 finite          :', int(np.isfinite(SLME_2).sum()))
    print('COMMON finite samples  :', N_COMMON)
    if N_COMMON < 100:
        raise RuntimeError('Không đủ common finite samples.')
    M1 = route_metrics(REF, PRED_1)
    M2 = route_metrics(REF, PRED_2)
    summary_df = pd.DataFrame([{'Route': 'Route 1: SAE-eps1 + SAE-eps2 -> alpha_derived -> SLME', 'n_common': N_COMMON, **M1}, {'Route': 'Route 2: SAE-alpha -> alpha_direct -> SLME', 'n_common': N_COMMON, **M2}])
    print()
    print('=' * 110)
    print('FINAL SLME_1 vs SLME_2 — SAME SAMPLES')
    print('=' * 110)
    print()
    print(summary_df.to_string(index=False, formatters={'R2': lambda x: f'{x:.6f}', 'Pearson': lambda x: f'{x:.6f}', 'Spearman': lambda x: f'{x:.6f}', 'MAE_pp': lambda x: f'{x:.6f}', 'RMSE_pp': lambda x: f'{x:.6f}', 'Max_predicted_SLME_percent': lambda x: f'{x:.6f}'}))
    topk_rows = []
    for k in TOP_K:
        c1, p1 = topk_overlap(REF, PRED_1, k)
        c2, p2 = topk_overlap(REF, PRED_2, k)
        topk_rows.append({'Top_k': k, 'Route1_count': c1, 'Route1_percent': p1, 'Route2_count': c2, 'Route2_percent': p2})
    topk_df = pd.DataFrame(topk_rows)
    print()
    print('=' * 110)
    print('TOP-k REFERENCE RETENTION — SAME SAMPLES')
    print('=' * 110)
    print()
    print(topk_df.to_string(index=False, formatters={'Route1_percent': lambda x: f'{x:.1f}%', 'Route2_percent': lambda x: f'{x:.1f}%'}))
    ERR1 = np.abs(PRED_1 - REF)
    ERR2 = np.abs(PRED_2 - REF)
    TOL = 1e-12
    route1_better = int(np.sum(ERR1 < ERR2 - TOL))
    route2_better = int(np.sum(ERR2 < ERR1 - TOL))
    ties = N_COMMON - route1_better - route2_better
    print()
    print('=' * 110)
    print('PER-SAMPLE SLME ERROR')
    print('=' * 110)
    print(f'Route 1 lower |error| : {route1_better}/{N_COMMON} ({100 * route1_better / N_COMMON:.2f}%)')
    print(f'Route 2 lower |error| : {route2_better}/{N_COMMON} ({100 * route2_better / N_COMMON:.2f}%)')
    print(f'Ties                  : {ties}/{N_COMMON}')
    print()
    print('=' * 110)
    print('ROUTE 2 MINUS ROUTE 1')
    print('=' * 110)
    print(f"ΔR²       = {M2['R2'] - M1['R2']:+.6f}")
    print(f"ΔMAE      = {M2['MAE_pp'] - M1['MAE_pp']:+.6f} pp")
    print(f"ΔRMSE     = {M2['RMSE_pp'] - M1['RMSE_pp']:+.6f} pp")
    print(f"ΔPearson  = {M2['Pearson'] - M1['Pearson']:+.6f}")
    print(f"ΔSpearman = {M2['Spearman'] - M1['Spearman']:+.6f}")
    print()
    print('=' * 110)
    print('INTERPRETATION')
    print('=' * 110)
    if M1['R2'] > M2['R2'] and M1['MAE_pp'] < M2['MAE_pp']:
        print('SLME RESULT: ROUTE 1 (dielectric-derived alpha) is better by both R² and MAE.')
    elif M2['R2'] > M1['R2'] and M2['MAE_pp'] < M1['MAE_pp']:
        print('SLME RESULT: ROUTE 2 (direct-alpha) is better by both R² and MAE.')
    else:
        print('SLME RESULT: mixed outcome. R² and MAE do not select the same route.')
    comparison_df = pd.DataFrame({'base_idx': BASE_IDX, 'optical_gap_cutoff_eV': GAP_EV, 'reference_SLME_percent': SLME_REF, 'route1_derived_alpha_SLME_percent': SLME_1, 'route2_direct_alpha_SLME_percent': SLME_2, 'route1_abs_error_pp': np.abs(SLME_1 - SLME_REF), 'route2_abs_error_pp': np.abs(SLME_2 - SLME_REF), 'common_valid': COMMON_VALID})
    summary_path = OUT_DIR / 'FINAL_SLME_route1_vs_route2_metrics.csv'
    topk_path = OUT_DIR / 'FINAL_SLME_route1_vs_route2_topk.csv'
    samples_path = OUT_DIR / 'FINAL_SLME_route1_vs_route2_samples.csv'
    summary_df.to_csv(summary_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    comparison_df.to_csv(samples_path, index=False)
    print()
    print('=' * 110)
    print('SAVED')
    print('=' * 110)
    print('Metrics :', summary_path)
    print('Top-k   :', topk_path)
    print('Samples :', samples_path)
    return {'metrics': summary_df, 'topk': topk_df, 'samples': comparison_df, 'n_common': N_COMMON, 'output_dir': OUT_DIR, 'paths': {'metrics_csv': summary_path, 'topk_csv': topk_path, 'samples_csv': samples_path}, 'reproduction': {'reference_MAE_pp': float(ref_repro_mae), 'direct_MAE_pp': float(direct_repro_mae)}}
