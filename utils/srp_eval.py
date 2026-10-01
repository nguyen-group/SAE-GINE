"""
SRP ranking evaluation helpers extracted from TASK3 notebook.
"""
import numpy as np
import pandas as pd

def corr_xy(x, y, rank=False):
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    ok = x.notna() & y.notna()

    if ok.sum() < 3:
        return np.nan

    x = x[ok].rank() if rank else x[ok]
    y = y[ok].rank() if rank else y[ok]

    return float(np.corrcoef(x, y)[0, 1])

def topk_overlap(df, ref_col, pred_col, k):
    valid = df[[ref_col, pred_col]].dropna()
    k = min(k, len(valid))

    ref_top = set(valid.sort_values(ref_col, ascending=False).head(k).index)
    pred_top = set(valid.sort_values(pred_col, ascending=False).head(k).index)

    n = len(ref_top & pred_top)
    return n, 100.0 * n / k

def add_robust_rank_score(df):
    out = df.copy()

    out["reference_SRP_robust_score"] = -0.5 * (
        out["reference_SRP_abs_power_percent"].rank(ascending=False)
        + out["reference_SRP_abs_photon_percent"].rank(ascending=False)
    )

    out["predicted_SRP_robust_score"] = -0.5 * (
        out["predicted_SRP_abs_power_percent"].rank(ascending=False)
        + out["predicted_SRP_abs_photon_percent"].rank(ascending=False)
    )

    return out

def evaluate_srp_score(df, thickness_nm, score_name, ref_col, pred_col):
    row = {
        "thickness_nm": thickness_nm,
        "score": score_name,
        "reference_col": ref_col,
        "predicted_col": pred_col,
        "Pearson": corr_xy(df[ref_col], df[pred_col]),
        "Spearman": corr_xy(df[ref_col], df[pred_col], rank=True),
    }

    for k in [10, 20, 50, 100]:
        n, pct = topk_overlap(df, ref_col, pred_col, k)
        row[f"Top{k}_overlap_count"] = n
        row[f"Top{k}_overlap_percent"] = pct

    return row
