"""
Spectrum normalization, losses, and metrics.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ============================================================
# Spectrum normalization / loss / metrics utilities
# ============================================================

EPS_SPECTRUM = 1e-8


def infer_allow_negative_from_target_key(target_key):
    """
    Default convention:
    - epsI_* : negative values are not allowed
    - epsR_* : negative values are allowed
    """
    if target_key is None:
        return True

    key = str(target_key).lower()
    if key.startswith("epsi"):
        return False
    if key.startswith("epsr"):
        return True

    return True


def get_default_negative_weight(target_key, fallback=0.02):
    """
    If target is epsI_*, a small negative penalty is used.
    If target is epsR_*, the negative penalty is disabled.
    """
    allow_negative = infer_allow_negative_from_target_key(target_key)

    if allow_negative:
        return 0.0

    return float(fallback)


def ensure_2d_tensor(x):
    if x.ndim == 1:
        return x.unsqueeze(0)

    return x


def safe_mask_like(ref, mask=None):
    if mask is None:
        return torch.ones_like(ref, dtype=ref.dtype, device=ref.device)

    mask = ensure_2d_tensor(mask).to(dtype=ref.dtype, device=ref.device)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0).to(ref.dtype)

    return mask


@torch.no_grad()
def compute_spectrum_stats(dataset, device="cpu"):
    """
    Compute per-bin mean/std of spectra in a mask-aware manner.
    Use this only on the train split.
    """
    sum_y = None
    sum_y2 = None
    sum_m = None

    for i in range(len(dataset)):
        data = dataset[i]

        y = ensure_2d_tensor(data.y)[0].to(torch.float32).cpu()
        m = getattr(data, "y_mask", None)

        if m is None:
            m = torch.ones_like(y, dtype=torch.float32)
        else:
            m = ensure_2d_tensor(m)[0].to(torch.float32).cpu()

        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = (m > 0).to(torch.float32)

        if sum_y is None:
            spectrum_length = y.numel()
            sum_y = torch.zeros(spectrum_length, dtype=torch.float64)
            sum_y2 = torch.zeros(spectrum_length, dtype=torch.float64)
            sum_m = torch.zeros(spectrum_length, dtype=torch.float64)

        sum_y += (y * m).to(torch.float64)
        sum_y2 += ((y * y) * m).to(torch.float64)
        sum_m += m.to(torch.float64)

    mean = sum_y / torch.clamp(sum_m, min=1.0)
    var = (sum_y2 / torch.clamp(sum_m, min=1.0)) - mean.pow(2)
    var = torch.clamp(var, min=1e-8)

    std = torch.sqrt(var)
    std = torch.clamp(std, min=1e-4)

    mean = mean.to(torch.float32).to(device)
    std = std.to(torch.float32).to(device)

    try:
        stats_df = pd.DataFrame(
            {
                "bin_idx": np.arange(len(mean)),
                "mean_train": mean.detach().cpu().numpy(),
                "std_train": std.detach().cpu().numpy(),
            }
        )
    except Exception:
        stats_df = None

    return mean, std, stats_df


def normalize_spectrum(y, mean, std):
    return (y - mean) / (std + EPS_SPECTRUM)


def denormalize_spectrum(y_norm, mean, std):
    return y_norm * (std + EPS_SPECTRUM) + mean


def spectrum_loss(
    pred_norm,
    target_norm,
    mask=None,
    pred_denorm=None,
    target_denorm=None,
    target_key=None,
    peak_weight=2.0,
    peak_mode="denorm",
    smoothness_weight=0.05,
    smoothness_domain="norm",
    negative_weight=None,
    allow_negative=None,
):
    """
    Peak-aware spectrum loss.

    Components:
    - SmoothL1 on normalized spectra
    - peak-aware weighting
    - first-derivative consistency
    - optional negative penalty for epsI_*
    """
    pred_norm = ensure_2d_tensor(pred_norm)
    target_norm = ensure_2d_tensor(target_norm)
    mask = safe_mask_like(target_norm, mask)

    if allow_negative is None:
        allow_negative = infer_allow_negative_from_target_key(target_key)

    if negative_weight is None:
        negative_weight = get_default_negative_weight(target_key, fallback=0.02)

    if peak_mode == "denorm" and target_denorm is not None:
        peak_ref = ensure_2d_tensor(target_denorm).detach().to(
            dtype=target_norm.dtype,
            device=target_norm.device,
        )
    else:
        peak_ref = target_norm.detach()

    peak_scale = torch.abs(peak_ref)
    peak_scale = peak_scale / (peak_scale.amax(dim=1, keepdim=True) + EPS_SPECTRUM)
    weights = 1.0 + peak_weight * peak_scale

    base = F.smooth_l1_loss(pred_norm, target_norm, reduction="none")
    base = base * weights * mask
    base_loss = base.sum() / torch.clamp(mask.sum(), min=1.0)

    smooth_loss = pred_norm.new_tensor(0.0)

    if pred_norm.shape[1] >= 2 and smoothness_weight > 0:
        if smoothness_domain == "denorm" and pred_denorm is not None and target_denorm is not None:
            pred_ref = ensure_2d_tensor(pred_denorm)
            true_ref = ensure_2d_tensor(target_denorm)
        else:
            pred_ref = pred_norm
            true_ref = target_norm

        dp = pred_ref[:, 1:] - pred_ref[:, :-1]
        dt = true_ref[:, 1:] - true_ref[:, :-1]
        dm = mask[:, 1:] * mask[:, :-1]

        smooth_loss = torch.abs(dp - dt)
        smooth_loss = (smooth_loss * dm).sum() / torch.clamp(dm.sum(), min=1.0)

    total = base_loss + smoothness_weight * smooth_loss

    neg_loss = pred_norm.new_tensor(0.0)

    if (not allow_negative) and negative_weight > 0 and pred_denorm is not None:
        pred_denorm = ensure_2d_tensor(pred_denorm)
        neg_part = torch.relu(-pred_denorm)
        neg_loss = (neg_part * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        total = total + negative_weight * neg_loss

    info = {
        "loss_total": float(total.detach().cpu().item()),
        "loss_base": float(base_loss.detach().cpu().item()),
        "loss_smooth": float(smooth_loss.detach().cpu().item()),
        "loss_negative": float(neg_loss.detach().cpu().item()),
        "allow_negative": bool(allow_negative),
        "negative_weight": float(negative_weight),
        "peak_mode": str(peak_mode),
        "smoothness_domain": str(smoothness_domain),
    }

    return total, info


@torch.no_grad()
def compute_spectrum_metrics(
    pred_denorm,
    true_denorm,
    mask=None,
    axis_grid=None,
):
    """
    Evaluation metrics on denormalized spectra.
    """
    pred_denorm = ensure_2d_tensor(pred_denorm)
    true_denorm = ensure_2d_tensor(true_denorm)
    mask = safe_mask_like(true_denorm, mask)

    valid = torch.clamp(mask.sum(dim=1), min=1.0)

    diff = (pred_denorm - true_denorm) * mask

    mae = diff.abs().sum(dim=1) / valid
    mse = diff.pow(2).sum(dim=1) / valid
    rmse = torch.sqrt(torch.clamp(mse, min=0.0))

    pred_m = pred_denorm * mask
    true_m = true_denorm * mask

    dot = (pred_m * true_m).sum(dim=1)
    pred_norm_v = torch.sqrt(pred_m.pow(2).sum(dim=1).clamp_min(EPS_SPECTRUM))
    true_norm_v = torch.sqrt(true_m.pow(2).sum(dim=1).clamp_min(EPS_SPECTRUM))
    cosine = dot / (pred_norm_v * true_norm_v + EPS_SPECTRUM)

    pred_peak_search = pred_denorm.masked_fill(mask <= 0, float("-inf"))
    true_peak_search = true_denorm.masked_fill(mask <= 0, float("-inf"))

    pred_peak = pred_peak_search.max(dim=1).values
    true_peak = true_peak_search.max(dim=1).values
    peak_mae = (pred_peak - true_peak).abs()

    pred_peak_idx = pred_peak_search.argmax(dim=1)
    true_peak_idx = true_peak_search.argmax(dim=1)
    peak_bin_shift = (pred_peak_idx - true_peak_idx).abs().to(torch.float32)

    neg_rate = ((pred_denorm < 0).to(torch.float32) * mask).sum(dim=1) / valid

    metrics = {
        "MAE_mean": float(mae.mean().detach().cpu().item()),
        "RMSE_mean": float(rmse.mean().detach().cpu().item()),
        "CosineSim_mean": float(cosine.mean().detach().cpu().item()),
        "PeakMAE_mean": float(peak_mae.mean().detach().cpu().item()),
        "PeakBinShift_mean": float(peak_bin_shift.mean().detach().cpu().item()),
        "PeakIndexError_mean": float(peak_bin_shift.mean().detach().cpu().item()),
        "NegativeBinRate_mean": float(neg_rate.mean().detach().cpu().item()),
    }

    if axis_grid is not None:
        axis_grid = ensure_2d_tensor(axis_grid).to(
            dtype=pred_denorm.dtype,
            device=pred_denorm.device,
        )

        if axis_grid.shape[0] == 1 and pred_denorm.shape[0] > 1:
            axis_grid = axis_grid.expand(pred_denorm.shape[0], -1)

        pred_peak_axis = torch.gather(axis_grid, 1, pred_peak_idx.view(-1, 1)).squeeze(1)
        true_peak_axis = torch.gather(axis_grid, 1, true_peak_idx.view(-1, 1)).squeeze(1)
        peak_axis_err = (pred_peak_axis - true_peak_axis).abs()

        metrics["PeakAxisError_mean"] = float(peak_axis_err.mean().detach().cpu().item())

    return metrics


@torch.no_grad()
def run_paired_spectrum_stats_smoke_test(
    epsR_train_ds,
    epsI_train_ds,
    device="cpu",
):
    """
    Compact smoke test for paired epsR_0 / epsI_0 train datasets.
    """
    outputs = {}

    for target_key, train_ds in [
        ("epsR_0", epsR_train_ds),
        ("epsI_0", epsI_train_ds),
    ]:
        mean_train, std_train, stats_df = compute_spectrum_stats(train_ds, device=device)

        y0 = ensure_2d_tensor(train_ds[0].y.to(torch.float32))
        m0 = ensure_2d_tensor(train_ds[0].y_mask.to(torch.float32))
        axis0 = ensure_2d_tensor(train_ds[0].axis_grid.to(torch.float32))

        metrics_debug = compute_spectrum_metrics(
            pred_denorm=y0.clone(),
            true_denorm=y0,
            mask=m0,
            axis_grid=axis0,
        )

        outputs[target_key] = {
            "mean_train": mean_train,
            "std_train": std_train,
            "stats_df": stats_df,
            "metrics_debug": metrics_debug,
            "allow_negative": infer_allow_negative_from_target_key(target_key),
            "negative_weight": get_default_negative_weight(target_key, fallback=0.02),
        }

    return outputs
