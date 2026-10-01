"""
Direct-alpha ablation utilities.

Target:
    log10(1 + alpha(E))
derived from paired epsR_0 and epsI_0 spectra.
"""
import torch
from torch.utils.data import Dataset

def dielectric_to_alpha_log10_1p(epsR, epsI, axis_eV):
    epsR = torch.as_tensor(epsR, dtype=torch.float32).view(-1)
    epsI = torch.as_tensor(epsI, dtype=torch.float32).view(-1)
    E    = torch.as_tensor(axis_eV, dtype=torch.float32).view(-1)

    eps_abs = torch.sqrt(epsR**2 + epsI**2 + 1e-12)
    k = torch.sqrt(torch.clamp((eps_abs - epsR) / 2.0, min=0.0))

    omega = E / HBAR_EV_S
    alpha_m = 2.0 * omega * k / C_LIGHT
    alpha_cm = alpha_m / 100.0

    alpha_cm = torch.nan_to_num(alpha_cm, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_cm = torch.clamp(alpha_cm, min=0.0)

    return torch.log10(1.0 + alpha_cm).float()

class DirectAlphaFromEpsilonDataset(Dataset):
    def __init__(self, epsR_ds, epsI_ds, name="alpha_log10_1p"):
        assert len(epsR_ds) == len(epsI_ds)
        self.epsR_ds = epsR_ds
        self.epsI_ds = epsI_ds
        self.name = name

    def __len__(self):
        return len(self.epsI_ds)

    def __getitem__(self, idx):
        dR = self.epsR_ds[idx]
        dI = self.epsI_ds[idx]

        if hasattr(dR, "sample_idx") and hasattr(dI, "sample_idx"):
            assert int(dR.sample_idx) == int(dI.sample_idx), f"sample_idx mismatch at {idx}"

        if hasattr(dR, "base_idx") and hasattr(dI, "base_idx"):
            assert int(dR.base_idx) == int(dI.base_idx), f"base_idx mismatch at {idx}"

        data = dI.clone()

        epsR = dR.y.view(-1).float()
        epsI = dI.y.view(-1).float()

        if hasattr(dI, "axis_grid"):
            axis = dI.axis_grid.view(-1).float()
        elif hasattr(dI, "spectral_axis"):
            axis = dI.spectral_axis.view(-1).float()
        else:
            axis = torch.linspace(0.0, 20.0, epsI.numel())

        alpha_log = dielectric_to_alpha_log10_1p(epsR, epsI, axis)

        # IMPORTANT: keep y as [1, 2001], not [2001]
        data.y = alpha_log.view(1, -1)

        if hasattr(dI, "y_mask"):
            data.y_mask = dI.y_mask.view(1, -1).clone()
        else:
            data.y_mask = torch.ones_like(data.y)

        data.target_name = self.name
        return data
