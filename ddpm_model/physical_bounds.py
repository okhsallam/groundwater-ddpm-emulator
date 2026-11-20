"""
Enforcement of the physical bounds on the forecast variables.

Why not a sigmoid on the output layer, and why not a log/logit transform of the
target either -- both were considered and rejected on evidence:

  * The network does not emit the state. Under the `pred_v` parameterisation it
    emits a velocity from which the clean sample is reconstructed, and the clean
    sample here is a RESIDUAL (state minus initial state), which is signed. A
    saturating nonlinearity on that head has no correct range and breaks the
    diffusion parameterisation, which assumes an approximately Gaussian target.

  * A smooth bounding transform (inverse-softplus for depth, logit for moisture)
    would guarantee admissible values, but it maps the bound itself to infinity,
    and in this dataset the bounds carry large point masses:
        water table depth is EXACTLY 0 in 4.2% of cell-days (27% below 0.05 m)
        soil moisture is EXACTLY at porosity (saturation) in 28.8% of cell-days
    Those are censoring points, not asymptotes -- ParFlow itself clips there. A
    transform that sends them to +-infinity makes the most common values in the
    distribution unrepresentable.

The right treatment for a censored variable is the censoring itself: project
onto the admissible set. The important detail is WHERE. Clamping the saved
output after the fact repairs the numbers but not the trajectory, because during
an autoregressive rollout an inadmissible state is fed back in and compounds.
Clamping inside the loop, at every hop, before the state becomes the next hop's
initial condition, is what actually prevents the drift into nonsense.

Bounds enforced:
    water_table_depth  >= 0                     (depth below land surface)
    0 <= soil_moisture <= porosity(x, z=4)      (volumetric content cannot
                                                 exceed pore space; verified
                                                 never violated in WY2003-2006,
                                                 and attained exactly)

Both are applied in NORMALISED space, which is exact because the normalisation
is affine per variable.
"""
import numpy as np
import torch


def _affine(bound, var, stats):
    """Physical bound -> normalised bound. Normalisation is (x - mean)/std."""
    m = stats[var]["mean"]
    s = stats[var]["std"]
    m = np.asarray(m, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    return (np.asarray(bound, dtype=np.float32) - m) / s


class PhysicalBounds:
    """Normalised-space clamp for the two forecast variables.

    porosity_top: (H, W) porosity of the shallowest layer (z=4) for the tiles in
    the order they are stacked, or a scalar upper bound if unavailable.
    """

    def __init__(self, norm_stats, porosity_top, device):
        self.wtd_lo = float(_affine(0.0, "water_table_depth", norm_stats))
        sm_lo = _affine(0.0, "soil_moisture", norm_stats)
        sm_hi = _affine(porosity_top, "soil_moisture", norm_stats)
        self.sm_lo = torch.as_tensor(sm_lo, dtype=torch.float32, device=device)
        self.sm_hi = torch.as_tensor(sm_hi, dtype=torch.float32, device=device)

    def clamp_wtd_sm(self, x):
        """x: (..., 2, H, W) normalised [water_table_depth, soil_moisture]."""
        wtd = x[..., 0:1, :, :].clamp(min=self.wtd_lo)
        sm = torch.maximum(x[..., 1:2, :, :], self.sm_lo)
        sm = torch.minimum(sm, self.sm_hi)
        return torch.cat([wtd, sm], dim=-3)

    def clamp_block(self, x, n_state, future_len):
        """A stacked multi-day block (..., future_len*n_state, H, W)."""
        if n_state != 2:
            raise ValueError("physical bounds are defined for the lean "
                             "(water_table_depth, soil_moisture) state only")
        parts = [self.clamp_wtd_sm(x[..., k * n_state:(k + 1) * n_state, :, :])
                 for k in range(future_len)]
        return torch.cat(parts, dim=-3)

    def report(self, x, n_state, future_len, label=""):
        """How much was out of bounds before clamping -- worth logging, because
        a run that needs a large correction is telling you something."""
        with torch.no_grad():
            below = above = total = 0
            for k in range(future_len):
                b = x[..., k * n_state:(k + 1) * n_state, :, :]
                below += (b[..., 0:1, :, :] < self.wtd_lo).sum().item()
                below += (b[..., 1:2, :, :] < self.sm_lo).sum().item()
                above += (b[..., 1:2, :, :] > self.sm_hi).sum().item()
                total += b.numel()
        return (f"{label}out of bounds before clamp: "
                f"{100.0 * below / max(total, 1):.2f}% below, "
                f"{100.0 * above / max(total, 1):.2f}% above")


def load_porosity_top(tiles_root, tile_size, tiles):
    """Porosity of the shallowest layer per tile, stacked like the static batch.

    Read from the tiled static archives so it is aligned with the tiles exactly,
    with no re-clipping.
    """
    import os
    out = []
    for tile_id in range(len(tiles)):
        npz = np.load(os.path.join(tiles_root, f"tile{tile_size}", "static",
                                   f"tile_{tile_id:03d}.npz"))
        out.append(npz["porosity"][4])          # z=4 is the shallowest 0.1 m layer
    return np.stack(out, axis=0).astype(np.float32)   # (n_tiles, H, W)
