"""
Compute per-lead-day residual normalization stats: mean/std of
state[t+k] - state[t] for k=1..future_len, from the TRAIN water years only
(WY2003-2004), for the 3 state variables (pressure_head per z-layer,
water_table_depth, soil_moisture).

Why this exists: dataset_groundwater.py/train.py normalize the absolute state
using stats computed on the absolute value's own variance (normalization_stats.json).
Reusing that same std to scale the *residual* (state[t+k] - state[t]) badly
undersizes it -- day-to-day change is much smaller than total spatial
variance, so the residual ends up with std ~0.05-0.06 in those units instead
of ~1. A diffusion model whose noise schedule assumes roughly unit-variance
targets then can't resolve the residual's actual spatial structure and
degenerates to matching only its marginal variance -- visible as fine-grained,
spatially-incoherent noise in sampled output (confirmed 2026-09-15 in the
first residual-training run's quick-look figures).

Fix: normalize the residual by *its own* per-lead-day mean/std, not the
absolute-value std. Since lead-day-1 change is much smaller than lead-day-15
change (see persistence baseline RMSE growing with lead day), stats are
computed separately per k, not pooled.

Output: data/residual_normalization_stats_flen{future_len}.json (or --out),
consumed by dataset_groundwater.py's to_residual_target()/from_residual_pred().

--train-years lets the 2026-09-17 4-fold cross-validation compute fold-specific
stats. IMPORTANT: deltas are computed WITHIN each water year separately, never
across a concatenated multi-year array -- WY2003+WY2004 happen to be exactly
calendar-contiguous (Sep30,2003 -> Oct1,2003) so the original hardcoded
[2003,2004] version got away with concatenating first, but a fold like
{2003,2006} is NOT contiguous, and sliding a k-day window across that
concatenation boundary would silently compute a delta spanning a 2-year gap.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "parflow_raw"


def stats_1layer(arr):
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def stats_5layer(arr):
    """arr: (..., z, y, x) -- returns per-z (mean, std) lists."""
    flat_by_z = np.moveaxis(arr, -3, 0).reshape(arr.shape[-3], -1)
    means = np.nanmean(flat_by_z, axis=1)
    stds = np.nanstd(flat_by_z, axis=1)
    return [float(m) for m in means], [float(s) for s in stds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--future-len", type=int, default=15)
    parser.add_argument("--train-years", type=int, nargs="+", default=[2003, 2004])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    future_len = args.future_len
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "data" / f"residual_normalization_stats_flen{future_len}.json"

    pressure_head_years, wtd_years, sm_years = [], [], []
    for wy in args.train_years:
        target3d = xr.open_dataset(RAW_DIR / f"target3d_wy{wy}.nc")
        target2d = xr.open_dataset(RAW_DIR / f"target2d_wy{wy}.nc")
        pressure_head_years.append(target3d["pressure_head"].values)
        wtd_years.append(target2d["water_table_depth"].values)
        sm_years.append(target2d["soil_moisture"].values)
        target3d.close(); target2d.close()
        print(f"loaded WY{wy} ({pressure_head_years[-1].shape[0]} days)")

    results = {"pressure_head": {"mean": [], "std": []},
               "water_table_depth": {"mean": [], "std": []},
               "soil_moisture": {"mean": [], "std": []}}

    for k in range(1, future_len + 1):
        # Per-year deltas, then pooled across years -- never slide a window
        # across a year-concatenation boundary (see module docstring).
        ph_delta = np.concatenate([ph[k:] - ph[:ph.shape[0] - k] for ph in pressure_head_years], axis=0)
        wtd_delta = np.concatenate([w[k:] - w[:w.shape[0] - k] for w in wtd_years], axis=0)
        sm_delta = np.concatenate([s[k:] - s[:s.shape[0] - k] for s in sm_years], axis=0)

        ph_mean, ph_std = stats_5layer(ph_delta)
        wtd_mean, wtd_std = stats_1layer(wtd_delta)
        sm_mean, sm_std = stats_1layer(sm_delta)

        results["pressure_head"]["mean"].append(ph_mean)
        results["pressure_head"]["std"].append(ph_std)
        results["water_table_depth"]["mean"].append(wtd_mean)
        results["water_table_depth"]["std"].append(wtd_std)
        results["soil_moisture"]["mean"].append(sm_mean)
        results["soil_moisture"]["std"].append(sm_std)
        print(f"k={k}: wtd std={wtd_std:.4f}, sm std={sm_std:.4f}, ph std (per z)={ph_std}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
