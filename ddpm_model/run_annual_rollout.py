"""
Autoregressive full-year rollout: chain future_len-day windows across all of
WY2006 (the held-out test year), for every tile, using each hop's predicted
final-day state as the next hop's init (see README's forecast-strategy
section -- this was always the intended long-horizon usage, never exercised
until now).

Forcing (tmin/tmax/precipitation/evapotranspiration/snowmelt) is taken from
the real historical record for every hop, never predicted -- this is a
hindcast assuming perfect forcing knowledge, consistent with the "Forced" in
the FSTR paper's Forced-SpatioTemporal-RNN, not an operational forecast claim.

Supports the leaner (2026-09-17) water_table_depth+soil_moisture-only models
(config.model.include_pressure_head: false): pressure_head is never forecast
in that case, but it's still needed as conditioning for the next hop's init,
so it's carried forward via simple persistence (held at its real day-0 value)
rather than actively predicted -- justified by pressure_head's own slow
day-to-day variability (see README/compute_residual_normalization_stats.py).

Tiles (25% overlap) are mosaicked into full-domain daily maps by averaging
overlapping predictions. Ground truth for comparison is read directly from
the untiled raw NetCDFs (data/parflow_raw/target{2d,3d}_wy2006.nc).

Output: <out-dir>/
  - predicted_state.npz: {var: (n_days, [5,] 637, 665)} float32, physical units
    (no "pressure_head" key if include_pressure_head is false)
  - domain_mean_timeseries.csv: predicted vs actual domain-mean per variable per day
  - rmse_by_lead_position.json: RMSE at each within-hop lead position (position 0 =
    day right after a real/chained init), showing whether error compounds across hops

Usage:
    python run_annual_rollout.py --config config_tile256_f5_lean_residual.yaml \
        --checkpoint Results/groundwater_tile256_f5_lean_residual_100ep_8gpu/checkpoints/best_model.pt \
        --out-dir ../data/rollout_wy2006_tile256_f5_lean
"""
import argparse
import json
import os

import numpy as np
import torch
import xarray as xr

from dataset_groundwater import (
    N_STATIC, N_STATE, FORCING_VARS, STATIC_VARS_1LAYER, STATIC_VARS_5LAYER,
    _normalize, _load_state, _load_forcing, ResidualNormalizer,
)
from denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch import (
    ConditionalUnet, ConditionalGaussianDiffusion,
)
from eval_utils import denorm
from utils import load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "parflow_raw")
DOMAIN_SHAPE = (637, 665)


def load_tile_grid(tiles_root, tile_size):
    with open(os.path.join(tiles_root, f"tile{tile_size}", "tile_grid.json")) as f:
        return json.load(f)["tiles"]


def load_static_batch(tiles_root, tile_size, tiles, norm_stats, device):
    parts = []
    for tile_id in range(len(tiles)):
        npz = np.load(os.path.join(tiles_root, f"tile{tile_size}", "static", f"tile_{tile_id:03d}.npz"))
        p = [_normalize(v, npz[v], norm_stats)[None, ...] for v in STATIC_VARS_1LAYER]
        p += [_normalize(v, npz[v], norm_stats) for v in STATIC_VARS_5LAYER]
        parts.append(np.concatenate(p, axis=0))
    return torch.from_numpy(np.stack(parts, axis=0).astype(np.float32)).to(device)  # (n_tiles, N_STATIC, H, W)


def load_dynamic_npz(tiles_root, tile_size, tile_id, global_day):
    return np.load(os.path.join(tiles_root, f"tile{tile_size}", "dynamic", f"tile_{tile_id:03d}", f"day_{global_day:04d}.npz"))


def mosaic(tile_values, tiles, domain_shape):
    """tile_values: (n_tiles, C, h, w) numpy -> (C, *domain_shape), averaging overlaps."""
    C = tile_values.shape[1]
    acc = np.zeros((C, *domain_shape), dtype=np.float64)
    count = np.zeros(domain_shape, dtype=np.float64)
    for i, t in enumerate(tiles):
        sl = (slice(None), slice(t["y0"], t["y1"]), slice(t["x0"], t["x1"]))
        acc[sl] += tile_values[i]
        count[t["y0"]:t["y1"], t["x0"]:t["x1"]] += 1
    return (acc / count[None, ...]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-hops", type=int, default=None, help="smoke-test: only run the first N hops")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_config(args.config)
    tile_size = config.data.size[0]
    future_len = config.model.future_len
    include_pressure = getattr(config.model, "include_pressure_head", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config.data.normalization_stats) as f:
        norm_stats = json.load(f)

    target_mode = config.training.target_mode
    residual_normalizer = None
    if target_mode == "residual":
        with open(config.data.residual_normalization_stats) as f:
            residual_stats = json.load(f)
        residual_normalizer = ResidualNormalizer(future_len, norm_stats, residual_stats, device=device,
                                                  include_pressure=include_pressure)
    n_state = N_STATE if include_pressure else 2
    print(f"target_mode: {target_mode}, include_pressure_head: {include_pressure}, n_state: {n_state}")

    # test_years is normally a single water year (the held-out fold); rollout only
    # supports one contiguous year at a time.
    test_years_cfg = getattr(config.data, "test_years", None) or [2006]
    assert len(test_years_cfg) == 1, f"run_annual_rollout.py supports exactly one test year, got {test_years_cfg}"
    test_year = test_years_cfg[0]

    with open(os.path.join(config.data.tiles_root, "day_index.json")) as f:
        day_index = json.load(f)
    test_days = [i for i, e in enumerate(day_index) if e["water_year"] == test_year]
    day0 = min(test_days)
    n_days = len(test_days)
    # day0 itself is the init state, not a forecastable day -- only n_days-1 days
    # are actually available to forecast into. Using n_days directly here undercounts
    # by nothing when future_len doesn't evenly divide n_days (the remainder absorbs
    # it), but overshoots by exactly one hop when it does (confirmed 2026-09-17:
    # future_len=5 divides 365 evenly, 73*5=365 needs day 1461, which doesn't exist).
    n_full_hops = (n_days - 1) // future_len
    print(f"WY{test_year}: global_day {day0}..{day0 + n_days - 1} ({n_days} days), {n_full_hops} full {future_len}-day hops "
          f"+ {n_days - 1 - n_full_hops * future_len} leftover days")

    tiles = load_tile_grid(config.data.tiles_root, tile_size)
    n_tiles = len(tiles)
    static_batch = load_static_batch(config.data.tiles_root, tile_size, tiles, norm_stats, device)

    ch_cond = config.model.channels_static + config.model.channels_init + config.model.channels_forcing
    model = ConditionalUnet(
        dim=config.model.dim, channels=config.model.channels_output, channels_cond=ch_cond,
        dim_mults=config.model.dim_mults, with_conditioning=True,
    )
    diffusion = ConditionalGaussianDiffusion(
        model, image_size=tuple(config.data.size), timesteps=config.model.timesteps,
        objective=config.model.objective,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    diffusion.model.load_state_dict(checkpoint["model_state_dict"])
    diffusion.eval()
    print(f"loaded checkpoint: {args.checkpoint}")

    # init state at day0: real ground truth (first hop only -- every later hop's
    # init is the previous hop's own prediction, per the chained-rollout design).
    # Always the full 7-channel state, regardless of include_pressure: `init` is
    # conditioning either way, and pressure_head is still useful context even
    # when it isn't a forecast target.
    init_full = torch.from_numpy(np.stack(
        [_load_state(load_dynamic_npz(config.data.tiles_root, tile_size, t, day0), norm_stats) for t in range(n_tiles)],
        axis=0,
    ).astype(np.float32)).to(device)

    if include_pressure:
        cond_init = init_full  # (n_tiles, 7, H, W); replaced by the model's own prediction each hop
        ph_carry = None
    else:
        # pressure_head is never forecast -- carry it forward via persistence
        # (held at its real day-0 value) rather than updating it from the model,
        # justified by its own slow day-to-day variability.
        ph_carry = init_full[:, 0:5]
        wtd_sm_state = init_full[:, 5:7]
        cond_init = torch.cat([ph_carry, wtd_sm_state], dim=1)

    predicted_days = {}  # global_day -> (n_tiles, n_state, H, W) absolute-normalized numpy
    physics = torch.zeros(n_tiles, 4, device=device)

    n_hops = n_full_hops
    if args.max_hops is not None:
        n_hops = min(n_hops, args.max_hops)
    with torch.no_grad():
        for hop in range(n_hops):
            start_day = day0 + hop * future_len
            forcing_days = []
            for k in range(1, future_len + 1):
                fk = np.stack([_load_forcing(load_dynamic_npz(config.data.tiles_root, tile_size, t, start_day + k), norm_stats)
                               for t in range(n_tiles)], axis=0)
                forcing_days.append(fk)
            forcing_batch = torch.from_numpy(np.concatenate(forcing_days, axis=1).astype(np.float32)).to(device)

            cond = torch.cat([static_batch, cond_init, forcing_batch], dim=1)
            # Sample in small sub-batches: training only ever ran batch_size=4 per
            # GPU (distributed across 8 for the full batch_size=32 effective), so a
            # single GPU has never held more than 4 tiles' worth of activations
            # through the full diffusion sampling loop -- batching all 12 tiles at
            # once here OOM'd a 40GB GPU (confirmed 2026-09-17).
            chunk_size = 4
            pred_chunks = []
            for c0 in range(0, n_tiles, chunk_size):
                c1 = min(c0 + chunk_size, n_tiles)
                pred_chunks.append(diffusion.sample(
                    batch_size=c1 - c0, cond=cond[c0:c1], physics_cond=physics[c0:c1],
                ))
            pred_sample = torch.cat(pred_chunks, dim=0)
            if target_mode == "residual":
                init_for_residual = cond_init if include_pressure else cond_init[:, 5:7]
                pred_absnorm = residual_normalizer.from_residual(pred_sample) + init_for_residual.repeat(1, future_len, 1, 1)
            else:
                pred_absnorm = pred_sample

            pred_np = pred_absnorm.cpu().numpy()
            for k in range(1, future_len + 1):
                gday = start_day + k
                if gday >= day0 + n_days:
                    break
                block = pred_np[:, (k - 1) * n_state:k * n_state]
                predicted_days[gday] = block

            # next hop's init: model's own predicted state at the last day of this hop
            last_day_state = pred_absnorm[:, (future_len - 1) * n_state:future_len * n_state]
            if include_pressure:
                cond_init = last_day_state
            else:
                wtd_sm_state = last_day_state
                cond_init = torch.cat([ph_carry, wtd_sm_state], dim=1)
            print(f"hop {hop+1}/{n_hops}: days {start_day+1}..{start_day+future_len} done")

    # --- Mosaic predictions into full-domain daily maps, denormalize ---
    sorted_days = sorted(predicted_days.keys())
    wtd_pred, sm_pred, ph_pred = [], [], []
    for gday in sorted_days:
        block = predicted_days[gday]  # (n_tiles, n_state, H, W), normalized
        full = mosaic(block, tiles, DOMAIN_SHAPE)  # (n_state, 637, 665), still normalized
        if include_pressure:
            ph_pred.append(denorm("pressure_head", full[0:5], norm_stats))
            wtd_pred.append(denorm("water_table_depth", full[5:6], norm_stats)[0])
            sm_pred.append(denorm("soil_moisture", full[6:7], norm_stats)[0])
        else:
            wtd_pred.append(denorm("water_table_depth", full[0:1], norm_stats)[0])
            sm_pred.append(denorm("soil_moisture", full[1:2], norm_stats)[0])
    wtd_pred = np.stack(wtd_pred, axis=0)
    sm_pred = np.stack(sm_pred, axis=0)

    save_kwargs = dict(water_table_depth=wtd_pred, soil_moisture=sm_pred, global_days=np.array(sorted_days))
    if include_pressure:
        save_kwargs["pressure_head"] = np.stack(ph_pred, axis=0)
    np.savez_compressed(os.path.join(args.out_dir, "predicted_state.npz"), **save_kwargs)
    print(f"saved predicted_state.npz: {len(sorted_days)} days")

    # --- Ground truth (direct from raw NetCDF, no tiling needed) ---
    target2d = xr.open_dataset(os.path.join(RAW_DIR, f"target2d_wy{test_year}.nc"))
    local_idx = [g - day0 for g in sorted_days]  # test-year-local day index
    wtd_true = target2d["water_table_depth"].values[local_idx]
    sm_true = target2d["soil_moisture"].values[local_idx]
    if include_pressure:
        target3d = xr.open_dataset(os.path.join(RAW_DIR, f"target3d_wy{test_year}.nc"))
        ph_true = target3d["pressure_head"].values[local_idx]
    dates = [day_index[g]["date"] for g in sorted_days]

    # --- Domain-mean timeseries CSV ---
    import csv
    with open(os.path.join(args.out_dir, "domain_mean_timeseries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "global_day", "wtd_pred", "wtd_true", "sm_pred", "sm_true"])
        for i, g in enumerate(sorted_days):
            w.writerow([dates[i], g, wtd_pred[i].mean(), wtd_true[i].mean(), sm_pred[i].mean(), sm_true[i].mean()])
    print(f"wrote domain_mean_timeseries.csv")

    # --- RMSE vs. lead-time-since-last-real-init (does error compound across hops?) ---
    rmse_by_lead = {"water_table_depth": [], "soil_moisture": []}
    if include_pressure:
        rmse_by_lead["pressure_head"] = []
    for k in range(future_len):
        idxs = [i for i in range(len(sorted_days)) if i % future_len == k]
        if not idxs:
            continue
        rmse_by_lead["water_table_depth"].append(float(np.sqrt(np.mean((wtd_pred[idxs] - wtd_true[idxs]) ** 2))))
        rmse_by_lead["soil_moisture"].append(float(np.sqrt(np.mean((sm_pred[idxs] - sm_true[idxs]) ** 2))))
        if include_pressure:
            rmse_by_lead["pressure_head"].append(float(np.sqrt(np.mean((ph_pred[idxs] - ph_true[idxs]) ** 2))))
    with open(os.path.join(args.out_dir, "rmse_by_lead_position.json"), "w") as f:
        json.dump(rmse_by_lead, f, indent=2)
    print("RMSE by within-hop lead position (position 0 = day right after a real/chained init):")
    for name, vals in rmse_by_lead.items():
        print(f"  {name}: {['%.4f' % v for v in vals]}")

    print("DONE")


if __name__ == "__main__":
    main()
