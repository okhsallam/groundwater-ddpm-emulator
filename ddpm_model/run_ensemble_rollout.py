"""
Ensemble autoregressive rollout: N independent members over a full water year.

A diffusion model is generative -- each call is a draw from a conditional
distribution, not a point estimate. Running the chained rollout once and
reporting that single trajectory discards the model's own statement about how
uncertain it is. This script draws N members from the same initial condition and
the same forcing, differing only in the sampling noise, so the spread across
members is the model's forecast uncertainty and grows with lead time as it
should.

Two implementation points that matter:

  * Members share the data loading. The tile archives, the static maps and the
    forcing for each window are read ONCE per hop and reused across all members,
    with only the diffusion sampling repeated. Members are folded into the batch
    dimension (N x n_tiles), so cost scales with N in compute but not in I/O.

  * Physical bounds are enforced INSIDE the loop, at every hop, before a state
    becomes the next hop's initial condition (see physical_bounds.py for why a
    clamp rather than a sigmoid or a log transform). Clamping only the saved
    output would leave the trajectory itself contaminated.

Outputs into --out-dir:
    ensemble_state.npz    water_table_depth / soil_moisture, both
                          (n_members, n_days, H, W), plus global_days
    ensemble_mean.npz     member mean and standard deviation per day
    rollout_meta.json     members, seed, clamp statistics per hop

Usage:
    python run_ensemble_rollout.py --config config_fstr_f15_lean_residual.yaml \
        --checkpoint Results/<run>/checkpoints/final_model.pt \
        --out-dir ../data/rollout_wy2006_fstr_f15_ens --n-members 16
"""
import argparse
import json
import os

import numpy as np
import torch

from run_annual_rollout import (
    load_tile_grid, load_static_batch, load_dynamic_npz, mosaic, DOMAIN_SHAPE,
)
from dataset_groundwater import (_load_state, _load_wtd_sm, _load_forcing,
                                 ResidualNormalizer, resolve_forcing_vars)
from denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch import (
    ConditionalUnet, ConditionalGaussianDiffusion,
)
from eval_utils import denorm
from utils import load_config
from physical_bounds import PhysicalBounds, load_porosity_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-members", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk-size", type=int, default=4,
                    help="tiles*members per diffusion sampling call; 4 fits a 40 GB GPU")
    ap.add_argument("--max-hops", type=int, default=None, help="smoke test")
    ap.add_argument("--no-clamp", action="store_true",
                    help="disable the physical bounds, for an ablation against them")
    ap.add_argument("--bounds", choices=["full", "lower"], default="full",
                    help="'lower' enforces only WTD>=0 and SM>=0; 'full' (default) also "
                         "caps SM at porosity. The cap is not cosmetic: in the fold1 "
                         "f15 rollout 16.6%% of cell-days exceeded porosity, against "
                         "0.2%% below zero, with a worst excess of 0.60 -- more than "
                         "double the available pore space.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_config(args.config)
    tile_size = config.data.size[0]
    future_len = config.model.future_len
    include_pressure = getattr(config.model, "include_pressure_head", True)
    if include_pressure:
        raise SystemExit("ensemble rollout is implemented for the lean "
                         "(water_table_depth + soil_moisture) models only")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_state = 2
    M = args.n_members
    # Conditioning provenance. 'nldas' keeps only tmin/tmax/precipitation, which
    # are NLDAS-2 reanalysis; it drops evapotranspiration and snowmelt, which are
    # ParFlow-CLM outputs and would otherwise be fed to the emulator at every hop.
    forcing_vars = resolve_forcing_vars(getattr(config.data, "forcing_set", "all"))
    ph_in_init = bool(getattr(config.model, "include_pressure_in_init", True))
    print(f"forcing channels ({len(forcing_vars)}): {', '.join(forcing_vars)}")
    print(f"pressure head in conditioning: {ph_in_init}")

    with open(config.data.normalization_stats) as f:
        norm_stats = json.load(f)
    with open(config.data.residual_normalization_stats) as f:
        residual_stats = json.load(f)
    residual_normalizer = ResidualNormalizer(future_len, norm_stats, residual_stats,
                                             device=device, include_pressure=False)

    test_years_cfg = getattr(config.data, "test_years", None) or [2006]
    assert len(test_years_cfg) == 1
    test_year = test_years_cfg[0]
    with open(os.path.join(config.data.tiles_root, "day_index.json")) as f:
        day_index = json.load(f)
    test_days = [i for i, e in enumerate(day_index) if e["water_year"] == test_year]
    day0, n_days = min(test_days), len(test_days)
    n_hops = (n_days - 1) // future_len
    if args.max_hops:
        n_hops = min(n_hops, args.max_hops)

    tiles = load_tile_grid(config.data.tiles_root, tile_size)
    T = len(tiles)
    static_batch = load_static_batch(config.data.tiles_root, tile_size, tiles, norm_stats, device)
    print(f"WY{test_year}: {n_days} days, {n_hops} hops of {future_len} d, "
          f"{T} tiles, {M} members")

    bounds = None
    if not args.no_clamp:
        import numpy as _np
        if args.bounds == "full":
            por = load_porosity_top(config.data.tiles_root, tile_size, tiles)  # (T,H,W)
            # The sampling batch is the members folded in front of the tiles and
            # flattened, so element i corresponds to tile i % T. The bound has to
            # be tiled to (M*T, 1, H, W) to line up; leaving it at (T, 1, H, W)
            # broadcast-fails as soon as M > 1.
            por = _np.tile(por[:, None, :, :], (M, 1, 1, 1))                   # (M*T,1,H,W)
            msg = "WTD >= 0, 0 <= SM <= porosity(z=4)"
        else:
            por = _np.full((1, 1, 1, 1), 1e6, dtype=_np.float32)               # cap disabled
            msg = "WTD >= 0, SM >= 0 (upper cap disabled by --bounds lower)"
        bounds = PhysicalBounds(norm_stats, por, device)
        print(f"physical bounds active: {msg}")

    ch_cond = (config.model.channels_static + config.model.channels_init
               + config.model.channels_forcing)
    model = ConditionalUnet(dim=config.model.dim, channels=config.model.channels_output,
                            channels_cond=ch_cond, dim_mults=config.model.dim_mults,
                            with_conditioning=True)
    diffusion = ConditionalGaussianDiffusion(
        model, image_size=tuple(config.data.size), timesteps=config.model.timesteps,
        objective=config.model.objective).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    diffusion.model.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()
    print(f"loaded checkpoint: {args.checkpoint}")

    loader = _load_state if ph_in_init else _load_wtd_sm
    init_full = torch.from_numpy(np.stack(
        [loader(load_dynamic_npz(config.data.tiles_root, tile_size, t, day0), norm_stats)
         for t in range(T)], axis=0).astype(np.float32)).to(device)   # (T,{7,2},H,W)
    if ph_in_init:
        ph_carry = init_full[:, 0:5].unsqueeze(0).expand(M, -1, -1, -1, -1)
        state = init_full[:, 5:7].unsqueeze(0).repeat(M, 1, 1, 1, 1)
    else:
        ph_carry = None
        state = init_full.unsqueeze(0).repeat(M, 1, 1, 1, 1)                # (M,T,2,H,W)

    static_M = static_batch.unsqueeze(0).expand(M, -1, -1, -1, -1)
    physics = torch.zeros(M * T, 4, device=device)
    torch.manual_seed(args.seed)

    predicted = {}                 # global_day -> (M, T, 2, H, W) normalised
    clamp_log = []
    with torch.no_grad():
        for hop in range(n_hops):
            start_day = day0 + hop * future_len
            forcing = np.concatenate([
                np.stack([_load_forcing(
                    load_dynamic_npz(config.data.tiles_root, tile_size, t, start_day + k),
                    norm_stats, forcing_vars) for t in range(T)], axis=0)
                for k in range(1, future_len + 1)], axis=1)
            forcing = torch.from_numpy(forcing.astype(np.float32)).to(device)   # (T,75,H,W)
            forcing_M = forcing.unsqueeze(0).expand(M, -1, -1, -1, -1)

            cond_init = torch.cat([ph_carry, state], dim=2) if ph_in_init else state
            cond = torch.cat([static_M, cond_init, forcing_M], dim=2)
            cond = cond.reshape(M * T, -1, *cond.shape[-2:])

            chunks = []
            for c0 in range(0, M * T, args.chunk_size):
                c1 = min(c0 + args.chunk_size, M * T)
                chunks.append(diffusion.sample(batch_size=c1 - c0, cond=cond[c0:c1],
                                               physics_cond=physics[c0:c1]))
            pred = torch.cat(chunks, dim=0)                                     # (M*T,30,H,W)

            init_flat = state.reshape(M * T, n_state, *state.shape[-2:])
            absnorm = (residual_normalizer.from_residual(pred)
                       + init_flat.repeat(1, future_len, 1, 1))

            if bounds is not None:
                clamp_log.append(bounds.report(absnorm, n_state, future_len,
                                               label=f"hop {hop+1}: "))
                absnorm = bounds.clamp_block(absnorm, n_state, future_len)

            absnorm = absnorm.reshape(M, T, future_len * n_state, *state.shape[-2:])
            for k in range(1, future_len + 1):
                gday = start_day + k
                if gday >= day0 + n_days:
                    break
                predicted[gday] = absnorm[:, :, (k - 1) * n_state:k * n_state].cpu().numpy()

            state = absnorm[:, :, (future_len - 1) * n_state:future_len * n_state].contiguous()
            if clamp_log:
                print(f"hop {hop+1}/{n_hops} done | {clamp_log[-1]}")
            else:
                print(f"hop {hop+1}/{n_hops} done")

    days = sorted(predicted)
    wtd = np.empty((M, len(days), *DOMAIN_SHAPE), dtype=np.float32)
    sm = np.empty_like(wtd)
    for j, g in enumerate(days):
        for m in range(M):
            full = mosaic(predicted[g][m], tiles, DOMAIN_SHAPE)
            wtd[m, j] = denorm("water_table_depth", full[0:1], norm_stats)[0]
            sm[m, j] = denorm("soil_moisture", full[1:2], norm_stats)[0]

    np.savez_compressed(os.path.join(args.out_dir, "ensemble_state.npz"),
                        water_table_depth=wtd, soil_moisture=sm,
                        global_days=np.array(days))
    np.savez_compressed(os.path.join(args.out_dir, "ensemble_mean.npz"),
                        water_table_depth=wtd.mean(0), soil_moisture=sm.mean(0),
                        water_table_depth_sd=wtd.std(0, ddof=1),
                        soil_moisture_sd=sm.std(0, ddof=1),
                        global_days=np.array(days))
    json.dump({"n_members": M, "seed": args.seed, "test_year": int(test_year),
               "future_len": future_len, "n_hops": n_hops,
               "clamped": not args.no_clamp, "clamp_log": clamp_log},
              open(os.path.join(args.out_dir, "rollout_meta.json"), "w"), indent=1)
    print(f"\nsaved {len(days)} days x {M} members -> {args.out_dir}")
    print(f"  min WTD across members: {wtd.min():.3f} m  (0 expected with clamping)")
    print(f"  SM range: {sm.min():.4f} to {sm.max():.4f}")


if __name__ == "__main__":
    main()
