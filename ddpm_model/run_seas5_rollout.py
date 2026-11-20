"""
Autoregressive rollout of the trained groundwater emulator driven by SEAS5
seasonal forecast forcing instead of NLDAS-2 reanalysis.

NOTHING IS RETRAINED AND NO CHECKPOINT IS TOUCHED. The current model conditions
on 22 static channels, 2 initial-state channels and 15 x 3 forcing channels
(tmin, tmax, precipitation), all three of which a seasonal forecast system also
predicts. This script swaps those 45 channels for a forecast and leaves the rest
of the rollout -- tiling, residual denormalisation, the in-loop physical-bounds
projection -- bit-identical to run_ensemble_rollout.py.

NORMALISATION IS DELIBERATELY *NOT* RECOMPUTED FROM SEAS5. The fold's own
NLDAS-2 statistics are used, unchanged. The network's weights were learned
against inputs on that scale; re-deriving mean/std from the forecast would
silently redefine what every channel means to the model and make the comparison
with arm A meaningless. Making SEAS5 live inside the training distribution is
the job of the bias correction in prepare_seas5_domain.py, not of the z-score.

ARMS -- each differs from the one above it in exactly one respect:

  A   NLDAS-2 at 1 km                     emulator error alone. Already run; this
                                          script does not reproduce it (use
                                          run_ensemble_rollout.py).
  B   NLDAS-2 coarsened to the 1 deg
      SEAS5 frame and interpolated back        + resolution loss alone
  B2  ERA5 through the same path, plus the
      stage-2 downscaling delta                + the whole correction chain,
                                                 driven by a perfect forecast
  C   SEAS5, bias-corrected                    + genuine forecast error
  D   SEAS5, raw                               what the correction buys
  E   NLDAS-2 day-of-year climatology      the no-forecast-information reference

Arm B2 is what makes arm C readable: if arm C is poor but arm B2 is too, the
limit is this pipeline, not seasonal predictability.

ENSEMBLE STRUCTURE. Arms C and D carry a 25-member forcing ensemble; every arm
also draws diffusion members from the sampler's own noise. The outer loop is
over FORCING members, because each one needs its own 14-hop chain with its own
carried state -- which is also what makes the work shardable: pass disjoint
--members to several jobs and merge afterwards. Within one forcing member the
diffusion members share all I/O and fold into the batch dimension, exactly as
run_ensemble_rollout.py does.

This is the experiment's other point of interest. The emulator is known to be
UNDER-DISPERSED (spread-to-error 0.43, U-shaped rank histogram) because sampler
noise is its only perturbation. A forcing ensemble supplies the physically
correct missing source, so spread-to-error moving toward 1 in arm C would close
a limitation rather than merely add an application. Keeping the two member
dimensions separate on disk is what allows that decomposition.

ONE INITIALISATION IS NOT A WATER YEAR. SEAS5 reaches 215 days, so 14 hops of
the 15-day window = 210 forecast days, roughly Oct-Apr for an Oct init. Every
arm therefore defaults to 14 hops so all of them are scored over identical days;
arm A's existing 360-day rollout is truncated to the same window at analysis
time.

STORAGE. Members are written float16: ~0.03 m resolution near a 50 m water
table and ~5e-4 in soil moisture, against RMSEs of 0.42 m and 0.035. That is
two orders of magnitude below the error being measured, and it halves 25
members x 210 days x 2 variables from 71 GB to 36 GB.

Usage (one arm, one job):
    python run_seas5_rollout.py --config config_f15_fold1.yaml \
        --checkpoint Results/groundwater_f15_200ep_fold1/checkpoints/final_model.pt \
        --arm B --out-dir Results/seas5/armB --n-diff 16

    # arm C, sharded over 8 GPUs: --members 0-2, 3-5, ... in 8 jobs
    python run_seas5_rollout.py ... --arm C --n-diff 4 --members 0-2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_annual_rollout import (load_tile_grid, load_static_batch,
                                load_dynamic_npz, mosaic, DOMAIN_SHAPE)
from dataset_groundwater import _load_wtd_sm, ResidualNormalizer
from denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch import (
    ConditionalUnet, ConditionalGaussianDiffusion,
)
from eval_utils import denorm
from utils import load_config
from physical_bounds import PhysicalBounds, load_porosity_top
from seas5_forcing import ArmForcing, CoarseMap, FORCING_VARS


def parse_members(spec, n_available):
    """'0-5', '0,3,7' or 'all' -> a list of forcing-member indices."""
    if spec in (None, "all"):
        return list(range(n_available))
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [m for m in out if m < 0 or m >= n_available]
    if bad:
        raise SystemExit(f"members {bad} outside 0..{n_available - 1}")
    return sorted(set(out))


def forcing_tensor(source, gdays, member, tiles, norm_stats, device):
    """(n_tiles, 3*len(gdays), h, w), day-major, normalised with the FOLD's
    NLDAS-2 statistics -- see the module docstring on why not SEAS5's own."""
    phys = source.fields(gdays, member)                 # {var: (K,637,665)}
    blocks = []
    for k in range(len(gdays)):
        chans = []
        for v in FORCING_VARS:
            s = norm_stats[v]
            chans.append((phys[v][k] - s["mean"]) / s["std"])
        day = np.stack(chans, axis=0)                   # (3,637,665)
        blocks.append(np.stack([day[:, t["y0"]:t["y1"], t["x0"]:t["x1"]]
                                for t in tiles], axis=0))
    arr = np.concatenate(blocks, axis=1).astype(np.float32)
    return torch.from_numpy(arr).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["B", "B2", "C", "D", "Dp", "E", "Dyear"])
    ap.add_argument("--init", default=None,
                    help="SEAS5 initialisation tag, e.g. 200510. Required when "
                         "more than one is prepared, so a forecast is never "
                         "scored against the wrong water year.")
    ap.add_argument("--n-diff", type=int, default=None,
                    help="diffusion members per forcing member; default 16 for "
                         "the single-trajectory arms (B, B2, E) and 4 for the "
                         "ensemble forecast arms (C, D, Dp)")
    ap.add_argument("--members", default="all",
                    help="forcing members, e.g. '0-2' -- for sharding arms C/D")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--max-hops", type=int, default=14,
                    help="14 x 15 d = 210 d, the reach of one SEAS5 init")
    ap.add_argument("--no-clamp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_config(args.config)
    tile_size = config.data.size[0]
    future_len = config.model.future_len
    if getattr(config.model, "include_pressure_head", True) or \
       getattr(config.model, "include_pressure_in_init", True):
        raise SystemExit("implemented for the lean (WTD + SM) models only")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_state = 2

    with open(config.data.normalization_stats) as f:
        norm_stats = json.load(f)
    with open(config.data.residual_normalization_stats) as f:
        residual_stats = json.load(f)
    residual_normalizer = ResidualNormalizer(future_len, norm_stats, residual_stats,
                                             device=device, include_pressure=False)

    cmap = CoarseMap()
    # The fold is implied by the config's test year, so the per-fold BCSD
    # disaggregation field is selected automatically rather than by a flag the
    # caller could get wrong.
    _fold_of = {2006: 1, 2005: 2, 2004: 3, 2003: 4}
    _tyears = getattr(config.data, "test_years", None) or [2006]
    source = ArmForcing(args.arm, cmap=cmap, init=args.init,
                        fold=_fold_of.get(int(_tyears[0])))
    members = parse_members(args.members, source.n_members)
    M = args.n_diff if args.n_diff else (4 if args.arm in ("C", "D", "Dp") else 16)

    test_year = (getattr(config.data, "test_years", None) or [2006])[0]

    # THE CHECK THAT MATTERS MOST HERE. A 1 October initialisation opens the
    # NEXT water year, and the config decides which year the truth comes from.
    # Pair the 2004-10 forecast with fold 1 by accident and every score is
    # computed against a different year's weather -- numbers appear, nothing
    # errors, and the result is meaningless. Fail instead.
    init_ts = getattr(source, "_init", None)
    if init_ts is not None:
        opens = init_ts.year + (1 if init_ts.month >= 10 else 0)
        if opens != test_year:
            raise SystemExit(
                f"initialisation {init_ts.date()} opens water year {opens}, but "
                f"{args.config} tests on WY{test_year}. Pass the matching "
                f"--init, or the matching config: the forecast would otherwise "
                f"be scored against the wrong year's truth.")
    with open(os.path.join(config.data.tiles_root, "day_index.json")) as f:
        day_index = json.load(f)
    test_days = [i for i, e in enumerate(day_index) if e["water_year"] == test_year]
    day0, n_days = min(test_days), len(test_days)
    n_hops = min((n_days - 1) // future_len, args.max_hops)
    if n_hops < 1:
        raise SystemExit(f"--max-hops {args.max_hops} leaves no hops to run")
    n_fc = n_hops * future_len

    tiles = load_tile_grid(config.data.tiles_root, tile_size)
    T = len(tiles)
    static_batch = load_static_batch(config.data.tiles_root, tile_size, tiles,
                                     norm_stats, device)
    print(f"arm {args.arm}: {source.label}")
    print(f"WY{test_year} from day {day0} ({day_index[day0]['date']}), "
          f"{n_hops} hops x {future_len} d = {n_fc} forecast days")
    print(f"forcing members {members} of {source.n_members}; "
          f"{M} diffusion members each; {T} tiles", flush=True)

    bounds = None
    if not args.no_clamp:
        por = load_porosity_top(config.data.tiles_root, tile_size, tiles)
        por = np.tile(por[:, None, :, :], (M, 1, 1, 1))
        bounds = PhysicalBounds(norm_stats, por, device)
        print("physical bounds active: WTD >= 0, 0 <= SM <= porosity(z=4)")

    ch_cond = (config.model.channels_static + config.model.channels_init
               + config.model.channels_forcing)
    model = ConditionalUnet(dim=config.model.dim,
                            channels=config.model.channels_output,
                            channels_cond=ch_cond,
                            dim_mults=config.model.dim_mults,
                            with_conditioning=True)
    diffusion = ConditionalGaussianDiffusion(
        model, image_size=tuple(config.data.size),
        timesteps=config.model.timesteps,
        objective=config.model.objective).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    diffusion.model.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()
    print(f"loaded checkpoint: {args.checkpoint}", flush=True)

    init_full = torch.from_numpy(np.stack(
        [_load_wtd_sm(load_dynamic_npz(config.data.tiles_root, tile_size, t, day0),
                      norm_stats) for t in range(T)],
        axis=0).astype(np.float32)).to(device)                   # (T,2,H,W)
    static_M = static_batch.unsqueeze(0).expand(M, -1, -1, -1, -1)
    physics = torch.zeros(M * T, 4, device=device)

    meta = {"arm": args.arm, "label": source.label, "members": members,
            "init": (str(init_ts.date()) if init_ts is not None else None),
            "n_diff": M, "seed": args.seed, "test_year": int(test_year),
            "future_len": future_len, "n_hops": n_hops, "day0": int(day0),
            "clamped": not args.no_clamp, "clamp_log": {}}

    for fm in members:
        torch.manual_seed(args.seed + 1000 * fm)
        state = init_full.unsqueeze(0).repeat(M, 1, 1, 1, 1)      # (M,T,2,H,W)
        predicted, clamp_log = {}, []
        with torch.no_grad():
            for hop in range(n_hops):
                start_day = day0 + hop * future_len
                gdays = list(range(start_day + 1, start_day + future_len + 1))
                forcing = forcing_tensor(source, gdays, fm, tiles,
                                         norm_stats, device)      # (T,45,H,W)
                cond = torch.cat([static_M, state,
                                  forcing.unsqueeze(0).expand(M, -1, -1, -1, -1)],
                                 dim=2)
                cond = cond.reshape(M * T, -1, *cond.shape[-2:])

                chunks = []
                for c0 in range(0, M * T, args.chunk_size):
                    c1 = min(c0 + args.chunk_size, M * T)
                    chunks.append(diffusion.sample(
                        batch_size=c1 - c0, cond=cond[c0:c1],
                        physics_cond=physics[c0:c1]))
                pred = torch.cat(chunks, dim=0)

                init_flat = state.reshape(M * T, n_state, *state.shape[-2:])
                absnorm = (residual_normalizer.from_residual(pred)
                           + init_flat.repeat(1, future_len, 1, 1))
                if bounds is not None:
                    clamp_log.append(bounds.report(absnorm, n_state, future_len,
                                                   label=f"hop {hop+1}: "))
                    absnorm = bounds.clamp_block(absnorm, n_state, future_len)
                absnorm = absnorm.reshape(M, T, future_len * n_state,
                                          *state.shape[-2:])
                for k in range(1, future_len + 1):
                    predicted[start_day + k] = absnorm[
                        :, :, (k - 1) * n_state:k * n_state].cpu().numpy()
                state = absnorm[:, :, (future_len - 1) * n_state:
                                      future_len * n_state].contiguous()
                tail = f" | {clamp_log[-1]}" if clamp_log else ""
                print(f"  member {fm}  hop {hop+1}/{n_hops}{tail}", flush=True)

        days = sorted(predicted)
        wtd = np.empty((M, len(days), *DOMAIN_SHAPE), dtype=np.float16)
        sm = np.empty_like(wtd)
        for j, g in enumerate(days):
            for m in range(M):
                full = mosaic(predicted[g][m], tiles, DOMAIN_SHAPE)
                wtd[m, j] = denorm("water_table_depth", full[0:1],
                                   norm_stats)[0].astype(np.float16)
                sm[m, j] = denorm("soil_moisture", full[1:2],
                                  norm_stats)[0].astype(np.float16)
        # Write to a temporary name and rename, so the final path never exists
        # in a half-written state. np.savez_compressed streams a zip archive
        # straight to its target, so a reader that opens the file mid-write gets
        # BadZipFile -- which is what happened to an analysis script reading
        # results while a later shard was still running. os.replace is atomic
        # within a filesystem, so a member file is either absent or complete.
        p = os.path.join(args.out_dir, f"member_{fm:03d}.npz")
        # The temporary name MUST end in .npz: np.savez_compressed appends
        # ".npz" to any path that does not, so a ".part" suffix produced
        # "member_000.npz.part.npz" on disk while os.replace looked for
        # "member_000.npz.part" and raised FileNotFoundError -- after the
        # expensive work was already done. This destroyed three fold-arms
        # before it was caught.
        tmp = p[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, water_table_depth=wtd, soil_moisture=sm,
                            global_days=np.array(days))
        os.replace(tmp, p)
        meta["clamp_log"][str(fm)] = clamp_log
        print(f"  member {fm} -> {p}  ({os.path.getsize(p)/1e9:.2f} GB)  "
              f"WTD {float(wtd.min()):.3f}..{float(wtd.max()):.3f} m  "
              f"SM {float(sm.min()):.4f}..{float(sm.max()):.4f}", flush=True)

    mp = os.path.join(args.out_dir, f"rollout_meta_{members[0]:03d}.json")
    json.dump(meta, open(mp, "w"), indent=1)
    print(f"\ndone: {len(members)} forcing member(s) x {M} diffusion members, "
          f"{n_fc} days -> {args.out_dir}")


if __name__ == "__main__":
    main()
