"""
Post-training evaluation on the held-out WY2006 test split.

Full-sampling evaluation over the whole test set is impractical (diffusion
`sample()` runs `timesteps` forward passes per batch, and the test set alone
has thousands of tile-windows) so this draws a random subset (`--n-eval`,
default 50) of test windows, runs the trained model's reverse-diffusion
sampling on them, and reports per-variable RMSE broken down by forecast lead
day (1..future_len) -- shows whether/how forecast skill degrades with lead
time, the standard diagnostic for this kind of multi-step model.

Usage:
    python evaluate_test.py --config config_groundwater_tile128.yaml \
        --checkpoint Results/groundwater_tile128_50ep_8gpu/checkpoints/best_model.pt \
        --out-dir Results/groundwater_tile128_50ep_8gpu/test_eval
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset_groundwater import (GroundwaterTileDataset, broadcast_init,
                                 ResidualNormalizer, resolve_forcing_vars)
from denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch import (
    ConditionalUnet, ConditionalGaussianDiffusion,
)
from eval_utils import rmse_by_lead_day, plot_rmse_curves, STATE_VAR_BLOCKS, STATE_VAR_BLOCKS_LEAN
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--persistence-baseline", default=None,
                         help="path to a persistence_rmse_by_lead_day.json from "
                              "evaluate_persistence_baseline.py, to overlay on the plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config.data.normalization_stats) as f:
        norm_stats = json.load(f)

    target_mode = config.training.target_mode
    include_pressure = getattr(config.model, "include_pressure_head", True)
    residual_normalizer = None
    if target_mode == "residual":
        with open(config.data.residual_normalization_stats) as f:
            residual_stats = json.load(f)
        residual_normalizer = ResidualNormalizer(config.model.future_len, norm_stats, residual_stats, device=device,
                                                  include_pressure=include_pressure)

    test_dataset = GroundwaterTileDataset(
        config.data.tiles_root, tile_size=config.data.size[0],
        future_len=config.model.future_len, split="test",
        normalization_stats=norm_stats,
        include_pressure_in_target=include_pressure,
        train_years=getattr(config.data, "train_years", None),
        val_years=getattr(config.data, "val_years", None),
        test_years=getattr(config.data, "test_years", None),
        forcing_vars=resolve_forcing_vars(getattr(config.data, "forcing_set", "all")),
        include_pressure_in_init=getattr(config.model, "include_pressure_in_init", True),
    )
    n_init_ch = int(config.model.channels_init)
    print(f"test_dataset: {len(test_dataset)} windows total, sampling {args.n_eval}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(test_dataset), size=min(args.n_eval, len(test_dataset)), replace=False)
    eval_loader = DataLoader(Subset(test_dataset, idx), batch_size=8, shuffle=False, num_workers=4)

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

    all_preds, all_targets = [], []
    with torch.no_grad():
        for target, cond in eval_loader:
            target, cond = target.to(device), cond.to(device)
            physics = torch.zeros(target.shape[0], 4, device=device)
            pred_sample = diffusion.sample(batch_size=target.shape[0], cond=cond, physics_cond=physics)
            if target_mode == "residual":
                # Model was trained on residual-normalized units; convert back to
                # absolute state for a physically meaningful RMSE against ground truth.
                pred = residual_normalizer.from_residual(pred_sample) + broadcast_init(cond, config.model.future_len, include_pressure, n_init_ch)
            else:
                pred = pred_sample
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            print(f"  sampled batch of {target.shape[0]} ({sum(p.shape[0] for p in all_preds)}/{len(idx)})")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    blocks = STATE_VAR_BLOCKS if include_pressure else STATE_VAR_BLOCKS_LEAN
    results = rmse_by_lead_day(all_preds, all_targets, config.model.future_len, norm_stats, state_var_blocks=blocks)
    with open(os.path.join(args.out_dir, "test_rmse_by_lead_day.json"), "w") as f:
        json.dump(results, f, indent=2)

    persistence = None
    if args.persistence_baseline and os.path.exists(args.persistence_baseline):
        with open(args.persistence_baseline) as f:
            persistence = json.load(f)

    plot_rmse_curves(results, os.path.join(args.out_dir, "test_rmse_by_lead_day.png"),
                      title=f"Test-set RMSE vs. forecast lead day -- {os.path.basename(args.out_dir)}",
                      other=persistence, other_label="persistence baseline")

    print("Test RMSE by lead day (physical units):")
    for name, values in results.items():
        line = f"  {name}: day1={values[0]:.4f} ... day{len(values)}={values[-1]:.4f}"
        if persistence:
            line += f"  (persistence: day1={persistence[name][0]:.4f} ... day{len(values)}={persistence[name][-1]:.4f})"
        print(line)
    print(f"wrote {args.out_dir}/test_rmse_by_lead_day.{{json,png}}")


if __name__ == "__main__":
    main()
