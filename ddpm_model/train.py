import os
import csv
import json
import time
import argparse
import shutil
from datetime import datetime

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from plotting_utils import plot_loss_curve, plot_profiling_breakdown, plot_groundwater_wtd
from utils import load_config

# Default worker IPC strategy ('file_descriptor') passes batches between
# DataLoader workers and the main process via /dev/shm, whose size on a PBS
# compute node's cgroup can be much smaller than on the login node -- with
# 8 accelerate processes x num_workers DataLoader workers each shuttling
# large multi-channel batches (case4's 119-channel conditioning tensor is the
# heaviest), this reliably exhausted it and killed workers with a bus error.
# 'file_system' routes the same IPC through temp files instead, sidestepping
# the /dev/shm size entirely.
torch.multiprocessing.set_sharing_strategy('file_system')

# IMPORT THE TILED GROUNDWATER DATASET CLASS
from dataset_groundwater import (GroundwaterTileDataset, broadcast_init,
                                 ResidualNormalizer, resolve_forcing_vars)

# Import Model
from denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch_modified.denoising_diffusion_pytorch import ConditionalUnet, ConditionalGaussianDiffusion


def create_directories(base_dir):
    dirs = {
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'figures': os.path.join(base_dir, 'Figures')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def create_model(image_size, channels, channels_cond, dim, dim_mults, timesteps, objective):
    model = ConditionalUnet(
        dim=dim,
        channels=channels,
        channels_cond=channels_cond,
        dim_mults=dim_mults,
        with_conditioning=True
    )
    diffusion = ConditionalGaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        objective=objective,
    )
    return model, diffusion


def save_checkpoint(accelerator, diffusion, save_path):
    """Save only the inner ConditionalUnet weights, matching the prior checkpoint format."""
    unwrapped_diffusion = accelerator.unwrap_model(diffusion)
    accelerator.save({'model_state_dict': unwrapped_diffusion.model.state_dict()}, save_path)


def train(config, diffusion, accelerator, save_dirs, run_name=None):
    lr = config.training.lr
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)

    best_loss = float('inf')
    log_file = None
    metrics_csv_path = os.path.join(save_dirs['logs'], 'metrics.csv')

    if accelerator.is_main_process:
        log_file = open(os.path.join(save_dirs['logs'], 'training_log.txt'), 'w', buffering=1)

        # Per-epoch metrics (loss + a lightweight data/forward/backward time
        # breakdown) written every epoch so progress/profiling can be monitored
        # from disk (CSV + regenerated figures) without waiting for the run to finish.
        metrics_fields = ['epoch', 'avg_loss', 'val_loss', 'epoch_time_s', 'data_ms_per_batch',
                           'fwd_ms_per_batch', 'bwd_ms_per_batch']
        with open(metrics_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(metrics_fields)


    with open(config.data.normalization_stats) as f:
        norm_stats = json.load(f)

    # target_mode: "residual" trains on (state - init), rescaled to the residual's
    # own per-lead-day variance (see dataset_groundwater.ResidualNormalizer);
    # "absolute" trains on the raw absolute-normalized state, no init subtraction
    # at all -- init is still given as conditioning either way, only the
    # diffusion TARGET differs. Comparing both is the point of this ablation
    # round (2026-09-17): does the residual reparameterization actually help
    # once the window-length confound is controlled for.
    target_mode = config.training.target_mode
    assert target_mode in ("residual", "absolute"), f"unknown target_mode {target_mode!r}"
    # include_pressure_head=False: the leaner (2026-09-17) models that forecast
    # only water_table_depth+soil_moisture -- pressure_head stays in `init`
    # conditioning but is dropped from the prediction target entirely.
    include_pressure = getattr(config.model, "include_pressure_head", True)
    residual_normalizer = None
    if target_mode == "residual":
        with open(config.data.residual_normalization_stats) as f:
            residual_stats = json.load(f)
        residual_normalizer = ResidualNormalizer(
            config.model.future_len, norm_stats, residual_stats, device=accelerator.device,
            include_pressure=include_pressure,
        )

    def make_dataset(split):
        return GroundwaterTileDataset(
            config.data.tiles_root,
            tile_size=config.data.size[0],
            future_len=config.model.future_len,
            split=split,
            normalization_stats=norm_stats,
            include_pressure_in_target=include_pressure,
            train_years=getattr(config.data, "train_years", None),
            val_years=getattr(config.data, "val_years", None),
            forcing_vars=resolve_forcing_vars(getattr(config.data, "forcing_set", "all")),
            include_pressure_in_init=getattr(config.model, "include_pressure_in_init", True),
            test_years=getattr(config.data, "test_years", None),
        )

    accelerator.print("Initializing Training Dataset from tiles...")
    train_dataset = make_dataset('train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Validation (WY2005, monitored every epoch: loss metric + periodic figure)
    # is loaded on ALL ranks and run through accelerator.prepare like the train
    # loader, so the loss is computed distributed rather than bottlenecked on
    # rank 0. WY2006 (test) is intentionally never loaded here -- it stays
    # untouched during training, for a final unbiased evaluation elsewhere
    # (proper train/val/test discipline, not train/test reused as both the
    # monitored set and the final one).
    # FSTR-style runs have no validation split: with four water years the paper
    # trains on three and tests on the fourth, and holding a year back for
    # monitoring costs a third of the training data. With use_validation off the
    # val loop is skipped and the final epoch is the checkpoint -- justified by
    # the fold1 curves, where validation tracked training to epoch 100 with no
    # divergence.
    n_init_ch = int(config.model.channels_init)
    use_validation = bool(getattr(config.training, "use_validation", True))
    accelerator.print(f"use_validation: {use_validation}")
    val_dataset = make_dataset('val') if use_validation else None
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    ) if use_validation else None

    # Small single-process loaders purely for the periodic train/val figure
    # (one random patch each) -- separate from the metric loaders above so
    # pulling a visualization sample doesn't disturb their iteration state.
    if accelerator.is_main_process:
        train_vis_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                           num_workers=2, pin_memory=True, drop_last=True)
        train_vis_iterator = iter(train_vis_dataloader)
        if use_validation:
            val_vis_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True,
                                             num_workers=2, pin_memory=True, drop_last=True)
            val_vis_iterator = iter(val_vis_dataloader)
        else:
            val_vis_dataloader = val_vis_iterator = None
    else:
        train_vis_dataloader = val_vis_dataloader = None
        train_vis_iterator = val_vis_iterator = None

    # accelerator.prepare does not accept None, so the val loader is prepared
    # only when one exists
    if use_validation:
        diffusion, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            diffusion, optimizer, train_dataloader, val_dataloader
        )
    else:
        diffusion, optimizer, train_dataloader = accelerator.prepare(
            diffusion, optimizer, train_dataloader
        )

    accelerator.print("DataLoaders ready.")

    for epoch in range(config.training.num_epochs):
        diffusion.train()
        total_loss = 0.0
        batch_count = 0
        sum_data_s, sum_fwd_s, sum_bwd_s = 0.0, 0.0, 0.0
        epoch_t0 = time.time()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        t_prev_end = time.time()
        for batch_idx, batch_data in enumerate(pbar):
            t_data = time.time() - t_prev_end  # time spent waiting on the DataLoader

            t_target, t_cond = batch_data
            if target_mode == "residual":
                t_target = residual_normalizer.to_residual(t_target - broadcast_init(t_cond, config.model.future_len, include_pressure, n_init_ch))
            # The model's scalar_embed unconditionally consumes a (batch, 4)
            # physics_cond tensor (a leftover HiPIMS "4 watershed scalars" slot,
            # not actually optional despite the None default) -- we have no
            # groundwater analog for it yet, so pass an explicit zero
            # placeholder rather than touch the vendored diffusion model code.
            t_physics = torch.zeros(t_target.shape[0], 4, device=t_target.device)

            optimizer.zero_grad(set_to_none=True)

            t_fwd_start = time.time()
            with accelerator.autocast():
                loss, _ = diffusion(t_target, t_cond, paired=True, physics_cond=t_physics)
            t_bwd_start = time.time()

            accelerator.backward(loss)
            optimizer.step()
            t_bwd_end = time.time()

            sum_data_s += t_data
            sum_fwd_s += t_bwd_start - t_fwd_start
            sum_bwd_s += t_bwd_end - t_bwd_start

            # Global mean loss across all ranks for accurate reporting.
            mean_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
            total_loss += mean_loss
            batch_count += 1

            if accelerator.is_main_process:
                pbar.set_postfix({'loss': f"{mean_loss:.4f}", 'data_ms': f"{t_data*1000:.0f}"})

            t_prev_end = time.time()

        # --- End of Epoch Logic ---
        avg_loss = total_loss / max(1, batch_count)
        epoch_time_s = time.time() - epoch_t0
        data_ms = 1000 * sum_data_s / max(1, batch_count)
        fwd_ms = 1000 * sum_fwd_s / max(1, batch_count)
        bwd_ms = 1000 * sum_bwd_s / max(1, batch_count)

        # --- Validation loss (every epoch, distributed across ranks, no backward) ---
        diffusion.eval()
        val_total_loss, val_batch_count = 0.0, 0
        with torch.no_grad():
            for val_batch in (val_dataloader or []):
                v_target, v_cond = val_batch
                if target_mode == "residual":
                    v_target = residual_normalizer.to_residual(v_target - broadcast_init(v_cond, config.model.future_len, include_pressure, n_init_ch))
                v_physics = torch.zeros(v_target.shape[0], 4, device=v_target.device)
                with accelerator.autocast():
                    v_loss, _ = diffusion(v_target, v_cond, paired=True, physics_cond=v_physics)
                v_mean_loss = accelerator.gather_for_metrics(v_loss.detach()).mean().item()
                val_total_loss += v_mean_loss
                val_batch_count += 1
        val_loss = (val_total_loss / val_batch_count) if val_batch_count else float("nan")
        diffusion.train()

        if accelerator.is_main_process:
            with open(metrics_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{val_loss:.6f}", f"{epoch_time_s:.1f}",
                                         f"{data_ms:.1f}", f"{fwd_ms:.1f}", f"{bwd_ms:.1f}"])
            try:
                plot_loss_curve(metrics_csv_path, os.path.join(save_dirs['figures'], 'loss_curve.png'), case_name=run_name)
                plot_profiling_breakdown(
                    {'Data loading': data_ms, 'Forward + loss': fwd_ms, 'Backward + step': bwd_ms},
                    os.path.join(save_dirs['figures'], 'profiling_breakdown.png'), case_name=run_name,
                )
            except Exception as e:
                print(f"  (monitoring figure generation failed, continuing: {e})")

            msg = (f'Epoch {epoch} Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | {epoch_time_s:.1f}s | '
                   f'data {data_ms:.0f}ms/batch, fwd {fwd_ms:.0f}ms/batch, bwd {bwd_ms:.0f}ms/batch')
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

            # Checkpoint on validation loss, not train loss -- train loss keeps
            # dropping even when the model is overfitting; val loss is the
            # actual generalization signal.
            if use_validation:
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(accelerator, diffusion,
                                    os.path.join(save_dirs['checkpoints'], 'best_model.pt'))
            else:
                # No held-out set to select on, so the last epoch IS the model.
                # Written every epoch so a truncated run is still usable.
                save_checkpoint(accelerator, diffusion,
                                os.path.join(save_dirs['checkpoints'], 'final_model.pt'))
                # ...and a numbered snapshot periodically. Without a validation
                # curve there is no signal that would tell us to stop early, so
                # the only protection against a long run going wrong late is to
                # keep earlier states. It also gives a free epoch-N vs epoch-M
                # comparison on an identical split, which is otherwise a second
                # training run.
                freq = int(getattr(config.save, 'checkpoint_save_freq', 0) or 0)
                if freq and (epoch + 1) % freq == 0:
                    save_checkpoint(accelerator, diffusion,
                                    os.path.join(save_dirs['checkpoints'],
                                                 f'epoch_{epoch + 1:04d}.pt'))

        accelerator.wait_for_everyone()

        # --- Inference / Visualization (main process only): one figure from a
        # train patch, one from a validation patch, so over/under-fitting is
        # visible directly rather than only inferred from the loss curves. ---
        if accelerator.is_main_process and epoch % config.save.inference_freq == 0:
            print("Running Inference for 4-day sequence...")
            unwrapped_diffusion = accelerator.unwrap_model(diffusion)
            unwrapped_diffusion.eval()
            device = accelerator.device
            with torch.no_grad():
                for split_name, vis_loader, vis_iter_name in (
                    ('train', train_vis_dataloader, 'train_vis_iterator'),
                    ('val', val_vis_dataloader, 'val_vis_iterator'),
                ):
                    if vis_loader is None:      # no validation split configured
                        continue
                    vis_iter = train_vis_iterator if split_name == 'train' else val_vis_iterator
                    try:
                        vis_batch = next(vis_iter)
                    except StopIteration:
                        vis_iter = iter(vis_loader)
                        vis_batch = next(vis_iter)
                    if split_name == 'train':
                        train_vis_iterator = vis_iter
                    else:
                        val_vis_iterator = vis_iter

                    v_target, v_cond = vis_batch
                    v_target = v_target.to(device, non_blocking=True)
                    v_cond = v_cond.to(device, non_blocking=True)
                    v_physics = torch.zeros(v_target.shape[0], 4, device=device)

                    pred_sample = unwrapped_diffusion.sample(
                        batch_size=v_target.shape[0],
                        cond=v_cond,
                        physics_cond=v_physics,
                    )
                    if target_mode == "residual":
                        # Model output is in residual-normalized units; convert back
                        # to absolute-normalized state for plotting.
                        pred_frame = residual_normalizer.from_residual(pred_sample) + broadcast_init(v_cond, config.model.future_len, include_pressure, n_init_ch)
                    else:
                        pred_frame = pred_sample

                    save_prefix = os.path.join(save_dirs['figures'], f'epoch_{epoch}_{split_name}_')
                    plot_groundwater_wtd(
                        pred_frame[0].cpu().numpy(), v_target[0].cpu().numpy(),
                        future_len=config.model.future_len, norm_stats=norm_stats,
                        save_prefix=save_prefix, epoch=epoch,
                        case_name=f"{run_name} ({split_name})" if run_name else split_name,
                        n_state=(7 if include_pressure else 2),
                    )

            diffusion.train()

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        log_file.close()

    accelerator.print("Training complete.")


def main(config_path, checkpoint_path=None, run_name=None):
    config = load_config(config_path)
    size = config.data.size
    accelerator = Accelerator(mixed_precision='fp16')

    if run_name is None:
        run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    base_directory_run = os.path.join(config.save.directory, run_name)
    if accelerator.is_main_process:
        if os.path.exists(base_directory_run):
            run_name = f"{run_name}_new"
            base_directory_run = os.path.join(config.save.directory, run_name)
        save_dirs = create_directories(base_directory_run)
        shutil.copy(config_path, os.path.join(base_directory_run, 'config.yaml'))
    else:
        save_dirs = {
            'checkpoints': os.path.join(base_directory_run, 'checkpoints'),
            'logs': os.path.join(base_directory_run, 'logs'),
            'figures': os.path.join(base_directory_run, 'Figures'),
        }
    accelerator.wait_for_everyone()

    ch_cond = config.model.channels_static + config.model.channels_init + config.model.channels_forcing

    _, diffusion = create_model(
        image_size=(size[0], size[1]),
        channels=config.model.channels_output,
        channels_cond=ch_cond,
        dim=config.model.dim,
        dim_mults=config.model.dim_mults,
        timesteps=config.model.timesteps,
        objective=config.model.objective
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        diffusion.model.load_state_dict(checkpoint['model_state_dict'])
        accelerator.print("Weights loaded successfully.")
    elif checkpoint_path:
        accelerator.print(f"Warning: checkpoint path {checkpoint_path} not found. Starting from scratch.")

    train(config, diffusion, accelerator, save_dirs, run_name=run_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to best_model.pt')
    parser.add_argument('--name', type=str, default=None, help='Name for the run')
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.name)
