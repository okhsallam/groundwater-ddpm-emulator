"""
PyTorch Dataset for the CESER groundwater diffusion model, reading the tiled,
per-day npz files produced by ../prepare_tiles.py.

Design mirrors ceser_ddpm_GloFAS's SR_Dataset_Daily (static + a single init
frame + known future forcing, all concatenated into one `cond` tensor; the
model predicts the future state trajectory directly, non-autoregressively,
within one window -- see ceser_groundwater_GloFAS/README.md's forecast
strategy discussion for why longer horizons are handled by chaining windows
rather than lengthening this Dataset's `future_len`) -- but storage is kept
at per-day/per-tile granularity so `future_len` (the rollout period) is a
Dataset constructor argument, not baked into the files on disk.

Channel layout:
  static  (22, H, W): elevation(1) + topographic_index(1) + permeability(5)
                      + porosity(5) + van_genuchten_alpha(5) + van_genuchten_n(5)
  init    (7, H, W):  pressure_head(5) + water_table_depth(1) + soil_moisture(1), at day t
  forcing (future_len*5, H, W): tmin+tmax+precipitation+evapotranspiration+snowmelt,
                                 days t+1..t+future_len, known in advance (this is the
                                 "Forced" in the FSTR paper's Forced-SpatioTemporal-RNN)
  cond = cat([static, init, forcing])                      -> (22+7+future_len*5, H, W)
  target = cat over days of [pressure_head(5)+water_table_depth(1)+soil_moisture(1)]
            -> (future_len*7, H, W), flat/day-major (channels 0-6 = day t+1's 7 vars,
            7-13 = day t+2's, ...) -- matches ceser_ddpm_GloFAS's flat
            channels_output convention (its 30 = 10 steps x 3 vars), since the
            U-Net takes a plain (C, H, W) tensor, not a (time, var, H, W) one.

Water-year split: train = WY2003-2004, val = WY2005 (held out for monitoring
during training), test = WY2006 (final evaluation, matches the FSTR paper's
train/test boundary exactly -- never touched during training/val).
Assigned by the init day's water year.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STATIC_VARS_1LAYER = ["elevation", "topographic_index"]
STATIC_VARS_5LAYER = ["permeability", "porosity", "van_genuchten_alpha", "van_genuchten_n"]
# Full set as originally downloaded. NOTE THE PROVENANCE: tmin, tmax and
# precipitation come from NLDAS-2 reanalysis and are genuine atmospheric forcing;
# evapotranspiration and snowmelt are ParFlow-CLM OUTPUTS. Conditioning on the
# latter during an autoregressive rollout feeds the emulator quantities the
# simulation itself produces -- and evapotranspiration is a function of soil
# moisture, which is a forecast target, so it leaks information about the future
# state. Use FORCING_VARS_NLDAS for a configuration that does not.
FORCING_VARS = ["tmin", "tmax", "precipitation", "evapotranspiration", "snowmelt"]
FORCING_VARS_NLDAS = ["tmin", "tmax", "precipitation"]

def resolve_forcing_vars(name):
    """'all' -> the five original channels; 'nldas' -> the three atmospheric ones."""
    return {"all": FORCING_VARS, "nldas": FORCING_VARS_NLDAS}[str(name).lower()]
STATE_VARS_5LAYER = ["pressure_head"]
STATE_VARS_1LAYER = ["water_table_depth", "soil_moisture"]

TRAIN_WATER_YEARS = {2003, 2004}
VAL_WATER_YEARS = {2005}
TEST_WATER_YEARS = {2006}

N_STATIC = len(STATIC_VARS_1LAYER) + 5 * len(STATIC_VARS_5LAYER)  # 22
N_STATE = 5 * len(STATE_VARS_5LAYER) + len(STATE_VARS_1LAYER)      # 7 (init block, and each day's target block)


def broadcast_init(cond, future_len, include_pressure=True, n_init=None):
    """Extract the init-state block from a batched `cond` tensor (B, C, H, W)
    and tile it future_len times along the channel dim, to align with the
    target's flat day-major layout. Used to convert between the absolute-state
    target dataset __getitem__ returns and a residual-from-init target:
        target_residual = target_absolute - broadcast_init(cond, future_len)
        target_absolute = target_residual + broadcast_init(cond, future_len)

    include_pressure=False extracts only the water_table_depth+soil_moisture
    slice (the last 2 of init's 7 channels) -- for the leaner models
    (2026-09-17 ablation) that don't forecast pressure_head at all, even
    though `init` itself always carries the full 7-channel state as
    conditioning (pressure_head is still useful context, just not a
    prediction target).
    """
    n_init = N_STATE if n_init is None else n_init
    if include_pressure:
        init = cond[:, N_STATIC:N_STATIC + n_init]
    elif n_init == 2:
        init = cond[:, N_STATIC:N_STATIC + 2]            # init block is already wtd, sm
    else:
        init = cond[:, N_STATIC + 5:N_STATIC + n_init]   # last 2 channels: wtd, sm
    return init.repeat(1, future_len, 1, 1)


class ResidualNormalizer:
    """Converts between:
      - diff_normalized = target_absnorm - init_absnorm (both normalized with
        normalization_stats.json's absolute-value stats -- what broadcast_init-based
        subtraction gives you), and
      - residual_finalnorm: the SAME physical residual (state[t+k]-state[t]),
        but rescaled by its OWN per-lead-day mean/std (compute_residual_normalization_stats.py's
        output) instead of the absolute-value std.

    Why: the absolute-value std is ~20x the residual's natural scale (day-to-day
    groundwater change is small relative to total spatial variance), so feeding
    diff_normalized straight into the diffusion model starves it of dynamic
    range and it degenerates to matching only the marginal variance of the
    residual rather than its spatial structure (confirmed empirically
    2026-09-15). Since abs_stats' mean cancels exactly in a difference,
    raw_diff = diff_normalized * abs_std, then residual_finalnorm =
    (raw_diff - res_mean_k) / res_std_k.

    Both methods take/return (B, future_len*n_state, H, W) tensors, day-major.
    With include_pressure=True (default), n_state=7 (pressure_head[5]+wtd[1]+sm[1]
    per day). With include_pressure=False (the leaner, 2026-09-17 water_table_depth
    + soil_moisture-only models), n_state=2 -- pressure_head is dropped from the
    forecast target entirely (it's still available as conditioning via `init`,
    just not something the model predicts; see broadcast_init's include_pressure).
    """

    def __init__(self, future_len, abs_stats, residual_stats, device, include_pressure=True):
        self.future_len = future_len
        self.include_pressure = include_pressure
        self.n_state = N_STATE if include_pressure else 2

        def layered(std_or_mean_list):
            # (future_len, 5) -> (1, future_len, 5, 1, 1)
            t = torch.tensor(std_or_mean_list, dtype=torch.float32, device=device)
            return t.view(1, future_len, 5, 1, 1)

        def scalar(std_or_mean_list):
            # (future_len,) -> (1, future_len, 1, 1, 1)
            t = torch.tensor(std_or_mean_list, dtype=torch.float32, device=device)
            return t.view(1, future_len, 1, 1, 1)

        if include_pressure:
            abs_std_ph = torch.tensor(abs_stats["pressure_head"]["std"], dtype=torch.float32, device=device)
            self.abs_std_ph = abs_std_ph.view(1, 1, 5, 1, 1)
            self.res_mean_ph = layered(residual_stats["pressure_head"]["mean"])
            self.res_std_ph = layered(residual_stats["pressure_head"]["std"]).clamp_min(1e-6)

        abs_std_wtd = torch.tensor(abs_stats["water_table_depth"]["std"], dtype=torch.float32, device=device)
        abs_std_sm = torch.tensor(abs_stats["soil_moisture"]["std"], dtype=torch.float32, device=device)
        # broadcastable over (B, future_len, {5,1}, H, W); same abs std for every lead day
        self.abs_std_wtd = abs_std_wtd.view(1, 1, 1, 1, 1)
        self.abs_std_sm = abs_std_sm.view(1, 1, 1, 1, 1)

        self.res_mean_wtd = scalar(residual_stats["water_table_depth"]["mean"])
        self.res_std_wtd = scalar(residual_stats["water_table_depth"]["std"]).clamp_min(1e-6)
        self.res_mean_sm = scalar(residual_stats["soil_moisture"]["mean"])
        self.res_std_sm = scalar(residual_stats["soil_moisture"]["std"]).clamp_min(1e-6)

    def _split(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.future_len, self.n_state, H, W)
        if self.include_pressure:
            return x[:, :, 0:5], x[:, :, 5:6], x[:, :, 6:7], H, W
        return None, x[:, :, 0:1], x[:, :, 1:2], H, W

    def to_residual(self, diff_normalized):
        ph, wtd, sm, H, W = self._split(diff_normalized)
        parts = []
        if self.include_pressure:
            res_ph = (ph * self.abs_std_ph - self.res_mean_ph) / self.res_std_ph
            parts.append(res_ph)
        res_wtd = (wtd * self.abs_std_wtd - self.res_mean_wtd) / self.res_std_wtd
        res_sm = (sm * self.abs_std_sm - self.res_mean_sm) / self.res_std_sm
        parts += [res_wtd, res_sm]
        out = torch.cat(parts, dim=2)
        B = diff_normalized.shape[0]
        return out.view(B, self.future_len * self.n_state, H, W)

    def from_residual(self, residual_finalnorm):
        ph, wtd, sm, H, W = self._split(residual_finalnorm)
        parts = []
        if self.include_pressure:
            diff_ph = (ph * self.res_std_ph + self.res_mean_ph) / self.abs_std_ph
            parts.append(diff_ph)
        diff_wtd = (wtd * self.res_std_wtd + self.res_mean_wtd) / self.abs_std_wtd
        diff_sm = (sm * self.res_std_sm + self.res_mean_sm) / self.abs_std_sm
        parts += [diff_wtd, diff_sm]
        out = torch.cat(parts, dim=2)
        B = residual_finalnorm.shape[0]
        return out.view(B, self.future_len * self.n_state, H, W)


def _normalize(name, arr, stats):
    """z-score normalize `arr` (any shape, variable `name` on axis 0 if layered).

    `stats[name]` is {"mean": scalar or 5-list, "std": scalar or 5-list} --
    per-layer for the 5-layer variables (pressure_head and the 4 5-layer
    static params), since e.g. pressure_head's deepest layer (~+50m,
    saturated) and shallowest layer (~0m, fluctuating) are on completely
    different scales (see README's z-layer-ordering section).
    """
    if stats is None or name not in stats:
        return arr
    mean = np.asarray(stats[name]["mean"], dtype=np.float32)
    std = np.asarray(stats[name]["std"], dtype=np.float32)
    # z=0 (deepest layer) is a spatially-uniform bedrock constant for
    # porosity/van_genuchten_alpha/van_genuchten_n (std=0, confirmed in
    # README) -- guard the divide, that layer just centers to 0 instead of NaN.
    std = np.where(std == 0, 1.0, std)
    if mean.ndim == 1:  # per-layer stats -> broadcast over (H, W)
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
    return (arr - mean) / std


def _load_state(npz, stats=None):
    """(7, H, W): pressure_head(5) + water_table_depth(1) + soil_moisture(1)."""
    parts = [_normalize("pressure_head", npz["pressure_head"], stats)]  # (5, H, W)
    for v in STATE_VARS_1LAYER:
        parts.append(_normalize(v, npz[v], stats)[None, ...])  # (1, H, W)
    return np.concatenate(parts, axis=0)


def _load_wtd_sm(npz, stats=None):
    """(2, H, W): water_table_depth(1) + soil_moisture(1) -- the leaner forecast
    target (2026-09-17) that drops pressure_head from the prediction entirely."""
    parts = [_normalize(v, npz[v], stats)[None, ...] for v in STATE_VARS_1LAYER]
    return np.concatenate(parts, axis=0)


def _load_forcing(npz, stats=None, forcing_vars=None):
    """(n_forcing, H, W). Channels are selected by name, so restricting the set
    needs no re-tiling: each variable is a separate array in the archive."""
    return np.stack([_normalize(v, npz[v], stats)
                     for v in (forcing_vars or FORCING_VARS)], axis=0)


class GroundwaterTileDataset(Dataset):
    def __init__(self, root, tile_size, future_len=10, split="train", normalization_stats=None,
                 include_pressure_in_target=True, train_years=None, val_years=None,
                 test_years=None, forcing_vars=None, include_pressure_in_init=True):
        """
        root: ceser_groundwater_GloFAS/data/tiles directory (from prepare_tiles.py)
        tile_size: 128 or 256 (must match a `tile{tile_size}/` subdir already prepared)
        future_len: rollout/forecast horizon in days for one training window
        split: "train", "val", "test", or "all"
        normalization_stats: optional dict of {var_name: {"mean":.., "std":..}} :
            per-variable z-score normalization applied before concatenation into
            cond/target; left unnormalized (raw physical units) if None. Load from
            compute_normalization_stats.py's output JSON in normal use.
        include_pressure_in_target: False for the leaner (2026-09-17) models that
            forecast only water_table_depth+soil_moisture -- pressure_head is
            dropped from `target` (2 channels/day instead of 7), but `init`
            (part of `cond`) always stays the full 7-channel state regardless,
            since pressure_head is still useful conditioning even when it's not
            a prediction target.
        train_years/val_years/test_years: optional iterables of water years,
            overriding the module-default TRAIN/VAL/TEST_WATER_YEARS -- used by
            the 2026-09-17 4-fold cross-validation (each water year takes a turn
            as the test year). Defaults preserve the original WY2003-2004/2005/2006
            split exactly when omitted.
        """
        self.root = Path(root)
        self.tile_dir = self.root / f"tile{tile_size}"
        self.future_len = future_len
        self.normalization_stats = normalization_stats
        self.include_pressure_in_target = include_pressure_in_target
        # pressure head in the INIT block is separate from pressure head as a
        # TARGET. Held at its day-0 value through a rollout it becomes stale and
        # inconsistent with the predicted state it is paired with, so it can be
        # dropped from the conditioning entirely.
        self.include_pressure_in_init = include_pressure_in_init
        self.forcing_vars = list(forcing_vars) if forcing_vars else list(FORCING_VARS)

        with open(self.root / "day_index.json") as f:
            self.day_index = json.load(f)
        with open(self.tile_dir / "tile_grid.json") as f:
            self.tile_grid = json.load(f)
        self.n_tiles = len(self.tile_grid["tiles"])

        train_years = set(train_years) if train_years is not None else TRAIN_WATER_YEARS
        val_years = set(val_years) if val_years is not None else VAL_WATER_YEARS
        test_years = set(test_years) if test_years is not None else TEST_WATER_YEARS

        if split == "train":
            years = train_years
        elif split == "val":
            years = val_years
        elif split == "test":
            years = test_years
        elif split == "all":
            years = train_years | val_years | test_years
        else:
            raise ValueError(f"unknown split {split!r}")

        n_days = len(self.day_index)
        # Every day in the window -- start_day AND start_day+1..start_day+future_len,
        # not just start_day itself -- must belong to this split's year set, or a
        # window starting near a split boundary would silently pull future days
        # from a DIFFERENT split (e.g. the held-out test year) into training. This
        # matters far more for the 2026-09-17 cross-validation folds with
        # non-contiguous train years (e.g. {2003,2006}) than it did for the
        # original contiguous {2003,2004}, where the leak was just minor blur
        # into the adjacent (and legitimately close) validation year.
        valid_start_days = [
            d for d in range(n_days - future_len)
            if all(self.day_index[d + k]["water_year"] in years for k in range(future_len + 1))
        ]
        self.samples = [
            (tile_id, start_day)
            for start_day in valid_start_days
            for tile_id in range(self.n_tiles)
        ]

        self._static_cache = {}

    def __len__(self):
        return len(self.samples)

    def _static(self, tile_id):
        if tile_id not in self._static_cache:
            npz = np.load(self.tile_dir / "static" / f"tile_{tile_id:03d}.npz")
            parts = [_normalize(v, npz[v], self.normalization_stats)[None, ...] for v in STATIC_VARS_1LAYER]
            parts += [_normalize(v, npz[v], self.normalization_stats) for v in STATIC_VARS_5LAYER]
            self._static_cache[tile_id] = np.concatenate(parts, axis=0)
        return self._static_cache[tile_id]

    def _dynamic(self, tile_id, global_day):
        path = self.tile_dir / "dynamic" / f"tile_{tile_id:03d}" / f"day_{global_day:04d}.npz"
        return np.load(path)

    def __getitem__(self, idx):
        tile_id, start_day = self.samples[idx]

        static = self._static(tile_id)
        load_init = _load_state if self.include_pressure_in_init else _load_wtd_sm
        init_state = load_init(self._dynamic(tile_id, start_day), self.normalization_stats)

        forcing_days = []
        target_days = []
        load_target = _load_state if self.include_pressure_in_target else _load_wtd_sm
        for k in range(1, self.future_len + 1):
            npz = self._dynamic(tile_id, start_day + k)
            forcing_days.append(_load_forcing(npz, self.normalization_stats,
                                              self.forcing_vars))
            target_days.append(load_target(npz, self.normalization_stats))
        forcing = np.concatenate(forcing_days, axis=0)  # (future_len*n_forcing, H, W)
        target = np.concatenate(target_days, axis=0)  # (future_len*{7,2}, H, W), day-major

        cond = np.concatenate([static, init_state, forcing], axis=0)

        return (
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(cond.astype(np.float32)),
        )
