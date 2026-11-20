"""
Prediction-vs-ground-truth figures for the CESER DDPM training loop.

Was missing entirely in this project (train.py does `from plotting_utils import
*` but no such file existed here -- copied the signature-matching version from
../../SR_daily_inundation_flood_ddpm/plotting_utils.py and rewrote plot_contour
for presentation-quality output: per-variable robust color scaling (so it isn't
silently miscalibrated against this project's log1p/robust normalization, unlike
the old fixed vmax=0.8), non-rainbow colormaps (sequential viridis for the
non-negative depth h, diverging RdBu_r for the signed momentum components
hUx/hUy -- 'jet' has no consistent perceptual ordering), and row/column labels.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch  # noqa: F401  (kept for parity with callers that may pass tensors)

plt.rcParams.update({
    'axes.labelsize': 11,
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.titlesize': 14,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
})

_VAR_SPECS = {
    'h':   {'label': 'h (depth)',            'cmap': 'viridis', 'diverging': False},
    'hUx': {'label': r'hU$_x$ (x-momentum)', 'cmap': 'RdBu_r',  'diverging': True},
    'hUy': {'label': r'hU$_y$ (y-momentum)', 'cmap': 'RdBu_r',  'diverging': True},
}
_TARGET_FIELD_VARS = {'h': ['h'], 'hU': ['hUx', 'hUy'], 'full': ['h', 'hUx', 'hUy']}

# Flood-depth hazard classes (meters) -- a continuous colormap on normalized h
# is dominated by two clusters (near-zero dry pixels, and wet pixels pegged at
# the NORM_CLIP_ABS ceiling from apply_norm's zero-inflation fix), so it reads
# as just two flat colors. Decoding to real depth and binning into standard
# flood-depth classes is far more legible -- but note the clip is lossy: any
# pixel that saturated during normalization decodes to the same ceiling depth
# rather than its true value, so the top class ('>2.0 m') under-resolves how
# deep the deepest water actually is.
_DEPTH_CLASS_EDGES_M = [0.01, 0.15, 0.3, 0.6, 1.0, 2.0]  # interior edges only
_DEPTH_CLASS_LABELS = ['Dry', '<0.15 m', '0.15-0.3 m', '0.3-0.6 m', '0.6-1.0 m', '1.0-2.0 m', '>2.0 m']
_DEPTH_CLASS_COLORS = ['#f0f0f0', '#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c', '#08306b']


def denormalize(arr, params):
    """Invert apply_norm's transform+scaler, best-effort -- apply_norm's clip
    step (see pipeline_utils.py / dataset_SR.py, NORM_CLIP_ABS) is lossy for
    saturated pixels: they decode to the same ceiling value rather than their
    true magnitude. Fine for classing depth into hazard bins; don't rely on
    this for exact depth recovery of the most extreme pixels."""
    out = np.asarray(arr, dtype=np.float64).copy()
    scaler = params.get('scaler', 'min-max')
    if scaler == 'min-max':
        out = out * (params['max'] - params['min'] + 1e-6) + params['min']
    elif scaler == 'robust':
        out = out * (params['iqr'] + 1e-6) + params['median']
    if params.get('transform') == 'log1p':
        out = np.expm1(np.clip(out, a_min=None, a_max=50))  # clip pre-expm1 to avoid float overflow
    return out


def _depth_to_class_idx(depth_m):
    return np.digitize(depth_m, _DEPTH_CLASS_EDGES_M)  # 0..6, matches the 7 classes/colors above


def _robust_limits(arr, diverging):
    """1st/99th percentile bounds -- robust to a handful of extreme pixels,
    and shared between pred/GT in a row so the two are visually comparable."""
    lo, hi = np.percentile(arr, [1, 99])
    if diverging:
        m = max(abs(lo), abs(hi), 1e-6)
        return -m, m
    return lo, max(hi, lo + 1e-6)


def _plot_var_block(fig, axes, row0, pred, gt, var_key, n_steps):
    spec = _VAR_SPECS[var_key]
    vmin, vmax = _robust_limits(gt, spec['diverging'])
    im = None
    for i in range(n_steps):
        im = axes[row0, i].imshow(pred[i], cmap=spec['cmap'], vmin=vmin, vmax=vmax)
        axes[row0 + 1, i].imshow(gt[i], cmap=spec['cmap'], vmin=vmin, vmax=vmax)
        for r in (row0, row0 + 1):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
        axes[row0, i].set_title(f"+{(i + 1) * 12}h", fontsize=9)
    axes[row0, 0].set_ylabel("Pred", fontsize=10)
    axes[row0 + 1, 0].set_ylabel("GT", fontsize=10)
    cb = fig.colorbar(im, ax=axes[row0:row0 + 2, :].ravel().tolist(), shrink=0.85, pad=0.015, aspect=30)
    cb.set_label(spec['label'], fontsize=10)


def _plot_depth_class_block(fig, axes, row0, pred_norm, gt_norm, n_steps, h_params):
    """Same layout as _plot_var_block, but decodes normalized h back to real
    depth (meters) and renders as discrete hazard classes instead of a
    continuous colormap -- see _DEPTH_CLASS_* above for why."""
    cmap = ListedColormap(_DEPTH_CLASS_COLORS)
    pred_m = denormalize(pred_norm, h_params)
    gt_m = denormalize(gt_norm, h_params)
    im = None
    for i in range(n_steps):
        im = axes[row0, i].imshow(_depth_to_class_idx(pred_m[i]), cmap=cmap, vmin=-0.5, vmax=6.5)
        axes[row0 + 1, i].imshow(_depth_to_class_idx(gt_m[i]), cmap=cmap, vmin=-0.5, vmax=6.5)
        for r in (row0, row0 + 1):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
        axes[row0, i].set_title(f"+{(i + 1) * 12}h", fontsize=9)
    axes[row0, 0].set_ylabel("Pred", fontsize=10)
    axes[row0 + 1, 0].set_ylabel("GT", fontsize=10)
    cb = fig.colorbar(im, ax=axes[row0:row0 + 2, :].ravel().tolist(), shrink=0.85, pad=0.015,
                       aspect=30, ticks=range(len(_DEPTH_CLASS_LABELS)))
    cb.ax.set_yticklabels(_DEPTH_CLASS_LABELS, fontsize=8)
    cb.set_label('h (depth class)', fontsize=10)


def plot_contour(pred_frame, t_target, target_fields, save_prefix, epoch=None, case_name=None,
                  batch_idx=0, dynamic_norm_params=None):
    """pred_frame, t_target: (B, C, H, W) numpy arrays, channels = n_steps x
    n_vars in step-major order (matches dataset_SR.py's
    target_full.reshape(-1, H, W)). Saves `{save_prefix}.png`: one Pred/GT
    row-pair per physical variable, one column per forecast step (12h cadence).

    dynamic_norm_params: the dict loaded from normalization_params_dynamic.json
    (e.g. a dataset's `.dynamic_norm_params`) -- when given, the depth channel
    (h) is decoded back to real meters and plotted as discrete hazard classes
    instead of a continuous colormap on normalized values (see
    _DEPTH_CLASS_* / _plot_depth_class_block). Without it, h falls back to the
    old continuous robust-percentile viridis scaling. hUx/hUy always use the
    continuous diverging scaling -- only h is classed."""
    if target_fields not in _TARGET_FIELD_VARS:
        raise ValueError(f"Invalid target_fields config: {target_fields}")
    var_keys = _TARGET_FIELD_VARS[target_fields]
    n_vars = len(var_keys)
    n_steps = t_target.shape[1] // n_vars

    pred_b, gt_b = pred_frame[batch_idx], t_target[batch_idx]

    fig, axes = plt.subplots(
        nrows=2 * n_vars, ncols=n_steps, figsize=(1.6 * n_steps, 2.4 * n_vars),
        dpi=200, constrained_layout=True, squeeze=False,
    )
    for v, var_key in enumerate(var_keys):
        pred_v = pred_b[v::n_vars]
        gt_v = gt_b[v::n_vars]
        if var_key == 'h' and dynamic_norm_params is not None and 'h' in dynamic_norm_params:
            _plot_depth_class_block(fig, axes, 2 * v, pred_v, gt_v, n_steps, dynamic_norm_params['h'])
        else:
            _plot_var_block(fig, axes, 2 * v, pred_v, gt_v, var_key, n_steps)

    title = "Prediction vs. ground truth"
    if case_name:
        title += f" -- {case_name}"
    if epoch is not None:
        title += f" (epoch {epoch})"
    fig.suptitle(title, fontsize=13)

    plt.savefig(f"{save_prefix}.png", bbox_inches='tight')
    plt.close(fig)


def plot_loss_curve(csv_path, save_path, case_name=None):
    """Reads the (epoch, avg_loss, val_loss, ...) metrics CSV train.py writes
    each epoch and saves a train-vs-validation loss-vs-epoch line chart --
    lets you monitor over/under-fitting visually without parsing the log file.
    `val_loss` is optional for backward compat with older metrics.csv files
    that predate the train/val/test split -- falls back to train-only."""
    import csv
    epochs, losses, val_losses = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        has_val = 'val_loss' in (reader.fieldnames or [])
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['avg_loss']))
            if has_val:
                val_losses.append(float(row['val_loss']))
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150, constrained_layout=True)
    ax.plot(epochs, losses, color='#2C6E9E', linewidth=2, label='Train')
    if val_losses:
        ax.plot(epochs, val_losses, color='#B5651D', linewidth=2, label='Validation')
        ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg. loss')
    title = 'Training / validation loss'
    if case_name:
        title += f' -- {case_name}'
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_profiling_breakdown(timings_ms, save_path, case_name=None):
    """timings_ms: dict, e.g. {'Data loading': 12.3, 'Forward + loss': 45.6,
    'Backward + step': 30.1} (average ms/batch). Overwritten each epoch so a
    current snapshot exists even if the job is cut short."""
    labels = list(timings_ms.keys())
    values = list(timings_ms.values())
    colors = ['#6BA3C5', '#2C6E9E', '#1B4965', '#0B2A44']

    fig, ax = plt.subplots(figsize=(5.5, 0.6 * len(labels) + 1.2), dpi=150, constrained_layout=True)
    bars = ax.barh(labels, values, color=colors[:len(labels)])
    ax.set_xlabel('ms / batch')
    title = 'Per-batch time breakdown'
    if case_name:
        title += f' -- {case_name}'
    ax.set_title(title, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {v:.1f} ms', va='center', fontsize=9)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_groundwater_wtd(pred, target, future_len, norm_stats, save_prefix, epoch=None, case_name=None, n_state=7):
    """Quick-look figure for one groundwater training/val sample: predicted vs.
    ground-truth water_table_depth on the first and last forecast day,
    denormalized back to meters. `pred`/`target`: (future_len*n_state, H, W)
    numpy arrays for one sample (day-major channel layout, see
    dataset_groundwater.py). water_table_depth is always the second-to-last
    channel in each day's block: index 5 of 7 (pressure_head[5]+wtd+sm) in the
    full layout, or index 0 of 2 (wtd+sm) in the leaner pressure_head-free one.
    """
    mean = norm_stats["water_table_depth"]["mean"]
    std = norm_stats["water_table_depth"]["std"]
    wtd_offset = n_state - 2

    def wtd_day(arr, day_idx):
        return arr[day_idx * n_state + wtd_offset] * std + mean

    days = [0, future_len - 1]
    fig, axes = plt.subplots(2, len(days), figsize=(4 * len(days), 8), dpi=130, constrained_layout=True)
    for col, day_idx in enumerate(days):
        gt = wtd_day(target, day_idx)
        pr = wtd_day(pred, day_idx)
        vmax = max(np.nanpercentile(gt, 95), np.nanpercentile(pr, 95), 1e-3)
        for row, (arr, label) in enumerate([(gt, "ground truth"), (pr, "predicted")]):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap="viridis_r", vmin=0, vmax=vmax)
            ax.set_title(f"day {day_idx + 1} -- {label}", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
    title = "water_table_depth: prediction vs. ground truth"
    if case_name:
        title += f" -- {case_name}"
    if epoch is not None:
        title += f" (epoch {epoch})"
    fig.suptitle(title, fontsize=12)
    plt.savefig(f"{save_prefix}wtd.png", bbox_inches="tight")
    plt.close(fig)
