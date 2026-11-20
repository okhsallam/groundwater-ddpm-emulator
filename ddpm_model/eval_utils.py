"""
Shared RMSE-by-lead-day scoring/plotting for evaluate_test.py (trained model)
and evaluate_persistence_baseline.py (zero-change baseline) -- kept identical
between the two so their numbers are directly comparable.
"""
import numpy as np
import matplotlib.pyplot as plt

STATE_VAR_BLOCKS = [
    ("pressure_head", 0, 5),        # (name, offset within each day's 7-channel block, width)
    ("water_table_depth", 5, 1),
    ("soil_moisture", 6, 1),
]

# The leaner (2026-09-17) models that drop pressure_head from the forecast
# target entirely -- each day's block is just water_table_depth+soil_moisture.
STATE_VAR_BLOCKS_LEAN = [
    ("water_table_depth", 0, 1),
    ("soil_moisture", 1, 1),
]


def denorm(name, arr, norm_stats):
    """arr: (..., z_or_1, H, W) for one variable's block, already sliced out."""
    stats = norm_stats[name]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
    return arr * std + mean


def rmse_by_lead_day(pred, target, future_len, norm_stats, state_var_blocks=None):
    """pred/target: (N, future_len*n_state, H, W) numpy, normalized, absolute-state
    (not residual) layout. Returns {var_name: [rmse_day1, ..., rmse_dayF]} in
    physical (denormalized) units. `state_var_blocks` defaults to the full
    7-channel/day layout (STATE_VAR_BLOCKS); pass STATE_VAR_BLOCKS_LEAN for the
    pressure_head-free 2-channel/day models."""
    blocks = state_var_blocks or STATE_VAR_BLOCKS
    n_state = sum(width for _, _, width in blocks)
    results = {name: [] for name, _, _ in blocks}
    for day in range(future_len):
        base = day * n_state
        for name, offset, width in blocks:
            sl = slice(base + offset, base + offset + width)
            p = denorm(name, pred[:, sl], norm_stats)
            t = denorm(name, target[:, sl], norm_stats)
            results[name].append(float(np.sqrt(np.mean((p - t) ** 2))))
    return results


def plot_rmse_curves(results, save_path, title, other=None, other_label=None):
    """`other` (optional): a second {var_name: [...]} results dict plotted
    alongside `results` for comparison (e.g. persistence baseline vs. model)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), dpi=140, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (name, values) in zip(axes, results.items()):
        days = list(range(1, len(values) + 1))
        ax.plot(days, values, marker="o", color="#2C6E9E", label="model" if other else None)
        if other is not None:
            ax.plot(days, other[name], marker="o", color="#B5651D", linestyle="--", label=other_label or "baseline")
            ax.legend(fontsize=9, frameon=False)
        ax.set_xlabel("forecast lead day")
        ax.set_ylabel("RMSE")
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
