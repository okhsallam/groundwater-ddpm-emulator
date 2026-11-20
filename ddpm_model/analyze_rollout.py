"""
Post-rollout spatial/temporal analysis of a run_annual_rollout.py output.

Produces, for one (fold, lead-time) case:
  1. rmse_evolution.{png,csv} -- domain-mean spatial RMSE per day across the
     whole rolled-out year, with a 7-day rolling mean and the day-0 persistence
     reference drawn on the same axes, so "is the emulator actually adding
     skill over doing nothing?" is answerable at a glance.
  2. map_<var>.png -- per-pixel RMSE and per-pixel Pearson correlation over the
     full CESER domain (FSTR-paper style), plus a skill-vs-persistence map.
  3. snapshots_water_table_depth.png -- actual / predicted / difference maps at
     dates spread across the rollout.
  4. summary.json -- headline scalars, consumed by the cross-case comparison
     scripts (compare_leads.py, compare_folds.py).

All maps are drawn north-up through viz_style.north_up_imshow: the conus1 y
index increases NORTHWARD, so a bare imshow renders the domain upside down.

Works for both the full (pressure_head + water_table_depth + soil_moisture)
and the lean, pressure_head-free rollouts -- pressure_head panels are omitted
when predicted_state.npz has no "pressure_head" key.

Usage:
    python analyze_rollout.py --rollout-dir ../data/rollout_wy2006_fold1_f10_lean \
        --out-dir ../data/rollout_wy2006_fold1_f10_lean/analysis \
        --label "fold1 f10 lean (test WY2006)"
"""
import argparse
import csv
import json
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import viz_style as vs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "parflow_raw")

VAR_UNITS = {"water_table_depth": "m", "soil_moisture": "-", "pressure_head": "m"}
VAR_PRETTY = {"water_table_depth": "Water table depth",
              "soil_moisture": "Soil moisture",
              "pressure_head": "Pressure head"}


def load_ground_truth(sorted_days, day0, test_year, include_pressure):
    target2d = xr.open_dataset(os.path.join(RAW_DIR, f"target2d_wy{test_year}.nc"))
    local_idx = [g - day0 for g in sorted_days]
    wtd_true = target2d["water_table_depth"].values[local_idx]
    sm_true = target2d["soil_moisture"].values[local_idx]
    ph_true = None
    if include_pressure:
        target3d = xr.open_dataset(os.path.join(RAW_DIR, f"target3d_wy{test_year}.nc"))
        ph_true = target3d["pressure_head"].values[local_idx]
    return wtd_true, sm_true, ph_true


def per_pixel_rmse(pred, true):
    """pred/true: (n_days, H, W) -> (H, W)."""
    return np.sqrt(np.nanmean((pred - true) ** 2, axis=0))


def per_pixel_corr(pred, true):
    """Pearson r per pixel across the time axis (axis 0)."""
    pm = np.nanmean(pred, axis=0, keepdims=True)
    tm = np.nanmean(true, axis=0, keepdims=True)
    p, t = pred - pm, true - tm
    num = np.nansum(p * t, axis=0)
    den = np.sqrt(np.nansum(p ** 2, axis=0) * np.nansum(t ** 2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den == 0] = np.nan
    return r


def rolling_mean(x, window=7):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def skill_score(rmse_model, rmse_persist):
    """1 - RMSE_model / RMSE_persistence: >0 beats persistence, 0 ties, <0 loses."""
    with np.errstate(invalid="ignore", divide="ignore"):
        s = 1.0 - rmse_model / rmse_persist
    return np.where(np.isfinite(s), s, np.nan)


def map_triptych(rmse_map, corr_map, skill_map, var, title, save_path):
    """RMSE | correlation | skill-vs-persistence, all north-up."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.0), dpi=150, constrained_layout=True)
    unit = VAR_UNITS[var]

    vmax = np.nanpercentile(rmse_map, 97)
    im0 = vs.north_up_imshow(axes[0], rmse_map, cmap=vs.CMAP_ERROR, vmin=0, vmax=vmax)
    vs.map_axes_off(axes[0], "Per-pixel RMSE")
    vs.add_colorbar(fig, im0, axes[0], f"RMSE ({unit})")

    im1 = vs.north_up_imshow(axes[1], corr_map, cmap=vs.CMAP_CORR, vmin=-1, vmax=1)
    vs.map_axes_off(axes[1], "Temporal correlation")
    vs.add_colorbar(fig, im1, axes[1], "Pearson r (predicted vs. actual)")

    # symmetric about 0 so "beats persistence" (blue) vs "loses" (red) reads instantly
    im2 = vs.north_up_imshow(axes[2], skill_map, cmap=vs.CMAP_SKILL, vmin=-1, vmax=1)
    frac = float(np.nanmean(skill_map > 0)) * 100
    vs.map_axes_off(axes[2], f"Skill vs. persistence\n{frac:.0f}% of pixels beat persistence")
    vs.add_colorbar(fig, im2, axes[2], "1 - RMSE$_{model}$ / RMSE$_{persist}$")

    fig.suptitle(title, fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--n-snapshots", type=int, default=6)
    args = parser.parse_args()
    label = args.label or os.path.basename(args.rollout_dir.rstrip("/"))

    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(os.path.join(args.rollout_dir, "predicted_state.npz"))
    wtd_pred, sm_pred = data["water_table_depth"], data["soil_moisture"]
    include_pressure = "pressure_head" in data
    ph_pred = data["pressure_head"] if include_pressure else None
    global_days = data["global_days"]

    with open(os.path.join(PROJECT_ROOT, "data", "tiles", "day_index.json")) as f:
        day_index = json.load(f)
    test_year = day_index[int(global_days[0])]["water_year"]
    test_year_days = [i for i, e in enumerate(day_index) if e["water_year"] == test_year]
    day0 = min(test_year_days)
    dates = [day_index[int(g)]["date"] for g in global_days]

    wtd_true, sm_true, ph_true = load_ground_truth(global_days.tolist(), day0, test_year, include_pressure)

    # Day-0 persistence: freeze the initial state for the whole year. This is
    # the reference an annual emulator has to beat -- the chained rollout never
    # sees truth again after day 0 either, so the two are compared fairly.
    wtd_persist = np.broadcast_to(wtd_true[0], wtd_true.shape)
    sm_persist = np.broadcast_to(sm_true[0], sm_true.shape)

    # --- 1. RMSE evolution across the year ---------------------------------
    wtd_rmse_t = np.sqrt(np.nanmean((wtd_pred - wtd_true) ** 2, axis=(1, 2)))
    sm_rmse_t = np.sqrt(np.nanmean((sm_pred - sm_true) ** 2, axis=(1, 2)))
    wtd_persist_t = np.sqrt(np.nanmean((wtd_persist - wtd_true) ** 2, axis=(1, 2)))
    sm_persist_t = np.sqrt(np.nanmean((sm_persist - sm_true) ** 2, axis=(1, 2)))

    series = [("water_table_depth", wtd_rmse_t, wtd_persist_t),
              ("soil_moisture", sm_rmse_t, sm_persist_t)]
    if include_pressure:
        ph_rmse_t = np.sqrt(np.nanmean((ph_pred - ph_true) ** 2, axis=(1, 2, 3)))
        ph_persist = np.broadcast_to(ph_true[0], ph_true.shape)
        ph_persist_t = np.sqrt(np.nanmean((ph_persist - ph_true) ** 2, axis=(1, 2, 3)))
        series.append(("pressure_head", ph_rmse_t, ph_persist_t))

    with open(os.path.join(args.out_dir, "rmse_evolution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["date", "global_day"]
        for name, _, _ in series:
            header += [f"{name}_rmse", f"{name}_persistence_rmse"]
        w.writerow(header)
        for i in range(len(dates)):
            row = [dates[i], int(global_days[i])]
            for _, vals, pvals in series:
                row += [vals[i], pvals[i]]
            w.writerow(row)

    fig, axes = plt.subplots(len(series), 1, figsize=(15, 4.2 * len(series)), dpi=140,
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    date_ticks = np.arange(len(dates))
    for ax, (name, vals, pvals) in zip(axes, series):
        ax.plot(date_ticks, pvals, color=vs.C_PERSIST, linewidth=vs.LW_SECONDARY,
                linestyle="--", label="persistence (day-0 state held)")
        ax.plot(date_ticks, vals, color=vs.C_PRED, linewidth=vs.LW_FAINT + 0.5,
                alpha=0.35, label="DDPM (daily)")
        ax.plot(date_ticks, rolling_mean(vals), color=vs.C_PRED, linewidth=vs.LW_MAIN,
                label="DDPM (7-day mean)")
        ax.fill_between(date_ticks, rolling_mean(vals), pvals,
                        where=rolling_mean(vals) < pvals, color=vs.C_PRED, alpha=0.12,
                        interpolate=True)
        vs.style_axes(ax, ylabel=f"{VAR_PRETTY[name]}\nRMSE ({VAR_UNITS[name]})")
        ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="upper left", ncol=3)
        # headroom for the legend: without it the 3-column key sits on top of
        # the curves in exactly the months where the two lines separate most
        top = np.nanmax([np.nanmax(vals), np.nanmax(pvals)])
        ax.set_ylim(0, top * 1.32)
    tick_idx = np.linspace(0, len(dates) - 1, 8).astype(int)
    axes[-1].set_xticks(date_ticks[tick_idx])
    axes[-1].set_xticklabels([dates[i] for i in tick_idx], rotation=25, ha="right")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Domain-mean RMSE through the WY{test_year} rollout\n{label}",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(os.path.join(args.out_dir, "rmse_evolution.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote rmse_evolution.{{csv,png}} ({len(dates)} days)")

    # --- 2. Per-pixel RMSE + correlation + skill maps -----------------------
    summary = {"label": label, "test_year": int(test_year), "n_days": len(dates),
               "rollout_dir": os.path.abspath(args.rollout_dir)}
    for var, pred, true, persist in [
        ("water_table_depth", wtd_pred, wtd_true, wtd_persist),
        ("soil_moisture", sm_pred, sm_true, sm_persist),
    ]:
        rmse_map = per_pixel_rmse(pred, true)
        corr_map = per_pixel_corr(pred, true)
        persist_map = per_pixel_rmse(persist, true)
        sk = skill_score(rmse_map, persist_map)
        map_triptych(rmse_map, corr_map, sk, var,
                     f"{VAR_PRETTY[var]} -- {label}",
                     os.path.join(args.out_dir, f"map_{var}.png"))
        np.savez_compressed(os.path.join(args.out_dir, f"maps_{var}.npz"),
                            rmse=rmse_map.astype(np.float32),
                            corr=corr_map.astype(np.float32),
                            persistence_rmse=persist_map.astype(np.float32))
        summary[var] = {
            "rmse_overall": float(np.sqrt(np.nanmean((pred - true) ** 2))),
            "persistence_rmse_overall": float(np.sqrt(np.nanmean((persist - true) ** 2))),
            "median_pixel_rmse": float(np.nanmedian(rmse_map)),
            "median_pixel_corr": float(np.nanmedian(corr_map)),
            "frac_pixels_beating_persistence": float(np.nanmean(sk > 0)),
            "rmse_first_30d": float(np.sqrt(np.nanmean((pred[:30] - true[:30]) ** 2))),
            "rmse_last_30d": float(np.sqrt(np.nanmean((pred[-30:] - true[-30:]) ** 2))),
            "rmse_by_day": [float(v) for v in
                            (wtd_rmse_t if var == "water_table_depth" else sm_rmse_t)],
            "persistence_rmse_by_day": [float(v) for v in
                                        (wtd_persist_t if var == "water_table_depth" else sm_persist_t)],
        }
        summary[var]["skill_vs_persistence"] = (
            1.0 - summary[var]["rmse_overall"] / summary[var]["persistence_rmse_overall"])
    if include_pressure:
        ph_sp, ph_st = ph_pred[:, 4], ph_true[:, 4]  # z=4: shallowest 0.1 m layer
        ph_persist_s = np.broadcast_to(ph_st[0], ph_st.shape)
        map_triptych(per_pixel_rmse(ph_sp, ph_st), per_pixel_corr(ph_sp, ph_st),
                     skill_score(per_pixel_rmse(ph_sp, ph_st), per_pixel_rmse(ph_persist_s, ph_st)),
                     "pressure_head",
                     f"Pressure head (shallowest layer, z=4) -- {label}",
                     os.path.join(args.out_dir, "map_pressure_head_shallow.png"))
    summary["dates"] = dates
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote map_*.png, maps_*.npz, summary.json")

    # --- 3. Snapshot maps ---------------------------------------------------
    snap_idx = np.linspace(0, len(dates) - 1, args.n_snapshots).astype(int)
    fig, axes = plt.subplots(3, args.n_snapshots,
                             figsize=(3.6 * args.n_snapshots, 11.0), dpi=140,
                             constrained_layout=True)
    vmax = np.nanpercentile(wtd_true, 95)
    dmax = max(np.nanpercentile(np.abs(wtd_pred - wtd_true), 97), 1e-3)
    for col, i in enumerate(snap_idx):
        im0 = vs.north_up_imshow(axes[0, col], wtd_true[i], cmap=vs.CMAP_STATE, vmin=0, vmax=vmax)
        vs.map_axes_off(axes[0, col], dates[i])
        im1 = vs.north_up_imshow(axes[1, col], wtd_pred[i], cmap=vs.CMAP_STATE, vmin=0, vmax=vmax)
        vs.map_axes_off(axes[1, col])
        im2 = vs.north_up_imshow(axes[2, col], wtd_pred[i] - wtd_true[i],
                                 cmap=vs.CMAP_DIFF, vmin=-dmax, vmax=dmax)
        vs.map_axes_off(axes[2, col])
    for row, lab in enumerate(["ParFlow\n(truth)", "DDPM\n(predicted)", "difference\n(pred - truth)"]):
        axes[row, 0].set_ylabel(lab, fontsize=vs.FS_LABEL, fontweight="bold")
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].set_yticks([])
    cb0 = plt.colorbar(im0, ax=axes[:2, :], fraction=0.016, pad=0.01)
    cb0.set_label("Water table depth (m)", fontsize=vs.FS_CBAR_LABEL, fontweight="bold")
    cb0.ax.tick_params(labelsize=vs.FS_CBAR_TICK)
    cb2 = plt.colorbar(im2, ax=axes[2, :], fraction=0.016, pad=0.01)
    cb2.set_label("Error (m)", fontsize=vs.FS_CBAR_LABEL, fontweight="bold")
    cb2.ax.tick_params(labelsize=vs.FS_CBAR_TICK)
    fig.suptitle(f"Water table depth through the WY{test_year} rollout\n{label}",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(os.path.join(args.out_dir, "snapshots_water_table_depth.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote snapshots_water_table_depth.png ({args.n_snapshots} dates)")
    print("DONE")


if __name__ == "__main__":
    main()
