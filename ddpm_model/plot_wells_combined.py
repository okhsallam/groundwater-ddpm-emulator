"""
Observation-well comparison for both networks and both forcings.

Each water year is predicted by the fold that held it out, so concatenating the
four folds gives a continuous 2003-2006 model series in which no year was seen
in training. NLDAS-driven rollouts cover the whole water year; SEAS5 arms reach
210 days from the 1 October initialisation, so their traces stop in late April.

THE OFFSET CONVENTION, carried over from the USGS comparison. ParFlow places the
water table systematically shallower than the wells observe -- a model-vs-reality
discrepancy, not an emulator error, so it must not be charged to the emulator.
Each observed series is therefore shifted by ONE CONSTANT per site, the
difference of means over the common period, and only the observation is moved.
The comparison that remains is of SHAPE: does the emulator reproduce the timing
and amplitude of the water table's movement.

Networks are drawn differently because they are different evidence: USGS wells
were selected for coverage across a wide depth range, UIUC wells are the denser
unconfined network with near-daily records through 2003-2006.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import viz_style as vs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "well_series")
FOLD_WY = {4: 2003, 3: 2004, 2: 2005, 1: 2006}


def load_wells():
    ws = []
    for f, net in ((os.path.join(ROOT, "data", "reference_wells.json"), "USGS"),
                   (os.path.join(ROOT, "data", "reference_wells_uiuc.json"), "UIUC")):
        if os.path.exists(f):
            for w in json.load(open(f))["wells"]:
                w["net"] = net; ws.append(w)
    return ws


def observations(ws):
    obs = {}
    p = os.path.join(ROOT, "data", "usgs_wells", "observations.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        dcol = next((c for c in d.columns if "date" in c.lower()), None)
        vcol = next((c for c in d.columns if "depth" in c.lower() or "value" in c.lower()), None)
        scol = next((c for c in d.columns if "site" in c.lower()), None)
        if dcol and vcol and scol:
            d[dcol] = pd.to_datetime(d[dcol], errors="coerce", utc=True).dt.tz_localize(None)
            for w in ws:
                if w["net"] != "USGS":
                    continue
                g = d[d[scol].astype(str) == str(w["site_no"])].dropna(subset=[dcol, vcol])
                if len(g):
                    obs[w["short_id"]] = pd.Series(g[vcol].values,
                                                   index=pd.DatetimeIndex(g[dcol])).sort_index()
    u = os.path.join(ROOT, "data", "UIUC", "DOE_GW_OBV_unconfined_2003_2006_depth.csv")
    if os.path.exists(u):
        d = pd.read_csv(u, parse_dates=["date_time"])
        d["date_time"] = d["date_time"].dt.tz_localize(None)
        for w in ws:
            if w["net"] != "UIUC":
                continue
            g = d[d.p_number == w["p_number"]].dropna(subset=["depth_to_water_m"])
            if len(g):
                obs[w["short_id"]] = pd.Series(g.depth_to_water_m.values,
                                               index=pd.DatetimeIndex(g.date_time)).sort_index()
    return obs


def model_series(prefix, ws):
    """{short_id: DataFrame(date -> mean, lo, hi)} concatenated over folds."""
    di = json.load(open(os.path.join(ROOT, "data", "tiles", "day_index.json")))
    out = {w["short_id"]: [] for w in ws}
    for fold in (4, 3, 2, 1):
        p = os.path.join(CACHE, f"{prefix}_fold{fold}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        s = z["series"]                       # (M, days, wells)
        gd = z["global_days"].astype(int)
        dates = pd.DatetimeIndex([di[int(g)]["date"] for g in gd])
        sid = [str(x) for x in z["sid"]]
        for j, k in enumerate(sid):
            if k not in out:
                continue
            out[k].append(pd.DataFrame(
                dict(mean=s[:, :, j].mean(0),
                     lo=np.percentile(s[:, :, j], 10, axis=0),
                     hi=np.percentile(s[:, :, j], 90, axis=0)), index=dates))
    def _assemble(v):
        if not v:
            return None
        df = pd.concat(v).sort_index()
        # Each fold contributes one rollout. Under reanalysis forcing those
        # tile the record end to end, but a SEAS5 rollout is 210 days from
        # 1 October, so May-September carries no forecast at all -- three
        # gaps of about 156 days. Re-indexing onto a complete daily axis
        # turns them into NaN, which breaks the plotted line instead of
        # drawing a straight segment across five months of nothing.
        full = pd.date_range(df.index.min(), df.index.max(), freq="D")
        return df.reindex(full)

    return {k: _assemble(v) for k, v in out.items()}


def panel(ax, ob, mod, title, subtitle, color, first_col=True):
    if mod is None or ob is None or not len(ob):
        ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes)
        vs.style_axes(ax, title=title); return None
    lo, hi = mod.index.min(), mod.index.max()
    o = ob[(ob.index >= lo) & (ob.index <= hi)]
    if not len(o):
        ax.text(.5, .5, "no overlapping observations", ha="center", va="center",
                transform=ax.transAxes)
        vs.style_axes(ax, title=title); return None
    # one constant per site; only the observation moves
    off = float(mod["mean"].mean(skipna=True) - o.mean())
    ax.fill_between(mod.index, mod["lo"], mod["hi"], color=color, alpha=.20, lw=0)
    ax.plot(mod.index, mod["mean"], color=color, lw=vs.LW_MAIN, label="emulator")
    ax.plot(o.index, o.values + off, color=vs.C_ACTUAL, lw=vs.LW_SECONDARY,
            label="observed (shifted to the model mean)")
    ax.invert_yaxis()
    # the label is long and the panels are narrow: only the left column carries
    # it, and the title and the site note no longer share the same corner
    vs.style_axes(ax)
    ax.margins(y=.18)
    # name above, site note under it: side by side they collided on every panel
    ax.text(0, 1.21, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=vs.FS_TITLE-6, fontweight="bold")
    ax.text(0, 1.03, subtitle.split("\n")[0], transform=ax.transAxes,
            fontsize=vs.FS_ANNOT-7, color="#6B8197", va="bottom", ha="left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for t in ax.get_xticklabels():
        t.set_fontsize(vs.FS_TICK - 6)
    # Three correlations, each removing a different nuisance.
    #
    #   raw        the two series as plotted, after the per-site constant
    #   detrended  a least-squares line in time removed from BOTH series. It
    #              is still computed, because the report's tables cite it, but
    #              it is NOT shown on the panels: over the 2003-2006 record it
    #              is within 0.05 of the raw value at 12 of the 14 wells, so
    #              printing it alongside the raw value said nothing twice
    #   30-day     increments, which ask whether month-to-month movements are
    #              timed correctly
    #
    # An earlier version of this comment justified detrending by saying a
    # straight line scores a median |r| of 0.47 against these hydrographs. That
    # number was measured on SINGLE-water-year segments, where a ramp does score
    # about 0.54. These wells are compared over the whole 2003-2006 record, and
    # there a ramp scores only 0.176 and a linear trend is 3.2% of the observed
    # variance. Detrending is therefore nearly a no-op here (median 0.563 vs
    # 0.564 raw); it changes the answer at 2 wells of 14, W8 and U2, where one
    # series carries a trend the other does not. The 30-day increment is the
    # column that actually discriminates between forcings.
    # nearest-neighbour matching alone would pair an observation in the middle
    # of a 156-day forecast gap with a model value two months away; the
    # tolerance drops those instead of scoring the emulator on months it never
    # forecast. It removes nothing under reanalysis forcing and about a third
    # of the observations under SEAS5.
    m = mod["mean"].reindex(o.index, method="nearest",
                            tolerance=pd.Timedelta("3D"))
    keep = m.notna().values
    m, o_ = m[keep], o[keep]
    a, b = m.values.astype(float), o_.values.astype(float)
    if len(a) < 10:
        return None
    raw = float(np.corrcoef(a, b)[0, 1])

    def _detrend(z):
        t = np.arange(len(z), dtype=float)
        return z - np.polyval(np.polyfit(t, z, 1), t)

    det = float(np.corrcoef(_detrend(a), _detrend(b))[0, 1])
    # increments must not straddle a gap either
    k = 30
    span = (o_.index[k:] - o_.index[:-k]).days.values
    near = span <= 45
    inc = (float(np.corrcoef((a[k:] - a[:-k])[near],
                             (b[k:] - b[:-k])[near])[0, 1])
           if near.sum() > 10 else np.nan)
    ax.text(.985, .04,
            f"$r$ = {raw:+.2f}" + (f"   30-day {inc:+.2f}" if np.isfinite(inc) else "")
            + f"   shift {off:+.2f} m",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=vs.FS_ANNOT-6, color="#4A555F",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#DDDDDD", alpha=.88))
    return dict(raw=raw, detrended=det, increment=inc)


def _null_ax():
    """panel() both draws and scores; this lets the pooled numbers be
    recomputed without producing a figure."""
    f = plt.figure()
    return f.add_subplot(111)


WATER_YEARS = (2003, 2004, 2005, 2006)


def window(wy):
    """The water year. A reanalysis-driven rollout fills it; a SEAS5 chain
    reaches only late April, and the panel clips itself to whichever it is."""
    return pd.Timestamp(f"{wy-1}-10-01"), pd.Timestamp(f"{wy}-10-01")


def short_label(w):
    """A row label short enough to sit rotated beside a 2-inch panel.
    The full titles ("U3  ICN BROWNSTOWN (UIUC 421833)") overlapped each
    other when stacked six deep."""
    t = w["title"]
    t = t.split("  ", 1)[-1]                 # drop the leading short id
    t = re.sub(r"\s*\(.*?\)", "", t)          # drop the agency identifier
    t = re.split(r"\s+--\s+|\s+-\s+", t)[0]   # drop the state
    t = t.replace("ICN ", "").strip()
    if len(t) > 14:
        t = t[:13].rstrip() + "."
    return f"{w['short_id']}  {t}"


def year_panel(ax, ob, mod, color, wy, off):
    """One well, one water year. The x axis is the forecast, nothing else."""
    lo, hi = window(wy)
    m = mod[(mod.index >= lo) & (mod.index <= hi)].dropna(subset=["mean"]) \
        if mod is not None else None
    o = ob[(ob.index >= lo) & (ob.index <= hi)] if ob is not None else None
    if m is None or not len(m) or o is None or not len(o):
        ax.text(.5, .5, "no forecast", ha="center", va="center",
                transform=ax.transAxes, fontsize=vs.FS_ANNOT - 6,
                color="#999999")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return None

    ax.fill_between(m.index, m["lo"], m["hi"], color=color, alpha=.20, lw=0)
    ax.plot(m.index, m["mean"], color=color, lw=vs.LW_MAIN - 0.6,
            label="emulator")
    ax.plot(o.index, o.values + off, color=vs.C_ACTUAL, lw=vs.LW_SECONDARY - 0.7,
            label="observed (shifted to the model mean)")
    ax.invert_yaxis()
    # clip to the rollout actually present, so a 210-day chain is not drawn
    # inside five months of blank axis
    pad = pd.Timedelta(days=6)
    ax.set_xlim(max(lo, m.index.min() - pad), min(hi, m.index.max() + pad))
    ax.margins(y=.16)
    vs.style_axes(ax)
    span = (m.index.max() - m.index.min()).days
    ax.xaxis.set_major_locator(mdates.MonthLocator(
        bymonth=(10, 1, 4, 7) if span > 300 else (10, 1, 4)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for t in ax.get_xticklabels():
        t.set_fontsize(vs.FS_TICK - 7)
    for t in ax.get_yticklabels():
        t.set_fontsize(vs.FS_TICK - 7)

    mm = m["mean"].reindex(o.index, method="nearest",
                           tolerance=pd.Timedelta("3D"))
    keep = mm.notna().values
    a = mm[keep].values.astype(float)
    b = o[keep].values.astype(float) + off
    # A correlation needs both series to vary. U2's record inside WY2003 sits
    # entirely within its 283-sample flat-lined run, so the observation is a
    # constant there and the coefficient is undefined, not zero.
    if len(a) < 30 or a.std() == 0 or b.std() == 0:
        ax.text(.03, .05, "no score", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=vs.FS_ANNOT - 8, color="#999999")
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    k = 30
    da, db = a[k:] - a[:-k], b[k:] - b[:-k]
    inc = (float(np.corrcoef(da, db)[0, 1])
           if len(a) > k + 10 and da.std() > 0 and db.std() > 0 else np.nan)
    if not np.isfinite(r):
        ax.text(.03, .05, "no score", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=vs.FS_ANNOT - 8, color="#999999")
        return None
    ax.text(.03, .05, f"$r$={r:+.2f}" + (f"  30d={inc:+.2f}"
                                         if np.isfinite(inc) else ""),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=vs.FS_ANNOT - 8, color="#4A555F",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#DDDDDD", alpha=.9))
    return dict(raw=r, increment=inc)


def figure_by_year(out_dir, net, sel, obs, mods, color, label, tag):
    """Rows are wells, columns are water years: every 210-day rollout gets
    its own axes instead of being stacked onto a four-year time line, where
    each one occupied a quarter of the width and the five summer months with
    no forecast read as missing data."""
    nr, nc = len(sel), len(WATER_YEARS)
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.15 * nr),
                             squeeze=False)
    rows = []
    for i, w in enumerate(sel):
        ob, mod = obs.get(w["short_id"]), mods.get(w["short_id"])
        off = 0.0
        if ob is not None and mod is not None and len(ob):
            ov = ob[(ob.index >= mod.index.min()) & (ob.index <= mod.index.max())]
            if len(ov):
                off = float(mod["mean"].mean(skipna=True) - ov.mean())
        for j, wy in enumerate(WATER_YEARS):
            r = year_panel(axes[i][j], ob, mod, color, wy, off)
            if r is not None:
                rows.append(dict(forcing=tag, network=net, well=w["short_id"],
                                 water_year=wy, **r))
            if i == 0:
                axes[i][j].set_title(f"WY{wy}", fontsize=vs.FS_TITLE - 4,
                                     fontweight="bold", pad=7)
        axes[i][0].set_ylabel(f"{short_label(w)}\nshift {off:+.1f} m",
                              fontsize=vs.FS_LABEL - 8, fontweight="bold",
                              labelpad=8)
        axes[i][0].yaxis.set_visible(True)
    h, l = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if h:
            break
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, .0), ncol=2,
               frameon=False, fontsize=vs.FS_LEGEND - 2)
    kind = "210-day forecast" if tag == "seas5" else "water-year rollout"
    fig.suptitle(f"{net} observation wells, one panel per {kind}"
                 f"\n{label}", fontsize=vs.FS_SUPTITLE - 4, fontweight="bold")
    fig.supylabel("Depth to water [m]", fontsize=vs.FS_LABEL - 2,
                  fontweight="bold")
    fig.tight_layout(rect=(.035, .03, 1, .945))
    fig.subplots_adjust(hspace=.40, wspace=.19)
    name = f"fig_wells_{tag}_{net.lower()}"
    vs.save_both(fig, os.path.join(out_dir, name)); plt.close(fig)
    print(f"  wrote {name}  ({nr} wells x {nc} water years)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    vs.apply_style(); os.makedirs(args.out_dir, exist_ok=True)
    ws = load_wells(); obs = observations(ws)
    sets = {"nldas": ("NLDAS-2 reanalysis forcing", vs.C_PRED, model_series("nldas", ws)),
            "seas5": ("raw SEAS5 seasonal forcing", "#1C7C54",
                      model_series("seas5_D", ws))}
    stats, stats_year = [], []
    for tag, (label, color, mods) in sets.items():
        for net in ("USGS", "UIUC"):
            sel = [w for w in ws if w["net"] == net]
            if not sel:
                continue
            # One panel per rollout, for both forcings: four folds are four
            # independent forecasts and do not belong on a shared time axis.
            stats_year.extend(figure_by_year(args.out_dir, net, sel, obs,
                                             mods, color, label, tag))
            # the pooled numbers the report's tables quote are still computed,
            # from the same matched days, without producing a figure
            for w in sel:
                r = panel(_null_ax(), obs.get(w["short_id"]),
                          mods.get(w["short_id"]), w["title"],
                          w.get("subtitle", ""), color)
                if r is not None:
                    stats.append(dict(forcing=tag, network=net,
                                      well=w["short_id"], **r))
            plt.close("all")
            continue
            n = len(sel); rows = (n + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(16, 3.3*rows),
                                     squeeze=False)
            for k, (ax, w) in enumerate(zip(axes.ravel(), sel)):
                r = panel(ax, obs.get(w["short_id"]), mods.get(w["short_id"]),
                          w["title"], w.get("subtitle", ""), color,
                          first_col=(k % 2 == 0))
                if r is not None:
                    stats.append(dict(forcing=tag, network=net,
                                      well=w["short_id"], **r))
            for ax in axes.ravel()[n:]:
                ax.axis("off")
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, .0), ncol=2,
                       frameon=False, fontsize=vs.FS_LEGEND-2)
            fig.suptitle(f"{net} observation wells against the emulator, {label}",
                         fontsize=vs.FS_SUPTITLE-3, fontweight="bold")
            fig.supylabel("Depth to water [m]", fontsize=vs.FS_LABEL-1,
                          fontweight="bold")
            fig.tight_layout(rect=(.015, .035, 1, .955))
            fig.subplots_adjust(hspace=.62, wspace=.13)
            name = f"fig_wells_{tag}_{net.lower()}"
            vs.save_both(fig, os.path.join(args.out_dir, name)); plt.close(fig)
            print(f"  wrote {name}")
    if stats_year:
        pd.DataFrame(stats_year).to_csv(
            os.path.join(args.out_dir, "wells_correlation_by_year.csv"),
            index=False)
        t = pd.DataFrame(stats_year).groupby(
            ["forcing", "network", "water_year"])[["raw", "increment"]].mean()
        print("\n  per-rollout correlation, averaged over the wells:")
        print(t.round(3).to_string())
    if stats:
        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(args.out_dir, "wells_correlation.csv"), index=False)
        print("\n  correlation with the observation, shifted by one constant")
        print("  per site. Over 2003-2006 a linear trend is ~3% of the observed")
        print("  variance, so raw and detrended agree closely; the 30-day")
        print("  increment is the column that separates the two forcings.")
        print(df.groupby(["forcing", "network"])[["raw", "detrended", "increment"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
