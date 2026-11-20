"""
The cross-fold story: 4-fold leave-one-water-year-out x 2 forecast windows.

Every water year in the archive (WY2003-2006) takes a turn as the unseen test
year, so the headline claim is not "it worked on 2006" but "it worked on every
year we have, including years whose hydrology the model never saw".

One caveat drives the whole design of these figures: the four test years are
NOT equally hard. A year with a big seasonal swing gives persistence a large
error and the emulator an easy target; a quiet year does the opposite. So raw
RMSE is not comparable across folds -- SKILL vs. persistence is, and that is
what the cross-fold panels lead with.

Figures written to --out-dir:
  cv_headline.png       -- annual RMSE per fold/lead against that year's persistence
  cv_skill.png          -- skill score + % of pixels beating persistence
  cv_error_growth.png   -- normalised error growth through the year, all 8 cases
  cv_correlation_maps.png -- per-pixel correlation, 4 folds x 2 leads, north-up
  cv_lead_verdict.png   -- paired 10-day vs 15-day outcome across folds
  cv_summary.csv        -- every number in the figures

Usage:
    python compare_folds.py --out-dir ../data/comparisons/cross_fold
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import viz_style as vs
from analyze_rollout import rolling_mean, VAR_UNITS, VAR_PRETTY

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# fold -> (train years, test year). Each year is held out exactly once.
FOLDS = {
    "fold1": {"train": [2003, 2004, 2005], "test": 2006},
    "fold2": {"train": [2003, 2004, 2006], "test": 2005},
    "fold3": {"train": [2003, 2005, 2006], "test": 2004},
    "fold4": {"train": [2004, 2005, 2006], "test": 2003},
}
LEADS = [10, 15]
VARS = ["water_table_depth", "soil_moisture"]
LEAD_COLORS = {10: vs.C_F10, 15: vs.C_F15}


def load_all():
    cases = {}
    for fold, spec in FOLDS.items():
        for lead in LEADS:
            an = os.path.join(DATA_DIR, f"rollout_wy{spec['test']}_{fold}_f{lead}_lean", "analysis")
            path = os.path.join(an, "summary.json")
            if not os.path.exists(path):
                print(f"WARNING: missing {path} -- skipping {fold} f{lead}")
                continue
            with open(path) as f:
                cases[(fold, lead)] = json.load(f)
            cases[(fold, lead)]["_analysis_dir"] = an
    return cases


def plot_headline(cases, save_path):
    """Annual RMSE per fold/lead, with that year's persistence drawn as the bar
    each model has to get under."""
    folds = list(FOLDS)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=140, constrained_layout=True)
    x = np.arange(len(folds))
    width = 0.26
    for ax, var in zip(axes, VARS):
        persist = [cases[(f, LEADS[0])][var]["persistence_rmse_overall"] for f in folds]
        ax.bar(x - width, persist, width, color=vs.C_PERSIST, label="persistence (do nothing)",
               edgecolor="white", linewidth=1.5)
        for j, lead in enumerate(LEADS):
            vals = [cases[(f, lead)][var]["rmse_overall"] for f in folds]
            ax.bar(x + j * width, vals, width, color=LEAD_COLORS[lead],
                   label=f"DDPM {lead}-day window", edgecolor="white", linewidth=1.5)
            for xi, v in zip(x + j * width, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=vs.FS_ANNOT - 2, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f}\ntest WY{FOLDS[f]['test']}" for f in folds])
        vs.style_axes(ax, ylabel=f"Annual RMSE ({VAR_UNITS[var]})", title=VAR_PRETTY[var])
        ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="upper left")
    fig.suptitle("Full-year autoregressive rollout: DDPM vs. persistence in every fold\n"
                 "lower is better; grey bar is the do-nothing baseline for that test year",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_skill(cases, save_path):
    """Skill score and pixel-win-rate -- the fold-comparable metrics."""
    folds = list(FOLDS)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=140, constrained_layout=True)
    x = np.arange(len(folds))
    width = 0.34
    for row, (metric, ylab, fmt, ref) in enumerate([
        ("skill_vs_persistence", "Skill score\n1 - RMSE$_{model}$ / RMSE$_{persist}$", "{:.3f}", 0.0),
        ("frac_pixels_beating_persistence", "Fraction of pixels\nbeating persistence", "{:.2f}", 0.5),
    ]):
        for col, var in enumerate(VARS):
            ax = axes[row, col]
            for j, lead in enumerate(LEADS):
                vals = [cases[(f, lead)][var][metric] for f in folds]
                ax.bar(x + (j - 0.5) * width, vals, width, color=LEAD_COLORS[lead],
                       label=f"{lead}-day window", edgecolor="white", linewidth=1.5)
                for xi, v in zip(x + (j - 0.5) * width, vals):
                    ax.text(xi, v, fmt.format(v), ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=vs.FS_ANNOT - 2, fontweight="bold")
            ax.axhline(ref, color="#333333", linewidth=2.0, linestyle="--")
            ax.text(len(folds) - 0.45, ref, " break-even", va="center",
                    fontsize=vs.FS_ANNOT - 1, fontweight="bold", color="#333333")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{f}\nWY{FOLDS[f]['test']}" for f in folds])
            vs.style_axes(ax, ylabel=ylab if col == 0 else None,
                          title=VAR_PRETTY[var] if row == 0 else None)
            if row == 0 and col == 0:
                ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="best")
    fig.suptitle("Skill against persistence -- the metric that IS comparable across folds\n"
                 "above the dashed line the emulator is adding information",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_error_growth(cases, save_path):
    """Error through the year, each case divided by its own persistence curve,
    which removes the 'some years are simply harder' confound."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=140, constrained_layout=True)
    for col, var in enumerate(VARS):
        ax_abs, ax_rel = axes[0, col], axes[1, col]
        for fold in FOLDS:
            for lead in LEADS:
                if (fold, lead) not in cases:
                    continue
                s = cases[(fold, lead)][var]
                vals = rolling_mean(np.array(s["rmse_by_day"]))
                persist = np.array(s["persistence_rmse_by_day"])
                style = "-" if lead == 10 else "--"
                color = vs.FOLD_COLORS[fold]
                ax_abs.plot(vals, color=color, linestyle=style, linewidth=vs.LW_MAIN,
                            label=f"{fold} (WY{FOLDS[fold]['test']}) f{lead}")
                with np.errstate(invalid="ignore", divide="ignore"):
                    ax_rel.plot(vals / np.maximum(persist, 1e-9), color=color,
                                linestyle=style, linewidth=vs.LW_MAIN)
        ax_rel.axhline(1.0, color="#333333", linewidth=2.2, linestyle=":")
        ax_rel.text(5, 1.02, "= persistence", fontsize=vs.FS_ANNOT, fontweight="bold",
                    color="#333333", va="bottom")
        vs.style_axes(ax_abs, ylabel=f"RMSE ({VAR_UNITS[var]})" if col == 0 else None,
                      title=VAR_PRETTY[var])
        vs.style_axes(ax_rel, xlabel="Day of the rolled-out water year",
                      ylabel="RMSE / persistence RMSE" if col == 0 else None)
        ax_abs.set_ylim(bottom=0)
        ax_rel.set_ylim(0, 2.0)
        if col == 0:
            ax_abs.legend(fontsize=vs.FS_LEGEND - 3, frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Error growth through the chained rollout -- solid = 10-day window, dashed = 15-day\n"
                 "top row: raw RMSE (years differ in difficulty).  bottom row: divided by that "
                 "year's persistence, so all 8 cases are on one scale",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_correlation_maps(cases, var, save_path):
    """4 folds x 2 leads of per-pixel temporal correlation. The point of this
    panel is that the SAME geography is easy/hard no matter which year was held
    out -- the model has learned terrain-controlled behaviour, not one year."""
    folds = list(FOLDS)
    fig, axes = plt.subplots(len(LEADS), len(folds),
                             figsize=(5.2 * len(folds), 5.6 * len(LEADS)),
                             dpi=140, constrained_layout=True)
    im = None
    for r, lead in enumerate(LEADS):
        for c, fold in enumerate(folds):
            ax = axes[r, c]
            if (fold, lead) not in cases:
                vs.map_axes_off(ax, "missing")
                continue
            m = np.load(os.path.join(cases[(fold, lead)]["_analysis_dir"],
                                     f"maps_{var}.npz"))["corr"]
            im = vs.north_up_imshow(ax, m, cmap=vs.CMAP_CORR, vmin=-1, vmax=1)
            # both summaries, because they answer different questions and the
            # per-pixel r distribution is strongly left-skewed: the median is the
            # TYPICAL pixel, the mean is the domain average and is pulled down by
            # a minority of badly-tracked pixels. The 10d/15d ranking is the same
            # under either (and under a Fisher-z mean) -- see decompose_error.py.
            med, mean = np.nanmedian(m), np.nanmean(m)
            stat = f"median r = {med:.2f}   mean r = {mean:.2f}"
            title = (f"{fold} -- test WY{FOLDS[fold]['test']}\n{stat}"
                     if r == 0 else stat)
            vs.map_axes_off(ax, title)
            if c == 0:
                ax.set_ylabel(f"{lead}-day window", fontsize=vs.FS_LABEL + 2,
                              fontweight="bold", color=LEAD_COLORS[lead])
                ax.yaxis.set_visible(True)
                ax.set_yticks([])
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label("Pearson r (predicted vs. actual, per pixel)",
                     fontsize=vs.FS_CBAR_LABEL, fontweight="bold")
        cb.ax.tick_params(labelsize=vs.FS_CBAR_TICK)
    fig.suptitle(f"{VAR_PRETTY[var]}: temporal correlation across every fold and window\n"
                 "the same terrain is easy (blue) and hard (red) regardless of which year was held out\n"
                 "r is computed on anomalies and is therefore blind to drift -- see decomp_mse_terms.png",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lead_verdict(cases, save_path):
    """Paired 10-vs-15 outcome: one line per fold, so the verdict is read off
    the direction of the lines rather than from eight separate bars."""
    folds = list(FOLDS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=140, constrained_layout=True)
    for ax, var in zip(axes, VARS):
        for fold in folds:
            if (fold, 10) not in cases or (fold, 15) not in cases:
                continue
            y = [cases[(fold, lead)][var]["skill_vs_persistence"] for lead in LEADS]
            ax.plot([0, 1], y, color=vs.FOLD_COLORS[fold], linewidth=vs.LW_MAIN + 0.6,
                    marker="o", markersize=15, markeredgecolor="white", markeredgewidth=2.5,
                    label=f"{fold} (test WY{FOLDS[fold]['test']})")
            ax.annotate(f"{y[1] - y[0]:+.3f}", (1, y[1]), xytext=(14, 0),
                        textcoords="offset points", va="center",
                        fontsize=vs.FS_ANNOT, fontweight="bold", color=vs.FOLD_COLORS[fold])
        ax.axhline(0, color="#333333", linewidth=2.0, linestyle="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["10-day window", "15-day window"])
        ax.set_xlim(-0.25, 1.35)
        vs.style_axes(ax, ylabel="Skill vs. persistence" if var == VARS[0] else None,
                      title=VAR_PRETTY[var])
        if var == VARS[0]:
            ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="best")
    fig.suptitle("Does a longer forecast window help?  Each line is one fold; "
                 "the number is the change in skill\nupward slope = the 15-day window is better for that year",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(cases, save_path):
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "train_years", "test_year", "lead_days", "variable",
                    "rmse_overall", "persistence_rmse", "skill_vs_persistence",
                    "median_pixel_rmse", "median_pixel_corr",
                    "frac_pixels_beating_persistence", "rmse_first_30d", "rmse_last_30d"])
        for fold, spec in FOLDS.items():
            for lead in LEADS:
                if (fold, lead) not in cases:
                    continue
                for var in VARS:
                    s = cases[(fold, lead)][var]
                    w.writerow([fold, "+".join(map(str, spec["train"])), spec["test"], lead, var,
                                f"{s['rmse_overall']:.5f}", f"{s['persistence_rmse_overall']:.5f}",
                                f"{s['skill_vs_persistence']:.4f}", f"{s['median_pixel_rmse']:.5f}",
                                f"{s['median_pixel_corr']:.4f}",
                                f"{s['frac_pixels_beating_persistence']:.4f}",
                                f"{s['rmse_first_30d']:.5f}", f"{s['rmse_last_30d']:.5f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    cases = load_all()
    if not cases:
        raise SystemExit("no analysis summaries found -- run analyze_rollout.py first")

    plot_headline(cases, os.path.join(args.out_dir, "cv_headline.png"))
    plot_skill(cases, os.path.join(args.out_dir, "cv_skill.png"))
    plot_error_growth(cases, os.path.join(args.out_dir, "cv_error_growth.png"))
    for var in VARS:
        plot_correlation_maps(cases, var, os.path.join(args.out_dir, f"cv_correlation_maps_{var}.png"))
    plot_lead_verdict(cases, os.path.join(args.out_dir, "cv_lead_verdict.png"))
    write_summary(cases, os.path.join(args.out_dir, "cv_summary.csv"))

    print(f"wrote cross-fold comparison ({len(cases)} cases) -> {args.out_dir}")
    for (fold, lead), s in sorted(cases.items()):
        wtd = s["water_table_depth"]
        print(f"  {fold} f{lead:2d} (WY{s['test_year']}): "
              f"WTD RMSE {wtd['rmse_overall']:.3f} m vs persistence "
              f"{wtd['persistence_rmse_overall']:.3f} m  ->  skill {wtd['skill_vs_persistence']:+.3f}")


if __name__ == "__main__":
    main()
