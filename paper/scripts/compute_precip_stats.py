"""Per-water-year precipitation statistics over the study domain.

Reads data/parflow_raw/forcing_wy<YYYY>.nc, keys every unit conversion off the
`units` attribute of the precipitation variable, and writes

    paper/tables/precip_stats.tex   booktabs tabular, no table wrapper
    paper/tables/precip_stats.json  the same numbers, plus the daily
                                    domain-mean series used by fig_precip_folds

Statistic definitions (stated in the .tex footnote as well):
  season total        sum over the water year of the domain-mean daily depth
  domain-mean daily   season total / number of days
  wet-day fraction    fraction of grid-cell-days with depth > 0.1 mm
  wet-day intensity   mean depth over those grid-cell-days
  p95                 95th percentile of all grid-cell-day depths

Grid-cell-days (the full space-time sample) are used for the last three so the
numbers follow the usual ETCCDI-style convention rather than being smoothed by
the 637x665 spatial average. The domain-mean-series equivalents are also
written to the JSON under "domain_mean_series_variants".
"""
import json
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P

P.apply_style()

TABLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
WET_THRESH_MM = 0.1


def to_mm_per_day(units):
    """Factor converting the stored precipitation units to mm per day.

    The forcing is daily, so a per-day rate and a daily accumulated depth are
    numerically the same number; anything else is rejected rather than guessed.
    """
    u = (units or "").strip().lower().replace(" ", "")
    if u in ("mm", "mm/d", "mm/day", "mmd-1", "kg/m2", "kg/m^2", "kgm-2"):
        return 1.0
    if u in ("m", "m/d", "m/day"):
        return 1000.0
    if u in ("mm/h", "mm/hr", "mm/hour", "kg/m2/s", "kgm-2s-1"):
        raise ValueError(f"sub-daily precipitation units {units!r}: aggregation "
                         "is ambiguous, refusing to guess")
    raise ValueError(f"unrecognised precipitation units {units!r}")


def stats_for(wy):
    path = os.path.join(P.RAW, f"forcing_wy{wy}.nc")
    with xr.open_dataset(path) as ds:
        pr = ds["precipitation"]
        units = pr.attrs.get("units")
        factor = to_mm_per_day(units)
        print(f"  WY{wy}: units={units!r} -> x{factor} mm/day, shape {pr.shape}")
        a = pr.values.astype(np.float32) * np.float32(factor)
        dates = np.datetime_as_string(ds["time"].values, unit="D")

    nan = int(np.isnan(a).sum())
    if nan:
        print(f"    WARNING {nan} NaN cells, excluded from the sample")
    flat = a[np.isfinite(a)].ravel()
    daily = np.nanmean(a, axis=(1, 2)).astype(np.float64)   # domain mean, mm/day

    total = float(daily.sum())
    ndays = int(daily.size)
    wet = flat > WET_THRESH_MM
    nwet = int(wet.sum())

    out = dict(
        water_year=int(wy),
        fold=int(P.WY_FOLD[wy]),
        start=str(dates[0]), end=str(dates[-1]), n_days=ndays,
        precip_units_in_file=units, unit_factor_to_mm_per_day=factor,
        season_total_mm=total,
        domain_mean_daily_mm=total / ndays,
        wet_day_fraction=nwet / float(flat.size),
        wet_day_intensity_mm=float(flat[wet].mean()) if nwet else 0.0,
        p95_daily_mm=float(np.percentile(flat, 95)),
        n_cell_days=int(flat.size), n_wet_cell_days=nwet,
        max_daily_cell_mm=float(flat.max()),
        max_daily_domain_mean_mm=float(daily.max()),
        domain_mean_series_variants=dict(
            wet_day_fraction=float((daily > WET_THRESH_MM).mean()),
            wet_day_intensity_mm=float(daily[daily > WET_THRESH_MM].mean())
            if (daily > WET_THRESH_MM).any() else 0.0,
            p95_daily_mm=float(np.percentile(daily, 95))),
        daily_domain_mean_mm=[round(float(v), 6) for v in daily],
        dates=[str(d) for d in dates],
    )
    del a, flat
    return out


# One column per statistic, and the maximum of each column set in bold so the
# wettest year in each sense can be found at a glance. Headers are split over
# two lines and abbreviated: spelled out on one line the tabular is 521 pt
# against a 470 pt \textwidth, which overruns the page.
KEYS = (("season_total_mm", "{:.1f}", "Season", r"total (mm)"),
        ("domain_mean_daily_mm", "{:.2f}", "Daily mean", r"(mm\,d$^{-1}$)"),
        ("wet_day_fraction", "{:.3f}", "Wet-day", "fraction"),
        ("wet_day_intensity_mm", "{:.2f}", "Wet-day", "intensity (mm)"),
        ("p95_daily_mm", "{:.2f}", r"$p_{95}$", r"(mm\,d$^{-1}$)"))


def write_tex(rows):
    hi = {k: max(r[k] for r in rows) for k, _, _, _ in KEYS}

    def cell(r, k, fmt):
        t = fmt.format(r[k])
        return "$\\mathbf{" + t + "}$" if r[k] == hi[k] else t

    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         "Water year & " + " & ".join(h for _, _, h, _ in KEYS) + r" \\",
         " & " + " & ".join(u for _, _, _, u in KEYS) + r" \\",
         r"\midrule"]
    for r in rows:
        L.append(f"WY{r['water_year']} & "
                 + " & ".join(cell(r, k, f) for k, f, _, _ in KEYS) + r" \\")
    L += [r"\midrule"]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    L.append("Mean & " + " & ".join(f.format(mean(k)) for k, f, _, _ in KEYS)
             + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += ["%",
          "% Season total and domain-mean daily depth are averages over the",
          "% 665 x 637 km study domain (665 cols x 637 rows at 1 km).",
          "% Wet-day fraction, wet-day intensity and the 95th percentile are",
          "% computed over all grid-cell-days of the water year.",
          "% Source: NLDAS-2 precipitation in",
          "% data/parflow_raw/forcing_wy<YYYY>.nc (units attribute: mm per day)."]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "precip_stats.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("  wrote precip_stats.tex")


if __name__ == "__main__":
    rows = [stats_for(wy) for wy in P.WATER_YEARS]       # chronological
    write_tex(rows)
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "precip_stats.json"), "w") as f:
        json.dump(dict(wet_day_threshold_mm=WET_THRESH_MM,
                       water_years=list(P.WATER_YEARS),
                       rows=rows), f, indent=1)
    print("  wrote precip_stats.json")
    for r in rows:
        print(f"WY{r['water_year']}  total={r['season_total_mm']:8.1f} mm  "
              f"mean={r['domain_mean_daily_mm']:5.2f}  "
              f"wetfrac={r['wet_day_fraction']:.3f}  "
              f"wetint={r['wet_day_intensity_mm']:5.2f}  "
              f"p95={r['p95_daily_mm']:5.2f}  max_cell={r['max_daily_cell_mm']:6.1f}")
