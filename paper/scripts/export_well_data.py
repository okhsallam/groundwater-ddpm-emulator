"""
Export the observation-well records as CSV, one table per well, for sharing.

What makes this worth producing rather than forwarding the source files: the
two networks arrive in different formats with different column names and
different conventions, the ISWS file holds wells outside the model domain and
stores many readings twice, and neither file carries the number a well is
given in the paper. The export is one schema for both networks, one row per
day, with the paper's well number attached so a table can be matched to a
figure.

NOTHING IS SILENTLY DROPPED. Every measured value appears; the `flag` column
says what was done with it:

  ok              used in the paper's figures and scores
  screened_spike  a reverting excursion rejected by the screening
  duplicate_day   a second, disagreeing reading on a day already represented;
                  the value kept is the one consistent with the surrounding
                  week, and this row carries the one not kept

so a recipient who disagrees with the screening can reinstate the points by
ignoring the column.

Writes paper/well_data_export/ and a .zip beside it.
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as P                                        # noqa: E402
import wells_common as C                                       # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "well_data_export")
SPAN = C.OBS_SPAN


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    ws = C.load_wells()
    raw = C.raw_observations(ws)
    clean, rep = C.clean_observations(raw, report=True)
    rep = rep.set_index("sid")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "per_well"))

    index, long_rows = [], []
    for w in ws:
        sid = w["short_id"]
        s = raw[sid]
        s = s[(s.index >= SPAN[0]) & (s.index < SPAN[1])]
        kept = clean[sid]                       # daily, screened
        spikes = set(pd.to_datetime(rep.loc[sid, "dates"]))

        # one row per measured value, collapsed to the day it belongs to
        by_day = {}
        for ts, v in s.items():
            by_day.setdefault(ts.normalize(), []).append(float(v))

        rows = []
        for day in sorted(by_day):
            vals = sorted(set(by_day[day]))     # verbatim repeats collapse
            chosen = kept.get(day, np.nan)
            for v in vals:
                if day in spikes:
                    flag = "screened_spike"
                elif len(vals) > 1 and not (np.isfinite(chosen)
                                            and abs(v - chosen) < 1e-9):
                    flag = "duplicate_day"
                else:
                    flag = "ok"
                rows.append(dict(
                    well_number=w["no"], short_id=sid, well_name=w["pretty"],
                    network="USGS" if w["net"] == "USGS" else "ISWS/UIUC",
                    latitude_wgs84=round(w["lat"], 6),
                    longitude_wgs84=round(w["lon"], 6),
                    date=day.date().isoformat(),
                    depth_to_water_m=round(v, 4), flag=flag))
        df = pd.DataFrame(rows)
        fn = f"well_{w['no']:02d}_{slug(w['pretty'])}.csv"
        df.to_csv(os.path.join(OUT, "per_well", fn), index=False)
        long_rows.append(df)

        used = df[df.flag == "ok"]
        index.append(dict(
            well_number=w["no"], short_id=sid, well_name=w["pretty"],
            network="USGS" if w["net"] == "USGS" else "ISWS/UIUC",
            source_id=w.get("p_number", w.get("site_no", "")),
            latitude_wgs84=round(w["lat"], 6),
            longitude_wgs84=round(w["lon"], 6),
            model_grid_row=w["y"], model_grid_col=w["x"],
            n_values_total=len(df), n_values_used=len(used),
            n_screened=int((df.flag == "screened_spike").sum()),
            n_duplicate_day=int((df.flag == "duplicate_day").sum()),
            first_date=df.date.min() if len(df) else "",
            last_date=df.date.max() if len(df) else "",
            median_depth_m=round(float(used.depth_to_water_m.median()), 3)
            if len(used) else "",
            min_depth_m=round(float(used.depth_to_water_m.min()), 3)
            if len(used) else "",
            max_depth_m=round(float(used.depth_to_water_m.max()), 3)
            if len(used) else "",
            file=os.path.join("per_well", fn)))

    idx = pd.DataFrame(index)
    idx.to_csv(os.path.join(OUT, "wells_index.csv"), index=False)
    pd.concat(long_rows, ignore_index=True).to_csv(
        os.path.join(OUT, "all_wells_long.csv"), index=False)

    open(os.path.join(OUT, "README.txt"), "w").write(f"""\
Observation-well water-level records, study domain, WY2003-WY2006
=================================================================

{len(ws)} wells: {sum(w['net'] == 'USGS' for w in ws)} USGS and
{sum(w['net'] != 'USGS' for w in ws)} ISWS/UIUC unconfined-network wells.
Numbering is the single 1..{len(ws)} sequence used in the paper's figures and
tables: the USGS sites first, then the ISWS sites in order of median observed
depth.

FILES
  wells_index.csv      one row per well: number, name, network, coordinates,
                       the model grid cell it falls in, record length and
                       observed depth range.
  per_well/*.csv       one file per well, one row per measured value.
  all_wells_long.csv   every per-well file concatenated.

COLUMNS in the per-well files
  well_number          the number used in the paper, 1..{len(ws)}
  short_id             internal identifier (W* = USGS, U* = ISWS/UIUC)
  well_name, network
  latitude_wgs84, longitude_wgs84   as reported by the operating network
  date                 YYYY-MM-DD
  depth_to_water_m     depth to water below land surface, metres.
                       POSITIVE DOWNWARD: a larger number is a deeper water
                       table. Negative values are water above land surface
                       and appear as measured.
  flag                 ok              used in the paper
                       screened_spike  rejected by the screening (see below)
                       duplicate_day   a second, disagreeing reading on a day
                                       already represented; the value kept is
                                       the one consistent with the surrounding
                                       week, and this row is the one not kept

Every measured value is present. To use the records exactly as the paper does,
keep rows with flag == "ok"; to use them as supplied, ignore the flag column.

SCREENING
Short excursions that depart from the local level and return to it are
flagged: at most three days, with the levels before and after agreeing to
within half the departure. The reverting condition is what distinguishes a
logging artefact from a recharge event, which is equally sudden but leaves the
series at a new level; steps are kept at any size. The size threshold is per
well and depends on what the sampling supports -- for a near-daily record,
twice that well's 99th-percentile daily change; for a monthly or quarterly
record, 1.5x its interquartile range.

{int(rep.n_spikes.sum())} values are flagged in total. Four of them are one
group: on 2005-01-03..05, four separately sampled ISWS wells (Crystal Lake,
Good Hope, Coffman-Barry, Greenfield) read 0.8 to 3.4 m deeper than the levels
either side of them, while none of the fifteen near-daily wells departs from
its own level on those dates. That pattern is not hydrological and looks like
one bad round of manual measurement; it may be worth checking against the
original field sheets.

COVERAGE
The network is very unevenly sampled: 15 wells near-daily, 18 monthly, and 12
only a handful of times in four years. Record length per well is in
wells_index.csv (n_values_used).

SOURCES
  USGS       data/usgs_wells/observations.csv
  ISWS/UIUC  DOE_GW_OBV_unconfined_2003_2006_depth.csv (unconfined wells only)
Three ISWS wells in the source file lie outside the model domain and are not
included here: Carbondale, Sparta and Boyleston.
""")

    base = shutil.make_archive(OUT, "zip", root_dir=os.path.dirname(OUT),
                               base_dir=os.path.basename(OUT))
    tot = int(idx.n_values_total.sum())
    print(f"  {len(ws)} wells, {tot} values "
          f"({int(idx.n_values_used.sum())} ok, "
          f"{int(idx.n_screened.sum())} screened, "
          f"{int(idx.n_duplicate_day.sum())} duplicate-day)")
    print(f"  folder: {OUT}")
    print(f"  zip   : {base}  ({os.path.getsize(base)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
