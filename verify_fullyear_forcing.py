"""
Check that the spliced October + May forcing covers each water year without a
gap, before any GPU time is spent on it.

A missing forcing day does not announce itself. The rollout would carry on and
the results would look merely disappointing rather than broken, so the gap is
checked here and treated as fatal.
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, ".")
from seas5_chained import ChainedForcing

WY_INIT = {2003: (2002, 10), 2004: (2003, 10), 2005: (2004, 10),
           2006: (2005, 10)}
N_DAYS = 360


# Check the COARSE files, which is what the rollout actually opens. An earlier
# version of this gate checked the RAW downloads instead. Those were present,
# so it passed, while the coarse files the rollout needs had never been
# written -- the inference job then started, failed on every member in seconds,
# and reported success. A gate must test the artefact that is consumed.
class Src:
    """Minimal stand-in exposing what ChainedForcing needs for routing."""
    def __init__(self, path):
        d = xr.open_dataset(path, decode_timedelta=False)
        n = int(d.sizes.get("lead_day", 0))
        if n == 0:
            raise SystemExit(f"{path} has no lead_day dimension")
        self._src = {"tmin": np.zeros((1, n, 1, 1))}
        self.n_member = int(d.sizes.get("member", 0))
        d.close()

    def dates(self, g):
        return pd.DatetimeIndex(g)


def main():
    ok = True
    for wy, (iy, im) in sorted(WY_INIT.items()):
        oct_p = f"data/seas5/forcing_coarse_seas5_{iy}{im:02d}.nc"
        may_p = f"data/seas5/forcing_coarse_seas5_{wy}05.nc"
        try:
            so, sm = Src(oct_p), Src(may_p)
        except Exception as e:
            print(f"  WY{wy}: MISSING {type(e).__name__}: {str(e)[:90]}")
            ok = False
            continue
        ch = ChainedForcing([(pd.Timestamp(f"{iy}-{im:02d}-01"), so),
                             (pd.Timestamp(f"{wy}-05-01"), sm)])
        days = pd.date_range(f"{iy}-{im:02d}-02", periods=N_DAYS)
        try:
            pick = ch._route(days)
        except SystemExit as e:
            print(f"  WY{wy}: GAP {str(e)[:110]}")
            ok = False
            continue
        sp = ch.splice_dates(days)
        if so.n_member != sm.n_member:
            print(f"  WY{wy}: member counts differ, "
                  f"Oct {so.n_member} vs May {sm.n_member}")
            ok = False
        print(f"  WY{wy}: OK  Oct {int((pick==0).sum())} d + "
              f"May {int((pick==1).sum())} d = {len(days)} d, "
              f"splice {sp[0].date() if sp else 'none'}, "
              f"{so.n_member} members, {days[0].date()}..{days[-1].date()}")
    if not ok:
        sys.exit("forcing is not continuous; not proceeding to inference")
    print("all four water years are continuous over 360 days")


if __name__ == "__main__":
    main()
