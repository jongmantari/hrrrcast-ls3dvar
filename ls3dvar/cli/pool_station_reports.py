#!/usr/bin/env python3
# ls3dvar/cli/pool_station_reports.py  (extracted from mantari/workflows/pool_station_reports.py)
"""
ls3dvar/cli/pool_station_reports.py

Pool the per-station reports of K-fold (or multi-seed) LS-3DVar runs and score
every station on the runs where it was HELD OUT.

    python -m ls3dvar.cli.pool_station_reports <dir>/station_report_fold*of5.csv [--top 15]

prints
  - pooled held-out O-B -> O-A rms per variable (every location held out once with --fold)
  - the same per network (asos / tempest / grenade ...)
  - the stations the analysis hurts most  (held-out |O-A| - |O-B| largest)
  - obs_qc rejections
writes pooled_heldout.csv next to the first input.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.nanmean(a ** 2))) if a.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in a.reports], ignore_index=True)
    rej = df[df.get("rejected", pd.Series(index=df.index, dtype=object)).notna()] \
        if "rejected" in df else df.iloc[:0]
    ho = df[(df["held_out"] == True) & df["oma"].notna()].copy()            # noqa: E712
    print(f"{len(a.reports)} reports, {len(ho)} held-out station-variables, "
          f"{ho.groupby(['variable', 'station']).ngroups} unique")

    def table(g):
        rows = []
        for key, d in g:
            rb, ra = rms(d["omb"]), rms(d["oma"])
            rows.append((key, len(d), rb, ra, 100 * (rb - ra) / rb if rb else np.nan,
                         d["omb"].mean(), d["oma"].mean()))
        return rows

    print(f"\n{'variable':10s} {'n':>5s} {'O-B rms':>9s} {'O-A rms':>9s} {'impr%':>7s} {'O-B mean':>9s} {'O-A mean':>9s}")
    for key, n, rb, ra, imp, mb, ma in table(ho.groupby("variable")):
        print(f"{key:10s} {n:5d} {rb:9.3f} {ra:9.3f} {imp:+7.2f} {mb:+9.3f} {ma:+9.3f}")

    print(f"\n{'variable':10s} {'network':10s} {'n':>5s} {'O-B rms':>9s} {'O-A rms':>9s} {'impr%':>7s}")
    for (v, net), n, rb, ra, imp, _, _ in table(ho.groupby(["variable", "network"])):
        print(f"{v:10s} {net:10s} {n:5d} {rb:9.3f} {ra:9.3f} {imp:+7.2f}")

    ho["damage"] = ho["oma"].abs() - ho["omb"].abs()
    print(f"\nstations the analysis hurts most (held out):  damage = |O-A| - |O-B|")
    cols = ["variable", "station", "network", "lat", "lon", "omb", "oma", "damage"]
    print(ho.sort_values("damage", ascending=False)[cols].head(a.top).to_string(index=False,
                                                                                float_format=lambda x: f"{x:8.3f}"))

    if len(rej):
        print(f"\nobs_qc rejections ({len(rej)}):")
        print(rej[["variable", "station", "network", "omb", "rejected"]].drop_duplicates()
              .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    out = Path(a.reports[0]).with_name("pooled_heldout.csv")
    ho.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
