# ls3dvar/obs/time_window.py  (extracted from mantari/ls3dvar/observation_time.py)
"""
mantari/ls3dvar/observation_time.py

Time handling for analysis-time LS-3DVar.

    1. window   : keep obs with |t_obs - t_analysis| <= half_width
    2. thinning : per (source, station, variable) keep the report nearest to
                  t_analysis  (repeated reports from one station are not
                  independent, and R treats them as if they were)

Config (all optional):

    observations:
      time_window:
        enabled: true
        half_width_minutes: 30      # 23:30-00:30Z for a 00Z analysis
        thinning: nearest           # nearest | none
        keep_untimed: false         # obs without dateTime
"""
import numpy as np


def analysis_time_from_npz(hrrr):
    return np.datetime64(str(hrrr["init_datetime"]).replace(" ", "T"), "s")


def describe_times(observations, t_ana, label):
    dt = np.array([(o.time - t_ana) / np.timedelta64(1, "m")
                   for o in observations if o.time is not None], dtype=np.float64)
    n_untimed = sum(o.time is None for o in observations)
    if dt.size == 0:
        print(f"  {label}: n={len(observations)}  (no times; untimed={n_untimed})")
        return
    n_stn = len({(o.source, o.station, o.variable) for o in observations})
    print(f"  {label}: n={len(observations)}  station-variables={n_stn}  "
          f"dt[min] min={dt.min():+.0f} p50={np.median(dt):+.0f} max={dt.max():+.0f}  "
          f"untimed={n_untimed}")


def apply_time_window(observations, t_ana, cfg):
    tw = cfg.get("observations", {}).get("time_window", {})
    if not tw.get("enabled", True):
        return observations

    half = np.timedelta64(int(tw.get("half_width_minutes", 30) * 60), "s")
    keep_untimed = bool(tw.get("keep_untimed", False))
    thinning = tw.get("thinning", "nearest")

    print()
    print("=" * 80)
    print(f"OBSERVATION TIME WINDOW  t_ana={t_ana}  +/-{tw.get('half_width_minutes', 30)} min  "
          f"thinning={thinning}")
    print("=" * 80)
    describe_times(observations, t_ana, "raw     ")

    if not any(o.time is not None for o in observations):
        print("  WARNING: no dateTime in any obs file -> window NOT applied, obs unchanged")
        return observations

    win = [o for o in observations
           if (o.time is None and keep_untimed)
           or (o.time is not None and abs(o.time - t_ana) <= half)]
    describe_times(win, t_ana, "window  ")

    if thinning == "nearest":
        best = {}
        for o in win:
            key = (o.source, o.station, o.variable)
            d = abs(o.time - t_ana) if o.time is not None else np.timedelta64(10 ** 9, "s")
            if key not in best or d < best[key][0]:
                best[key] = (d, o)
        win = [o for _, o in best.values()]
        describe_times(win, t_ana, "thinned ")

    return win
