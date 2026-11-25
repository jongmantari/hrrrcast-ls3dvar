#!/usr/bin/env python3
# ls3dvar/cli/verify_state.py  (extracted from mantari/workflows/verify_state.py)
"""
Verify one or more model states (hrrr-npz format, normalized model_input)
against observations and an optional gridded reference, at the state's
valid time (init_datetime of the npz).

    obs  : T2M, D2M, station pressure (height-corrected), 10-m wind U10/V10
           (earth-relative), +/- window, nearest; T2M/D2M lapse-rate height correction
           if the --config enables observations.temperature_height_correction
    --subset heldout|assimilated --holdout-stations k/K : score only the stations that the
           analysis never saw (or only the assimilated ones)
    --qc   : same QC as the analysis (gross threshold + obs_qc buddy/network/blacklist),
           decided ONCE from the reference state (label "bg" if present, else the first)
           so all states are scored on identical observations
    truth: rms(state - truth) for T2M / D2M / PRES / U10 / V10 / TMP_850 on all / land / sea
           (U10/V10/TMP_850 need their stats: --config with channel_stats, else sigma units)

    python -m ls3dvar.cli.verify_state \
        --state an=cycles/20260811T00Z/fcst_an_f01.npz bg=cycles/20260811T00Z/fcst_bg_f01.npz \
        --obs-root /home/jonggyunkim/runs/s3/HRRRCast-live/obsdb \
        --truth-npz /home/jonggyunkim/runs/s3/HRRRCast-live/reference/hrrr_ana/20260811T01Z/hrrr_20260811_01.npz \
        --csv cycles/verification.csv --meta cycle=20260811T00Z lead=1
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


from ls3dvar.obs.ioda import ObservationProvider  # noqa: E402
from ls3dvar.obs.types import SurfaceObservationSet  # noqa: E402
from ls3dvar.obs.grid import build_station_indices  # noqa: E402
from ls3dvar.obs.time_window import apply_time_window  # noqa: E402
from ls3dvar.obs.operators import (  # noqa: E402
    t2m_observation_operator, d2m_observation_operator,
    station_height_offsets, make_station_pressure_operator,
    T2M_CHANNEL, T2M_STD, D2M_CHANNEL, D2M_STD, PRES_CHANNEL, PRES_STD,
    make_wind_operator, make_lapse_operator, lapse_settings,
)
from ls3dvar.obs.holdout import parse_spec, is_heldout  # noqa: E402
from ls3dvar.obs.qc import ObsQC  # noqa: E402
from ls3dvar.hrrrcast.channels import (  # noqa: E402
    channel_names, resolve_name, ChannelStats, grid_rotation,
)

VARS = {
    "airTemperatureAt2M": "T2M",
    "dewPointTemperatureAt2M": "D2M",
    "stationPressure": "PS",
    "windEastwardAt10M": "U10",
    "windNorthwardAt10M": "V10",
}
SHORTS = ("T2M", "D2M", "PS", "U10", "V10", "TMP850")


def rms(a):
    a = np.asarray(a, np.float64)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else np.nan


def valid_time(npz):
    return np.datetime64(str(npz["init_datetime"]).replace(" ", "T"), "s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs="+", required=True, help="label=path.npz ...")
    ap.add_argument("--obs-root", required=True, help="obsdb root (<type>/<YYYYMMDDTHHZ>/*.nc4)")
    ap.add_argument("--obs-types", nargs="+", default=["asos", "tempest", "grenade"])
    ap.add_argument("--half-width-minutes", type=float, default=30)
    ap.add_argument("--truth-npz", default=None)
    ap.add_argument("--csv", default=None, help="append rows to this CSV")
    ap.add_argument("--meta", nargs="*", default=[], help="key=value columns added to CSV rows")
    ap.add_argument("--config", default=None,
                    help="yaml with channel_stats / normalize_stats_nc / observations.wind_rotation")
    ap.add_argument("--wind-rotation", default=None, choices=["earth", "grid"],
                    help="model U10/V10 frame (default: config observations.wind_rotation, else earth)")
    ap.add_argument("--holdout-stations", default=None, help="k/K (as in the analysis)")
    ap.add_argument("--subset", default="all", choices=["all", "heldout", "assimilated"])
    ap.add_argument("--qc", action="store_true",
                    help="apply the analysis QC (thresholds + obs_qc from --config) before scoring")
    args = ap.parse_args()

    states = {}
    for item in args.state:
        label, path = item.split("=", 1)
        states[label] = (path, np.load(path, allow_pickle=True))
    first_path, first = next(iter(states.values()))
    t_valid = valid_time(first)
    for label, (path, d) in states.items():
        if valid_time(d) != t_valid:
            raise SystemExit(f"{label}: valid {valid_time(d)} != {t_valid}")
    sv = str(t_valid)                                   # 2026-08-11T01:00:00
    cyc = f"{sv[0:4]}{sv[5:7]}{sv[8:10]}T{sv[11:13]}Z"
    meta = dict(m.split("=", 1) for m in args.meta)
    meta["valid"] = cyc

    # ---- observations at the valid time -----------------------------------
    provider = ObservationProvider()
    obs = []
    for typ in args.obs_types:
        d = Path(args.obs_root) / typ / cyc
        if d.is_dir():
            obs += provider.from_ioda_directory(str(d), typ).observations
        else:
            print(f"  (no {d})")
    cfg = {"observations": {"time_window": {"enabled": True,
                                            "half_width_minutes": args.half_width_minutes,
                                            "thinning": "nearest"}}}
    obs = apply_time_window(obs, t_valid, cfg)
    hspec = parse_spec(args.holdout_stations)
    if args.subset != "all" and hspec is None:
        raise SystemExit("--subset needs --holdout-stations k/K")
    orog = np.asarray(first["OROG_raw"], np.float32) if "OROG_raw" in first.files else None

    cfg_y = {}
    if args.config:
        import yaml
        cfg_y = yaml.safe_load(open(args.config)) or {}
    names = channel_names(first)
    stats = ChannelStats(cfg_y, names)
    rot = args.wind_rotation or str(cfg_y.get("observations", {}).get("wind_rotation", "earth"))
    if rot == "auto":
        rot = "earth"
        print("  (wind_rotation auto -> earth in verify_state; pass --wind-rotation from the analysis log)")
    chan = {"T2M": (T2M_CHANNEL, T2M_STD), "D2M": (D2M_CHANNEL, D2M_STD), "PS": (PRES_CHANNEL, PRES_STD)}
    wind = None
    for short, key in (("U10", "U10"), ("V10", "V10"), ("TMP850", "TMP_850")):
        try:
            nm = resolve_name(names, key)
        except KeyError:
            continue
        try:
            chan[short] = (names.index(nm), stats.get(nm)[1])
        except KeyError:
            chan[short] = (names.index(nm), np.nan)       # truth in sigma units, no obs operator
    if "U10" in chan and "V10" in chan and np.isfinite(chan["U10"][1]) and np.isfinite(chan["V10"][1]):
        nu, nv = resolve_name(names, "U10"), resolve_name(names, "V10")
        kw = {}
        if rot == "grid":
            c, s_ = grid_rotation(first["lats"], first["lons"])
            kw = dict(cos_t=c, sin_t=s_)
        wind = {comp: make_wind_operator(comp, chan["U10"][0], chan["V10"][0],
                                         stats.get(nu), stats.get(nv), **kw) for comp in "uv"}

    lapse = lapse_settings(cfg_y)
    groups = {}
    for var, short in VARS.items():
        lst = [o for o in obs if o.variable == var]
        if not lst:
            continue
        iy, ix = build_station_indices(SurfaceObservationSet(lst), first_path)
        iy, ix = tf.constant(iy, tf.int32), tf.constant(ix, tf.int32)
        y = np.array([o.value for o in lst], np.float32)
        if short in ("U10", "V10"):
            if wind is None:
                continue
            op = wind["u" if short == "U10" else "v"]
        elif short == "PS":
            if orog is None:
                continue
            dz, _ = station_height_offsets(lst, iy.numpy(), ix.numpy(), orog)
            op = make_station_pressure_operator(dz)
        else:
            op = t2m_observation_operator if short == "T2M" else d2m_observation_operator
            key = short.lower()
            if key in lapse and orog is not None:
                dz, _ = station_height_offsets(lst, iy.numpy(), ix.numpy(), orog)
                op = make_lapse_operator(op, dz, lapse[key])
        groups[short] = (op, iy, ix, y, lst, var)

    def regroup(short, lst2, y2):
        """rebuild indices / height-corrected operator for a subset of a group's obs"""
        if not lst2:
            groups.pop(short, None)
            return
        op0, _, _, _, _, var_ = groups[short]
        iy2, ix2 = build_station_indices(SurfaceObservationSet(lst2), first_path)
        iy2, ix2 = tf.constant(iy2, tf.int32), tf.constant(ix2, tf.int32)
        op2 = op0
        if short == "PS":
            dz, _ = station_height_offsets(lst2, iy2.numpy(), ix2.numpy(), orog)
            op2 = make_station_pressure_operator(dz)
        elif short.lower() in lapse and orog is not None:
            dz, _ = station_height_offsets(lst2, iy2.numpy(), ix2.numpy(), orog)
            op2 = make_lapse_operator(t2m_observation_operator if short == "T2M"
                                      else d2m_observation_operator, dz, lapse[short.lower()])
        groups[short] = (op2, iy2, ix2, np.asarray(y2, np.float32), lst2, var_)

    # ---- QC on identical obs for every state (decided from the reference state) --------
    if args.qc and groups:
        ref_label = "bg" if "bg" in states else next(iter(states))
        xr_ = tf.convert_to_tensor(np.asarray(states[ref_label][1]["model_input"][..., :138], np.float32))
        ls = cfg_y.get("ls3dvar", {}) or {}
        thr = ((ls.get("observation_filter") or {}).get("thresholds") or {})
        errs = ls.get("observation_errors") or {}
        oqc = ObsQC(cfg_y)
        print(f"  QC from state '{ref_label}': thresholds + obs_qc={'on' if oqc.enabled else 'off'}")
        for short, (op, iy, ix, y, lst, var) in list(groups.items()):
            d = op(xr_, iy, ix).numpy() - y
            keep = np.ones(len(y), bool)
            if var in thr:
                keep &= np.abs(d) <= float(thr[var])
            if oqc.enabled:
                kept_ids = {id(o) for o in oqc.apply(var, [o for o, k in zip(lst, keep) if k],
                                                     d[keep], float(errs.get(var, 1.0)))}
                keep &= np.array([id(o) in kept_ids for o in lst])
            if not keep.all():
                regroup(short, [o for o, k in zip(lst, keep) if k], y[keep])
            print(f"    {short:4s} {len(y)} -> {int(keep.sum())}")

    truth = None
    if args.truth_npz:
        tz = np.load(args.truth_npz, allow_pickle=True)
        if valid_time(tz) != t_valid:
            print(f"WARNING truth valid {valid_time(tz)} != {t_valid}")
        truth = np.asarray(tz["model_input"][0], np.float32)
        land = (np.asarray(first["LAND_raw"]) > 0.5) if "LAND_raw" in first.files \
            else np.ones(truth.shape[:2], bool)

    # ---- station subset (after QC, so QC used every station as a buddy) --------------
    if args.subset != "all":
        want = args.subset == "heldout"
        for short in list(groups):
            _, _, _, y_, lst_, _ = groups[short]
            m = np.array([is_heldout(o.station, hspec) == want for o in lst_])
            if not m.all():
                regroup(short, [o for o, k in zip(lst_, m) if k], y_[m])
        meta["subset"] = args.subset
        print(f"  subset {args.subset} ({args.holdout_stations}): "
              + " ".join(f"{k}={len(v[3])}" for k, v in groups.items()))

    # ---- table ------------------------------------------------------------
    print()
    print("=" * 96)
    print(f"VERIFICATION  valid {cyc}  " + "  ".join(f"{k}={v}" for k, v in meta.items() if k != "valid"))
    print("=" * 96)
    hdr = f"{'state':8s} {'var':6s} {'n':>4s} {'O-F bias':>10s} {'O-F rms':>10s}"
    if truth is not None:
        hdr += f" {'truth all':>10s} {'land':>8s} {'sea':>8s}"
    print(hdr)

    rows = []
    for label, (path, d) in states.items():
        x = np.asarray(d["model_input"][..., :truth.shape[-1] if truth is not None else 138],
                       np.float32)
        x_tf = tf.convert_to_tensor(x)
        for short in SHORTS:
            if short not in chan:
                continue
            row = {**meta, "state": label, "var": short, "n": 0, "bias": np.nan, "rms": np.nan,
                   "truth_all": np.nan, "truth_land": np.nan, "truth_sea": np.nan, "path": path}
            line = f"{label:8s} {short:6s}"
            if short in groups:
                op, iy, ix, y = groups[short][:4]
                hx = op(x_tf, iy, ix).numpy()
                dif = hx - y
                row.update(n=len(y), bias=float(dif.mean()), rms=rms(dif))
                line += f" {len(y):4d} {dif.mean():+10.3f} {rms(dif):10.3f}"
            else:
                line += f" {'-':>4s} {'-':>10s} {'-':>10s}"
            if truth is not None:
                ch, sd = chan[short]
                e = (x[0, ..., ch] - truth[..., ch]) * (sd if np.isfinite(sd) else 1.0)
                ta, tl, ts = rms(e), rms(e[land]), rms(e[~land])
                row.update(truth_all=ta, truth_land=tl, truth_sea=ts)
                line += f" {ta:10.3f} {tl:8.3f} {ts:8.3f}"
            print(line)
            rows.append(row)

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        new = not p.exists()
        if not new:
            with open(p) as f:
                keys = next(csv.reader(f))
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nappended {len(rows)} rows -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
