#!/usr/bin/env python3
"""
ls3dvar/cli/run_forecasts.py

HRRRCast forecasts of a given length for every init time in a range.

    python -m ls3dvar.cli.run_forecasts \\
        --start 2026-08-10T23 --end 2026-08-11T03 --length 3 \\
        --reference-dir $DATA/reference/hrrr_ana \\
        --out $DATA/forecasts/ref_f03

    # from the analyses of an experiment instead of the reference HRRR analyses
    python -m ls3dvar.cli.run_forecasts ... --init-from $DATA/experiments/frozen_v1_pkg

Inputs per init time T (cycle folder C = YYYYMMDDTHHZ, tag = YYYYMMDD_HH):
    state   --init-from reference (default): <reference-dir>/C/hrrr_<tag>.npz
            --init-from EXPERIMENT_DIR:      EXPERIMENT_DIR/C/analysis.npz
    GFS     <reference-dir>/C/gfs_<tag>.npz   (always: forcing at the init time)

Output:
    <out>/C/fcst_f01.npz ... fcst_f<length>.npz     one model load per init time
    <out>/C/forecast.log
    <out>/forecast_manifest.yaml                    settings + code version

The layout is what run_cycles.py reads with `background: {source: forecast_dir}`: the
background for cycle H is <out>/<H-lead>/fcst_f<lead>.npz. With --init-from reference and
--length >= 1, this is the same F01 cold-start background that cycle_ls3dvar.sh computes
(same tool, same inputs; bit-identical with --deterministic).

--length is limited to 6 h: make_forecast_background reproduces fcst.py only up to the
first 6-hourly state refresh. Existing outputs are kept (resume); a manifest written with
different settings is refused unless --force.
"""
import argparse
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

from ls3dvar.workflow.core import (MODULES, REPO_ROOT, StepFailed, Steps, code_version, cycstr,
                                   fmt_seconds, git_dirty, hours, load_yaml, parse_time, save_yaml,
                                   ymd_hh)

WATCH = re.compile(r"^(HRRRCast F\d+\s+valid|FAIL|WARNING|Traceback|\w*Error\b)")

MAX_LENGTH = 6
SETTINGS_KEYS = ("init_from", "reference_dir", "length", "member", "noise_rho", "sampler",
                 "deterministic", "code")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first init time, YYYY-MM-DDTHH (UTC)")
    ap.add_argument("--end", required=True, help="last init time")
    ap.add_argument("--every", type=int, default=1, help="hours between init times (default 1)")
    ap.add_argument("--length", type=int, default=3, help=f"forecast length in hours, 1..{MAX_LENGTH} (default 3)")
    ap.add_argument("--reference-dir", required=True, help="<dir>/YYYYMMDDTHHZ/{hrrr,gfs}_YYYYMMDD_HH.npz")
    ap.add_argument("--init-from", default="reference",
                    help="'reference' (HRRR analyses in --reference-dir) or an experiment folder "
                         "(uses its <cycle>/analysis.npz)")
    ap.add_argument("--out", required=True, help="output root: <out>/<init>/fcst_fLL.npz")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--noise-rho", type=float, default=0.9)
    ap.add_argument("--sampler", choices=["dpmpp-2m", "ddim"], default="dpmpp-2m")
    ap.add_argument("--code", choices=sorted(MODULES), default="ls3dvar")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=False,
                    help="bit-reproducible TF (default: plain TF, as run_cycles)")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--timeout", type=int, default=900, help="[s] per init time; then killed and retried")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--python", default=None, help="python for the steps (default: this one)")
    ap.add_argument("--redo", action="store_true", help="recompute even if the outputs exist")
    ap.add_argument("--force", action="store_true", help="accept a manifest with other settings")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    a = ap.parse_args(argv)

    if not 1 <= a.length <= MAX_LENGTH:
        raise SystemExit(f"--length must be 1..{MAX_LENGTH} h (make_forecast_background reproduces "
                         f"fcst.py only up to the first 6-hourly state refresh)")
    start, end = parse_time(a.start), parse_time(a.end)
    if end < start:
        raise SystemExit("--end is before --start")
    ref = Path(a.reference_dir)
    init_exp = None if a.init_from == "reference" else Path(a.init_from)
    if init_exp is not None and not init_exp.is_dir():
        raise SystemExit(f"--init-from: not 'reference' and not a folder: {a.init_from}")
    out = Path(a.out)
    leads = list(range(1, a.length + 1))

    settings = dict(init_from=str(init_exp or "reference"), reference_dir=str(ref), length=a.length,
                    member=a.member, noise_rho=a.noise_rho, sampler=a.sampler,
                    deterministic=bool(a.deterministic), code=a.code)
    man_path = out / "forecast_manifest.yaml"
    if man_path.exists():
        old = load_yaml(man_path).get("settings", {})
        diff = {k: (old.get(k), settings[k]) for k in SETTINGS_KEYS if old.get(k) != settings[k]}
        if diff and not a.force:
            raise SystemExit(f"{man_path} was written with other settings {diff}; "
                             f"use another --out, or --force")

    # ------------------------------------------------------------------ plan
    plan, missing = [], []
    for t in hours(start, end, a.every):
        c, tag = cycstr(t), ymd_hh(t)
        state = (ref / c / f"hrrr_{tag}.npz") if init_exp is None else (init_exp / c / "analysis.npz")
        gfs = ref / c / f"gfs_{tag}.npz"
        targets = [out / c / f"fcst_f{L:02d}.npz" for L in leads]
        if not (state.exists() and gfs.exists()):
            missing.append((c, [str(p) for p in (state, gfs) if not p.exists()]))
            continue
        done = all(p.exists() and p.stat().st_size > 0 for p in targets)
        plan.append((t, c, state, gfs, targets, done and not a.redo))

    print(f"HRRRCast forecasts  init {cycstr(start)} .. {cycstr(end)} every {a.every} h   "
          f"length {a.length} h   from {settings['init_from']}")
    print(f"  out  {out}")
    print(f"  code {code_version(a.code, REPO_ROOT, a.deterministic)}"
          f"{'  (uncommitted changes)' if git_dirty(REPO_ROOT) else ''}")
    for c, what in missing:
        print(f"  !! {c}: missing {', '.join(what)}")
    todo = [p for p in plan if not p[5]]
    print(f"  {len(plan)} init times, {len(plan) - len(todo)} already done, {len(todo)} to run"
          + (f", {len(missing)} skipped (missing inputs)" if missing else ""))
    if a.dry_run or not todo:
        return 0 if not missing else 1

    out.mkdir(parents=True, exist_ok=True)
    save_yaml({"settings": settings,
               "code_version": code_version(a.code, REPO_ROOT, a.deterministic),
               "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "layout": "<out>/<init YYYYMMDDTHHZ>/fcst_f<LL>.npz (valid = init + LL h)"}, man_path)

    steps = Steps(REPO_ROOT, python=a.python, deterministic=a.deterministic, timeout=a.timeout,
                  retries=a.retries)
    module = MODULES[a.code]["forecast"]
    failed = []
    for t, c, state, gfs, targets, _ in todo:
        d = out / c
        d.mkdir(parents=True, exist_ok=True)
        valid = ", ".join(cycstr(t + timedelta(hours=L)) for L in leads)
        print(f"{time.strftime('%H:%M:%S')} [fc] init {c} -> valid {valid}")
        try:
            sec = steps.run(f"forecast {c}", module,
                            ["--hrrr-npz", state, "--gfs-npz", gfs, "--lead", *leads,
                             "--out", d / "fcst_f{lead:02d}.npz", "--member", a.member,
                             "--noise-rho", a.noise_rho, "--sampler", a.sampler],
                            d / "forecast.log", watch=WATCH)
            print(f"  done in {fmt_seconds(sec)}", flush=True)
        except StepFailed as e:
            print(f"  !! {e}")
            failed.append(c)
    if failed:
        print(f"FAILED init times: {' '.join(failed)}")
        return 1
    print("done")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
