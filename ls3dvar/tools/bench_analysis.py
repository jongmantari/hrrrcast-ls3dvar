#!/usr/bin/env python3
"""
ls3dvar/tools/bench_analysis.py

Where does the analysis time go? Runs a few L-BFGS evaluations of ONE cycle's analysis
under several variants and reports setup time and seconds per evaluation.

    python -m ls3dvar.tools.bench_analysis \\
        --config  $EXP/20260812T19Z/config.yaml \\
        --truth-npz $REF/20260812T19Z/hrrr_20260812_19.npz \\
        --evals 6

Variants (--variants, comma separated; default: all four)
    det              TF_DETERMINISTIC_OPS=1 + tools/det_runner.py   (what run_cycles uses)
    nondet           plain TF (python -m)
    det-nopseudo     det, with ls3dvar.pseudo_obs.enabled: false
    nondet-nopseudo  nondet, with ls3dvar.pseudo_obs.enabled: false

The cycle's config is COPIED with outputs.directory redirected into the bench folder, so
the experiment is never touched. Each variant runs the analysis with --outer 1
--maxiter <evals>. Lines are timestamped as they arrive (the steps run unbuffered).
Needs the GPU to itself: stop or finish the running experiment first.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ls3dvar.paths import REPO_ROOT

VARIANTS = {
    "det": dict(deterministic=True, pseudo=True),
    "nondet": dict(deterministic=False, pseudo=True),
    "det-nopseudo": dict(deterministic=True, pseudo=False),
    "nondet-nopseudo": dict(deterministic=False, pseudo=False),
}
ITER = re.compile(r"^ITER=(\d+) J=\s*([-\d.eE+]+)")


def run_variant(name, v, a, out):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    cfg.setdefault("outputs", {})["directory"] = str(d / "ls3dvar")
    if not v["pseudo"]:
        cfg.setdefault("ls3dvar", {}).setdefault("pseudo_obs", {})["enabled"] = False
    cfg_path = d / "config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    py = a.python or sys.executable
    if v["deterministic"]:
        env.update(TF_DETERMINISTIC_OPS="1", TF_CUDNN_DETERMINISTIC="1")
        launch = [py, str(REPO_ROOT / "ls3dvar" / "tools" / "det_runner.py")]
    else:
        env.pop("TF_DETERMINISTIC_OPS", None)
        env.pop("TF_CUDNN_DETERMINISTIC", None)
        launch = [py, "-m"]
    cmd = [*launch, "ls3dvar.cli.run_analysis", "--config", str(cfg_path),
           "--outer", "1", "--maxiter", str(a.evals), *a.extra.split()]
    if a.truth_npz:
        cmd += ["--truth-npz", a.truth_npz]

    print(f"\n[{name}] {' '.join(cmd[-8:])}", flush=True)
    t0 = time.monotonic()
    t_iter, J = [], []
    with open(d / "analysis.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            now = time.monotonic() - t0
            log.write(f"{now:9.2f} {line}")
            m = ITER.match(line)
            if m:
                t_iter.append(now)
                J.append(float(m.group(2)))
                print(f"  {now:7.1f}s  ITER={int(m.group(1)):03d}  J={J[-1]:.4f}", flush=True)
        rc = p.wait()
    total = time.monotonic() - t0
    per = [b - a_ for a_, b in zip(t_iter, t_iter[1:])]
    res = dict(variant=name, rc=rc, setup_s=t_iter[0] if t_iter else float("nan"),
               s_per_eval=statistics.median(per) if per else float("nan"),
               first_eval_s=per[0] if per else float("nan"), evals=len(t_iter), total_s=total,
               J_last=J[-1] if J else float("nan"), log=str(d / "analysis.log"))
    if rc != 0:
        print(f"  !! exit {rc}: see {d / 'analysis.log'}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a cycle's config.yaml (e.g. <exp>/<cycle>/config.yaml)")
    ap.add_argument("--truth-npz", default=None, help="reference analysis at the cycle (pseudo-obs need it)")
    ap.add_argument("--evals", type=int, default=6, help="L-BFGS evaluations per variant (default 6)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--extra", default="", help="more analysis flags, e.g. '--control latent'")
    ap.add_argument("--out", default=None, help="bench folder (default <repo>/gate/bench_<time>)")
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)

    names = [n.strip() for n in a.variants.split(",") if n.strip()]
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {list(VARIANTS)}")
    out = Path(a.out or REPO_ROOT / "gate" / time.strftime("bench_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    rows = [run_variant(n, VARIANTS[n], a, out) for n in names]

    print("\n" + "=" * 92)
    print(f"{'variant':16s} {'setup s':>8s} {'1st eval s':>10s} {'s/eval':>8s} {'evals':>6s} "
          f"{'total s':>8s} {'J last':>12s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['variant']:16s} {r['setup_s']:8.1f} {r['first_eval_s']:10.1f} {r['s_per_eval']:8.2f} "
              f"{r['evals']:6d} {r['total_s']:8.1f} {r['J_last']:12.4f}{'   !! exit ' + str(r['rc']) if r['rc'] else ''}")
    print("=" * 92)
    print("setup = start .. first ITER (model load, latent, obs, B); s/eval = median between ITER lines")
    print(f"logs: {out}/<variant>/analysis.log")
    yaml.safe_dump(rows, open(out / "bench.yaml", "w"), sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
