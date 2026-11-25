#!/usr/bin/env python3
"""
ls3dvar/tools/gate.py

Safety gate for the incremental LS-3DVar cleanup. Before any change, it records
a BASELINE of
    1. mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps   (driver, fixed explicit args)
    2. mantari.workflows.test_decode_parity                   (exit code + PASS/FAIL lines)
    3. mantari.workflows.test_encode_decode_roundtrip         (exit code + summary numbers)
After every change, it reruns the same commands and compares the results with
the baseline.

    # once, on the untouched code
    python -m ls3dvar.tools.gate snapshot --name base --driver-args ""      # settings from the config
    python -m ls3dvar.tools.gate snapshot --name base \
        --driver-args "--control blocks --block REFC:15:2 \
                       --block T2M:1.0:40:land --holdout-stations 0/5 --outer 2 --maxiter 40"
    # determinism check: rerun and compare with itself (no code change in between)
    python -m ls3dvar.tools.gate compare --baseline base
    # after each cleanup step
    python -m ls3dvar.tools.gate compare --baseline base        # exact
    # the stand-alone ls3dvar package against the baseline of the old code
    python -m ls3dvar.tools.gate compare --baseline base --use ls3dvar
    # two cycling experiments (PKG=mantari vs PKG=ls3dvar), every npz and csv
    python -m ls3dvar.tools.gate diffdirs EXP_OLD EXP_NEW
    python -m ls3dvar.tools.gate compare --baseline base --atol 1e-6 --rtol 1e-5

Nothing in the repository is modified:
  * the driver gets a COPY of your config (default $MANTARI_CONFIG or
    mantari/config/test.yaml) with only outputs.directory redirected into
    gate/<run>/driver_out/. The copy is frozen at snapshot time, and compare reuses it,
    so later edits of test.yaml do not change what is compared.
  * a snapshot with a failed step (non-zero exit, or a driver that wrote no npz) is
    marked INVALID, and compare refuses it
  * the round-trip test writes its npz into gate/<run>/ (--out)
  * the parity tests read TEST_CONFIG as they always do
Results live under gate/<name>/ (git-ignore it): logs, a copy of the driver
outputs, and manifest.json (the commands, the git commit, and a fingerprint of
every array and CSV).

Comparison rules
  * exit codes must match
  * every .npz in the driver output: same keys, shapes and dtypes; values
    bit-identical (default) or allclose(atol, rtol) when tolerances are given
  * every .csv: same columns and rows; numeric columns compared like arrays,
    others compared exactly. *_history_* CSVs (iteration logs) are compared too:
    a different iteration count is reported, because it means the minimization
    path changed.
  * parity logs: the same PASS/FAIL verdict per check. The numbers on those lines
    (max_abs, rmse, ...) are compared with the same tolerances.
  * files present in only one run are reported (new files, e.g. plots, are OK
    and are only listed; a missing file is a FAIL)

TF on GPU is not bit-reproducible by default: the forward pass already varies (~5e-5)
and L-BFGS amplifies it (0.3 sigma in x_a after 10 iterations, measured). Snapshots
therefore run DETERMINISTIC by default: TF_DETERMINISTIC_OPS=1, and every module runs
through a small runner that replaces exact-integer nearest-neighbour resize (the
decoder's up-sampling) by tf.repeat (forward checked bit-identical; its gradient is
deterministic, ResizeNearestNeighborGrad is not). No repository file is changed.
--no-deterministic runs plain TF; then use the run-to-run noise as the tolerance.
Parity-test logs are compared by PASS/FAIL per check (--strict-logs: numbers too).
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PRESETS = {
    # old code (mantari) and the stand-alone extraction (ls3dvar): same CLI, same outputs
    "mantari": {"driver": "mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps",
                "parity": "mantari.workflows.test_decode_parity",
                "roundtrip": "mantari.workflows.test_encode_decode_roundtrip"},
    "ls3dvar": {"driver": "ls3dvar.cli.run_analysis",
                "parity": "ls3dvar.tests.test_decode_parity",
                "roundtrip": "ls3dvar.tests.test_encode_decode_roundtrip"},
}
NUM = re.compile(r"([A-Za-z_][\w\[\]()|-]*)\s*[=:]\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")
PARITY_LINE = re.compile(r"^(PASS|FAIL)\s+(\S+)")


DET_RUNNER = r'''"""
gate deterministic runner (written by ls3dvar/tools/gate.py; changes no repository file)

    python _det_runner.py MODULE [args...]      ==   python -m MODULE [args...]

TF has no deterministic GPU kernel for the gradient of nearest-neighbour resize, which the
LS-3DVar decoder uses. For EXACT integer up-sampling (the decoder: 34->68->136->272),
nearest resize is a pure copy, so tf.repeat gives a bit-identical forward result and a
deterministic gradient. This runner swaps it in, checks the swap against the original
resize once per shape (stops on any mismatch), then runs MODULE as __main__.
Other resizes are passed through unchanged.
"""
import os
import runpy
import sys

sys.path[0] = os.getcwd()                         # like python -m: repo root first
mod = sys.argv[1]
sys.argv = [mod] + sys.argv[2:]

import tensorflow as tf  # noqa: E402

_orig = tf.image.resize
_checked = set()


def _resize(images, size, method="bilinear", preserve_aspect_ratio=False, antialias=False, name=None):
    kw = dict(method=method, preserve_aspect_ratio=preserve_aspect_ratio, antialias=antialias, name=name)
    if str(method).lower() == "nearest" and not preserve_aspect_ratio and images.shape.rank == 4:
        h, w = images.shape[1], images.shape[2]
        try:
            H, W = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            H = W = None
        if h and w and H and W and H % h == 0 and W % w == 0:
            out = tf.repeat(tf.repeat(images, H // h, axis=1), W // w, axis=2)
            key = (tuple(images.shape), H, W, str(images.dtype))
            if key not in _checked:
                ref = _orig(images, size, **kw)
                if ref.dtype != out.dtype or ref.shape != out.shape or not bool(tf.reduce_all(tf.equal(ref, out))):
                    raise RuntimeError(f"gate det runner: repeat != nearest resize for {key}")
                _checked.add(key)
                print(f"[gate det] nearest {h}x{w} -> {H}x{W}: tf.repeat, forward bit-identical", file=sys.stderr)
            return out
    return _orig(images, size, **kw)


tf.image.resize = _resize
runpy.run_module(mod, run_name="__main__", alter_sys=True)
'''


# ---------------------------------------------------------------------------- helpers
def sh(cmd, cwd, log, env):
    t0 = time.time()
    with open(log, "w") as f:
        f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    shown = [c for c in cmd[1:] if not c.endswith("_det_runner.py") and c != "-m"][:2]
    print(f"  exit {p.returncode}  {time.time() - t0:7.1f}s  {' '.join(shown)}  -> {log}")
    return p.returncode


def git_info(root):
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()[:16]


def fingerprint_npz(p):
    out = {}
    with np.load(p, allow_pickle=True) as z:
        for k in sorted(z.files):
            a = z[k]
            d = {"shape": list(a.shape), "dtype": str(a.dtype)}
            if a.dtype.kind in "fiucb":
                d["sha"] = sha(a)
                if a.size and a.dtype.kind in "fiu":
                    af = a.astype(np.float64)
                    d["mean"] = float(np.nanmean(af))
                    d["absmax"] = float(np.nanmax(np.abs(af)))
            else:
                d["repr"] = str(a)[:200]
            out[k] = d
    return out


def parse_log(p):
    """parity / roundtrip lines worth comparing: PASS/FAIL checks and summary numbers"""
    checks, numbers = {}, {}
    if not Path(p).exists():
        return checks, numbers
    lines = [l.strip() for l in Path(p).read_text(errors="replace").splitlines()]
    rule = lambda t: bool(t) and set(t) == {"="}
    section = ""
    for i, s in enumerate(lines):
        if rule(s):
            continue
        if 0 < i < len(lines) - 1 and rule(lines[i - 1]) and rule(lines[i + 1]):
            section = s                            # banner title: ===== / TITLE / =====
            continue
        m = PARITY_LINE.match(s)
        if m:
            checks[m.group(2)] = m.group(1)
            for k, v in NUM.findall(s):
                numbers[f"{m.group(2)}.{k}"] = float(v)
            continue
        if section.startswith(("PATH SUMMARY", "ROUND TRIP")) or s.startswith(("ALL ", "failing channels")):
            for k, v in NUM.findall(s):
                numbers[f"{section[:20]}|{s.split()[0]}.{k}"] = float(v)
    return checks, numbers


def read_csv(p):
    import pandas as pd
    return pd.read_csv(p)


# ---------------------------------------------------------------------------- run
def run_all(a, run_dir):
    root = Path(a.root).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="0")
    if a.deterministic:
        env.update(TF_DETERMINISTIC_OPS="1", TF_CUDNN_DETERMINISTIC="1")
    env.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")      # PTX-JIT cache, as in cycle_ls3dvar.sh
    py = a.python or sys.executable

    import yaml
    cfg_src = Path(a.config or os.environ.get("MANTARI_CONFIG") or root / "mantari/config/test.yaml")
    cfg = yaml.safe_load(open(cfg_src))
    drv_out = run_dir / "driver_out"
    if drv_out.exists():
        shutil.rmtree(drv_out)
    cfg.setdefault("outputs", {})["directory"] = str(drv_out)
    gate_cfg = run_dir / "config.yaml"
    yaml.safe_dump(cfg, open(gate_cfg, "w"), sort_keys=False)

    mods = PRESETS[a.use]
    if a.deterministic:
        runner = run_dir / "_det_runner.py"
        runner.write_text(DET_RUNNER)
        launch = [py, str(runner)]
    else:
        launch = [py, "-m"]
    cmds = {
        "driver": [*launch, mods["driver"], "--config", str(gate_cfg), *shlex.split(a.driver_args)],
        "parity": [*launch, mods["parity"]],
        "roundtrip": [*launch, mods["roundtrip"], "--anchor", "--out", str(run_dir / "roundtrip.npz")],
    }
    skip = set(a.skip or [])
    codes = {}
    for name, cmd in cmds.items():
        if name in skip:
            continue
        codes[name] = sh(cmd, root, run_dir / f"{name}.log", env)

    files = {}
    if drv_out.exists():
        for p in sorted(drv_out.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(drv_out))
                if p.suffix == ".npz":
                    files[rel] = {"kind": "npz", "arrays": fingerprint_npz(p)}
                elif p.suffix == ".csv":
                    files[rel] = {"kind": "csv", "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
                else:
                    files[rel] = {"kind": p.suffix.lstrip(".") or "file"}
    logs = {}
    for name in ("parity", "roundtrip"):
        c, n = parse_log(run_dir / f"{name}.log")
        logs[name] = {"checks": c, "numbers": n}

    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), git=git_info(root), use=a.use,
                    deterministic=bool(a.deterministic),
                    config_source=str(getattr(a, "config_origin", None) or cfg_src),
                    config_frozen=str(gate_cfg), driver_args=a.driver_args,
                    commands={k: v for k, v in cmds.items() if k not in skip},
                    exit_codes=codes, files=files, logs=logs)
    json.dump(manifest, open(run_dir / "manifest.json", "w"), indent=1)
    return manifest


def problems(m):
    """reasons a run cannot serve as a baseline (empty list = usable)"""
    out = [f"{k} exited {v}" for k, v in m["exit_codes"].items() if v != 0]
    if "driver" in m["exit_codes"] and not any(f.endswith(".npz") for f in m["files"]):
        out.append("driver wrote no npz files")
    return out


def check_driver_args(s):
    if re.search(r"[<>]", s or ""):
        raise SystemExit(f"--driver-args looks like a placeholder: {s!r}\n"
                         "give real flags, e.g. \"--control blocks --block REFC:15:2 --outer 2 --maxiter 40\", "
                         "or \"\" to use the settings in the (frozen) config")


# ---------------------------------------------------------------------------- compare
class Report:
    def __init__(self):
        self.fail, self.info = [], []

    def bad(self, msg):
        self.fail.append(msg)
        print(f"  FAIL  {msg}")

    def note(self, msg):
        self.info.append(msg)
        print(f"  info  {msg}")


def _maxdiff(x, y):
    if not x.size:
        return 0.0
    with np.errstate(invalid="ignore"):
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else 0.0


def close(x, y, atol, rtol):
    if atol == 0 and rtol == 0:
        return bool(np.array_equal(x, y, equal_nan=True)), _maxdiff(x, y)
    return bool(np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)), _maxdiff(x, y)


COMPARED = {".npz", ".csv"}           # other files (logs, yaml, png, txt) are only listed


def compare_dirs(base_dir, new_dir, mb, mn, a, R):
    compare_trees(base_dir / "driver_out", new_dir / "driver_out", a, R)


def compare_trees(bo, no, a, R, ignore_cols=()):
    """compare every .npz / .csv under bo and no (same relative paths)"""
    def listing(root):
        return {str(p.relative_to(root)): p for p in sorted(Path(root).rglob("*")) if p.is_file()} \
            if Path(root).exists() else {}
    fb, fn = listing(bo), listing(no)
    for rel in sorted(set(fb) - set(fn)):
        (R.bad if Path(rel).suffix in COMPARED else R.note)(f"missing output {rel}")
    for rel in sorted(set(fn) - set(fb)):
        R.note(f"new output {rel}")
    n_cmp = 0
    for rel in sorted(set(fb) & set(fn)):
        suf = Path(rel).suffix
        if suf == ".npz":
            n_cmp += 1
            with np.load(bo / rel, allow_pickle=True) as zb, np.load(no / rel, allow_pickle=True) as zn:
                for k in sorted(set(zb.files) - set(zn.files)):
                    R.bad(f"{rel}: key {k} missing")
                for k in sorted(set(zn.files) - set(zb.files)):
                    R.note(f"{rel}: new key {k}")
                for k in sorted(set(zb.files) & set(zn.files)):
                    x, y = zb[k], zn[k]
                    if x.shape != y.shape or x.dtype != y.dtype:
                        R.bad(f"{rel}:{k} shape/dtype {x.shape} {x.dtype} -> {y.shape} {y.dtype}")
                    elif x.dtype.kind in "fiucb":
                        ok, dmax = close(x, y, a.atol, a.rtol)
                        if not ok:
                            R.bad(f"{rel}:{k} differs  max|diff|={dmax:.3e}")
                        elif dmax:
                            R.note(f"{rel}:{k} within tolerance  max|diff|={dmax:.3e}")
                    elif str(x) != str(y) and k not in a.ignore_keys:
                        R.bad(f"{rel}:{k} value changed: {str(x)[:60]!r} -> {str(y)[:60]!r}")
        elif suf == ".csv":
            n_cmp += 1
            if (bo / rel).read_bytes() == (no / rel).read_bytes():
                continue
            db, dn = read_csv(bo / rel), read_csv(no / rel)
            tag = "history " if "history" in rel else ""
            if list(db.columns) != list(dn.columns):
                miss = [c for c in db.columns if c not in dn.columns]
                add = [c for c in dn.columns if c not in db.columns]
                (R.bad if miss else R.note)(f"{rel}: columns -{miss} +{add}")
            if len(db) != len(dn):
                R.bad(f"{tag}{rel}: rows {len(db)} -> {len(dn)}")
                continue
            for c in [c for c in db.columns if c in dn.columns and c not in ignore_cols]:
                x, y = db[c].to_numpy(), dn[c].to_numpy()
                if db[c].dtype.kind in "fiub" and dn[c].dtype.kind in "fiub":
                    ok, dmax = close(x.astype(np.float64), y.astype(np.float64), a.atol, a.rtol)
                    if not ok:
                        R.bad(f"{rel}:{c} differs  max|diff|={dmax:.3e}")
                elif not (db[c].astype(str) == dn[c].astype(str)).all():
                    R.bad(f"{rel}:{c} text values changed")
    return n_cmp


def compare_logs(mb, mn, a, R):
    for name in ("parity", "roundtrip"):
        lb, ln = mb["logs"].get(name, {}), mn["logs"].get(name, {})
        cb, cn = lb.get("checks", {}), ln.get("checks", {})
        for k in sorted(set(cb) | set(cn)):
            if cb.get(k) != cn.get(k):
                R.bad(f"{name}: check {k} {cb.get(k)} -> {cn.get(k)}")
        nb, nn = lb.get("numbers", {}), ln.get("numbers", {})
        if not a.strict_logs:
            continue                    # the tests judge their own numbers (PASS/FAIL above)
        for k in sorted(set(nb) & set(nn)):
            x, y = nb[k], nn[k]
            tol = max(a.atol, a.rtol * abs(x)) if (a.atol or a.rtol) else 0.0
            if abs(x - y) > tol:
                R.bad(f"{name}: {k} {x:.6g} -> {y:.6g}")
        for k in sorted(set(nb) - set(nn)):
            R.note(f"{name}: number no longer printed: {k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("snapshot", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".", help="repository root (where `python -m mantari...` works)")
        p.add_argument("--gate-dir", default="gate")
        p.add_argument("--python", default=None)
        p.add_argument("--config", default=None, help="config to copy (default $MANTARI_CONFIG or test.yaml)")
        p.add_argument("--skip", nargs="*", choices=["driver", "parity", "roundtrip"])
        p.add_argument("--deterministic", dest="deterministic", action="store_true", default=None,
                       help="deterministic TF (default for snapshot): TF_DETERMINISTIC_OPS=1 + exact-2x "
                            "nearest resize via tf.repeat (bit-identical forward). compare follows the baseline")
        p.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                       help="plain TF (runs are NOT reproducible on GPU)")
        p.add_argument("--use", choices=sorted(PRESETS), default=None,
                       help="which code to run: mantari (old, default for snapshot) or ls3dvar "
                            "(stand-alone package); compare defaults to the baseline's choice")
    s = sub.choices["snapshot"]
    s.add_argument("--name", default="base")
    s.add_argument("--driver-args", required=True,
                   help="EXPLICIT driver settings, so config defaults cannot drift")
    c = sub.choices["compare"]
    c.add_argument("--baseline", default="base")
    c.add_argument("--name", default=None, help="run name (default cmp_<time>)")
    c.add_argument("--atol", type=float, default=0.0)
    c.add_argument("--rtol", type=float, default=0.0)
    c.add_argument("--ignore-keys", nargs="*", default=["ls3dvar_meta"],
                   help="npz string keys allowed to change (e.g. new run metadata)")
    c.add_argument("--reuse", action="store_true", help="compare an existing run --name, do not rerun")
    c.add_argument("--strict-logs", action="store_true",
                   help="also compare the numbers printed by the parity tests (default: only PASS/FAIL)")
    c.add_argument("--allow-invalid-baseline", action="store_true",
                   help="compare even if the baseline had a failed step (debugging only)")
    d = sub.add_parser("diffdirs", help="compare two experiment / output directories (e.g. two cycling runs)")
    d.add_argument("old")
    d.add_argument("new")
    d.add_argument("--atol", type=float, default=0.0)
    d.add_argument("--rtol", type=float, default=0.0)
    d.add_argument("--ignore-keys", nargs="*", default=["ls3dvar_meta"])
    d.add_argument("--ignore-cols", nargs="*", default=["path"],
                   help="CSV columns that hold run-specific paths (verify_state writes 'path')")
    argv = sys.argv[1:]
    for i in range(len(argv) - 1):            # allow --driver-args "--outer 1" and "--outer"
        if argv[i] == "--driver-args":
            argv[i:i + 2] = [f"--driver-args={argv[i + 1]}"]
            break
    a = ap.parse_args(argv)

    if a.cmd == "diffdirs":
        print(f"DIFFDIRS {a.old} -> {a.new}  atol={a.atol:g} rtol={a.rtol:g}")
        R = Report()
        n = compare_trees(Path(a.old), Path(a.new), a, R, ignore_cols=set(a.ignore_cols))
        print(f"\n{n} npz/csv files compared")
        print("DIFFDIRS PASS" if not R.fail else f"DIFFDIRS FAIL ({len(R.fail)} differences)")
        return 0 if not R.fail else 1

    gate = Path(a.root).resolve() / a.gate_dir
    if a.cmd == "snapshot":
        check_driver_args(a.driver_args)
        a.use = a.use or "mantari"
        if a.deterministic is None:
            a.deterministic = True
        d = gate / a.name
        if (d / "manifest.json").exists():
            raise SystemExit(f"{d} exists; pick another --name or delete it")
        print(f"SNAPSHOT -> {d}")
        m = run_all(a, d)
        print(f"\nexit codes {m['exit_codes']}   driver files {len(m['files'])}")
        print(f"git {m['git']}")
        if m["git"].get("dirty"):
            print("WARNING working tree has uncommitted changes: commit first so the baseline is reproducible")
        bad = problems(m)
        if bad:
            print(f"\nSNAPSHOT INVALID: {'; '.join(bad)}  -> see {d}/*.log")
            print(f"compare will refuse this baseline; fix the cause, then: rm -rf {d} and snapshot again")
            return 1
        print("\nSNAPSHOT OK")
        return 0

    bdir = gate / a.baseline
    mb = json.load(open(bdir / "manifest.json"))
    bad = problems(mb)
    if bad and not a.allow_invalid_baseline:
        raise SystemExit(f"baseline '{a.baseline}' is not usable: {'; '.join(bad)}\n"
                         f"rm -rf {bdir} and take a new snapshot (see {bdir}/*.log)")
    name = a.name or time.strftime("cmp_%Y%m%d_%H%M%S")
    ndir = gate / name
    if a.reuse:
        mn = json.load(open(ndir / "manifest.json"))
    else:
        print(f"RUN {name}  (driver args from baseline: {mb['driver_args']})")
        a.driver_args = mb["driver_args"]
        a.use = a.use or mb.get("use", "mantari")
        if a.deterministic is None:
            a.deterministic = mb.get("deterministic", False)
        print(f"code: {a.use} (baseline: {mb.get('use', 'mantari')})")
        if a.config is None:                      # the config as it was at snapshot time
            a.config = mb.get("config_frozen") or str(bdir / "config.yaml")
            a.config_origin = mb.get("config_source")
        mn = run_all(a, ndir)

    print(f"\nCOMPARE {a.baseline} ({mb['git'].get('commit', '')[:10]}) -> {name} "
          f"({mn['git'].get('commit', '')[:10]}{' dirty' if mn['git'].get('dirty') else ''})  "
          f"atol={a.atol:g} rtol={a.rtol:g}")
    R = Report()
    for k, v in mn["exit_codes"].items():
        if v != 0:
            R.bad(f"{k} exited {v} in this run (see {ndir}/{k}.log)")
    for k in sorted(set(mb["exit_codes"]) | set(mn["exit_codes"])):
        if mb["exit_codes"].get(k) != mn["exit_codes"].get(k):
            R.bad(f"exit code {k}: {mb['exit_codes'].get(k)} -> {mn['exit_codes'].get(k)}")
    compare_dirs(bdir, ndir, mb, mn, a, R)
    compare_logs(mb, mn, a, R)
    print()
    print("GATE PASS" if not R.fail else f"GATE FAIL ({len(R.fail)} differences)")
    json.dump({"fail": R.fail, "info": R.info}, open(ndir / "compare.json", "w"), indent=1)
    return 0 if not R.fail else 1


if __name__ == "__main__":
    sys.exit(main())
