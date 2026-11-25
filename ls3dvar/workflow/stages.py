"""
ls3dvar/workflow/stages.py

Prep and post stages of run_cycles.py: they run the data-directory scripts as they are, for the
hours an experiment needs, and skip what already exists.

    prep  (before the cycles)
      reference   stage_reference.sh START END     HRRR analysis npz (hrrr + gfs) per hour
                                                   (runs run_cycle.sh for hours not staged yet)
      obs         build_obsdb_cycles.py CONFIG --start --end --types --output-root
      mrms        mrms_stage.py --start --end --ref-dir --out     MRMS REFC on the HRRRCast grid
      background  python -m ls3dvar.cli.run_forecasts ...         for background forecast_dir:
                                                                  every cycle (source) or only the
                                                                  first (first_cycle; later cycles
                                                                  cycle from the analysis)
    post  (after the cycles)
      plots        plot_ls3dvar_cycling.py   verification / precip summaries (+ maps if asked)
      fields       plot_ls3dvar_cycling.py   compare panels: HRRR | MRMS | background | analysis
                                             + increments + error reduction, every lead
      diagnostics  ls3dvar_diagnostics.py    with vs without DA: stations (bootstrap), grid rms,
                                             REFC/APCP skill vs MRMS and HRRR, maps, summary.md
      threat       plot_mrms_threat.py       REFC threat scores vs MRMS by lead and threshold

Hours (one cycle every hour, start..end, forecast leads L):
    reference  start-1 .. end+maxL     (cold-start background at start-1, truth at every valid time)
    obs, mrms  start   .. end+maxL     (assimilation, verification at every valid time)
    background init start-lead .. end-lead
The post configs are written to <exp>/post/*.yaml from the templates, with the experiment's
paths, cycles and leads filled in, then `set:` from the experiment YAML on top.
"""
import copy
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from ls3dvar.workflow.core import (REPO_ROOT, StepFailed, cycstr, deep_merge, fmt_seconds, hours,
                                   load_yaml, save_yaml, ymd_hh)

PREP_PARTS = ("reference", "obs", "mrms", "background")
POST_PARTS = ("plots", "fields", "diagnostics", "threat")

PREP_DEFAULTS = {
    "reference": {"enabled": False, "script": "${data}/stage_reference.sh", "run_missing": True,
                  "copy": False, "env": {}},
    "obs": {"enabled": False, "script": "${data}/build_obsdb_cycles.py", "config": "${data}/obsdb_cycles.yaml",
            "python": None},
    "mrms": {"enabled": False, "via": "mrms_stage", "script": "${data}/mrms_stage.py", "dir": "${data}/mrms",
             "args": "", "python": None},
    "background": {"enabled": False},
}
POST_DEFAULTS = {
    "dir": None,                    # default <exp>/figures
    "python": None,                 # python with matplotlib, pandas, scipy, xarray, netCDF4
    "plots": {"enabled": False, "script": "${data}/plot_ls3dvar_cycling.py",
              "template": "${data}/plot_ls3dvar.yaml", "only": ["verification", "precip"],
              "verification_csv": "auto", "set": {}},
    "fields": {"enabled": False, "script": "${data}/plot_ls3dvar_cycling.py",
               "template": "${data}/fine_fields.yaml", "set": {}},
    "diagnostics": {"enabled": False, "script": "${data}/ls3dvar_diagnostics.py",
                    "template": "${data}/diag_fine_da.yaml", "only": None, "set": {}},
    "threat": {"enabled": False, "script": "${data}/plot_mrms_threat.py", "hrrr": False},
}
PUSH_DEFAULTS = {"enabled": False, "dest": None, "what": "all", "tool": "aws", "args": "",
                 "exclude": ["*/driver.pid", "*/progress.tmp", "*/__pycache__/*"]}
FREE_FORM = {("prep", "reference", "env"), ("post", "plots", "set"), ("post", "fields", "set"),
             ("post", "diagnostics", "set")}

# imports each script needs (any one of a group is enough) - checked by `check`
NEEDS = {
    "obs": [("numpy",), ("netCDF4",), ("yaml",)],
    "mrms": [("numpy",), ("scipy",), ("pygrib", "grib2io", "cfgrib")],
    "post": [("numpy",), ("pandas",), ("matplotlib",), ("scipy",), ("xarray",), ("netCDF4",), ("yaml",)],
}


def ranges(times):
    """consecutive hourly runs: [t0, t1, t2, t5] -> [(t0, t2), (t5, t5)]"""
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] == timedelta(hours=1):
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def iso_h(t):
    return f"{t:%Y-%m-%dT%H}"


def script_cmd(script, python):
    s = str(script)
    return ["bash", s] if s.endswith(".sh") else [python, s]


def check_imports(python, groups, timeout=90):
    """[] if `python` can import one module of every group, else the missing groups"""
    code = ("import importlib.util as u, sys\n"
            f"g = {groups!r}\n"
            "bad = [x for x in g if not any(u.find_spec(m) for m in x)]\n"
            "print(repr(bad)); sys.exit(1 if bad else 0)")
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [(f"cannot run {python}: {e}",)]
    if r.returncode == 0:
        return []
    try:
        return [tuple(x) for x in eval(r.stdout.strip().splitlines()[-1])]  # noqa: S307 (our own output)
    except Exception:  # noqa: BLE001
        return [(r.stderr.strip()[-200:] or "import check failed",)]


class Stages:
    """prep / post of one experiment (x: run_cycles.Experiment, steps: core.Steps)"""

    def __init__(self, x, steps, echo=print):
        self.x, self.steps, self.echo = x, steps, echo
        self.prep, self.post = x.cfg["prep"], x.cfg["post"]
        self.python = x.rt["python"] or steps.python
        self.maxlead = max(x.leads) if x.leads else 0
        self.mrms_dir = Path(self.prep["mrms"]["dir"])
        self.figdir = Path(self.post["dir"] or x.exp / "figures")

    # ------------------------------------------------------------------ hours and files
    def hours_for(self, part):
        x = self.x
        if part == "reference":
            return list(hours(x.start - timedelta(hours=1), x.end + timedelta(hours=self.maxlead)))
        if part in ("obs", "mrms"):
            return list(hours(x.start, x.end + timedelta(hours=self.maxlead)))
        if part == "background":
            L = x.bg_lead
            if x.bg["source"] != "forecast_dir":     # only the first cycle reads forecast_dir
                return [x.start - timedelta(hours=L)]
            return list(hours(x.start - timedelta(hours=L), x.end - timedelta(hours=L)))
        raise KeyError(part)

    def done(self, part, t):
        """the hour's outputs exist (obs: every type)"""
        x, c = self.x, cycstr(t)
        if part == "reference":
            return all((x.ref / c / f"{k}_{ymd_hh(t)}.npz").exists() for k in ("hrrr", "gfs"))
        if part == "obs":
            return all(any((x.obs / ty / c).glob("*.nc4")) for ty in x.obs_types)
        if part == "mrms":
            return (self.mrms_dir / c / f"mrms_{ymd_hh(t)}.npz").exists()
        if part == "background":
            return (Path(x.bg["forecast_dir"]) / c / f"fcst_f{x.bg_lead:02d}.npz").exists()
        raise KeyError(part)

    def missing(self, part):
        return [t for t in self.hours_for(part) if not self.done(part, t)]

    def needed(self, part):
        """hours the cycles cannot run without (missing elsewhere = verification skipped there)"""
        x = self.x
        if part == "reference":
            return set(hours(x.start - timedelta(hours=1), x.end))
        if part == "background":
            return set(self.hours_for("background"))
        if part == "mrms":
            rs = (x.analysis_base().get("ls3dvar", {}) or {}).get("refc_source") or {}
            return set(hours(x.start, x.end)) if rs.get("source") == "mrms" and x.an["enabled"] else set()
        return set()                            # obs: the analysis decides (a network may have no data)

    def any_reference(self):
        """a staged reference npz (the HRRRCast grid for mrms_stage.py)"""
        for t in self.hours_for("reference"):
            f = self.x.ref / cycstr(t) / f"hrrr_{ymd_hh(t)}.npz"
            if f.exists():
                return f
        return None

    def wants_background(self):
        b = self.x.bg
        return "forecast_dir" in (b["first_cycle"], b["source"])

    def enabled_prep(self):
        out = [p for p in PREP_PARTS if self.prep[p]["enabled"]]
        if "background" in out and not self.wants_background():
            out.remove("background")
        return out

    def enabled_post(self):
        return [p for p in POST_PARTS if self.post[p]["enabled"]]

    # ------------------------------------------------------------------ check
    def check_lines(self):
        """(lines, problems) for `check`"""
        lines, bad = [], []
        prep_on, post_on = self.enabled_prep(), self.enabled_post()
        lines.append(f"prep       : {', '.join(prep_on) or 'off'}")
        for p in prep_on:
            miss = self.missing(p)
            hrs = self.hours_for(p)
            rng = ", ".join(f"{cycstr(a)}..{cycstr(b)}" if a != b else cycstr(a) for a, b in ranges(miss))
            lines.append(f"  {p:10s} {len(hrs) - len(miss)}/{len(hrs)} hours present"
                         + (f"; to make: {rng}" if miss else ""))
            s = self.prep[p].get("script")
            if p == "mrms" and self.prep["mrms"].get("via") == "obsdb":
                s = self.prep["obs"]["script"]
                if not (Path(s).parent / "mrms_stage.py").exists():
                    bad.append(f"prep.mrms via obsdb: mrms_stage.py must sit next to {s}")
                if not Path(self.prep["obs"]["config"]).exists():
                    bad.append(f"prep.mrms via obsdb: prep.obs.config not found: {self.prep['obs']['config']}")
            elif p == "mrms" and self.prep["mrms"].get("via") not in ("mrms_stage", None):
                bad.append("prep.mrms.via must be mrms_stage | obsdb")
            if s and not Path(s).exists():
                bad.append(f"prep.{p}.script not found: {s}")
            if p == "obs" and not Path(self.prep["obs"]["config"]).exists():
                bad.append(f"prep.obs.config not found: {self.prep['obs']['config']}")
            if p == "reference" and self.x.ref.name != "hrrr_ana":
                bad.append(f"prep.reference: stage_reference.sh writes <REF_ROOT>/hrrr_ana, but "
                           f"paths.reference_dir is {self.x.ref} (must end in /hrrr_ana)")
        for p, key in (("obs", "obs"), ("mrms", "mrms")):
            if p in prep_on and self.missing(p):
                py = self.prep[p].get("python") or self.python
                for g in check_imports(py, NEEDS[key]):
                    bad.append(f"prep.{p}: {py} cannot import {' or '.join(g)}")
        lines.append(f"post       : {', '.join(post_on) or 'off'}"
                     + (f"   -> {self.figdir}" if post_on else ""))
        for p in post_on:
            for k in ("script", "template"):
                f = self.post[p].get(k)
                if f and not Path(f).exists():
                    bad.append(f"post.{p}.{k} not found: {f}")
        if post_on:
            py = self.post["python"] or self.python
            for g in check_imports(py, NEEDS["post"]):
                bad.append(f"post: {py} cannot import {' or '.join(g)}")
        if "threat" in post_on and "diagnostics" not in post_on and not self.post["diagnostics"]["template"]:
            bad.append("post.threat needs post.diagnostics.template (it reads the diagnostics config)")
        # the analysis reads MRMS from its own config: warn when it is not the staged folder
        rs = (self.x.analysis_base().get("ls3dvar", {}) or {}).get("refc_source") or {}
        if rs.get("source") == "mrms" and "mrms" in prep_on and Path(rs.get("dir", "")) != self.mrms_dir:
            bad.append(f"analysis reads MRMS from {rs.get('dir')} but prep.mrms.dir is {self.mrms_dir}")
        return lines, bad

    # ------------------------------------------------------------------ prep
    def _run(self, name, cmd, log, cwd=None, env=None, timeout_key="prep_timeout", retry=True):
        self.echo(f"{time.strftime('%H:%M:%S')} [{name}] " + " ".join(str(c) for c in cmd))
        sec = self.steps.run(name, None, None, log, cmd=cmd, cwd=cwd, env=env, retry=retry,
                             timeout=int(self.x.rt.get(timeout_key) or 0), timeout_name=f"runtime.{timeout_key}")
        self.echo(f"  done in {fmt_seconds(sec)}   (log: {log})")
        return sec

    def run_prep(self, only=None, dry_run=False):
        x = self.x
        logdir = x.exp / "prep"
        logdir.mkdir(parents=True, exist_ok=True)
        parts = [p for p in self.enabled_prep() if not only or p in only]
        self.echo(f"\n================================ prep: {', '.join(parts) or 'nothing'} "
                  "================================")
        for part in parts:                      # order matters: mrms takes its grid from the reference
            miss = self.missing(part)
            if not miss:
                self.echo(f"  {part}: all {len(self.hours_for(part))} hours present")
                continue
            self.steps.progress(cycle="prep", step=part, state="running")
            for a, b in ranges(miss):
                tag = cycstr(a) if a == b else f"{cycstr(a)}-{cycstr(b)}"
                cmd, cwd, env = self.prep_command(part, a, b)
                if dry_run:
                    self.echo(f"  [dry-run] {part} {tag}: " + " ".join(str(c) for c in cmd))
                    continue
                try:
                    self._run(f"prep {part} {tag}", cmd, logdir / f"{part}_{tag}.log", cwd=cwd, env=env)
                except StepFailed as e:
                    if any(t in self.needed(part) for t in hours(a, b)):
                        raise
                    self.echo(f"  !! {e}   (only verification hours: continuing)")
            if dry_run:
                continue
            if part == "reference" and self.prep["reference"].get("copy"):
                self.dereference()
            still = self.missing(part)
            if still:
                lst = ", ".join(cycstr(t) for t in still[:8]) + (" ..." if len(still) > 8 else "")
                msg = f"  !! {part}: {len(still)} hour(s) still missing after the script: {lst}"
                if part == "obs":
                    absent = {ty for t in still for ty in x.obs_types
                              if not any((x.obs / ty / cycstr(t)).glob("*.nc4"))}
                    msg += f"   (types without files: {sorted(absent)}; no raw data for those hours?)"
                self.echo(msg)
                if any(t in self.needed(part) for t in still):
                    raise StepFailed(f"prep {part}: inputs the cycles need are missing (see {logdir})")
        return 0

    def dereference(self):
        """replace the links stage_reference.sh makes by copies (a self-contained data folder)"""
        n = 0
        for t in self.hours_for("reference"):
            d = self.x.ref / cycstr(t)
            for f in (d.iterdir() if d.is_dir() else []):
                if f.is_symlink() and f.exists():
                    src = f.resolve()
                    tmp = f.with_name(f.name + ".copying")
                    shutil.copy2(src, tmp)
                    tmp.replace(f)
                    n += 1
        if n:
            self.echo(f"  reference: {n} linked file(s) replaced by copies")

    def prep_command(self, part, a, b):
        x = self.x
        if part == "reference":
            p = self.prep["reference"]
            cmd = ["bash", p["script"], iso_h(a), iso_h(b)] + ([] if p["run_missing"] else ["--no-run"])
            env = {"REF_ROOT": str(x.ref.parent), **{k: str(v) for k, v in (p["env"] or {}).items()}}
            return cmd, Path(p["script"]).parent, env
        if part == "obs":
            p = self.prep["obs"]
            cmd = script_cmd(p["script"], p["python"] or self.python) + [
                p["config"], "--start", cycstr(a), "--end", cycstr(b),
                "--types", *x.obs_types, "--output-root", x.obs]
            return cmd, Path(p["script"]).parent, None
        if part == "mrms" and self.prep["mrms"].get("via") == "obsdb":
            p, o = self.prep["mrms"], self.prep["obs"]      # build_obsdb_cycles.py, its `mrms:` obs type
            cmd = script_cmd(o["script"], p["python"] or o["python"] or self.python) + [
                o["config"], "--start", cycstr(a), "--end", cycstr(b), "--types", self.mrms_dir.name,
                "--output-root", self.mrms_dir.parent, *str(p["args"] or "").split()]
            return cmd, Path(o["script"]).parent, None
        if part == "mrms":
            p = self.prep["mrms"]
            grid = self.any_reference()
            cmd = script_cmd(p["script"], p["python"] or self.python) + [
                "--start", iso_h(a), "--end", iso_h(b), "--ref-dir", x.ref, "--out", self.mrms_dir,
                "--repo", REPO_ROOT, *(["--grid-npz", grid] if grid and "--grid-npz" not in str(p["args"]) else []),
                *str(p["args"] or "").split()]
            return cmd, Path(p["script"]).parent, None
        if part == "background":
            cmd = [sys.executable, "-m", "ls3dvar.cli.run_forecasts", "--start", iso_h(a), "--end", iso_h(b),
                   "--length", x.bg_lead, "--reference-dir", x.ref, "--out", x.bg["forecast_dir"],
                   "--code", x.code, "--timeout", x.rt["step_timeout"], "--retries", x.rt["retries"],
                   "--deterministic" if x.deterministic else "--no-deterministic"]
            if x.rt["python"]:
                cmd += ["--python", x.rt["python"]]
            return cmd, REPO_ROOT, None
        raise KeyError(part)

    # ------------------------------------------------------------------ post
    def cycles_done(self):
        return [cycstr(t) for t in hours(self.x.start, self.x.end)
                if (self.x.exp / cycstr(t) / "analysis.npz").exists()]

    def post_config(self, part):
        """the script config for one post part, from its template; written to <exp>/post/"""
        x, p = self.x, self.post[part if part != "threat" else "diagnostics"]
        c = load_yaml(p["template"]) or {}
        cyc = self.cycles_done()
        if part in ("plots", "fields"):
            c.update(experiment_dir=str(x.exp), reference_dir=str(x.ref), obsdb_root=str(x.obs),
                     obs_types=list(x.obs_types), cycles=cyc, leads=list(x.leads) or [1])
            if part == "plots":
                csv = p["verification_csv"]
                if csv == "auto":
                    csv = "verification_heldout.csv" if x.holdout else "verification.csv"
                c.update(output_dir=str(self.figdir), verification_csv=csv)
                c.setdefault("plots", {})["compare"] = False
                c["compare"] = dict(c.get("compare") or {}, enabled=False)
            else:
                c.update(output_dir=str(self.figdir / "fields"))
                c["plots"] = {k: False for k in ("increment", "forecast_impact", "spectra", "verification",
                                                  "precip")} | {"compare": True,
                                                                "obs_overlay": (c.get("plots") or {}).get("obs_overlay", True)}
                cc = dict(c.get("compare") or {})
                cc.update(enabled=True, experiments=[{"label": "with LS-3DVar", "dir": str(x.exp)}],
                          leads=[0] + list(x.leads), mrms_dir=str(self.mrms_dir))
                cc.pop("background_from", None)
                c["compare"] = cc
        else:                                   # diagnostics / threat
            data = x.paths.get("data") or str(x.exp.parent.parent)
            c.update(data=str(data), reference_dir=str(x.ref), obsdb_root=str(x.obs),
                     mrms_dir=str(self.mrms_dir), experiments_dir=str(x.exp.parent),
                     output_dir=str(self.figdir / "da_impact"), obs_types=list(x.obs_types),
                     holdout=x.holdout or None, cycles=cyc, leads=[0] + list(x.leads))
            c["experiments"] = {f"{x.name}@bg": "without DA (HRRRCast from background)",
                                x.name: "with LS-3DVar"}
            c.pop("baseline", None)
        c = deep_merge(c, copy.deepcopy(p.get("set") or {}))
        out = x.exp / "post" / f"{'diagnostics' if part == 'threat' else part}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(c, out)
        return out, cyc

    def run_post(self, only=None):
        x = self.x
        parts = [p for p in self.enabled_post() if not only or p in only]
        self.echo(f"\n================================ post: {', '.join(parts) or 'nothing'} "
                  "================================")
        failed = []
        py = self.post["python"] or self.python
        for part in parts:
            try:
                cfg, cyc = self.post_config(part)
            except Exception as e:  # noqa: BLE001
                self.echo(f"  !! post {part}: cannot write its config: {type(e).__name__}: {e}")
                failed.append(part)
                continue
            if not cyc:
                self.echo(f"  !! post {part}: no cycle with analysis.npz in {x.exp}")
                failed.append(part)
                continue
            p = self.post[part]
            cmd = [py, p["script"], cfg]
            if part == "plots" and p.get("only"):
                cmd += ["--only", *p["only"]]
            if part == "diagnostics" and p.get("only"):
                cmd += ["--only", *p["only"]]
            if part == "threat" and p.get("hrrr"):
                cmd += ["--hrrr"]
            self.steps.progress(cycle="post", step=part, state="running")
            try:
                self._run(f"post {part}", cmd, x.exp / "post" / f"{part}.log", cwd=Path(p["script"]).parent,
                          timeout_key="post_timeout", retry=False)
            except StepFailed as e:
                self.echo(f"  !! post {part} failed: {e}")
                failed.append(part)
        if parts:
            self.echo(f"figures: {self.figdir}" + (f"   (failed: {', '.join(failed)})" if failed else ""))
        return failed


# ============================================================================ push
def push_plan(x):
    """[(source dir, destination URL)] for push.what: all = the data folder, experiment = <exp> only"""
    p = x.cfg["push"]
    dest = str(p["dest"] or "").rstrip("/")
    if p["what"] == "experiment":
        return [(x.exp, f"{dest}/{x.exp.parent.name}/{x.exp.name}")]
    if p["what"] == "all":
        root = Path(x.paths.get("data") or x.exp.parent.parent)
        return [(root, dest)]
    raise SystemExit("push.what must be all | experiment")


def push_check(x):
    """problems that would stop the push"""
    p, bad = x.cfg["push"], []
    if not p["enabled"]:
        return bad
    if not str(p["dest"] or "").startswith("s3://"):
        bad.append(f"push.dest must be an s3:// URL (got {p['dest']!r})")
    if not shutil.which(p["tool"]):
        bad.append(f"push: '{p['tool']}' not found on PATH (aws CLI)")
    else:
        try:
            r = subprocess.run([p["tool"], "sts", "get-caller-identity"], capture_output=True, text=True, timeout=60)
            if r.returncode:
                bad.append(f"push: no AWS credentials ({(r.stderr or r.stdout).strip().splitlines()[-1][:150]})")
            elif str(p["dest"] or "").startswith("s3://"):
                bucket = str(p["dest"])[5:].split("/")[0]
                r = subprocess.run([p["tool"], "s3", "ls", f"s3://{bucket}/"], capture_output=True, text=True,
                                   timeout=120)
                if r.returncode:
                    msg = (r.stderr or r.stdout or "").strip().splitlines()
                    bad.append(f"push: cannot access s3://{bucket}/ ({msg[-1][:150] if msg else 'aws s3 ls failed'})")
        except (OSError, subprocess.TimeoutExpired) as e:
            bad.append(f"push: cannot run {p['tool']}: {e}")
    return bad


def run_push(x, steps, echo=print, dry_run=False):
    p = x.cfg["push"]
    echo(f"\n================================ push -> {p['dest']} ================================")
    for src, dest in push_plan(x):
        cmd = [p["tool"], "s3", "sync", str(src), dest, "--no-progress"]
        for e in p["exclude"] or []:
            cmd += ["--exclude", e]
        cmd += str(p["args"] or "").split() + (["--dryrun"] if dry_run else [])
        log = x.exp / "post" / "push.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        echo(f"{time.strftime('%H:%M:%S')} [push] " + " ".join(cmd))
        sec = steps.run("push", None, None, log, cmd=cmd, cwd=src, retry=True,
                        timeout=int(x.rt.get("post_timeout") or 0), timeout_name="runtime.post_timeout")
        echo(f"  {'dry run ' if dry_run else ''}done in {fmt_seconds(sec)}   (log: {log})")
    return 0
