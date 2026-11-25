#!/usr/bin/env python3
"""
ls3dvar/cli/run_cycles.py

LS-3DVar cycling experiment from ONE YAML file: background, optional GFS blend,
analysis, forecasts and verification for every hour in [start, end].

    python -m ls3dvar.cli.run_cycles init  --from $DATA/configs/frozen_v1/test.yaml --name frozen_v1_pkg > exp.yaml
    python -m ls3dvar.cli.run_cycles check   exp.yaml     # settings, inputs, code, GPU free?
    python -m ls3dvar.cli.run_cycles run     exp.yaml     # foreground; --detach: background
    python -m ls3dvar.cli.run_cycles status  exp.yaml     # current cycle / step, newest files
    python -m ls3dvar.cli.run_cycles tail    exp.yaml     # follow the log
    python -m ls3dvar.cli.run_cycles stop    exp.yaml     # stop this experiment (only this one)
    python -m ls3dvar.cli.run_cycles results exp.yaml     # summary over cycles (all + held-out)

    python -m ls3dvar.cli.run_cycles prep     exp.yaml    # stage inputs only (prep: section)
    python -m ls3dvar.cli.run_cycles post     exp.yaml    # diagnostic plots only (post: section)
    python -m ls3dvar.cli.run_cycles pipeline exp.yaml    # prep -> cycles -> post -> push (--detach)
    python -m ls3dvar.cli.run_cycles push     exp.yaml    # aws s3 sync to push.dest (--dry-run)

prep / post run the data-directory scripts (stage_reference.sh, build_obsdb_cycles.py,
mrms_stage.py, plot_ls3dvar_cycling.py, ls3dvar_diagnostics.py, plot_mrms_threat.py) for the
hours this experiment needs, skipping what exists: see ls3dvar/workflow/stages.py.

See ls3dvar/configs/experiment_example.yaml for every key. Per cycle H, exactly as
scripts/cycle_ls3dvar.sh:

    background   first cycle: background.first_cycle      later: background.source
                   reference_forecast  F01 from the reference HRRR analysis at H-1
                   cycled              F01 from the previous cycle's analysis
                   forecast_dir        <forecast_dir>/<H-lead>/fcst_f<lead>.npz (run_forecasts.py)
    blend        optional large-scale blend toward GFS at H (raw F01 kept as background_raw.npz)
    analysis     LS-3DVar with obs <obsdb_root>/<type>/<H>   (analysis.enabled)
    forecasts    F<leads> from the analysis (an) and the background (bg)   (forecast.enabled)
    verification lead 0 at H, lead L at H+L, vs obs and vs the reference analysis
                 -> <exp>/verification.csv (+ verification_heldout.csv)   (verification.enabled)

Every step runs as its own process (timeout + retries; plain TF by default, bit-reproducible
with experiment.deterministic: true). Resume
skips finished steps, and refuses to continue an experiment made with other settings or
other code (--force overrides). Unlike cycle_ls3dvar.sh, a resumed run does not verify
finished cycles again (the bash script appended their verification rows a second time);
a fresh run is step-for-step identical to cycle_ls3dvar.sh.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from ls3dvar.workflow.core import (MODULES, REPO_ROOT, StepFailed, Steps, code_version, cycstr,
                                   deep_merge, expand, fmt_seconds, git_dirty, hours, load_yaml,
                                   parse_time, pid_alive, resolve_paths, save_yaml, ymd_hh)
from ls3dvar.workflow import stages as ST

# ============================================================================ schema
DEFAULTS = {
    "experiment": {"name": None, "start": None, "end": None, "output_dir": "${data}/experiments",
                   "resume": True, "code": "ls3dvar", "deterministic": False},
    "paths": {"data": None, "reference_dir": "${data}/reference/hrrr_ana", "obsdb_root": "${data}/obsdb"},
    "background": {"first_cycle": "reference_forecast", "source": "cycled", "forecast_dir": None,
                   "lead": 1,
                   "blend": {"enabled": False, "args": "--cutoff-km 1000 --pmax 850 --weight 1.0"}},
    "observations": {"types": ["asos", "tempest", "grenade"], "holdout_stations": None},
    "analysis": {"enabled": True, "base_config": None, "overrides": {}, "extra_args": "", "truth": True},
    "forecast": {"enabled": True, "leads": [1, 2, 3], "from": ["an", "bg"]},
    "verification": {"enabled": True, "qc": True, "config": None},
    "runtime": {"step_timeout": 900, "analysis_timeout": 1800, "retries": 3, "retry_wait": 20, "python": None, "progress": "iterations",
                "prep_timeout": 0, "post_timeout": 0,
                "env": {"CUDA_CACHE_MAXSIZE": "4294967296", "TF_FORCE_GPU_ALLOW_GROWTH": "true"}},
    "prep": ST.PREP_DEFAULTS,
    "post": ST.POST_DEFAULTS,
    "push": ST.PUSH_DEFAULTS,
}
FREE_FORM = {("analysis", "overrides"), ("runtime", "env"), ("paths",)} | ST.FREE_FORM   # any keys allowed
BG_SOURCES = ("reference_forecast", "cycled", "forecast_dir")
# settings that may change between runs of the same experiment (resume)
RESUMABLE = {("experiment", "start"), ("experiment", "end"), ("experiment", "resume"), ("runtime",),
             ("prep",), ("post",), ("push",)}

# anchored: only processes that ARE python (not a `bash -c "... python -m ..."` wrapper)
# live progress: lines of the step logs copied into run.log while a step runs
WATCH = {
    "iterations": re.compile(
        r"^(ITER=|EARLY STOP|.*OUTER LOOP \d+|.*: O-B -> O-A rms|\s+\w+\s+assimilated[=:]|"
        r"station hold-out|after QC:|\s+-> wind_rotation|\s+\S+\s+pseudo-obs:|\s+latent increment rms|"
        r"\s+PRES increment rms|\s+\w+\s+mean\(H x - y\)|\s+vs TRUTH|\s+[A-Z]\w*\s+(\d+\.\d+|nan)->|.*STRUCTURED B:|HYBRID:|"
        r"PSEUDO-OBS from|REFC source:|FAIL|WARNING|Traceback|\w*Error\b)"),
    "outer": re.compile(
        r"^(EARLY STOP|.*OUTER LOOP \d+|.*: O-B -> O-A rms|\s+\w+\s+assimilated[=:]|\s+-> wind_rotation|"
        r"\s+\S+\s+pseudo-obs:|\s+latent increment rms|FAIL|WARNING|Traceback|\w*Error\b)"),
}
WATCH_FORECAST = re.compile(r"^(HRRRCast F\d+\s+valid|FAIL|WARNING|Traceback|\w*Error\b)")

STEPS_PATTERN = r"^[^ ]*python[^ ]* (-m (ls3dvar\.cli|mantari\.workflows|mantari\.ls3dvar)\.|[^ ]*det_runner\.py )"
SUMMARY_VARS = ("T2M", "D2M", "PS", "U10", "V10", "TMP850")


def unknown_keys(user, ref, path=()):
    out = []
    for k, v in (user or {}).items():
        p = path + (k,)
        if any(p[:len(f)] == f for f in FREE_FORM):
            continue
        if not isinstance(ref, dict) or k not in ref:
            out.append(".".join(p))
        elif isinstance(v, dict) and isinstance(ref[k], dict):
            out += unknown_keys(v, ref[k], p)
    return out


class Experiment:
    """the resolved experiment settings"""

    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = load_yaml(self.path)
        bad = unknown_keys(raw, DEFAULTS)
        if bad:
            raise SystemExit(f"{self.path}: unknown keys {bad} (see ls3dvar/configs/experiment_example.yaml)")
        cfg = deep_merge(DEFAULTS, raw)
        if "paths" in raw:                               # paths: user keys only + defaults for the rest
            cfg["paths"] = {**DEFAULTS["paths"], **raw["paths"]}
        try:
            self.paths = resolve_paths({k: v for k, v in cfg["paths"].items() if v is not None})
        except ValueError as err:
            raise SystemExit(f"paths: {err}  (set paths.data, or reference_dir and obsdb_root)")
        stage_cfg = {k: cfg.pop(k) for k in ("prep", "post", "push")}
        cfg = expand({k: v for k, v in cfg.items() if k != "paths"}, self.paths)
        cfg["paths"] = self.paths
        for k, v in stage_cfg.items():          # ${data} needed only by the parts that are on
            v = expand(v, self.paths, strict=False)
            for part, pv in ({"push": v} if k == "push" else v).items():
                if isinstance(pv, dict) and pv.get("enabled") and "${" in json.dumps(pv):
                    raise SystemExit(f"{k}.{part}: unresolved ${{...}} in {pv} (set paths.data or the full paths)")
            cfg[k] = v
        self.cfg = cfg
        e, b, o, an, fc, vf, rt = (cfg[k] for k in ("experiment", "background", "observations", "analysis",
                                                     "forecast", "verification", "runtime"))
        for k in ("name", "start", "end"):
            if not e.get(k):
                raise SystemExit(f"experiment.{k} is required")
        self.name = str(e["name"])
        self.start, self.end = parse_time(e["start"]), parse_time(e["end"])
        if self.end < self.start:
            raise SystemExit("experiment.end is before experiment.start")
        self.exp = Path(e["output_dir"]) / self.name
        self.code = e["code"]
        if self.code not in MODULES:
            raise SystemExit(f"experiment.code must be one of {sorted(MODULES)}")
        self.mod = MODULES[self.code]
        self.deterministic = bool(e["deterministic"])
        self.resume = bool(e["resume"])
        self.ref = Path(self.paths["reference_dir"])
        self.obs = Path(self.paths["obsdb_root"])
        for k in ("first_cycle", "source"):
            if b[k] not in BG_SOURCES:
                raise SystemExit(f"background.{k} must be one of {BG_SOURCES}")
        if "forecast_dir" in (b["first_cycle"], b["source"]) and not b["forecast_dir"]:
            raise SystemExit("background.forecast_dir is required for source forecast_dir")
        self.bg = b
        self.bg_lead = int(b["lead"])
        self.obs_types = [str(t) for t in o["types"]]
        hold = o["holdout_stations"]
        self.holdout = "" if hold in (None, "", "none", False) else str(hold)
        self.an = an
        if an["enabled"] and not an["base_config"]:
            raise SystemExit("analysis.base_config is required (the ls3dvar: / observations: config)")
        self.fc, self.vf, self.rt = fc, vf, rt
        self.leads = sorted({int(L) for L in fc["leads"]}) if fc["enabled"] else []
        self.fc_from = [s for s in fc["from"] if s in ("an", "bg")] if fc["enabled"] else []
        if not an["enabled"]:
            self.fc_from = [s for s in self.fc_from if s == "bg"]          # no analysis: bg only
        # cycling from the analysis needs its F01, even if forecasts are off or skip lead 1
        self.need_an_f01 = an["enabled"] and "cycled" in (b["source"],)
        self.need_bg_f01 = (not an["enabled"]) and b["source"] == "cycled"

    # ------------------------------------------------------------------ analysis config
    def analysis_base(self):
        """the analysis config every cycle starts from: base_config + overrides"""
        base = load_yaml(self.an["base_config"]) if self.an["base_config"] else {}
        return deep_merge(base, self.an["overrides"] or {})

    def comparable(self):
        """settings that must not change when an experiment is resumed"""
        c = json.loads(json.dumps(self.cfg, default=str))
        for path in RESUMABLE:
            d = c
            for k in path[:-1]:
                d = d.get(k, {})
            d.pop(path[-1], None)
        c["analysis_config"] = json.loads(json.dumps(self.analysis_base(), default=str))
        return c


# ============================================================================ helpers
def log_print(msg=""):
    print(msg, flush=True)


def steps_running():
    """command lines of LS-3DVar steps and drivers running on this machine (not this one)"""
    try:
        r = subprocess.run(["pgrep", "-af", STEPS_PATTERN], capture_output=True, text=True)
    except OSError:
        return []
    me = {str(os.getpid()), str(os.getppid())}      # this driver (and its detaching parent)
    return [l for l in r.stdout.splitlines() if l.strip() and l.split()[0] not in me]


def flat_diff(a, b, path=""):
    out = []
    for k in sorted(set(a) | set(b), key=str):
        p = f"{path}.{k}" if path else str(k)
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out += flat_diff(a[k], b[k], p)
        elif a.get(k) != b.get(k):
            out.append(f"{p}: {a.get(k)!r} -> {b.get(k)!r}")
    return out


def compatibility(x):
    """reasons the existing experiment folder must not be resumed with these settings"""
    if not x.exp.exists() or not any(x.exp.iterdir()):
        return []
    if not (x.exp / "code_version.txt").exists() and not any(x.exp.glob("[0-9]*T[0-9]*Z")):
        return []                               # only prep / post output so far
    out = []
    cv = x.exp / "code_version.txt"
    now = code_version(x.code, REPO_ROOT, x.deterministic)
    if not cv.exists():
        out.append("no code_version.txt (made by the old scripts?)")
    elif cv.read_text().strip() != now:
        out.append(f"code: {cv.read_text().strip()}  ->  {now}")
    saved = x.exp / "experiment_resolved.json"
    if saved.exists():
        diff = flat_diff(json.loads(saved.read_text()), x.comparable())
        out += [f"setting {d}" for d in diff]
    else:
        out.append("no experiment_resolved.json (not made by run_cycles.py)")
    return out


def nonempty(p):
    return p.exists() and p.stat().st_size > 0


def section(text, start_pat):
    """the part of `text` from the first line matching start_pat (sed -n '/pat/,$p')"""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if re.search(start_pat, l):
            return "\n".join(lines[i:])
    return ""


# ============================================================================ run
class Cycler:
    def __init__(self, x, force=False):
        self.x = x
        self.force = force
        rt = x.rt
        self.steps = Steps(REPO_ROOT, python=rt["python"], deterministic=x.deterministic,
                           timeout=rt["step_timeout"], retries=rt["retries"], retry_wait=rt["retry_wait"],
                           env=rt["env"], progress_file=x.exp / "progress.json", echo=log_print)
        prog = rt.get("progress", "iterations")
        if prog not in ("iterations", "outer", "quiet"):
            raise SystemExit("runtime.progress must be iterations | outer | quiet")
        self.watch_an = WATCH.get(prog)
        self.watch_fc = WATCH_FORECAST if prog != "quiet" else None
        self.csv = x.exp / "verification.csv"
        self.csv_ho = x.exp / "verification_heldout.csv"

    def have(self, p):
        return self.x.resume and nonempty(p)

    # ------------------------------------------------------------------ steps of one cycle
    def background(self, t, d, first):
        x, ref = self.x, self.x.ref
        bg = d / "background.npz"
        if self.have(bg):
            return bg
        src = x.bg["first_cycle"] if first else x.bg["source"]
        prev = t - timedelta(hours=1)
        prev_an_f01 = x.exp / cycstr(prev) / ("fcst_an_f01.npz" if x.an["enabled"] else "fcst_bg_f01.npz")
        if src == "cycled" and not nonempty(prev_an_f01):
            log_print(f"  (no {prev_an_f01.relative_to(x.exp)}: cold start instead of cycling)")
            src = "reference_forecast"
        if src == "reference_forecast":
            log_print(f"{time.strftime('%H:%M:%S')} [bg] cold start: F01 from reference HRRR analysis {cycstr(prev)}")
            self.steps.progress(step="background")
            sec = self.steps.run("background", x.mod["forecast"],
                                 ["--hrrr-npz", ref / cycstr(prev) / f"hrrr_{ymd_hh(prev)}.npz",
                                  "--gfs-npz", ref / cycstr(prev) / f"gfs_{ymd_hh(prev)}.npz",
                                  "--lead", 1, "--out", bg], d / "background.log", retry=False)
            log_print(f"  background done in {fmt_seconds(sec)}")
        elif src == "cycled":
            log_print(f"{time.strftime('%H:%M:%S')} [bg] cycled: F01 from {'analysis' if x.an['enabled'] else 'background'} {cycstr(prev)}")
            shutil.copyfile(prev_an_f01, bg)
        else:
            L = x.bg_lead
            init = t - timedelta(hours=L)
            f = Path(x.bg["forecast_dir"]) / cycstr(init) / f"fcst_f{L:02d}.npz"
            if not nonempty(f):
                raise StepFailed(f"background from forecast_dir: missing {f} (run_forecasts.py)")
            log_print(f"{time.strftime('%H:%M:%S')} [bg] F{L:02d} from forecast_dir, init {cycstr(init)}")
            shutil.copyfile(f, bg)
        (d / "background_raw.npz").unlink(missing_ok=True)
        return bg

    def blend(self, t, d, bg):
        x = self.x
        raw = d / "background_raw.npz"
        if not x.bg["blend"]["enabled"] or nonempty(raw):
            return raw if nonempty(raw) else None
        log_print(f"{time.strftime('%H:%M:%S')} [bl] large-scale blend toward GFS")
        bg.replace(raw)
        try:
            self.steps.run("blend", x.mod["blend"],
                           ["--hrrr", raw, "--gfs", x.ref / cycstr(t) / f"gfs_{ymd_hh(t)}.npz", "--out", bg,
                            *str(x.bg["blend"]["args"]).split(),
                            "--truth", x.ref / cycstr(t) / f"hrrr_{ymd_hh(t)}.npz"],
                           d / "blend.log", retry=False)
        except StepFailed:
            raw.replace(bg)
            raise
        hits = [l for l in (d / "blend.log").read_text(errors="replace").splitlines()
                if re.match(r"^(TMP_850|UGRD_850|HGT_500) ", l)]
        for l in hits[-3:]:
            log_print(l)
        return raw

    def cycle_config(self, t, d, bg, base):
        """per-cycle analysis config (same edits as cycle_ls3dvar.sh)"""
        x = self.x
        c = json.loads(json.dumps(base))
        c.setdefault("ls3dvar", {})["holdout_stations"] = x.holdout or None
        if not x.holdout:
            c["ls3dvar"]["holdout"] = 0.0          # cycling without hold-out assimilates everything
        c["hrrr_npz"], c["gfs_npz"] = str(bg), str(x.ref / cycstr(t) / f"gfs_{ymd_hh(t)}.npz")
        c.setdefault("observations", {})["providers"] = [
            {"type": ty, "path": f"{x.obs}/{ty}/{cycstr(t)}"} for ty in x.obs_types]
        c.setdefault("outputs", {})["directory"] = str(d / "ls3dvar")
        save_yaml(c, d / "config.yaml")
        return d / "config.yaml"

    def analysis(self, t, d, cfg_path):
        x = self.x
        an = d / "analysis.npz"
        if self.have(an):
            return an
        log_print(f"{time.strftime('%H:%M:%S')} [an] LS-3DVar")
        args = ["--config", cfg_path, *str(x.an["extra_args"] or "").split()]
        if x.an["truth"]:
            args += ["--truth-npz", x.ref / cycstr(t) / f"hrrr_{ymd_hh(t)}.npz"]
        args += ["--analysis-out", an]
        sec = self.steps.run("analysis", x.mod["analysis"], args, d / "analysis.log", watch=self.watch_an,
                             timeout=x.rt.get("analysis_timeout") or x.rt["step_timeout"],
                             timeout_name="runtime.analysis_timeout")
        if self.watch_an is None:                  # quiet: the short summary afterwards, as the bash cycle
            txt = (d / "analysis.log").read_text(errors="replace").splitlines()
            for i, l in enumerate(txt):
                if "O-B -> O-A rms" in l:
                    for m in txt[i:i + 13]:
                        if re.search(r"assimilated|mean\(H|T2M |D2M |PRES ", m):
                            log_print(m)
                    break
        log_print(f"  analysis done in {fmt_seconds(sec)}   (log: {(d / 'analysis.log').relative_to(x.exp)})")
        return an

    def forecasts(self, t, d, states):
        x = self.x
        for src in ("an", "bg"):
            leads = list(x.leads) if src in x.fc_from else []
            if src == "an" and x.need_an_f01 and 1 not in leads:
                leads = [1] + leads
            if src == "bg" and x.need_bg_f01 and 1 not in leads:
                leads = [1] + leads
            if not leads or states.get(src) is None:
                continue
            last = d / f"fcst_{src}_f{leads[-1]:02d}.npz"
            if self.have(last):
                continue
            log_print(f"{time.strftime('%H:%M:%S')} [fc] F0{{{','.join(map(str, leads))}}} from {src}")
            sec = self.steps.run(f"forecast {src}", x.mod["forecast"],
                                 ["--hrrr-npz", states[src], "--gfs-npz", x.ref / cycstr(t) / f"gfs_{ymd_hh(t)}.npz",
                                  "--lead", *leads, "--out", d / f"fcst_{src}_f{{lead:02d}}.npz"],
                                 d / f"fcst_{src}.log", watch=self.watch_fc)
            log_print(f"  forecasts from {src} done in {fmt_seconds(sec)}")

    def verify(self, t, d, states0, cfg_path):
        x = self.x
        rot = None
        alog = d / "analysis.log"
        if alog.exists():
            m = re.findall(r"wind_rotation = ([a-z]*)", alog.read_text(errors="replace"))
            rot = m[-1] if m and m[-1] else None
        vargs = ["--config", x.vf["config"] or cfg_path]
        if rot:
            vargs += ["--wind-rotation", rot]
        if x.vf["qc"]:
            vargs += ["--qc"]
        hoargs = ["--holdout-stations", x.holdout, "--subset", "heldout"] if x.holdout else []
        common = ["--obs-root", x.obs, "--obs-types", *x.obs_types]

        def one(states, truth, lead, heldout):
            args = ["--state", *[f"{k}={v}" for k, v in states.items()], *vargs]
            if heldout:
                args += hoargs
            args += [*common, *(["--truth-npz", truth] if truth else []),
                     "--csv", self.csv_ho if heldout else self.csv, "--meta", f"cycle={cycstr(t)}", f"lead={lead}"]
            log = d / ("verify_heldout.log" if heldout else "verify.log")
            before = log.stat().st_size if log.exists() else 0
            self.steps.run(f"verify lead {lead}{' heldout' if heldout else ''}", x.mod["verify"], args,
                           log, append=True, retry=False)       # a retry could append CSV rows twice
            if not heldout:
                with open(log, errors="replace") as f:
                    f.seek(before)
                    sec = section(f.read(), r"^VERIFICATION")
                if sec:
                    log_print(sec)

        truth0 = x.ref / cycstr(t) / f"hrrr_{ymd_hh(t)}.npz"
        one(states0, truth0, 0, False)
        if hoargs:
            one(states0, truth0, 0, True)
        for L in x.leads:
            tv = t + timedelta(hours=L)
            truth = x.ref / cycstr(tv) / f"hrrr_{ymd_hh(tv)}.npz"
            states = {s: d / f"fcst_{s}_f{L:02d}.npz" for s in ("an", "bg") if s in x.fc_from}
            if not states:
                continue
            one(states, truth if truth.exists() else None, L, False)
            if hoargs:
                one(states, truth if truth.exists() else None, L, True)

    # ------------------------------------------------------------------ all cycles
    def run(self):
        x = self.x
        x.exp.mkdir(parents=True, exist_ok=True)
        base = x.analysis_base()
        # record what this experiment runs with
        if x.path != (x.exp / "experiment.yaml").resolve():
            shutil.copyfile(x.path, x.exp / "experiment.yaml")
        save_yaml(base, x.exp / "base_config.yaml")
        (x.exp / "experiment_resolved.json").write_text(json.dumps(x.comparable(), indent=1, default=str))
        (x.exp / "code_version.txt").write_text(code_version(x.code, REPO_ROOT, x.deterministic) + "\n")

        log_print(f"LS-3DVar cycling {cycstr(x.start)} .. {cycstr(x.end)}   leads: {' '.join(map(str, x.leads)) or '-'}"
                  f"   holdout: {x.holdout or 'none'}   verify_qc: {int(bool(x.vf['qc']))}")
        log_print(f"  exp  {x.exp}\n  ref  {x.ref}\n  obs  {x.obs}")
        log_print(f"  code {code_version(x.code, REPO_ROOT, x.deterministic)}   ({x.mod['analysis']})")
        log_print(f"  background first={x.bg['first_cycle']} then={x.bg['source']}   blend {int(bool(x.bg['blend']['enabled']))}")
        log_print(f"  analysis {'on' if x.an['enabled'] else 'OFF'}   forecast {'on' if x.fc['enabled'] else 'OFF'}"
                  f"   verification {'on' if x.vf['enabled'] else 'OFF'}")

        first = True
        for t in hours(x.start, x.end):
            c = cycstr(t)
            d = x.exp / c
            d.mkdir(parents=True, exist_ok=True)
            self.steps.progress(cycle=c, step="start", state="running")
            log_print(f"\n================================ {c} ================================")
            ref_hrrr = x.ref / c / f"hrrr_{ymd_hh(t)}.npz"
            ref_gfs = x.ref / c / f"gfs_{ymd_hh(t)}.npz"
            if not (ref_hrrr.exists() and ref_gfs.exists()):
                raise StepFailed(f"missing reference {x.ref / c}")
            bg = self.background(t, d, first)
            raw = self.blend(t, d, bg)
            cfg_path = self.cycle_config(t, d, bg, base)
            an = self.analysis(t, d, cfg_path) if x.an["enabled"] else None
            self.forecasts(t, d, {"an": an, "bg": bg})
            done_marker = d / ".verified"
            if x.vf["enabled"] and self.have(done_marker):
                log_print("  (verification of this cycle already done: skipped on resume)")
            elif x.vf["enabled"]:
                states0 = {}
                if an is not None:
                    states0["an"] = an
                states0["bg"] = bg
                if raw is not None and nonempty(raw):
                    states0["raw"] = raw
                self.verify(t, d, states0, cfg_path)
                done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            first = False
        self.steps.progress(step="done", state="done")
        summary(x)
        return 0


# ============================================================================ summary
def summary(x):
    import pandas as pd
    for path in (x.exp / "verification.csv", x.exp / "verification_heldout.csv"):
        if not nonempty(path):
            continue
        d = pd.read_csv(path)
        print("\n#### " + ("HELD-OUT STATIONS ONLY (never assimilated)" if "heldout" in path.name else "ALL STATIONS"))
        d["lead"] = d["lead"].astype(int)
        d = d.drop_duplicates(["cycle", "lead", "state", "var"], keep="last")
        print("\n" + "=" * 90 + "\nSUMMARY (mean over cycles)   O-F rms vs obs   |   rms vs HRRR analysis (all)\n" + "=" * 90)
        for var in SUMMARY_VARS:
            s = d[d["var"] == var].pivot_table(index="lead", columns="state", values=["rms", "truth_all"], aggfunc="mean")
            if s.empty:
                continue
            print(f"\n{var}")
            print(f"{'lead':>4s} {'obs bg':>9s} {'obs an':>9s} {'impr%':>7s}   {'ref bg':>9s} {'ref an':>9s} {'impr%':>7s}")
            for L, r in s.iterrows():
                ob, oa = r.get(("rms", "bg"), float("nan")), r.get(("rms", "an"), float("nan"))
                tb, ta = r.get(("truth_all", "bg"), float("nan")), r.get(("truth_all", "an"), float("nan"))
                print(f"{L:4d} {ob:9.3f} {oa:9.3f} {100 * (ob - oa) / ob:+7.2f}   {tb:9.3f} {ta:9.3f} {100 * (tb - ta) / tb:+7.2f}")
        print(f"\n(cycles: {d['cycle'].nunique()}; csv: {path})")


# ============================================================================ commands
def cmd_check(x, quiet=False, pipeline=False, need_gpu=True):
    ok = True
    stg = ST.Stages(x, type("S", (), {"python": x.rt["python"] or sys.executable})())
    prep_on = stg.enabled_prep() if pipeline else []
    print(f"experiment : {x.name}   {cycstr(x.start)} .. {cycstr(x.end)}")
    print(f"yaml       : {x.path}")
    print(f"exp        : {x.exp}")
    print(f"ref        : {x.ref}")
    print(f"obs        : {x.obs}   types {x.obs_types}   holdout {x.holdout or 'none'}")
    print(f"code       : {code_version(x.code, REPO_ROOT, x.deterministic)}"
          f"{'   (uncommitted changes in the repo)' if git_dirty(REPO_ROOT) else ''}")
    b = x.bg
    print(f"background : first={b['first_cycle']} then={b['source']}"
          f"{'   forecast_dir=' + str(b['forecast_dir']) + f' (F{x.bg_lead:02d})' if 'forecast_dir' in (b['first_cycle'], b['source']) else ''}"
          f"   blend={'on ' + b['blend']['args'] if b['blend']['enabled'] else 'off'}")
    print(f"analysis   : {'on' if x.an['enabled'] else 'OFF'}   base {x.an['base_config']}"
          f"   overrides {x.an['overrides'] or '-'}   extra [{x.an['extra_args']}]")
    print(f"forecast   : {'on  leads ' + str(x.leads) + ' from ' + str(x.fc_from) if x.fc['enabled'] else 'OFF'}"
          f"{'   (+F01 from an for cycling)' if x.need_an_f01 and 1 not in x.leads else ''}")
    print(f"verify     : {'on  qc=' + str(bool(x.vf['qc'])) + '  config=' + str(x.vf['config'] or 'per-cycle') if x.vf['enabled'] else 'OFF'}")
    if x.an["enabled"]:
        if not Path(x.an["base_config"]).exists():
            print(f"  !! base_config not found: {x.an['base_config']}")
            return False
        l = x.analysis_base().get("ls3dvar", {})
        print("ls3dvar    :", {k: l.get(k) for k in ("control_vector", "nmodes", "corr_length", "sigma_b",
                                                     "outer_loops", "max_iterations")})
        for blk in l.get("blocks", []) or []:
            print("  block    ", blk)
    if x.vf["config"] and not Path(x.vf["config"]).exists():
        print(f"  !! verification.config not found: {x.vf['config']}")
        ok = False
    # inputs
    maxlead = max(x.leads) if x.leads else 0
    print("inputs     : ref = reference hrrr+gfs npz, obs = obsdb types present"
          + (", bgf = forecast_dir background" if "forecast_dir" in (b["first_cycle"], b["source"]) else ""))
    t, te = x.start - timedelta(hours=1), x.end + timedelta(hours=maxlead)
    missing_ref, missing_bgf = 0, 0
    while t <= te:
        c = cycstr(t)
        nref = sum((x.ref / c / f"{k}_{ymd_hh(t)}.npz").exists() for k in ("hrrr", "gfs"))
        nobs = sum((x.obs / ty / c).is_dir() for ty in x.obs_types)
        line = f"  {c}  ref:{nref}  obs:{nobs}/{len(x.obs_types)}"
        in_run = x.start <= t <= x.end
        if in_run and "forecast_dir" in (b["first_cycle"], b["source"]):
            uses = (b["source"] == "forecast_dir") or (t == x.start and b["first_cycle"] == "forecast_dir")
            if uses:
                f = Path(b["forecast_dir"]) / cycstr(t - timedelta(hours=x.bg_lead)) / f"fcst_f{x.bg_lead:02d}.npz"
                line += f"  bgf:{int(f.exists())}"
                missing_bgf += not f.exists()
        print(line)
        if nref < 2 and (x.start - timedelta(hours=1) <= t <= x.end):
            missing_ref += 1
        t += timedelta(hours=1)
    if missing_ref and "reference" not in prep_on:
        print("  !! reference files missing inside the cycling period: the run will stop there"
              + ("" if pipeline else "  (pipeline with prep.reference.enabled stages them)"))
        ok = False
    if missing_bgf and "background" not in prep_on:
        print("  !! forecast_dir backgrounds missing (run_forecasts.py, or prep.background.enabled)")
        ok = False
    lines, bad = stg.check_lines()
    for l in lines:
        print(l)
    if x.cfg["push"]["enabled"]:
        print(f"push       : {', '.join(f'{s} -> {d}' for s, d in ST.push_plan(x))}"
              + ("" if pipeline else "   (pipeline / push)"))
        pb = ST.push_check(x)
        if pipeline:
            bad += pb
        else:
            for b in pb:
                print(f"  (!) {b}")
    for b in bad:
        print(f"  !! {b}")
    if bad and pipeline:                     # prep / post problems matter when they will run
        ok = False
    # resume compatibility
    if x.resume:
        prob = compatibility(x)
        if prob:
            print(f"resume     : !! {x.exp} exists and differs:")
            for p in prob[:20]:
                print(f"               {p}")
            print("             use another experiment.name, experiment.resume: false, or --force")
            ok = False
        elif x.exp.exists() and any(x.exp.glob("[0-9]*T[0-9]*Z")):
            print("resume     : existing experiment matches; finished steps are skipped")
    # busy?
    pidf = x.exp / "driver.pid"
    if pidf.exists() and pid_alive(pidf.read_text().strip()):
        print(f"GPU        : !! this experiment is already running (pid {pidf.read_text().strip()})")
        ok = False
    elif need_gpu:
        busy = steps_running()
        if busy:
            print("GPU        : !! LS-3DVar steps are running:")
            for l in busy[:5]:
                print("              ", l[:150])
            ok = False
        else:
            print("GPU        : free")
    return ok


def dict_enabled(x):
    """x with push.enabled forced on (an explicit `push` command checks the push settings)"""
    y = type("X", (), {})()
    y.__dict__.update(x.__dict__)
    y.cfg = dict(x.cfg, push=dict(x.cfg["push"], enabled=True))
    return y


def cmd_run(x, a, mode="run"):
    if mode == "push" and not a.force:
        print("push       : " + ", ".join(f"{s} -> {d}" for s, d in ST.push_plan(x)))
        bad = ST.push_check(dict_enabled(x))
        for b in bad:
            print(f"  !! {b}")
        if bad:
            return 1
    elif not a.force and not cmd_check(x, pipeline=(mode != "run"), need_gpu=(mode not in ("post", "push"))):
        print("\ncheck failed: fix the !! items above, or --force")
        return 1
    pidf = x.exp / "driver.pid"
    if pidf.exists() and pid_alive(pidf.read_text().strip()) and int(pidf.read_text()) != os.getpid():
        print(f"already running (pid {pidf.read_text().strip()})")
        return 1
    x.exp.mkdir(parents=True, exist_ok=True)
    log = x.exp / "run.log"
    if a.detach:
        cmd = [sys.executable, "-m", "ls3dvar.cli.run_cycles", mode, str(x.path), "--force", "--attached-child"]
        with open(log, "a") as f:
            p = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        pidf.write_text(str(p.pid))
        print(f"started pid {p.pid}   log: {log}")
        print(f"follow: python -m ls3dvar.cli.run_cycles tail {x.path}")
        return 0
    pidf.write_text(str(os.getpid()))
    cyc = None
    try:
        cyc = Cycler(x, force=a.force)
        if mode == "push":
            return ST.run_push(x, cyc.steps, echo=log_print, dry_run=getattr(a, "dry_run", False))
        if mode in ("prep", "post", "pipeline"):
            stg = ST.Stages(x, cyc.steps, echo=log_print)
            if mode == "post":
                return 1 if stg.run_post(getattr(a, "only", None)) else 0
            stg.run_prep(getattr(a, "only", None) if mode == "prep" else None,
                         dry_run=getattr(a, "dry_run", False))
            if mode == "prep":
                cyc.steps.progress(cycle="prep", step="done", state="done")
                return 0
            rc = cyc.run()
            failed = stg.run_post()
            cyc.steps.progress(cycle="post", step="done", state="done" if not failed else "post failed")
            if x.cfg["push"]["enabled"] and not rc:
                ST.run_push(x, cyc.steps, echo=log_print)      # also after a failed plot: data first
                cyc.steps.progress(cycle="push", step="done", state="done")
            return rc or (1 if failed else 0)
        return cyc.run()
    except StepFailed as e:
        print(f"\n!! {e}", flush=True)
        if cyc is not None:
            cyc.steps.progress(state="failed", error=str(e))
        return 1
    finally:
        if pidf.exists() and pidf.read_text().strip() == str(os.getpid()):
            pidf.unlink()


def cmd_status(x):
    pidf = x.exp / "driver.pid"
    alive = pidf.exists() and pid_alive(pidf.read_text().strip())
    print("RUNNING" if alive else "not running", f"   {x.exp}")
    cv = x.exp / "code_version.txt"
    if cv.exists():
        print("code:", cv.read_text().strip())
    pf = x.exp / "progress.json"
    if pf.exists():
        p = json.loads(pf.read_text())
        print(f"cycle {p.get('cycle')}   step {p.get('step')}   {p.get('state')}   "
              f"try {p.get('attempt', '-')}   updated {p.get('updated')}")
        if alive and p.get("state") == "running" and p.get("log") and Path(p["log"]).exists():
            tail = [l for l in Path(p["log"]).read_text(errors="replace").splitlines() if l.strip()][-4:]
            print(f"last lines of {p['log']}:")
            for l in tail:
                print("  | " + l[:150])
    log = x.exp / "run.log"
    if log.exists():
        lines = [l for l in log.read_text(errors="replace").splitlines()
                 if re.search(r"^={5,} |^[0-9:]{8} \[|failed|FAILED|TIMEOUT|SUMMARY|^\s*!!", l)]
        print("\n".join(lines[-10:]))
    cycles = sorted(p for p in x.exp.glob("[0-9]*T[0-9]*Z") if p.is_dir()) if x.exp.exists() else []
    if cycles:
        files = sorted((f for f in cycles[-1].iterdir() if f.is_file()), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        print(f"newest files in {cycles[-1]}:")
        for f in files:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(f.stat().st_mtime))}  {f.name}")
    return 0


def cmd_stop(x):
    pidf = x.exp / "driver.pid"
    if not pidf.exists() or not pid_alive(pidf.read_text().strip()):
        print("not running")
        return 0
    pid = int(pidf.read_text())
    os.kill(pid, signal.SIGTERM)                 # the driver kills its running step, then exits
    for _ in range(40):
        if not pid_alive(pid):
            break
        time.sleep(1)
    if pid_alive(pid):
        print("still alive -> kill -9")
        os.kill(pid, signal.SIGKILL)
        pf = x.exp / "progress.json"
        if pf.exists():
            sp = json.loads(pf.read_text()).get("step_pid")
            if sp:
                try:
                    os.killpg(int(sp), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    pidf.unlink(missing_ok=True)
    print("stopped")
    return 0


def cmd_init(a):
    base = load_yaml(a.from_config)
    cy = base.get("cycling", {}) or {}
    data = cy.get("data", "/home/jonggyunkim/runs/s3/HRRRCast-live")
    holdout = cy.get("holdout_stations")
    exp = {
        "experiment": {"name": a.name, "start": a.start, "end": a.end,
                       "output_dir": "${data}/experiments", "resume": True,
                       "code": "ls3dvar", "deterministic": False},
        "paths": {"data": data,
                  "reference_dir": cy.get("reference_dir", "${data}/reference/hrrr_ana"),
                  "obsdb_root": cy.get("obsdb_root", "${data}/obsdb")},
        "background": {"first_cycle": "reference_forecast", "source": "cycled", "forecast_dir": None,
                       "lead": 1,
                       "blend": {"enabled": bool(cy.get("blend", False)),
                                 "args": cy.get("blend_args", "--cutoff-km 1000 --pmax 850 --weight 1.0")}},
        "observations": {"types": cy.get("obs_types", ["asos", "tempest", "grenade"]),
                         "holdout_stations": a.holdout_stations if a.holdout_stations is not None else holdout},
        "analysis": {"enabled": True, "base_config": str(Path(a.from_config).resolve()),
                     "overrides": {}, "extra_args": "", "truth": True},
        "forecast": {"enabled": True, "leads": cy.get("leads", [1, 2, 3]), "from": ["an", "bg"]},
        "verification": {"enabled": True, "qc": bool(cy.get("verify_qc", True)),
                         "config": cy.get("verify_config")},
        "runtime": {"step_timeout": cy.get("step_timeout", 900), "analysis_timeout": 1800, "retries": 3, "retry_wait": 20,
                    "python": None, "prep_timeout": 0, "post_timeout": 0,
                    "env": {"CUDA_CACHE_MAXSIZE": "4294967296", "TF_FORCE_GPU_ALLOW_GROWTH": "true"}},
        "prep": json.loads(json.dumps(ST.PREP_DEFAULTS)),
        "post": json.loads(json.dumps(ST.POST_DEFAULTS)),
    }
    text = (f"# experiment generated by run_cycles.py init from {a.from_config}\n"
            "# (its cycling: section; run_experiment.sh overrides such as HOLDOUT_STATIONS=0/5 are NOT\n"
            "#  applied, set them here). Every key: ls3dvar/configs/experiment_example.yaml\n")
    import yaml
    text += yaml.safe_dump(exp, sort_keys=False)
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


# ============================================================================ import-old
OLD_HEADER = {   # cycle_ls3dvar.sh prints these lines at the top of $DATA/$NAME.out
    "period": re.compile(r"^LS-3DVar cycling (\S+) \.\. (\S+)\s+leads: (.*?)\s+holdout: (\S+)\s+verify_qc: (\S+)"),
    "cfg": re.compile(r"^  cfg  (.+)$"),
    "ref": re.compile(r"^  ref  (.+)$"),
    "obs": re.compile(r"^  obs  (.+)$"),
    "extra": re.compile(r"^  args from config\s*(?:\+ extra: (.*?))?\s*$"),
    "blend": re.compile(r"^  blend (\d)\s*(.*)$"),
}
SETTINGS_VARS = ("NAME", "START", "END", "DATA", "REPO", "CONFIG", "LEADS", "HOLDOUT_STATIONS", "VERIFY_QC",
                 "VERIFY_CONFIG", "BLEND", "STEP_TIMEOUT", "LS3DVAR_ARGS")


def old_settings(script):
    """the SETTINGS block of an old run_experiment.sh, evaluated by bash in a clean environment"""
    text = Path(script).read_text(errors="replace")
    m = re.search(r"^# =+ SETTINGS =+\s*$(.*?)^# =+ END SETTINGS =+\s*$", text, re.S | re.M)
    if not m:
        return {}
    block = "\n".join(l for l in m.group(1).splitlines() if not l.lstrip().startswith("export "))
    prog = block + "\n" + "".join(f'printf "%s\\0" "${{{v}-}}"\n' for v in SETTINGS_VARS)
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
    vals = r.stdout.split("\0")
    return {k: v for k, v in zip(SETTINGS_VARS, vals)}


def old_log_header(log):
    """what cycle_ls3dvar.sh reported it ran with (the last run's header in the log)"""
    out = {}
    if not Path(log).exists():
        return out
    for line in Path(log).read_text(errors="replace").splitlines():
        for key, rx in OLD_HEADER.items():
            m = rx.match(line)
            if not m:
                continue
            if key == "period":
                out.update(start=m.group(1), end=m.group(2), leads=m.group(3).split(), holdout=m.group(4),
                           verify_qc=m.group(5))
            elif key == "extra":
                out["extra"] = (m.group(1) or "").strip()
            elif key == "blend":
                out["blend"], out["blend_args"] = m.group(1), m.group(2).strip()
            else:
                out[key] = m.group(1).strip()
        if line.startswith("================================ "):
            break                                          # header ends at the first cycle
    return out


def cmd_import_old(a):
    old = Path(a.old_experiment).resolve()
    if not old.is_dir():
        raise SystemExit(f"not a folder: {old}")
    base_cfg = old / "base_config.yaml"
    if not base_cfg.exists():
        raise SystemExit(f"{base_cfg} missing (written by cycle_ls3dvar.sh at the start of a run)")
    st = old_settings(old / "run_experiment.sh") if (old / "run_experiment.sh").exists() else {}
    data = Path(st.get("DATA") or old.parent.parent)
    log = data / f"{old.name}.out"
    hd = old_log_header(log)
    cy = (load_yaml(base_cfg).get("cycling") or {})
    notes, src = [], {}

    def pick(key, *cands):
        for origin, v in cands:
            if v not in (None, ""):
                src[key] = origin
                return v
        src[key] = "default"
        return None

    cycles = sorted(p.name for p in old.glob("[0-9]*T[0-9]*Z") if p.is_dir())
    # what ran: the log, else the cycle folders; the copied settings block may hold stale defaults
    start = pick("start", ("log", hd.get("start")), ("cycle folders", cycles[0] if cycles else None),
                 ("settings", st.get("START")))
    end = pick("end", ("log", hd.get("end")), ("cycle folders", cycles[-1] if cycles else None),
               ("settings", st.get("END")))
    if cycles and start and end:
        first, last = cycstr(parse_time(start)), cycstr(parse_time(end))
        if (first, last) != (cycles[0], cycles[-1]):
            notes.append(f"period {first}..{last} ({src['start']}) but cycle folders {cycles[0]}..{cycles[-1]}")
    if st.get("START") and start and cycstr(parse_time(st["START"])) != cycstr(parse_time(start)):
        notes.append(f"the copied run_experiment.sh says START={st['START']} (stale default?); "
                     f"using {start} from the {src['start']}")
    leads = pick("leads", ("log", hd.get("leads")), ("settings", (st.get("LEADS") or "").split()),
                 ("config", cy.get("leads"))) or [1, 2, 3]
    hold = pick("holdout", ("log", hd.get("holdout")), ("settings", st.get("HOLDOUT_STATIONS")),
                ("config", cy.get("holdout_stations")))
    hold = None if hold in (None, "", "none") else str(hold)
    vqc = pick("verify_qc", ("log", hd.get("verify_qc")), ("settings", st.get("VERIFY_QC")),
               ("config", cy.get("verify_qc")))
    extra = hd.get("extra") if "extra" in hd else st.get("LS3DVAR_ARGS", "")
    src["extra_args"] = "log" if "extra" in hd else ("settings" if st else "default")
    blend = pick("blend", ("log", hd.get("blend")), ("settings", st.get("BLEND")), ("config", cy.get("blend")))
    blend_args = hd.get("blend_args") or cy.get("blend_args") or DEFAULTS["background"]["blend"]["args"]
    ref = pick("reference_dir", ("log", hd.get("ref")), ("config", cy.get("reference_dir"))) or f"{data}/reference/hrrr_ana"
    obsr = pick("obsdb_root", ("log", hd.get("obs")), ("config", cy.get("obsdb_root"))) or f"{data}/obsdb"
    types = cy.get("obs_types", DEFAULTS["observations"]["types"])
    vcfg = st.get("VERIFY_CONFIG") or cy.get("verify_config") or None
    if not hd:
        notes.append(f"no run log {log}: settings from the copied run_experiment.sh; values passed through "
                     "the environment at launch (NAME=... ./run_experiment.sh run) are not visible there")
    if st.get("REPO") and Path(st["REPO"]).resolve() != REPO_ROOT:
        notes.append(f"the old run used repo {st['REPO']}; this driver runs {REPO_ROOT}")

    exp = {
        "experiment": {"name": a.name or f"{old.name}_py", "start": start, "end": end,
                       "output_dir": str(old.parent), "resume": True, "code": "ls3dvar",
                       "deterministic": False},
        "paths": {"data": str(data), "reference_dir": ref, "obsdb_root": obsr},
        "background": {"first_cycle": "reference_forecast", "source": "cycled", "forecast_dir": None, "lead": 1,
                       "blend": {"enabled": str(blend) in ("1", "True", "true"), "args": blend_args}},
        "observations": {"types": types, "holdout_stations": hold},
        "analysis": {"enabled": True, "base_config": str(base_cfg), "overrides": {},
                     "extra_args": extra or "", "truth": True},
        "forecast": {"enabled": True, "leads": [int(L) for L in leads], "from": ["an", "bg"]},
        "verification": {"enabled": True, "qc": str(vqc) in ("1", "True", "true", "None"), "config": vcfg},
        "runtime": {"step_timeout": int(st.get("STEP_TIMEOUT") or cy.get("step_timeout") or 900),
                    "analysis_timeout": max(1800, int(st.get("STEP_TIMEOUT") or cy.get("step_timeout") or 900)),
                    "retries": 3, "retry_wait": 20, "python": None,
                    "env": dict(DEFAULTS["runtime"]["env"])},
    }
    import yaml
    head = [f"# imported from {old}",
            f"#   analysis config: {base_cfg} (the copy that run used)",
            f"#   run log: {log if hd else '(none)'}",
            "#   sources: " + ", ".join(f"{k}={v}" for k, v in src.items())]
    head += [f"#   NOTE: {n}" for n in notes]
    head += ["#   The old run was not deterministic and used other code: compare its scores with this",
             "#   one approximately (run-to-run noise was up to ~0.4 points of improvement)."]
    text = "\n".join(head) + "\n" + yaml.safe_dump(exp, sort_keys=False)
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
        for h in head[1:]:
            print(h, file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "status", "tail", "stop", "results"):
        p = sub.add_parser(name)
        p.add_argument("experiment", help="experiment YAML (results: also an experiment folder)")
    r = sub.choices["run"]
    r.add_argument("--detach", action="store_true", help="run in the background (log: <exp>/run.log)")
    r.add_argument("--force", action="store_true", help="run even if check reports problems")
    r.add_argument("--attached-child", action="store_true", help=argparse.SUPPRESS)
    for name, parts in (("prep", ST.PREP_PARTS), ("post", ST.POST_PARTS), ("pipeline", None), ("push", None)):
        q = sub.add_parser(name)
        q.add_argument("experiment", help="experiment YAML")
        q.add_argument("--detach", action="store_true", help="run in the background (log: <exp>/run.log)")
        q.add_argument("--force", action="store_true", help="run even if check reports problems")
        q.add_argument("--attached-child", action="store_true", help=argparse.SUPPRESS)
        if parts:
            q.add_argument("--only", nargs="+", choices=parts, help="only these parts")
        if name in ("prep", "push"):
            q.add_argument("--dry-run", action="store_true", help="list what would run / upload, change nothing")
    i = sub.add_parser("init", help="write an experiment YAML from an existing analysis/cycling config")
    i.add_argument("--from", dest="from_config", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--start", default="2026-08-11T00")
    i.add_argument("--end", default="2026-08-11T03")
    i.add_argument("--holdout-stations", default=None)
    i.add_argument("--out", default=None)
    o = sub.add_parser("import-old", help="experiment YAML that repeats an experiment of the old bash scripts")
    o.add_argument("old_experiment", help="folder of the old experiment (with base_config.yaml)")
    o.add_argument("--name", default=None, help="new experiment name (default: <old name>_py)")
    o.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd in ("init", "import-old", "check", "status", "results", "tail"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `... | head` ends quietly, no traceback
    if a.cmd == "init":
        return cmd_init(a)
    if a.cmd == "import-old":
        return cmd_import_old(a)
    if a.cmd == "results" and Path(a.experiment).is_dir():       # any experiment folder, also old ones
        summary(type("X", (), {"exp": Path(a.experiment)})())
        return 0
    x = Experiment(a.experiment)
    if a.cmd == "check":
        return 0 if cmd_check(x) else 1
    if a.cmd == "run":
        return cmd_run(x, a)
    if a.cmd in ("prep", "post", "pipeline", "push"):
        if a.cmd == "push" and not x.cfg["push"]["enabled"] and not x.cfg["push"]["dest"]:
            raise SystemExit("set push.dest (s3://bucket/prefix) in the experiment YAML")
        if a.cmd == "prep" and getattr(a, "dry_run", False):
            a.force, a.detach = True, False
        return cmd_run(x, a, mode=a.cmd)
    if a.cmd == "status":
        return cmd_status(x)
    if a.cmd == "tail":
        log = x.exp / "run.log"
        os.execvp("tail", ["tail", "-n", "40", "-f", str(log)])
    if a.cmd == "stop":
        return cmd_stop(x)
    if a.cmd == "results":
        summary(x)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
