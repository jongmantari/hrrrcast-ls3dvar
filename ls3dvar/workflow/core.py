"""
ls3dvar/workflow/core.py

Shared pieces of the experiment drivers:

    Steps         runs one module per process: timeout, retries, process-group kill,
                  deterministic launch (tools/det_runner.py), stop-signal handling
    MODULES       which modules a step runs (code: ls3dvar | mantari)
    times         cycle names / tags, hour ranges
    config        ${name} expansion, deep merge, YAML I/O
    code_version  "<code> <commit> deterministic=<0|1>"  (same format as the bash scripts)
"""
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ls3dvar.paths import REPO_ROOT  # noqa: F401  (re-exported for the drivers)

MODULES = {
    "ls3dvar": {"forecast": "ls3dvar.cli.make_forecast_background",
                "blend": "ls3dvar.cli.large_scale_blend",
                "analysis": "ls3dvar.cli.run_analysis",
                "verify": "ls3dvar.cli.verify_state"},
    "mantari": {"forecast": "mantari.workflows.make_forecast_background",
                "blend": "mantari.ls3dvar.large_scale_blend",
                "analysis": "mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps",
                "verify": "mantari.workflows.verify_state"},
}


# ============================================================================ times
def parse_time(s):
    """'2026-08-11T00', '2026-08-11T00Z', '2026-08-11 00:00', '20260811T00Z' -> aware UTC datetime"""
    s = str(s).strip().rstrip("Z").replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y%m%dT%H", "%Y%m%d%H"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"cannot parse time {s!r} (use YYYY-MM-DDTHH)")


def cycstr(t):
    """20260811T00Z: cycle folder names (obsdb, reference, experiments)"""
    return t.strftime("%Y%m%dT%HZ")


def ymd_hh(t):
    """20260811_00: npz file tags (hrrr_20260811_00.npz)"""
    return t.strftime("%Y%m%d_%H")


def hours(start, end, every=1):
    t = start
    while t <= end:
        yield t
        t += timedelta(hours=every)


# ============================================================================ config
_VAR = re.compile(r"\$\{([A-Za-z_][\w.]*)\}")


def _sub(value, names, strict):
    def one(m):
        k = m.group(1)
        if k in names:
            return str(names[k])
        if k in os.environ:
            return os.environ[k]
        return m.group(0)
    for _ in range(20):
        new = _VAR.sub(one, value)
        if new == value:
            break
        value = new
    if strict and _VAR.search(value):
        raise ValueError(f"unresolved ${{...}} in {value!r} (known names: {sorted(names)})")
    return value


def expand(value, names, strict=True):
    """replace ${name} by names[name] (or the environment), recursively, in every string"""
    if isinstance(value, dict):
        return {k: expand(v, names, strict) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, names, strict) for v in value]
    return _sub(value, names, strict) if isinstance(value, str) else value


def resolve_paths(paths):
    """expand the paths section against itself (paths may refer to each other, in any order)"""
    out = dict(paths or {})
    for _ in range(20):
        new = expand(out, out, strict=False)
        if new == out:
            break
        out = new
    return expand(out, out, strict=True)


def deep_merge(base, over):
    """dicts merge recursively; anything else in `over` replaces"""
    out = dict(base or {})
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# ============================================================================ code version
def git_commit(repo):
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def git_dirty(repo):
    try:
        r = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())
    except OSError:
        return False


def code_version(code, repo, deterministic):
    """same format as scripts/cycle_ls3dvar.sh writes into code_version.txt"""
    return f"{code} {git_commit(repo)} deterministic={int(bool(deterministic))}"


# ============================================================================ steps
class StepFailed(RuntimeError):
    pass


class Steps:
    """
    Run one module per process, as `python -m MODULE args` (or through det_runner.py).

    * timeout: the whole process group gets SIGTERM, then SIGKILL after `kill_after` s
    * retries: a failed or timed-out step is retried (cycle_ls3dvar.sh: 3 tries, 20 s apart)
    * stop:    SIGTERM / SIGINT to the driver kills the running step's process group
    * progress: the running step is recorded in progress.json (for `status` / `stop`)
    """

    def __init__(self, repo, python=None, deterministic=True, timeout=900, retries=3,
                 retry_wait=20, kill_after=30, env=None, progress_file=None, echo=print):
        self.repo = Path(repo)
        self.python = python or sys.executable
        self.deterministic = bool(deterministic)
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_wait = retry_wait
        self.kill_after = kill_after
        self.progress_file = Path(progress_file) if progress_file else None
        self.echo = echo
        self.env = dict(os.environ)
        self.env.update({k: str(v) for k, v in (env or {}).items()})
        self.env["PYTHONUNBUFFERED"] = "1"         # step logs fill line by line (live progress)
        if self.deterministic:
            self.env.update(TF_DETERMINISTIC_OPS="1", TF_CUDNN_DETERMINISTIC="1")
        self.runner = self.repo / "ls3dvar" / "tools" / "det_runner.py"
        if self.deterministic and not self.runner.exists():
            raise SystemExit(f"deterministic runs need {self.runner}")
        self.current = None
        self.context = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._stop)

    # --------------------------------------------------------------------- internals
    def _stop(self, signum, frame):
        if self.current is not None:
            self._kill(self.current)
        self.progress(step="stopped", state="stopped")
        raise SystemExit(128 + signum)

    def _kill(self, p):
        for sig, wait in ((signal.SIGTERM, self.kill_after), (signal.SIGKILL, 5)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                return
            try:
                p.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    def command(self, module, args):
        launch = [self.python, str(self.runner)] if self.deterministic else [self.python, "-m"]
        return [*launch, module, *[str(a) for a in args]]

    def progress(self, **kw):
        if not self.progress_file:
            return
        self.context.update(kw)
        self.context.update(pid=os.getpid(), updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = self.progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.context, indent=1))
        tmp.replace(self.progress_file)

    # --------------------------------------------------------------------- public
    def _echo_new(self, log, pos, partial, watch, prefix, final=False):
        """echo the lines added to `log` since byte `pos` that match `watch`"""
        try:
            with open(log, "rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            return pos, partial
        text = partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        partial = "" if final else lines.pop()          # keep an unfinished last line for later
        stamp = time.strftime("%H:%M:%S")             # when the line reached the log (+- poll interval)
        for l in lines:
            if l.strip() and watch.search(l):
                self.echo(f"{prefix}{stamp} {l.rstrip()}")
        return pos, partial

    def run(self, name, module, args, log, append=False, retry=True, watch=None, prefix="    ",
            poll=1.0, timeout=None, timeout_name="runtime.step_timeout", cmd=None, cwd=None, env=None):
        """
        run MODULE ARGS with stdout+stderr into `log`; raises StepFailed after the last try.
        watch: compiled regex; matching log lines are echoed while the step runs.
        cmd:   a full command line instead of MODULE ARGS (scripts: prep / post stages);
               cwd / env: its working directory and extra environment variables.
        Returns the run time [s] of the successful try.
        """
        cmd = [str(c) for c in cmd] if cmd is not None else self.command(module, args)
        penv = dict(self.env, **{k: str(v) for k, v in (env or {}).items()})
        tries = self.retries if retry else 1
        limit = self.timeout if timeout is None else timeout
        for attempt in range(1, tries + 1):
            t0 = time.time()
            with open(log, "a" if append else "w") as f:
                f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
                f.flush()
                pos, partial = f.tell(), ""
                p = subprocess.Popen(cmd, cwd=cwd or self.repo, stdout=f, stderr=subprocess.STDOUT,
                                     env=penv, start_new_session=True)
                self.current = p
                self.progress(step=name, step_pid=p.pid, attempt=attempt, log=str(log), state="running")
                rc = None
                try:
                    while True:
                        try:
                            rc = p.wait(timeout=poll if watch is not None else limit)
                            break
                        except subprocess.TimeoutExpired:
                            if watch is not None:
                                pos, partial = self._echo_new(log, pos, partial, watch, prefix)
                            if limit and time.time() - t0 > limit:
                                self._kill(p)
                                rc = None
                                break
                finally:
                    self.current = None
                if watch is not None:
                    self._echo_new(log, pos, partial, watch, prefix, final=True)
            if rc == 0:
                return time.time() - t0
            ran = fmt_seconds(time.time() - t0)
            if rc is None:
                self.echo(f"  {time.strftime('%H:%M:%S')} TIMEOUT (try {attempt}/{tries}): {name} killed after {ran}"
                          f" (limit {timeout_name} = {limit} s; raise it in the experiment YAML"
                          f" if the step is just slow) -> {log}")
            else:
                self.echo(f"  {time.strftime('%H:%M:%S')} FAILED (try {attempt}/{tries}): {name} exit {rc}"
                          f" after {ran} -> {log}")
            if attempt < tries:
                time.sleep(self.retry_wait)
        raise StepFailed(f"{name} failed after {tries} tries: see {log}")


def fmt_seconds(s):
    s = int(round(s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False
