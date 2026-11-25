#!/usr/bin/env python3
"""
ls3dvar/tools/inventory.py

Read-only inventory of the mantari package for the LS-3DVar cleanup. Nothing is
moved or deleted.

    python -m ls3dvar.tools.inventory --root . \
        [--entry mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps ...] \
        [--csv inventory.csv]

Reports
  1. modules reachable from the entry points (static import graph via ast)
  2. modules NOT reachable -> candidates for mantari/legacy/
  3. top-level class / function names defined in more than one module
  4. per-file smells: hard-coded /home paths, CUSTOM_OBJECTS copies,
     sys.path.insert, np.save to the working directory (debug dumps)

Limits: only static imports are followed (no importlib / __import__ strings), and
`from pkg import name` counts as an import of pkg.name when pkg/name.py exists.
"""
import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ENTRIES = [
    "mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps",
    "mantari.workflows.make_forecast_background",
    "mantari.workflows.verify_state",
    "mantari.workflows.pool_station_reports",
    "mantari.ls3dvar.large_scale_blend",
    "mantari.workflows.test_encode_decode_roundtrip",
    "mantari.workflows.test_decode_parity",
]

SMELLS = {
    "home_path": re.compile(r"/home/\w+"),
    "custom_objects": re.compile(r"^CUSTOM_OBJECTS\s*=\s*\{", re.M),
    "sys_path_insert": re.compile(r"sys\.path\.insert"),
    "cwd_np_save": re.compile(r"np\.save\(\s*f?[\"'](?!/)[\w./{}:]+[\"']"),
}


def module_name(root, path):
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def collect(root, package):
    mods = {}
    for p in sorted((root / package).rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            continue
        mods[module_name(root, p)] = p
    return mods


def imports_of(path, mods, this):
    is_pkg = path.name == "__init__.py"
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError as e:
        print(f"  WARNING syntax error in {path}: {e}", file=sys.stderr)
        return set(), {}
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                                   # relative import
                parts = this.split(".")
                drop = node.level - 1 if is_pkg else node.level
                base = parts[: len(parts) - drop]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            for a in node.names:
                out.add(f"{mod}.{a.name}")
    found = set()
    for m in out:
        while m:                                             # a.b.c -> a.b.c, a.b, a
            if m in mods:
                found.add(m)
            m = m.rpartition(".")[0]
    defs = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
            defs[node.name] = node.lineno
    return found, defs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--package", default="mantari")
    ap.add_argument("--entry", nargs="+", default=DEFAULT_ENTRIES)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    mods = collect(root, a.package)
    graph, defs = {}, {}
    for m, p in mods.items():
        graph[m], defs[m] = imports_of(p, mods, m)

    missing = [e for e in a.entry if e not in mods]
    for e in missing:
        print(f"WARNING entry point not found: {e}")
    reach, todo = set(), [e for e in a.entry if e in mods]
    while todo:
        m = todo.pop()
        if m in reach:
            continue
        reach.add(m)
        todo.extend(graph[m] - reach)
    # importing a.b.c also runs a/__init__ and a/b/__init__
    for m in list(reach):
        parts = m.split(".")
        for i in range(1, len(parts)):
            if ".".join(parts[:i]) in mods:
                reach.add(".".join(parts[:i]))

    users = defaultdict(set)
    for m, deps in graph.items():
        for d in deps:
            users[d].add(m)

    print(f"{len(mods)} modules under {a.package}/   reachable from entries: {len(reach)}")
    print("\n== REACHABLE (keep) ==")
    for m in sorted(reach):
        print(f"  {m}")
    print("\n== NOT REACHABLE (legacy candidates; 'used by' lists other unreachable importers) ==")
    for m in sorted(set(mods) - reach):
        u = sorted(users[m])
        print(f"  {m:60s} used by: {', '.join(u[:4]) + (' ...' if len(u) > 4 else '') or '-'}")

    print("\n== NAMES DEFINED IN MORE THAN ONE MODULE ==")
    where = defaultdict(list)
    for m, d in defs.items():
        for name in d:
            where[name].append(m)
    skip = {"main", "load_config", "banner", "rms", "stats", "compare", "summarize"}
    for name, ms in sorted(where.items()):
        if len(ms) > 1 and name not in skip:
            print(f"  {name:32s} {len(ms):3d}  {', '.join(sorted(ms)[:5])}{' ...' if len(ms) > 5 else ''}")
    common = {n: len(ms) for n, ms in where.items() if n in skip and len(ms) > 1}
    if common:
        print("  (boilerplate helpers: " + ", ".join(f"{n} x{c}" for n, c in sorted(common.items())) + ")")

    print("\n== SMELLS ==")
    rows = []
    for m, p in mods.items():
        txt = p.read_text(errors="replace")
        hit = {k: len(r.findall(txt)) for k, r in SMELLS.items()}
        rows.append(dict(module=m, path=str(p.relative_to(root)), reachable=m in reach,
                         lines=txt.count("\n") + 1, imported_by=len(users[m]), **hit))
    tot = {k: sum(1 for r in rows if r[k]) for k in SMELLS}
    print("  files with: " + "  ".join(f"{k}={v}" for k, v in tot.items()))
    for r in sorted(rows, key=lambda r: (not r["reachable"], r["module"])):
        flags = [k for k in SMELLS if r[k]]
        if flags and r["reachable"]:
            print(f"  [keep] {r['module']:55s} {', '.join(flags)}")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["module"]))
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
