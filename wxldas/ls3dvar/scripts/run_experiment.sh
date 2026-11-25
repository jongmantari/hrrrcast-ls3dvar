#!/bin/bash
# =============================================================================
# ls3dvar/scripts/run_experiment.sh — one LS-3DVar cycling experiment with the
# stand-alone ls3dvar package: settings + run + monitor
#
#   1. copy this file per experiment and edit the SETTINGS block; NAME, START, END,
#      CONFIG, LS3DVAR_ARGS, PKG, DETERMINISTIC can also come from the environment:
#        NAME=refc_insert CONFIG=$DATA/configs/refc/refc_insert.yaml ./run_experiment.sh run
#   2. ./run_experiment.sh check       # settings, inputs present, code, GPU free?
#      ./run_experiment.sh run         # start in background (nohup), log -> $DATA/$NAME.out
#      ./run_experiment.sh status      # current cycle / step / newest files
#      ./run_experiment.sh tail        # follow the log (Ctrl-C stops only the tail)
#      ./run_experiment.sh stop        # kill the cycling and its python steps
#      ./run_experiment.sh resume      # restart; finished steps are skipped
#      ./run_experiment.sh results     # held-out + all-station summaries
#      ./run_experiment.sh plot        # figures (verification on held-out stations, precip)
#      ./run_experiment.sh diff OTHER  # compare every npz/csv with experiments/OTHER (exact)
#
# Settings left "" fall back to the config's `cycling:` section; the analysis
# itself is configured by the config's `ls3dvar:` section (LS3DVAR_ARGS adds
# command-line overrides on top).
#
# Each experiment records its code in $EXP/code_version.txt ("<pkg> <commit>
# deterministic=<0|1>"). run/resume refuse to continue an experiment that was made
# with different code or a different DETERMINISTIC setting, so results of two code
# versions are never mixed. FORCE=1 overrides this (for example, after a commit that
# changes nothing numerically).
# =============================================================================
set -uo pipefail

# ============================== SETTINGS =====================================
# ---- experiment -------------------------------------------------------------
NAME=${NAME:-ls3dvar_v1}                # label: experiments/$NAME and $DATA/$NAME.out
START=${START:-2026-08-11T00}           # first cycle (YYYY-MM-DDTHH, UTC)
END=${END:-2026-08-11T03}               # last cycle

# ---- paths ------------------------------------------------------------------
DATA=/home/jonggyunkim/runs/s3/HRRRCast-live
REPO=/home/jonggyunkim/runs/s3/wxbox_cylc/external/HRRRCast-live     # contains ls3dvar/
CONFIG=${CONFIG:-$DATA/configs/frozen_v1/test.yaml}  # ls3dvar: + cycling: settings
REF=${REF:-""}                          # reference HRRR analyses; "" = config cycling.reference_dir,
                                        #   else <cycling.data or $DATA>/reference/hrrr_ana
OBS=${OBS:-""}                          # obsdb root; "" = config cycling.obsdb_root, else <...>/obsdb

# ---- code -------------------------------------------------------------------
PKG=${PKG:-ls3dvar}                     # ls3dvar (package) | mantari (old modules)
DETERMINISTIC=${DETERMINISTIC:-0}       # 0 = plain TF (default); 1 = bit-reproducible (~1.5x slower)
                                        #   (run-to-run noise up to ~0.4 points of improvement)

# ---- overrides of the config's cycling: section ("" = use the config) -------
EXP=$DATA/experiments/$NAME            # output directory
LEADS="1 2 3"                          # forecast leads verified [h]
HOLDOUT_STATIONS=0/5                   # fixed never-assimilated stations; "none" = assimilate all
VERIFY_QC=1                            # 1 = verify with the analysis QC (same obs for an and bg)
VERIFY_CONFIG=""                       # config used for scoring; set the same file for every
                                       #   experiment you want to compare (e.g. $DATA/configs/base_new.yaml)
BLEND=0                                # 1 = large-scale GFS blending of the background
STEP_TIMEOUT=${STEP_TIMEOUT:-1800}     # [s] a hung GPU step is killed and retried (3 tries);
                                        #   the analysis takes ~3.5 min on AC power, ~10x on battery
RESUME=1                               # 1 = skip steps whose outputs exist, 0 = redo everything

# ---- analysis overrides on top of ls3dvar: in the config (normally empty) ---
LS3DVAR_ARGS=${LS3DVAR_ARGS:-""}        # e.g. "--sigma-b 5" or "--block T2M:0.6:10:land"

# ---- environment ------------------------------------------------------------
CONDA_ENV=hrrrcast
export CUDA_CACHE_MAXSIZE=4294967296   # PTX-JIT cache for the Blackwell GPU (avoids long stalls)
export TF_FORCE_GPU_ALLOW_GROWTH=true
# ============================ END SETTINGS ===================================

LOG=$DATA/$NAME.out
PLOT_TEMPLATE=$DATA/plot_ls3dvar.yaml
PLOT_SCRIPT=$DATA/plot_ls3dvar_cycling.py
CYCLE=$REPO/ls3dvar/scripts/cycle_ls3dvar.sh
# every process a cycle runs: the cycle script itself (bash), package steps (python -m
# ls3dvar.cli..., or python .../det_runner.py) and old-code steps (python -m mantari...).
# Only bash/python command lines match, so editors or pagers on these files are not killed.
# anchored at the start of the command line: a `bash -c "..."` wrapper or an editor never matches
STEPS="^[^ ]*python[^ ]* (-m (ls3dvar\.cli|mantari\.workflows|mantari\.ls3dvar)\.|[^ ]*det_runner\.py )"
PROCS="^[^ ]*bash [^ ]*cycle_ls3dvar\.sh|$STEPS"

FORCE=${FORCE:-0}

# REF / OBS exactly as cycle_ls3dvar.sh would choose them, then exported, so that `check`
# looks at the same files the cycle reads
resolve_paths() {
    [[ -f "$CONFIG" ]] || return 0
    local r o
    read -r r o < <(python - "$CONFIG" "$DATA" <<'EOF'
import sys, yaml
c = (yaml.safe_load(open(sys.argv[1])) or {}).get("cycling", {}) or {}
base = c.get("data") or sys.argv[2]
print(c.get("reference_dir") or f"{base}/reference/hrrr_ana", c.get("obsdb_root") or f"{base}/obsdb")
EOF
)
    REF=${REF:-$r}; OBS=${OBS:-$o}
}

# export what cycle_ls3dvar.sh reads (an exported empty HOLDOUT_STATIONS disables hold-out)
export_settings() {
    resolve_paths
    export DATA REPO EXP REF OBS LEADS VERIFY_QC VERIFY_CONFIG BLEND STEP_TIMEOUT RESUME LS3DVAR_ARGS CONDA_ENV
    export PKG DETERMINISTIC
    if [[ "$HOLDOUT_STATIONS" == "none" ]]; then export HOLDOUT_STATIONS=""; else export HOLDOUT_STATIONS; fi
}

activate() {
    # shellcheck disable=SC1091
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate "$CONDA_ENV"
}

running() { pgrep -f "$PROCS" >/dev/null; }

# what this run would write into code_version.txt (same format as cycle_ls3dvar.sh)
code_now() { echo "$PKG $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit) deterministic=$DETERMINISTIC"; }

# refuse to resume an experiment made by other code (unless FORCE=1 or RESUME=0)
check_code_match() {
    local f="$EXP/code_version.txt" now old
    now=$(code_now)
    if [[ ! -d "$EXP" ]] || [[ -z "$(ls -A "$EXP" 2>/dev/null)" ]]; then return 0; fi
    if [[ ! -f "$f" ]]; then
        echo "  !! $EXP exists but has no code_version.txt (made by the old run_experiment.sh / cycle script)"
    else
        old=$(cat "$f")
        [[ "$old" == "$now" ]] && return 0
        echo "  !! $EXP was made by:  $old"
        echo "     this run would be:  $now"
        [[ "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]] && \
            echo "     (note: the working tree has uncommitted changes; the commit id may not describe the code)"
    fi
    if [[ "$RESUME" == "1" && "$FORCE" != "1" ]]; then
        echo "     RESUME=1 would mix results of two code versions. Use a new NAME, or RESUME=0"
        echo "     to redo everything, or FORCE=1 if the change does not affect results."
        return 1
    fi
    echo "     continuing (RESUME=$RESUME FORCE=$FORCE)"
    return 0
}

# ---- check: settings, config summary, inputs present, code, nothing else on the GPU --------
do_check() {
    activate
    local ok=1
    echo "run     : $NAME   $START .. $END"
    echo "config  : $CONFIG"
    [[ -f "$CONFIG" ]] || { echo "  !! config not found"; return 1; }
    resolve_paths
    echo "exp     : $EXP"
    echo "ref     : $REF"
    echo "obs     : $OBS"
    echo "log     : $LOG"
    echo "code    : PKG=$PKG  DETERMINISTIC=$DETERMINISTIC  ($(code_now))"
    case "$PKG" in ls3dvar|mantari) ;; *) echo "  !! PKG must be ls3dvar or mantari"; return 1 ;; esac
    [[ -x "$CYCLE" ]] || { echo "  !! cycle script not found or not executable: $CYCLE"; return 1; }
    [[ "$PKG" == "ls3dvar" && ! -f "$REPO/ls3dvar/cli/run_analysis.py" ]] && { echo "  !! $REPO/ls3dvar not found"; return 1; }
    [[ "$DETERMINISTIC" == "1" && ! -f "$REPO/ls3dvar/tools/det_runner.py" ]] && \
        { echo "  !! DETERMINISTIC=1 needs $REPO/ls3dvar/tools/det_runner.py"; return 1; }
    grep -q PYRUN "$CYCLE" || { echo "  !! $CYCLE has no DETERMINISTIC switch (old version)"; return 1; }
    echo "cycling : leads=[$LEADS] holdout=[$HOLDOUT_STATIONS] verify_qc=$VERIFY_QC blend=$BLEND resume=$RESUME"
    echo "verify  : ${VERIFY_CONFIG:-per-cycle config}"
    echo "extra   : [${LS3DVAR_ARGS}]"
    python - "$CONFIG" <<'EOF' || ok=0
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); l = c["ls3dvar"]
print("ls3dvar :", {k: l.get(k) for k in ("control_vector", "nmodes", "corr_length", "sigma_b",
                                          "outer_loops", "max_iterations")})
for b in l.get("blocks", []):
    print("  block  ", b)
print("errors  :", l.get("observation_errors"))
print("obs_qc  :", (l.get("obs_qc") or {}).get("enabled"),
      "  height corr:", (c["observations"].get("temperature_height_correction") or {}).get("enabled"),
      "  wind:", c["observations"].get("wind_rotation"))
EOF
    # inputs: reference HRRR analysis for H-1 .. END+max(lead), obsdb for each cycle
    local maxlead t te c nref nobs refok=1
    maxlead=$(echo $LEADS | tr ' ' '\n' | sort -n | tail -1)
    t=$(( $(date -u -d "${START/T/ }:00" +%s) - 3600 ))
    te=$(( $(date -u -d "${END/T/ }:00" +%s) + maxlead * 3600 ))
    echo "inputs  : (ref = reference HRRR analysis + gfs npz, obs = obsdb types present)"
    while (( t <= te )); do
        c=$(date -u -d @$t +%Y%m%dT%HZ)
        nref=$(ls "$REF"/$c/hrrr_*.npz "$REF"/$c/gfs_*.npz 2>/dev/null | wc -l)
        nobs=$(ls -d "$OBS"/*/$c 2>/dev/null | wc -l)
        printf "  %s  ref:%s  obs:%s\n" "$c" "$nref" "$nobs"
        (( nref < 2 )) && refok=0
        t=$(( t + 3600 ))
    done
    (( refok )) || echo "  !! missing reference files (stage_reference.sh) — verification at those hours is skipped"
    check_code_match || ok=0
    if running; then echo "GPU     : !! a cycling / python step is already running:"; pgrep -af "$PROCS" | cut -c1-160; ok=0
    else echo "GPU     : free"; fi
    (( ok ))
}

# ---- run / resume ------------------------------------------------------------------------
do_run() {
    if running; then echo "already running — './run_experiment.sh stop' first"; pgrep -af "$PROCS" | cut -c1-160; return 1; fi
    check_code_match || return 1
    export_settings
    cd "$DATA" || return 1
    mkdir -p "$EXP"
    cp "$0" "$EXP/run_experiment.sh"            # keep the exact settings with the results
    nohup "$CYCLE" "$START" "$END" "$CONFIG" > "$LOG" 2>&1 &
    echo "started pid $!   log: $LOG   code: $(code_now)"
    echo "follow: ./run_experiment.sh tail     progress: ./run_experiment.sh status"
}

# ---- status: current cycle, step lines, newest files ------------------------------------
do_status() {
    running && echo "RUNNING" || echo "not running"
    [[ -f "$EXP/code_version.txt" ]] && echo "code: $(cat "$EXP/code_version.txt")"
    [[ -f "$LOG" ]] || { echo "no log $LOG"; return; }
    grep -E "^=====.*Z ====|^[0-9:]{8} \[|failed|timed out|SUMMARY" "$LOG" | tail -n 10
    local c
    c=$(ls -d "$EXP"/[0-9]*T[0-9]*Z 2>/dev/null | tail -1)
    [[ -n "$c" ]] && { echo "newest files in $c:"; ls -lt --time-style=+%H:%M:%S "$c" | sed -n '2,6p'; }
    pgrep -af "$STEPS" | cut -c1-160
}

do_tail() { tail -f "$LOG"; }

do_stop() {
    pkill -f "$PROCS"; sleep 3
    running && { echo "still alive -> kill -9"; pkill -9 -f "$PROCS"; }
    echo "stopped"
}

# ---- results: held-out and all-station summaries ----------------------------------------
do_results() {
    [[ -f "$LOG" ]] || { echo "no log $LOG"; return 1; }
    [[ -f "$EXP/code_version.txt" ]] && echo "code: $(cat "$EXP/code_version.txt")"
    echo "=================== HELD-OUT STATIONS (never assimilated) ==================="
    sed -n '/#### HELD-OUT/,$p' "$LOG"
    echo
    echo "=================== ALL STATIONS ==================="
    sed -n '/#### ALL STATIONS/,/#### HELD-OUT/p' "$LOG" | sed '$d'
}

# ---- plot: verification on held-out stations + precipitation ----------------------------
do_plot() {
    activate
    cd "$DATA" || return 1
    [[ -f "$PLOT_SCRIPT" && -f "$PLOT_TEMPLATE" ]] || { echo "needs $PLOT_SCRIPT and $PLOT_TEMPLATE"; return 1; }
    local csv=verification.csv
    [[ -s "$EXP/verification_heldout.csv" ]] && csv=verification_heldout.csv
    sed -e "s#^experiment_dir:.*#experiment_dir: $EXP#" \
        -e "s#^output_dir:.*#output_dir: $EXP/figures#" \
        -e "s#^verification_csv:.*#verification_csv: $csv#" "$PLOT_TEMPLATE" > "$EXP/plot.yaml"
    python "$PLOT_SCRIPT" "$EXP/plot.yaml" --only verification precip
    echo "figures: $EXP/figures  (scored on $csv)"
}

# ---- diff: every npz / csv against another experiment ------------------------------------
do_diff() {
    local other=${1:?usage: ./run_experiment.sh diff OTHER_NAME [--atol X]}; shift
    local odir=$DATA/experiments/$other
    [[ -d "$odir" ]] || { echo "no experiment $odir"; return 1; }
    activate
    for d in "$odir" "$EXP"; do printf "%-60s %s\n" "$d" "$(cat "$d/code_version.txt" 2>/dev/null || echo '(no code_version.txt)')"; done
    cd "$REPO" && python -m ls3dvar.tools.gate diffdirs "$odir" "$EXP" "$@"
}

case "${1:-}" in
    check)   do_check ;;
    run)     do_check && do_run ;;
    resume)  RESUME=1; do_run ;;
    status)  do_status ;;
    tail)    do_tail ;;
    stop)    do_stop ;;
    results) do_results ;;
    plot)    do_plot ;;
    diff)    shift; do_diff "$@" ;;
    *)       sed -n '2,29p' "$0"; exit 1 ;;
esac
