#!/bin/bash
# =============================================================================
# stage_reference.sh  — pre-stage HRRR analyses (reference) + GFS forcing
#
# For each hour H in [START, END]:
#   1. if subrun/YYYYMMDD/HH/hrrr_*.npz is missing, run run_cycle.sh at H
#      (lead 1, 1 member; plain forecast, no DA) -> hrrr/gfs npz + F01 nc
#   2. link into the reference tree, same layout as obsdb/:
#
#   $REF_ROOT/hrrr_ana/YYYYMMDDTHHZ/hrrr_YYYYMMDD_HH.npz   HRRR analysis at H (reference / init)
#   $REF_ROOT/hrrr_ana/YYYYMMDDTHHZ/gfs_YYYYMMDD_HH.npz    GFS forcing for init H
#   $REF_ROOT/hrrr_ana/YYYYMMDDTHHZ/hrrrcast_m00_f01.nc    operational F01 from H (checks)
#   $REF_ROOT/manifest.txt                                 one line per cycle
#
# usage:
#   ./stage_reference.sh 2026-08-10T22 2026-08-11T06
#   ./stage_reference.sh 2026-08-10T22 2026-08-11T06 --no-run   # link only, never run
# =============================================================================
set -euo pipefail

START=${1:?usage: stage_reference.sh START END [--no-run]   e.g. 2026-08-10T22 2026-08-11T06}
END=${2:?usage: stage_reference.sh START END [--no-run]}
NO_RUN=${3:-}

# ---- settings (override by env) --------------------------------------------
HRRRCAST_REPO=${HRRRCAST_REPO:-/opt/hrrrcast}
RUN_BASE=${RUN_BASE:-${DATA:-/data}/runs}          # run_cycle.sh output base
SUBRUN=${SUBRUN:-$RUN_BASE/subrun}                                      # where run_cycle.sh writes
REF_ROOT=${REF_ROOT:-$RUN_BASE/reference}
SUB_BBOX=${SUB_BBOX:?set SUB_BBOX: source domains/<name>.env}
CONDA_ENV=${CONDA_ENV:-hrrrcast}

export SUB_BBOX SUB_HALO=${SUB_HALO:-40}
if [[ -n "${DA_CONFIG:-}" ]]; then export DA_CONFIG; else unset DA_CONFIG; fi   # optional JEDI NPZ injection; unset = reference without DA
#unset DA_CONFIG                                                         # reference = no DA

if [[ "$NO_RUN" != "--no-run" ]]; then
    # shellcheck disable=SC1091
    [[ -f ~/miniforge3/etc/profile.d/conda.sh ]] && source ~/miniforge3/etc/profile.d/conda.sh
    command -v conda >/dev/null && conda activate "$CONDA_ENV" || true
fi

mkdir -p "$REF_ROOT/hrrr_ana"
MANIFEST="$REF_ROOT/manifest.txt"
echo "# cycle  hrrr_npz  gfs_npz  f01_nc   (staged $(date -u +%Y-%m-%dT%H:%MZ))" > "$MANIFEST"

t=$(date -u -d "${START/T/ }:00" +%s)
t_end=$(date -u -d "${END/T/ }:00" +%s)

while (( t <= t_end )); do
    ymd=$(date -u -d "@$t" +%Y%m%d)
    hh=$(date -u -d "@$t" +%H)
    iso=$(date -u -d "@$t" +%Y-%m-%dT%H)
    cyc="${ymd}T${hh}Z"
    src="$SUBRUN/$ymd/$hh"
    hrrr="$src/hrrr_${ymd}_${hh}.npz"
    gfs="$src/gfs_${ymd}_${hh}.npz"
    f01="$src/hrrrcast_m00_f01.nc"

    echo
    echo "=== $cyc ==="
    if [[ ! -f "$hrrr" || ! -f "$gfs" ]]; then
        if [[ "$NO_RUN" == "--no-run" ]]; then
            echo "  MISSING $hrrr (skip, --no-run)"
            t=$(( t + 3600 )); continue
        fi
        echo "  running run_cycle.sh $iso (lead 1)"
        ( cd "$HRRRCAST_REPO" && ./run_cycle.sh "$iso" 1 1 1 "$HRRRCAST_REPO" "$RUN_BASE" ) \
            > "$REF_ROOT/run_cycle_${cyc}.log" 2>&1 \
            || { echo "  FAILED (see $REF_ROOT/run_cycle_${cyc}.log)"; t=$(( t + 3600 )); continue; }
    else
        echo "  exists: $src"
    fi

    if [[ ! -f "$hrrr" ]]; then
        echo "  ERROR: run finished but $hrrr not found (check SUBRUN=$SUBRUN)"
        t=$(( t + 3600 )); continue
    fi

    dst="$REF_ROOT/hrrr_ana/$cyc"
    mkdir -p "$dst"
    ln -sfr "$hrrr" "$dst/"
    ln -sfr "$gfs"  "$dst/"
    [[ -f "$f01" ]] && ln -sfr "$f01" "$dst/"

    python - "$hrrr" <<'EOF' || true
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
print(f"  init_datetime={d['init_datetime']}  model_input={d['model_input'].shape}")
EOF
    echo "$cyc  $dst/$(basename "$hrrr")  $dst/$(basename "$gfs")  $([[ -f $f01 ]] && echo "$dst/$(basename "$f01")" || echo -)" >> "$MANIFEST"
    t=$(( t + 3600 ))
done

echo
echo "reference tree: $REF_ROOT/hrrr_ana"
echo "manifest:       $MANIFEST"
