#!/usr/bin/env bash
# hrrrcast-ls3dvar container entrypoint
#   check                      GPU / env / mounts sanity
#   stage  START [END]         HRRR+GFS reference (cropped to domain, no DA)
#   mrms   START [END]         MRMS REFC staging
#   asos   START [END]         IEM ASOS mirror for domain networks -> IODA
#   obsdb  START [END]         render obsdb template + build_obsdb_cycles
#   cycle  START [END]         stage -> mrms -> asos -> obsdb -> run_cycles (analysis, forecasts, verification)
#                              SKIP_PREP=1 reuses staged inputs
#   shell | <any command>
# Times: YYYY-MM-DDTHH (UTC). END defaults to START.
set -euo pipefail

REPO=${HRRRCAST_REPO:-/opt/hrrrcast}
RT=$REPO/scripts/realtime
export DATA=${DATA:-/data}
DOMAIN=${DOMAIN:-socal}

# ---- domain -----------------------------------------------------------------
DOMFILE=${DOMAIN_FILE:-$REPO/domains/$DOMAIN.env}
[[ -f $DOMFILE ]] || { echo "no domain file $DOMFILE" >&2; exit 2; }
set -a; source "$DOMFILE"; set +a
export SUB_BBOX SUB_HALO=${SUB_HALO:-40}
export RUN_BASE=${RUN_BASE:-$DATA/runs/$DOMAIN}
export ASOS_IODA_DIR=${ASOS_IODA_DIR:-$DATA/ioda/asos/$DOMAIN}   # per domain; tempest/grenade shared
mkdir -p "$RUN_BASE" "$DATA/obsdb/$DOMAIN" "$DATA/ioda" "${CUDA_CACHE_PATH:-/cache/nv}"

# ---- model weights: /models -> $REPO/net-diffusion (run_cycle.sh reads it there; weights are not in the image)
NET_DIR=${NET_DIR:-$REPO/net-diffusion}
if [[ -d ${MODELS:-/models} && -e ${MODELS:-/models}/model.keras && ${MODELS:-/models} != "$NET_DIR" ]]; then
  mkdir -p "$NET_DIR"
  for f in "${MODELS:-/models}"/*; do
    t=$NET_DIR/$(basename "$f")
    # replace missing files and git-lfs pointer stubs, keep real local files
    if [[ ! -e $t ]] || { [[ -f $t && ! -L $t ]] && head -c 40 "$t" | grep -q "git-lfs"; }; then ln -sfn "$f" "$t"; fi
  done
fi

# ---- hooks (override by env if a script's CLI differs) ------------------------
STAGE_CMD=${STAGE_CMD:-"$RT/stage_reference.sh"}
MRMS_CMD=${MRMS_CMD:-"python $RT/mrms_stage.py"}
ASOS_SYNC_CMD=${ASOS_SYNC_CMD:-"python $RT/sync_iem_asos.py"}
ASOS_IODA_CMD=${ASOS_IODA_CMD:-"python $RT/iem/iem_to_ioda.py"}   # IEM csv -> iem_asos_obs_*.nc4
ASOS_SYNC_ARGS=${ASOS_SYNC_ARGS:-"--months 1 --workers 3"}
OBSDB_CMD=${OBSDB_CMD:-"python $RT/build_obsdb_cycles.py"}
CYCLE_CMD=${CYCLE_CMD:-"python -m ls3dvar.cli.run_cycles run"}   # wxldas cycle driver (foreground), takes the experiment yaml
export MODELS=${MODELS:-/models}            # HRRRCast net-diffusion: model.keras, normalize-stats.nc
export LEADS=${LEADS:-"[1, 2, 3]"}          # forecast leads [h] from each analysis
export HOLDOUT=${HOLDOUT:-null}             # "0/5" = withhold 1/5 of stations for verification; null = use all
export VERIFY=${VERIFY:-true}

hours() {  # START END -> list of YYYY-MM-DDTHH
  local t; t=$(date -u -d "${1/T/ } UTC" +%s); local e; e=$(date -u -d "${2/T/ } UTC" +%s)
  while (( t <= e )); do date -u -d "@$t" +%Y-%m-%dT%H; t=$((t+3600)); done
}

render() {  # render TEMPLATE OUT: fill ${UPPER_CASE} from env; ${lower} left for the tools
  python - "$1" "$2" <<'PY2'
import os, re, sys
src = open(sys.argv[1]).read()
names = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", src))
miss = sorted(n for n in names if n not in os.environ)
if miss: sys.exit(f"{sys.argv[1]}: unset {miss}")
out = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: os.environ[m.group(1)], src)
open(sys.argv[2], "w").write(out)
PY2
}

shift_h() { date -u -d "${1/T/ }:00 UTC $2 hour" +%Y-%m-%dT%H; }   # shift_h 2026-11-24T12 -1

find_ref_npz() {  # first staged npz for START
  local d=${1:0:4}${1:5:2}${1:8:2} h=${1:11:2}
  find "$RUN_BASE" -name "hrrr_${d}_${h}.npz" -print -quit
}

do_stage() { $STAGE_CMD "$1" "$2"; }
do_mrms()  { $MRMS_CMD --start "$1" --end "$2" --out "$DATA/mrms/$DOMAIN" --grid-npz "$(find_ref_npz "$1")"; }

npz_bbox() {  # staged npz -> "lat_min lon_min lat_max lon_max" (+pad deg)
  python - "$1" "${ASOS_BBOX_PAD:-0.25}" <<'PY2'
import sys, numpy as np
z = np.load(sys.argv[1]); pad = float(sys.argv[2])
keys = {k.lower(): k for k in z.files}
la = next((z[keys[k]] for k in ("lat", "latitude", "xlat", "lats") if k in keys), None)
lo = next((z[keys[k]] for k in ("lon", "longitude", "xlong", "lons") if k in keys), None)
if la is None or lo is None: sys.exit(f"no lat/lon in {sys.argv[1]}: {z.files}")
lo = np.where(lo > 180, lo - 360, lo)
print(f"{la.min()-pad:.3f} {lo.min()-pad:.3f} {la.max()+pad:.3f} {lo.max()+pad:.3f}")
PY2
}

do_asos() {
  # mirror layout required by iem_to_ioda: $WXBOX_DATA_ROOT/datasets/iem-asos
  export WXBOX_DATA_ROOT=${WXBOX_DATA_ROOT:-$DATA/wxbox}
  export IEM_ASOS_ROOT=${IEM_ASOS_ROOT:-$WXBOX_DATA_ROOT/datasets/iem-asos}
  mkdir -p "$IEM_ASOS_ROOT" "$ASOS_IODA_DIR"
  if [[ -z ${ASOS_SKIP_SYNC:-} ]]; then
    for net in ${ASOS_NETWORKS:?set ASOS_NETWORKS in $DOMFILE}; do
      $ASOS_SYNC_CMD --network "$net" --local-root "$IEM_ASOS_ROOT" $ASOS_SYNC_ARGS
    done
  fi
  local npz; npz=$(find_ref_npz "$1")
  [[ -n $npz ]] || { echo "no staged npz for $1 (run 'stage' first)" >&2; exit 3; }
  local bb; bb=$(npz_bbox "$npz"); echo "[asos] bbox (S W N E) $bb"
  $ASOS_IODA_CMD --start "$1" --end "$2" --step 1h --half-width "${ASOS_HALF_WIDTH:-30m}" \
      --bbox $bb --outdir "$ASOS_IODA_DIR" --at-level-names
}

do_obsdb() {
  export START_CYC=$(date -u -d "${1/T/ }:00" +%Y%m%dT%HZ) END_CYC=$(date -u -d "${2/T/ }:00" +%Y%m%dT%HZ)
  export REF_NPZ=${REF_NPZ:-$(find_ref_npz "$1")}
  [[ -n $REF_NPZ ]] || { echo "no staged npz for $1 under $RUN_BASE (run 'stage' first)" >&2; exit 3; }
  export DATA="$DATA"   # obsdb root -> $DATA/obsdb (template)
  local cfg=$RUN_BASE/obsdb_cycles.yaml
  render "$RT/obsdb_cycles.template.yaml" "$cfg"
  for t in tempest grenade; do
    [[ -d $DATA/ioda/$t ]] || echo "[obsdb] $DATA/ioda/$t not mounted: $t skipped"
  done
  $OBSDB_CMD "$cfg" --output-root "$DATA/obsdb/$DOMAIN"
}

do_cycle() {
  # inputs: reference from START-1h (first background = F01 of START-1h) to END+max lead (verification)
  local lmax; lmax=$(echo "$LEADS" | tr -d '[] ' | tr ',' '\n' | sort -n | tail -1)
  local s0; s0=$(shift_h "$1" -1); local e1; e1=$(shift_h "$2" "${VERIFY_LEAD:-$lmax}")
  [[ $VERIFY == true ]] || e1=$2
  if [[ -z ${SKIP_PREP:-} ]]; then
    do_stage "$s0" "$e1"; do_mrms "$1" "$e1"; do_asos "$1" "$e1"; do_obsdb "$1" "$e1"
  fi
  # obs types = obsdb sub-folders that exist (asos always; tempest/grenade if their feed was mounted)
  export OBS_TYPES="[$(ls -d "$DATA/obsdb/$DOMAIN"/*/ 2>/dev/null | xargs -n1 basename | grep -vx mrms | paste -sd, -)]"
  export START_ISO=$1 END_ISO=$2 EXP_NAME=${EXP_NAME:-${DOMAIN}_${1//[-:]/}}
  local cd=$RUN_BASE/config; mkdir -p "$cd"
  export ANALYSIS_CONFIG=$cd/analysis_blocks.yaml
  render "$REPO/docker/config/analysis_blocks.template.yaml" "$ANALYSIS_CONFIG"
  render "$REPO/docker/config/experiment.template.yaml" "$cd/experiment_$EXP_NAME.yaml"
  echo "[cycle] $EXP_NAME  obs=$OBS_TYPES  leads=$LEADS  holdout=$HOLDOUT  -> $DATA/experiments/$DOMAIN/$EXP_NAME"
  $CYCLE_CMD "$cd/experiment_$EXP_NAME.yaml"
}

do_check() {
  echo "domain=$DOMAIN bbox=$SUB_BBOX halo=$SUB_HALO run_base=$RUN_BASE"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "!! no GPU"
  python - <<'PY'
import tensorflow as tf, numpy, importlib
print("tf", tf.__version__, "gpus", tf.config.list_physical_devices("GPU"))
for m in ("ls3dvar", "requests", "boto3", "yaml", "netCDF4", "pandas", "xarray", "numcodecs"):
    try: importlib.import_module(m); print("ok", m)
    except Exception as e: print("!!", m, e)
import ls3dvar.analysis.bmatrix as b
assert not hasattr(b, "LatentBMatrix"), "PCA code present"
print("bmatrix: blocks-only")
PY
  command -v wgrib2 >/dev/null && echo "ok wgrib2" || echo "!! wgrib2 missing"
  for f in model.keras normalize-stats.nc; do [[ -e $MODELS/$f ]] && echo "ok $MODELS/$f" || echo "!! missing $MODELS/$f"; done
  python -c "import ls3dvar.cli.run_cycles" 2>/dev/null && echo "ok run_cycles" || echo "!! ls3dvar.cli.run_cycles not importable"
}

cmd=${1:-check}; shift || true
s=${1:-}; e=${2:-${1:-}}
case $cmd in
  check) do_check ;;
  stage|mrms|asos|obsdb|cycle) [[ -n $s ]] || { echo "usage: $cmd START [END]" >&2; exit 1; }; do_$cmd "$s" "$e" ;;
  shell) exec bash ;;
  *) exec "$cmd" "$@" ;;
esac
