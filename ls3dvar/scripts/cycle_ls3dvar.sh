#!/bin/bash
# =============================================================================
# cycle_ls3dvar.sh — hourly LS-3DVar cycling + forecast-impact verification
#
# For each cycle H in [START, END]:
#   background  first cycle : HRRRCast F01 from the reference HRRR analysis at H-1 (cold start)
#               later cycles : F01 from the LS-3DVar analysis at H-1        (cycled)
#               BLEND=1      : upper-air large scales (TMP/UGRD/VGRD/HGT, p<=850 hPa,
#                              > ~1000 km) blended toward GFS at H  (large_scale_blend.py)
#                              raw F01 kept as background_raw.npz ("raw" in verification)
#   analysis    LS-3DVar (baseline config) with obsdb/<type>/<H>, truth = HRRR analysis at H
#   forecasts   F01..F03 from the analysis  (an)  and from the background (bg, no-DA control)
#   verify      lead 0 at H, lead L at H+L: vs obsdb and vs reference HRRR analysis
#               -> $EXP/verification.csv  (one row per cycle/lead/state/var)
#               HOLDOUT_STATIONS=k/K : stations with crc32(id)%K==k are NEVER assimilated (any
#               cycle); every lead is also scored on them alone -> $EXP/verification_heldout.csv
#               VERIFY_QC=1 : verification uses the analysis QC (same obs for an and bg)
#               VERIFY_CONFIG=file : verify every experiment with this config (identical obs/QC)
#
# code: PKG=ls3dvar (default, stand-alone package) | PKG=mantari (old modules, for parity runs)
#       DETERMINISTIC=1 : bit-reproducible runs (TF_DETERMINISTIC_OPS + tools/det_runner.py), ~1.5x slower
#
# usage (all settings come from the config; environment variables still override):
#   ./cycle_ls3dvar.sh 2026-08-11T00 2026-08-11T03 [config.yaml]
#   RESUME=1 ./cycle_ls3dvar.sh ...        # skip steps whose outputs exist
#
#   config sections used:
#     ls3dvar:  control_vector, nmodes, corr_length, sigma_b, blocks, outer_loops, max_iterations,
#               holdout_stations, ...         (read by the analysis driver itself)
#     cycling:  repo, data, reference_dir, obsdb_root, obs_types, experiment_dir, leads,
#               holdout_stations, verify_qc, verify_config, blend, blend_args, step_timeout
# =============================================================================
set -euo pipefail
export CUDA_CACHE_MAXSIZE=${CUDA_CACHE_MAXSIZE:-4294967296}     # PTX-JIT cache (Blackwell cc 12.0)

START=${1:?usage: cycle_ls3dvar.sh START END [config.yaml]   e.g. 2026-08-11T00 2026-08-11T03}
END=${2:?usage: cycle_ls3dvar.sh START END [config.yaml]}
BASE_CFG=$(readlink -f "${3:-${BASE_CFG:-/home/jonggyunkim/runs/s3/wxbox_cylc/external/HRRRCast-live/mantari/config/test.yaml}}")

# shellcheck disable=SC1091
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-hrrrcast}"

# ---- settings: config 'cycling:' section, environment overrides -------------
read_cycling_cfg() {
python - "$1" <<'PYCFG'
import sys, shlex, yaml
c = (yaml.safe_load(open(sys.argv[1])) or {}).get("cycling", {}) or {}
def j(v): return " ".join(map(str, v)) if isinstance(v, (list, tuple)) else ("" if v is None else str(v))
def b(v): return "1" if v in (True, 1, "1", "true", "yes") else "0"
data = c.get("data", "/home/jonggyunkim/runs/s3/HRRRCast-live")
for k, v in {
    "CFG_REPO": c.get("repo", "/home/jonggyunkim/runs/s3/wxbox_cylc/external/HRRRCast-live"),
    "CFG_DATA": data,
    "CFG_REF": c.get("reference_dir", f"{data}/reference/hrrr_ana"),
    "CFG_OBS": c.get("obsdb_root", f"{data}/obsdb"),
    "CFG_OBS_TYPES": j(c.get("obs_types", ["asos", "tempest", "grenade"])),
    "CFG_EXP": c.get("experiment_dir", f"{data}/experiments/ls3dvar_cycling"),
    "CFG_LEADS": j(c.get("leads", [1, 2, 3])),
    "CFG_HOLDOUT_STATIONS": j(c.get("holdout_stations")),
    "CFG_VERIFY_QC": b(c.get("verify_qc", True)),
    "CFG_VERIFY_CONFIG": j(c.get("verify_config")),
    "CFG_BLEND": b(c.get("blend", False)),
    "CFG_BLEND_ARGS": c.get("blend_args", "--cutoff-km 1000 --pmax 850 --weight 1.0"),
    "CFG_STEP_TIMEOUT": c.get("step_timeout", 900),
}.items():
    print(f"{k}={shlex.quote(str(v))}")
PYCFG
}
eval "$(read_cycling_cfg "$BASE_CFG")"
REPO=${REPO:-$CFG_REPO}
DATA=${DATA:-$CFG_DATA}
REF=${REF:-$CFG_REF}
OBS=${OBS:-$CFG_OBS}
OBS_TYPES=${OBS_TYPES:-$CFG_OBS_TYPES}
EXP=${EXP:-$CFG_EXP}
LEADS=${LEADS:-$CFG_LEADS}
HOLDOUT_STATIONS=${HOLDOUT_STATIONS-$CFG_HOLDOUT_STATIONS}   # HOLDOUT_STATIONS= (empty) disables
VERIFY_QC=${VERIFY_QC:-$CFG_VERIFY_QC}
VERIFY_CONFIG=${VERIFY_CONFIG:-$CFG_VERIFY_CONFIG}
BLEND=${BLEND:-$CFG_BLEND}
BLEND_ARGS=${BLEND_ARGS:-$CFG_BLEND_ARGS}
STEP_TIMEOUT=${STEP_TIMEOUT:-$CFG_STEP_TIMEOUT}
RESUME=${RESUME:-0}
LS3DVAR_ARGS=${LS3DVAR_ARGS:-""}      # optional extra analysis flags (override the config)

# ---- which code runs the cycle: ls3dvar (stand-alone package) or mantari (old) --------
PKG=${PKG:-ls3dvar}
case "$PKG" in
    ls3dvar) MOD_BG=ls3dvar.cli.make_forecast_background;   MOD_BLEND=ls3dvar.cli.large_scale_blend
             MOD_AN=ls3dvar.cli.run_analysis;                MOD_VERIFY=ls3dvar.cli.verify_state ;;
    mantari) MOD_BG=mantari.workflows.make_forecast_background; MOD_BLEND=mantari.ls3dvar.large_scale_blend
             MOD_AN=mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps; MOD_VERIFY=mantari.workflows.verify_state ;;
    *) echo "PKG must be ls3dvar or mantari (got '$PKG')"; exit 2 ;;
esac

# ---- DETERMINISTIC=1: bit-reproducible TF (about 1.5x slower), same mechanism as the gate --
# TF_DETERMINISTIC_OPS + exact-2x nearest up-sampling via tf.repeat (ls3dvar/tools/det_runner.py)
if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
    export TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
    PYRUN=(python "$REPO/ls3dvar/tools/det_runner.py")
else
    PYRUN=(python -m)
fi

cd "$REPO"
mkdir -p "$EXP"
cp "$BASE_CFG" "$EXP/base_config.yaml"            # record what this experiment ran with
echo "$PKG $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit) deterministic=${DETERMINISTIC:-0}" > "$EXP/code_version.txt"
CSV="$EXP/verification.csv"
CSV_HO="$EXP/verification_heldout.csv"

cycstr() { date -u -d "@$1" +%Y%m%dT%HZ; }
ymd_hh() { date -u -d "@$1" +%Y%m%d_%H; }
have()   { [[ "$RESUME" == "1" && -s "$1" ]]; }
run_step() {   # kill + retry a step that hangs (GPU / PTX-JIT stalls)
    local try
    for try in 1 2 3; do
        timeout --kill-after=30 "$STEP_TIMEOUT" "$@" && return 0
        echo "  $(date +%T) step failed/timed out (try $try): ${*:1:4} ..." >&2
        sleep 20
    done
    return 1
}

echo "LS-3DVar cycling $START .. $END   leads: $LEADS   holdout: ${HOLDOUT_STATIONS:-none}   verify_qc: $VERIFY_QC"
echo "  cfg  $BASE_CFG"
echo "  repo $REPO"
echo "  ref  $REF"
echo "  obs  $OBS"
echo "  exp  $EXP"
echo "  code $PKG   ($MOD_AN)   deterministic ${DETERMINISTIC:-0}"
echo "  args from config ${LS3DVAR_ARGS:+ + extra: $LS3DVAR_ARGS}"
echo "  blend $BLEND  $BLEND_ARGS"

t=$(date -u -d "${START/T/ }:00" +%s)
t_end=$(date -u -d "${END/T/ }:00" +%s)
first=1

while (( t <= t_end )); do
    cyc=$(cycstr "$t");          tag=$(ymd_hh "$t")
    prev=$(cycstr $(( t - 3600 ))); ptag=$(ymd_hh $(( t - 3600 )))
    dir="$EXP/$cyc"; mkdir -p "$dir"
    ref_hrrr="$REF/$cyc/hrrr_${tag}.npz"; ref_gfs="$REF/$cyc/gfs_${tag}.npz"

    echo; echo "================================ $cyc ================================"
    [[ -f "$ref_hrrr" && -f "$ref_gfs" ]] || { echo "missing reference $REF/$cyc"; exit 1; }

    # ---- background -------------------------------------------------------
    bg="$dir/background.npz"
    if ! have "$bg"; then
        if (( first )) || [[ ! -s "$EXP/$prev/fcst_an_f01.npz" ]]; then
            echo "$(date +%T) [bg] cold start: F01 from reference HRRR analysis $prev"
            "${PYRUN[@]}" "$MOD_BG" \
                --hrrr-npz "$REF/$prev/hrrr_${ptag}.npz" --gfs-npz "$REF/$prev/gfs_${ptag}.npz" \
                --lead 1 --out "$bg" > "$dir/background.log" 2>&1
        else
            echo "$(date +%T) [bg] cycled: F01 from analysis $prev"
            cp "$EXP/$prev/fcst_an_f01.npz" "$bg"
        fi
        rm -f "$dir/background_raw.npz"
    fi

    # ---- large-scale blending toward GFS (upper air) --------------------------
    raw="$dir/background_raw.npz"
    if [[ "$BLEND" == "1" && ! -s "$raw" ]]; then
        echo "$(date +%T) [bl] large-scale blend toward GFS"
        mv "$bg" "$raw"
        # shellcheck disable=SC2086
        "${PYRUN[@]}" "$MOD_BLEND" --hrrr "$raw" --gfs "$ref_gfs" --out "$bg" \
            $BLEND_ARGS --truth "$ref_hrrr" > "$dir/blend.log" 2>&1 \
            || { echo "blend failed: $dir/blend.log"; mv "$raw" "$bg"; exit 1; }
        grep -E "^(TMP_850|UGRD_850|HGT_500) " "$dir/blend.log" | tail -3 || true
    fi

    # ---- per-cycle config -------------------------------------------------
    python - "$BASE_CFG" "$dir/config.yaml" "$bg" "$ref_gfs" "$OBS" "$cyc" "$dir/ls3dvar" "$HOLDOUT_STATIONS" $OBS_TYPES <<'EOF'
import sys, yaml
base, out, bg, gfs, obs, cyc, odir, hold, *types = sys.argv[1:]
c = yaml.safe_load(open(base))
c.setdefault("ls3dvar", {})["holdout_stations"] = hold or None
if not hold:
    c["ls3dvar"]["holdout"] = 0.0          # cycling without hold-out assimilates everything
c["hrrr_npz"], c["gfs_npz"] = bg, gfs
c.setdefault("observations", {})["providers"] = [
    {"type": t, "path": f"{obs}/{t}/{cyc}"} for t in types]
c.setdefault("outputs", {})["directory"] = odir
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
EOF

    # ---- analysis ---------------------------------------------------------
    an="$dir/analysis.npz"
    if ! have "$an"; then
        echo "$(date +%T) [an] LS-3DVar"
        # shellcheck disable=SC2086
        run_step "${PYRUN[@]}" "$MOD_AN" --config "$dir/config.yaml" \
            $LS3DVAR_ARGS --truth-npz "$ref_hrrr" --analysis-out "$an" \
            > "$dir/analysis.log" 2>&1 || { echo "analysis failed: $dir/analysis.log"; exit 1; }
        grep -A12 "O-B -> O-A rms" "$dir/analysis.log" | grep -E "assimilated|mean\(H|T2M |D2M |PRES " || true
    fi

    # ---- forecasts from analysis and background ---------------------------
    for src in an bg; do
        init=$([[ $src == an ]] && echo "$an" || echo "$bg")
        last="$dir/fcst_${src}_f$(printf %02d "${LEADS##* }").npz"
        if ! have "$last"; then
            echo "$(date +%T) [fc] F0{${LEADS// /,}} from $src"
            # shellcheck disable=SC2086
            run_step "${PYRUN[@]}" "$MOD_BG" --hrrr-npz "$init" --gfs-npz "$ref_gfs" \
                --lead $LEADS --out "$dir/fcst_${src}_f{lead:02d}.npz" > "$dir/fcst_${src}.log" 2>&1 \
                || { echo "forecast failed: $dir/fcst_${src}.log"; exit 1; }
        fi
    done

    # ---- verification -----------------------------------------------------
    rot=$(grep -o "wind_rotation = [a-z]*" "$dir/analysis.log" 2>/dev/null | tail -1 | awk '{print $3}')
    vargs=(--config "${VERIFY_CONFIG:-$dir/config.yaml}"); [[ -n "$rot" ]] && vargs+=(--wind-rotation "$rot")
    [[ "$VERIFY_QC" == "1" ]] && vargs+=(--qc)
    hoargs=(); [[ -n "$HOLDOUT_STATIONS" ]] && hoargs=(--holdout-stations "$HOLDOUT_STATIONS" --subset heldout)
    states0=(an="$an" bg="$bg"); [[ -s "$raw" ]] && states0+=(raw="$raw")
    "${PYRUN[@]}" "$MOD_VERIFY" --state "${states0[@]}" "${vargs[@]}" --obs-root "$OBS" \
        --obs-types $OBS_TYPES --truth-npz "$ref_hrrr" --csv "$CSV" --meta cycle="$cyc" lead=0 \
        2>>"$dir/verify.log" | sed -n '/^VERIFICATION/,$p'
    if (( ${#hoargs[@]} )); then
        "${PYRUN[@]}" "$MOD_VERIFY" --state "${states0[@]}" "${vargs[@]}" "${hoargs[@]}" \
            --obs-root "$OBS" --obs-types $OBS_TYPES --truth-npz "$ref_hrrr" --csv "$CSV_HO" \
            --meta cycle="$cyc" lead=0 2>>"$dir/verify.log" >> "$dir/verify_heldout.log"
    fi
    for L in $LEADS; do
        tv=$(( t + L * 3600 )); vcyc=$(cycstr "$tv"); vtag=$(ymd_hh "$tv")
        truth="$REF/$vcyc/hrrr_${vtag}.npz"
        targs=(); [[ -f "$truth" ]] && targs=(--truth-npz "$truth")
        "${PYRUN[@]}" "$MOD_VERIFY" \
            --state an="$dir/fcst_an_f$(printf %02d "$L").npz" bg="$dir/fcst_bg_f$(printf %02d "$L").npz" \
            "${vargs[@]}" --obs-root "$OBS" --obs-types $OBS_TYPES "${targs[@]}" --csv "$CSV" \
            --meta cycle="$cyc" lead="$L" 2>>"$dir/verify.log" | sed -n '/^VERIFICATION/,$p'
        if (( ${#hoargs[@]} )); then
            "${PYRUN[@]}" "$MOD_VERIFY" \
                --state an="$dir/fcst_an_f$(printf %02d "$L").npz" bg="$dir/fcst_bg_f$(printf %02d "$L").npz" \
                "${vargs[@]}" "${hoargs[@]}" --obs-root "$OBS" --obs-types $OBS_TYPES "${targs[@]}" \
                --csv "$CSV_HO" --meta cycle="$cyc" lead="$L" 2>>"$dir/verify.log" >> "$dir/verify_heldout.log"
        fi
    done

    first=0
    t=$(( t + 3600 ))
done

# ---- summary over cycles ----------------------------------------------------
for C in "$CSV" "$CSV_HO"; do
[[ -s "$C" ]] || continue
python - "$C" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
print("\n#### " + ("HELD-OUT STATIONS ONLY (never assimilated)" if "heldout" in sys.argv[1] else "ALL STATIONS"))
d["lead"] = d["lead"].astype(int)
d = d.drop_duplicates(["cycle", "lead", "state", "var"], keep="last")
print("\n" + "=" * 90 + "\nSUMMARY (mean over cycles)   O-F rms vs obs   |   rms vs HRRR analysis (all)\n" + "=" * 90)
for var in ("T2M", "D2M", "PS", "U10", "V10", "TMP850"):
    s = d[d["var"] == var].pivot_table(index="lead", columns="state", values=["rms", "truth_all"], aggfunc="mean")
    if s.empty:
        continue
    print(f"\n{var}")
    print(f"{'lead':>4s} {'obs bg':>9s} {'obs an':>9s} {'impr%':>7s}   {'ref bg':>9s} {'ref an':>9s} {'impr%':>7s}")
    for L, r in s.iterrows():
        ob, oa = r.get(("rms", "bg"), float("nan")), r.get(("rms", "an"), float("nan"))
        tb, ta = r[("truth_all", "bg")], r[("truth_all", "an")]
        print(f"{L:4d} {ob:9.3f} {oa:9.3f} {100 * (ob - oa) / ob:+7.2f}   {tb:9.3f} {ta:9.3f} {100 * (tb - ta) / tb:+7.2f}")
print(f"\n(cycles: {d['cycle'].nunique()}; csv: {sys.argv[1]})")
EOF
done
