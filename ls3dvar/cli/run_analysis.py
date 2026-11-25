#!/usr/bin/env python3
# ls3dvar/cli/run_analysis.py  (extracted from mantari/workflows/test_ls3dvar_analysis_t2m_d2m_ps.py)
"""
Analysis-time LS-3DVar with T2M / D2M / station-pressure observations.

    background  x_b      = hrrr model_input (normalized)          -- NO forecast
    control     z        = encoder latent of the analysis input
    state map   x(z)     = x_b + unpad(refine(z) - refine(z_b))    (anchored)
    innovation  d        = H(x_b) - y                              (true O-B)
    cost        J(z)     = 1/2 |(z - z_b)/sigma_b|^2 + sum_k 1/2 |(H_k x(z) - y_k)/sigma_k|^2

Outer loops: z_b stays fixed; only the first guess moves to the previous
analysis.

Writes:
    <outdir>/analysis_t2m_d2m_ps.npz   hrrr npz with model_input[..., :138] = x_a
                                       (+ latent_analysis, latent_background)
    <outdir>/increment_t2m_d2m_ps.npz  x_a - x_b, latents
    <outdir>/lbfgs_history_outer_XX.csv, outer_loop_history_t2m_d2m_ps.csv

Run:
    python -m ls3dvar.cli.run_analysis
    python -m ls3dvar.cli.run_analysis \
        --sigma-b 1 3 10 30 100 --holdout 0.2

--ps-sigma S (Pa) adds a direct, Gaussian-correlated PRES increment to the
control (hybrid). --ps-corr-length in model grid cells. With --control ps the
PRES block is the only control (latent untouched):
    python -m mantari.workflows.test_ls3dvar_analysis_t2m_d2m_ps \
        --control blocks --ps-sigma 40 --ps-corr-length 10

--block CH:SIGMA:L[:land] adds a direct Gaussian-correlated increment on channel CH
(T2M, D2M, PRES; SIGMA in K or Pa, L in model grid cells). Repeatable;
--ps-sigma S --ps-corr-length L is shorthand for --block PRES:S:L:
    --control blocks --block T2M:1.0:40 --block D2M:1.5:40 --block PRES:40:10
    --block T2M:0.8:40:land    (amplitude x land fraction from hrrr npz LAND_raw)

Observations: T2M, D2M, station pressure and 10-m wind (windEastwardAt10M /
windNorthwardAt10M, earth-relative). Wind operator: model U10/V10 channels
(names from the npz, stats from channel_stats / normalize_stats_nc), rotated
grid->earth if observations.wind_rotation = grid (auto: picks the lower O-B).
    --block U10:1.5:10:land --block V10:1.5:10:land     (m/s, model cells)
Any channel can be a block: --block TMP_850:0.5:60 (needs its stats).

Pseudo-observations / insertion from the --truth-npz reference (fake OSSE; e.g.
the HRRR analysis valid at the analysis time). Config ls3dvar.pseudo_obs / ls3dvar.insert,
or on the command line:
    --pseudo-obs REFC:5:0      superobs of reference REFC, error 5 dBZ, floor 0 dBZ
    --pseudo-obs SPFH_850:0.001   (any channel with stats; error in physical units)
    --pseudo-box 8             superob tile size, model cells (8 x 3 km = 24 km)
    --insert REFC --insert-weight 1.0   overwrite channel(s) with the reference after the solve
REFC from MRMS instead of the reference (ls3dvar.refc_source or --refc-source mrms):
    --refc-source mrms --mrms-dir DATA/mrms [--mrms-field refc_t_mean]
reads DIR/YYYYMMDDTHHZ/mrms_YYYYMMDD_HH.npz (mrms_stage.py) at the analysis time and uses it
for REFC in --pseudo-obs and --insert (no radar coverage -> REFC left as is / tile dropped).
Pseudo-obs need a block on the same channel to move it directly (--block REFC:10:4); the
latent part alone barely changes REFC. Pseudo-obs are never held out and do not enter the
station reports; scores against the reference at lead 0 become circular for those channels.

--holdout F withholds a random fraction F of observation LOCATIONS (same
grid points removed for all variables). The minimization never sees them;
O-B -> O-A on the held-out set is the honest score for choosing sigma_b.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from ls3dvar.paths import DEFAULT_CONFIG, add_hrrrcast_src  # noqa: E402

add_hrrrcast_src()

import resnet  # noqa: E402

from ls3dvar.hrrrcast.analysis_input import build_analysis_input  # noqa: E402
from ls3dvar.analysis.state_map import AnalysisStateRuntimeTF  # noqa: E402
from ls3dvar.hrrrcast.latent import extract_latent  # noqa: E402
from ls3dvar.obs.pipeline import load_observations  # noqa: E402
from ls3dvar.obs.operators import (  # noqa: E402
    t2m_observation_operator,
    d2m_observation_operator,
    pres_observation_operator,
    station_height_offsets,
    make_station_pressure_operator,
)
from ls3dvar.analysis.solver import LS3DVARMultiSolver  # noqa: E402
from ls3dvar.obs.qc import ObsQC  # noqa: E402
from ls3dvar.obs.holdout import parse_spec, is_heldout  # noqa: E402
from ls3dvar.obs.operators import make_lapse_operator, lapse_settings  # noqa: E402
from ls3dvar.obs.operators import make_wind_operator, make_channel_operator  # noqa: E402
from ls3dvar.obs.operators import make_superob_operator, superob_boxes  # noqa: E402
from ls3dvar.hrrrcast.channels import (  # noqa: E402
    channel_names, resolve_name, ChannelStats, grid_rotation, unit_of,
)
from ls3dvar.analysis.bmatrix import (  # noqa: E402
    HybridControlRuntime,
)
from ls3dvar.obs.operators import (  # noqa: E402
    PRES_CHANNEL, PRES_STD, PRES_MEAN, T2M_CHANNEL, T2M_STD, D2M_CHANNEL, D2M_STD,
)
from ls3dvar.analysis.diagnostics import plot_convergence  # noqa: E402
from ls3dvar.obs.types import SurfaceObservationSet  # noqa: E402
from ls3dvar.obs.grid import build_station_indices  # noqa: E402

OUTER_LOOPS = 2
INNER_MAXITER = 40

CUSTOM_OBJECTS = {
    "CastLayer": resnet.CastLayer,
    "ChannelPoolAvg": resnet.ChannelPoolAvg,
    "ChannelSliceLayer": resnet.ChannelSliceLayer,
    "OutputMaskLayer": resnet.OutputMaskLayer,
    "RecomputeSubModel": resnet.RecomputeSubModel,
    "ReflectPadLayer": resnet.ReflectPadLayer,
    "SpatialGroupedLayer": resnet.SpatialGroupedLayer,
    "TimeCondLayer": resnet.TimeCondLayer,
    "UnpadLayer": resnet.UnpadLayer,
}

# variable name -> (operator, obs-dict key prefix)
VARIABLES = {
    "airTemperatureAt2M": (t2m_observation_operator, "t2m"),
    "dewPointTemperatureAt2M": (d2m_observation_operator, "d2m"),
    "stationPressure": (pres_observation_operator, "ps"),
    "windEastwardAt10M": (None, "u10"),          # operator built at run time (channels, rotation)
    "windNorthwardAt10M": (None, "v10"),
}
DEFAULT_OBS_ERROR = {"windEastwardAt10M": 2.0, "windNorthwardAt10M": 2.0}


def banner(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def obs_vectors(obs_list, sigma):
    """sigma: scalar, or per-obs array (network error scaling)"""
    y = tf.constant(np.array([o.value for o in obs_list], np.float32))
    s = np.asarray(sigma, np.float32)
    s = tf.constant(np.full(len(obs_list), s, np.float32) if s.ndim == 0 else s)
    return y, s


# yaml key (under ls3dvar:)  ->  argparse dest.  Command-line flags win over the yaml.
CONFIG_ARGS = {
    "control_vector": "control", "nmodes": "nmodes", "corr_length": "corr_length",
    "sigma_b": "sigma_b", "blocks": "block", "outer_loops": "outer", "max_iterations": "maxiter",
    "holdout": "holdout", "holdout_stations": "holdout_stations", "fold": "fold", "seed": "seed",
    "ps_sigma": "ps_sigma", "ps_corr_length": "ps_corr_length", "near_radius": "near_radius",
    "stop_rel_dj": "stop_rel", "stop_window": "stop_window", "stop_reduction": "stop_reduction",
}
LIST_ARGS = {"corr_length", "sigma_b", "ps_corr_length"}


def block_spec(b):
    """yaml block -> 'CH:SIGMA:L[:land]'  (accepts the string form or {var, sigma, L, mask})"""
    if isinstance(b, str):
        return b
    spec = f"{b['var']}:{b['sigma']}:{b['L']}"
    return spec + (":land" if str(b.get("mask", "")).lower() == "land" else "")


def apply_config_defaults(ap, args, cfg):
    """fill every option NOT given on the command line from cfg['ls3dvar'] (see CONFIG_ARGS)"""
    given = set()
    for a in ap._actions:
        for opt in a.option_strings:
            if any(t == opt or t.startswith(opt + "=") for t in sys.argv[1:]):
                given.add(a.dest)
    ls = cfg.get("ls3dvar", {}) or {}
    used = []
    for key, dest in CONFIG_ARGS.items():
        if dest in given or key not in ls or ls[key] is None:
            continue
        v = ls[key]
        if dest == "block":
            v = [block_spec(b) for b in (v or [])]
        elif dest in LIST_ARGS and not isinstance(v, (list, tuple)):
            v = [v]
        setattr(args, dest, v)
        used.append(key)
    print("run settings: " + "  ".join(
        f"{d}={getattr(args, d)}" for d in ("control", "nmodes", "corr_length", "sigma_b", "outer",
                                           "maxiter", "holdout", "holdout_stations", "fold", "seed")))
    for b in args.block:
        print(f"  block {b}")
    print(f"  (from config: {', '.join(used) or 'none'}; from command line: "
          f"{', '.join(sorted(given - {'config', 'truth_npz', 'analysis_out'})) or 'none'})")


def pseudo_settings(args, cfg):
    """
    pseudo-obs / insertion settings; command-line flags replace the config lists.
      ls3dvar.pseudo_obs: {enabled, box, channels: [{var, error, floor}]}
      ls3dvar.insert:     {channels: [..], weight}
    returns (po_specs [(name, error, floor|None)], box, insert_names [..], insert_weight)
    """
    ls = cfg.get("ls3dvar", {}) or {}
    pcfg = ls.get("pseudo_obs") or {}
    icfg = ls.get("insert") or {}
    po = []
    if args.pseudo_obs:
        for spec in args.pseudo_obs:
            p = spec.split(":")
            po.append((p[0], float(p[1]), float(p[2]) if len(p) > 2 and p[2] != "" else None))
    elif pcfg.get("enabled", False):
        for c in pcfg.get("channels") or []:
            fl = c.get("floor")
            po.append((c["var"], float(c["error"]), None if fl is None else float(fl)))
    box = int(args.pseudo_box or pcfg.get("box", 8))
    ins = list(args.insert) if args.insert else list(icfg.get("channels") or [])
    w = float(args.insert_weight if args.insert_weight is not None else icfg.get("weight", 1.0))
    return po, box, ins, w


def refc_source_settings(args, cfg):
    """(source 'truth'|'mrms', mrms_dir, field) from --refc-source/--mrms-dir or ls3dvar.refc_source"""
    rc = (cfg.get("ls3dvar", {}) or {}).get("refc_source") or {}
    if isinstance(rc, str):
        rc = {"source": rc}
    src = args.refc_source or rc.get("source", "truth")
    return src, args.mrms_dir or rc.get("dir"), args.mrms_field or rc.get("field", "refc_t_mean")


def load_mrms_refc(mrms_dir, when, field, shape):
    """MRMS REFC (dBZ, NaN = no coverage) on the model grid at analysis time `when`."""
    t = np.datetime64(str(when).replace(" ", "T"), "h").astype(object)
    p = Path(mrms_dir) / f"{t:%Y%m%dT%HZ}" / f"mrms_{t:%Y%m%d_%H}.npz"
    if not p.exists():
        raise SystemExit(f"MRMS file missing: {p}  (run mrms_stage.py for this hour)")
    z = np.load(p, allow_pickle=True)
    f = np.asarray(z[field], np.float32)
    if f.shape != tuple(shape):
        raise SystemExit(f"MRMS grid {f.shape} != model grid {tuple(shape)} ({p})")
    return f, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-b", type=float, nargs="+", default=None,
                    help="one or more sigma_b values (default: config ls3dvar.sigma_b)")
    ap.add_argument("--holdout-stations", default=None,
                    help="k/K: never assimilate stations with crc32(id) %% K == k (stable across "
                         "cycles); overrides --fold / --holdout. Verify them with verify_state --subset heldout")
    ap.add_argument("--fold", default=None,
                    help="k/K: deterministic K-fold split of locations, hold out fold k (0-based); "
                         "overrides --holdout. Run k=0..K-1 and pool with ls3dvar_station_report")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fraction of obs locations withheld for verification (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outer", type=int, default=OUTER_LOOPS)
    ap.add_argument("--maxiter", type=int, default=INNER_MAXITER)
    ap.add_argument("--stop-rel", type=float, default=None,
                    help="stall stop: best J improves < this (relative) over --stop-window "
                         "evaluations, e.g. 5e-3 (default off; config ls3dvar.stop_rel_dj)")
    ap.add_argument("--stop-window", type=int, default=5)
    ap.add_argument("--stop-reduction", type=float, default=None,
                    help="stop once J <= (1 - this) x J at the first guess, e.g. 0.9; the "
                         "remaining outer loops are skipped (default off; ls3dvar.stop_reduction)")
    ap.add_argument("--control", choices=["latent", "ps", "blocks"], default="latent",
                    help="latent: diagonal B on all latent values; blocks: Gaussian state blocks")
    ap.add_argument("--nmodes", type=int, default=0, help=argparse.SUPPRESS)        # unused
    ap.add_argument("--corr-length", type=float, nargs="+", default=[2.0], help=argparse.SUPPRESS)
    ap.add_argument("--ps-sigma", type=float, default=0.0,
                    help="PRES background error std [Pa] for the hybrid PRES block (0 = off)")
    ap.add_argument("--config", default=None,
                    help="config yaml (default: $MANTARI_CONFIG or mantari/config/test.yaml)")
    ap.add_argument("--analysis-out", default=None,
                    help="also copy the (last) analysis npz to this path (for cycling)")
    ap.add_argument("--block", action="append", default=[],
                    help="CH:SIGMA:L[:land] direct state-space block (repeatable), "
                         "e.g. T2M:1.0:40 or T2M:0.8:40:land")
    ap.add_argument("--truth-npz", default=None,
                    help="independent gridded reference at the analysis time (e.g. HRRR "
                         "analysis npz): reports rms(x_b - truth) vs rms(x_a - truth)")
    ap.add_argument("--near-radius", type=int, default=10,
                    help="grid cells: 'near' = within this distance of an assimilated obs")
    ap.add_argument("--ps-corr-length", type=float, nargs="+", default=[10.0],
                    help="PRES correlation length(s) [model grid cells]")
    ap.add_argument("--pseudo-obs", action="append", default=[],
                    help="CH:ERROR[:FLOOR] superobs of channel CH from --truth-npz (repeatable), "
                         "e.g. REFC:5:0 or SPFH_850:0.001; replaces ls3dvar.pseudo_obs")
    ap.add_argument("--pseudo-box", type=int, default=None,
                    help="superob tile size in model cells (default ls3dvar.pseudo_obs.box or 8)")
    ap.add_argument("--insert", action="append", default=[],
                    help="channel to overwrite with --truth-npz after the solve (repeatable), "
                         "e.g. REFC; replaces ls3dvar.insert.channels")
    ap.add_argument("--insert-weight", type=float, default=None,
                    help="x_a = (1-w) x_a + w truth on inserted channels (default 1)")
    ap.add_argument("--refc-source", choices=["truth", "mrms"], default=None,
                    help="REFC for --pseudo-obs/--insert from the reference (truth) or MRMS")
    ap.add_argument("--mrms-dir", default=None, help="mrms_stage.py output dir")
    ap.add_argument("--mrms-field", default=None, help="refc_t_mean (default) | refc_t (max)")
    args = ap.parse_args()

    import os
    import shutil
    cfg_path = args.config or os.environ.get("MANTARI_CONFIG", str(DEFAULT_CONFIG))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"config: {cfg_path}")
    apply_config_defaults(ap, args, cfg)

    outdir = Path(cfg["outputs"]["directory"])
    outdir.mkdir(parents=True, exist_ok=True)
    obs_err_cfg = cfg["ls3dvar"]["observation_errors"]

    # ------------------------------------------------------------------
    # Background + latent (analysis time, denoise mode)
    # ------------------------------------------------------------------
    hrrr = np.load(cfg["hrrr_npz"], allow_pickle=True)
    gfs = np.load(cfg["gfs_npz"], allow_pickle=True)
    X, x_b = build_analysis_input(hrrr, gfs, lead_hour=1, diffusion_step=0)

    model = tf.keras.models.load_model(
        cfg["forecast_engine"]["model_path"],
        custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False,
    )
    latent_b, state = extract_latent(model, X)
    latent_b = tf.convert_to_tensor(latent_b, tf.float32)

    runtime = AnalysisStateRuntimeTF(model, x_b, latent_b, state)

    banner("ANALYSIS-STATE RUNTIME CHECKS")
    print(f"x_b {x_b.shape}   latent {tuple(latent_b.shape)}")
    anchor = runtime.anchor_error(latent_b, state)
    gnorm = runtime.gradient_norm(latent_b, state)
    print(f"anchor   max|x(z_b) - x_b|  = {anchor:.3e}   (must be 0)")
    print(f"gradient |d<w,x(z)>/dz|     = {gnorm}        (must be > 0)")
    if anchor > 1e-5 or not gnorm:
        print("FAIL: runtime checks")
        return 2

    # ------------------------------------------------------------------
    # Observations, innovations against x_b, QC
    # ------------------------------------------------------------------
    obs = load_observations(cfg)
    obs_filter = obs["filter"]
    x_b_tf = tf.convert_to_tensor(x_b, tf.float32)

    height_corr = bool(cfg.get("observations", {}).get("pressure_height_correction", True))
    orog = np.asarray(hrrr["OROG_raw"], np.float32) if "OROG_raw" in hrrr.files else None
    if height_corr and orog is None:
        print("WARNING: no OROG_raw in hrrr npz -> pressure height correction disabled")
        height_corr = False

    lapse = lapse_settings(cfg)
    if lapse and orog is None:
        print("WARNING: no OROG_raw -> T2M/D2M height correction disabled")
        lapse = {}

    def height_operator(key, base, obs_list, iy, ix, verbose=False):
        """
        station-height corrected operator for this exact obs list / order:
          ps      hydrostatic (pressure_height_correction)
          t2m/d2m lapse rate (observations.temperature_height_correction)
        returns (op, dz, rebuild) ; (base, None, None) when no correction applies
        """
        if key == "ps" and height_corr:
            rebuild = make_station_pressure_operator
        elif key in lapse:
            rebuild = (lambda b, g: (lambda dz: make_lapse_operator(b, dz, g)))(base, lapse[key])
        else:
            return base, None, None
        dz, n_missing = station_height_offsets(obs_list, iy.numpy(), ix.numpy(), orog)
        if verbose:
            extra = f"  lapse {lapse[key] * 1000:.1f} K/km" if key in lapse else ""
            print(f"  height correction: dz = z_stn - z_model  "
                  f"mean={dz.mean():+.1f} m  rms={np.sqrt(np.mean(dz ** 2)):.1f} m  "
                  f"min={dz.min():+.1f}  max={dz.max():+.1f}  (no elevation: {n_missing}){extra}")
        return rebuild(dz), dz, rebuild

    # ---- 10-m wind operators ---------------------------------------------
    names = channel_names(hrrr)
    stats = ChannelStats(cfg, names)
    wind_ops = {}
    n_wind = len(obs["u10_obs"]) + len(obs["v10_obs"])
    if n_wind:
        try:
            nu, nv = resolve_name(names, "U10"), resolve_name(names, "V10")
            su, sv = stats.get(nu), stats.get(nv)
        except KeyError as e:
            print(f"\nWARNING: wind obs present but {e}\n-> wind NOT assimilated")
            nu = None
        if nu is not None:
            iu, iv = names.index(nu), names.index(nv)
            cos_t, sin_t = grid_rotation(hrrr["lats"], hrrr["lons"])
            cand = {"earth": dict(), "grid": dict(cos_t=cos_t, sin_t=sin_t)}
            mode = str(cfg.get("observations", {}).get("wind_rotation", "auto")).lower()
            banner(f"10-M WIND  channels {nu}[{iu}] {nv}[{iv}]  stats u={su} v={sv}")
            print(f"  grid rotation angle: min {np.degrees(np.arctan2(sin_t, cos_t)).min():+.1f}  "
                  f"max {np.degrees(np.arctan2(sin_t, cos_t)).max():+.1f} deg")
            score = {}
            for m_, kw in cand.items():
                r2 = []
                for comp, key in (("u", "u10"), ("v", "v10")):
                    if obs[f"{key}_obs"]:
                        op_ = make_wind_operator(comp, iu, iv, su, sv, **kw)
                        y_ = np.array([o.value for o in obs[f"{key}_obs"]], np.float32)
                        r2.append(((op_(x_b_tf, obs[f"{key}_y"], obs[f"{key}_x"]).numpy() - y_) ** 2))
                score[m_] = float(np.sqrt(np.mean(np.concatenate(r2))))
                print(f"  O-B rms (u,v pooled) with model winds treated as {m_:5s}-relative: "
                      f"{score[m_]:.3f} m/s")
            if mode == "auto":
                mode = min(score, key=score.get)
            print(f"  -> wind_rotation = {mode}")
            # implied normalization from obs (sanity check of the stats)
            for key, ch, (mu, sd) in (("u10", iu, su), ("v10", iv, sv)):
                if len(obs[f"{key}_obs"]) > 30 and mode == "earth":
                    xn = x_b[0, obs[f"{key}_y"].numpy(), obs[f"{key}_x"].numpy(), ch]
                    y_ = np.array([o.value for o in obs[f"{key}_obs"]])
                    a, b = np.polyfit(xn, y_, 1)
                    print(f"  {key}: obs ~ {a:.3f} * x_norm + {b:+.3f}   (stats: std {sd:.3f}, "
                          f"mean {mu:+.3f}; regression slope is biased low by obs/bg error)")
            for comp, key in (("u", "u10"), ("v", "v10")):
                wind_ops[key] = make_wind_operator(comp, iu, iv, su, sv, **cand[mode])

    obs_qc = ObsQC(cfg)
    if obs_qc.enabled:
        banner("OBS QC (obs_qc): blacklist / network check / buddy check / error scale")
    data = {}
    for var, (op, key) in VARIABLES.items():
        obs_list = obs[f"{key}_obs"]
        if key in ("u10", "v10"):
            if key not in wind_ops:
                continue
            op = wind_ops[key]
        err = obs_err_cfg.get(var, DEFAULT_OBS_ERROR.get(var, 1.0))
        y, s = obs_vectors(obs_list, err)
        dz, rebuild, op_base = None, None, op
        if (key == "ps" and height_corr) or key in lapse:
            hx_nc = op(x_b_tf, obs[f"{key}_y"], obs[f"{key}_x"])
            print()
            print(f"{var} O-B without height correction: rms={rms((hx_nc - y).numpy()):.3f}")
            op, dz, rebuild = height_operator(key, op_base, obs_list, obs[f"{key}_y"], obs[f"{key}_x"],
                                              verbose=True)
            print(f"{var} O-B with    height correction: rms="
                  f"{rms((op(x_b_tf, obs[f'{key}_y'], obs[f'{key}_x']) - y).numpy()):.3f}")
            if dz is not None and key in lapse:
                # empirical lapse: minimise |(H - y) - g dz| -> g* = sum(d dz) / sum(dz^2)
                d_ = (hx_nc - y).numpy().astype(np.float64)
                dzz = np.asarray(dz, np.float64)
                m_ = np.abs(d_) < 5 * np.std(d_)
                g = float(np.sum(d_[m_] * dzz[m_]) / max(np.sum(dzz[m_] ** 2), 1e-9))
                r = float(np.corrcoef(d_[m_], dzz[m_])[0, 1])
                print(f"  empirical lapse from O-B vs dz: {g * 1000:+.2f} K/km  (corr {r:+.2f}, "
                      f"rms at fit {rms(d_ - g * dzz):.3f}; standard atmosphere +6.5)")
        hx = op(x_b_tf, obs[f"{key}_y"], obs[f"{key}_x"])
        innov = (hx - y).numpy()

        banner(f"{var}: O-B INNOVATIONS (n={len(obs_list)})")
        print(obs_filter.innovation_stats(innov))

        def reindex(kept):
            idx_ = {}
            yy, xx = build_station_indices(SurfaceObservationSet(kept), cfg["hrrr_npz"])
            idx_["iy"] = tf.constant(yy, tf.int32)
            idx_["ix"] = tf.constant(xx, tf.int32)
            return idx_

        if obs_filter.enabled:
            kept = obs_filter.apply(var, obs_list, innov)
            if len(kept) != len(obs_list):
                ix_ = reindex(kept)
                obs[f"{key}_y"], obs[f"{key}_x"] = ix_["iy"], ix_["ix"]
                obs_list = kept
                if dz is not None:
                    op, dz, rebuild = height_operator(key, op_base, obs_list, obs[f"{key}_y"], obs[f"{key}_x"])
            print(f"after QC: {len(obs_list)}")

        if obs_qc.enabled and obs_list:
            y, s = obs_vectors(obs_list, err)
            innov = (op(x_b_tf, obs[f"{key}_y"], obs[f"{key}_x"]) - y).numpy()
            ObsQC.network_summary(var, obs_list, innov)
            kept = obs_qc.apply(var, obs_list, innov, err)
            if len(kept) != len(obs_list):
                ix_ = reindex(kept)
                obs[f"{key}_y"], obs[f"{key}_x"] = ix_["iy"], ix_["ix"]
                obs_list = kept
                if dz is not None:
                    op, dz, rebuild = height_operator(key, op_base, obs_list, obs[f"{key}_y"], obs[f"{key}_x"])

        y, s = obs_vectors(obs_list, obs_qc.sigmas(var, obs_list, err) if obs_qc.enabled else err)

        data[key] = dict(y=y, s=s, iy=obs[f"{key}_y"], ix=obs[f"{key}_x"], op=op,
                         n=len(obs_list), dz=dz, obs=obs_list, rebuild=rebuild)

    # ------------------------------------------------------------------
    # Train / held-out split by grid location (shared across variables)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    locs = sorted({(int(a), int(b)) for d in data.values()
                   for a, b in zip(d["iy"].numpy(), d["ix"].numpy())})
    hspec = parse_spec(args.holdout_stations)
    if hspec:
        held = {("stn", o.station) for d in data.values() for o in d["obs"] if is_heldout(o.station, hspec)}
        args.holdout = len(held)            # report: number of held-out stations
    elif args.fold:
        k_, K_ = (int(v) for v in args.fold.split("/"))
        perm = np.random.default_rng(12345).permutation(len(locs))   # fixed: folds never overlap
        held = {locs[i] for j, i in enumerate(perm) if j % K_ == k_}
        args.holdout = len(held) / max(len(locs), 1)
    else:
        n_hold = int(round(args.holdout * len(locs)))
        held = {locs[i] for i in rng.choice(len(locs), n_hold, replace=False)} if n_hold else set()

    def held_flag(o, a, b):
        return ("stn", o.station) in held if hspec else (int(a), int(b)) in held

    def split(d, keep_held):
        m = np.array([held_flag(o, a, b) == keep_held
                      for o, a, b in zip(d["obs"], d["iy"].numpy(), d["ix"].numpy())], bool)
        sub = lambda t: tf.boolean_mask(t, m)
        op, dz = d["op"], d.get("dz")
        if dz is not None:                      # operator must follow the subset
            dz = dz[m]
            op = d["rebuild"](dz)
        return dict(y=sub(d["y"]), s=sub(d["s"]), iy=sub(d["iy"]), ix=sub(d["ix"]),
                    op=op, n=int(m.sum()), dz=dz)

    train = {k: split(d, False) for k, d in data.items()}
    test = {k: split(d, True) for k, d in data.items()} if held else {}

    banner("OBSERVATION SPLIT")
    if hspec:
        print(f"station hold-out {args.holdout_stations}: {len(held)} stations never assimilated")
    else:
        print(f"locations: {len(locs)}  held out: {len(held)}  (fraction {args.holdout}, seed {args.seed})")
    for k in data:
        print(f"  {k:4s} assimilated={train[k]['n']:5d}  held-out={test[k]['n'] if test else 0:5d}")

    def evaluate(x, subset):
        x = tf.convert_to_tensor(x, tf.float32)
        return {k: rms((d["op"](x, d["iy"], d["ix"]) - d["y"]).numpy()) if d["n"] else np.nan
                for k, d in subset.items()}

    fit_b_tr = evaluate(x_b, train)
    fit_b_te = evaluate(x_b, test) if test else {}

    # ------------------------------------------------------------------
    # sigma_b sweep; outer loops keep z_b fixed, first guess moves
    # ------------------------------------------------------------------
    sweep = []
    sigmas = args.sigma_b or [cfg["ls3dvar"]["sigma_b"]]
    lengths = [None]
    if args.control == "pca":
        raise SystemExit("control_vector pca is not available in this release: use blocks")
    if args.control == "ps":                     # backward compatible alias
        args.control = "blocks"
    class _BlockStd(dict):
        """name -> (channel, std); any channel with known stats (aliases U10/V10 ok)"""
        def __missing__(self, key):
            nm = resolve_name(names, key)
            val = (names.index(nm), stats.get(nm)[1])
            self[key] = val
            return val

        def __contains__(self, key):
            try:
                self[key]
                return True
            except KeyError:
                return False
    BLOCK_STD = _BlockStd()
    block_specs = []                              # list of (name, sigma_phys, [L...])
    land = None
    if "LAND_raw" in hrrr.files:
        land = np.clip(np.asarray(hrrr["LAND_raw"], np.float32), 0.0, 1.0)
    for spec in args.block:
        parts = spec.split(":")
        name, sig, L = parts[:3]
        masked = len(parts) > 3 and parts[3].lower() == "land"
        if name.upper() not in BLOCK_STD:
            raise SystemExit(f"--block {spec}: unknown channel or no stats (see channel_stats)")
        if masked and land is None:
            raise SystemExit("--block ...:land needs LAND_raw in the hrrr npz")
        block_specs.append((name.upper(), float(sig), float(L), masked))
    ps_on = bool(block_specs) or args.ps_sigma > 0
    if args.control == "blocks" and not ps_on:
        raise SystemExit("--control blocks needs at least one --block (or --ps-sigma)")
    if args.control == "latent" and ps_on:
        raise SystemExit("--block / --ps-sigma need --control blocks")
    ps_lengths = args.ps_corr_length if args.ps_sigma > 0 else [None]
    if args.control == "blocks":
        sigmas = [0.0]

    # grid spacing (for reporting correlation lengths in km)
    lat, lon = np.deg2rad(hrrr["lats"]), np.deg2rad(hrrr["lons"])
    dx_km = float(np.median(6371.0 * np.hypot(np.diff(lat, axis=1)[:-1, :],
                                              np.cos(lat[:-1, :-1]) * np.diff(lon, axis=1)[:-1, :])))
    print(f"\nmodel grid spacing ~ {dx_km:.1f} km  (latent cell ~ {8 * dx_km:.0f} km)")

    truth = None
    if args.truth_npz:
        tz = np.load(args.truth_npz, allow_pickle=True)
        truth = np.asarray(tz["model_input"][..., :x_b.shape[-1]], np.float32)
        t_truth = str(tz["init_datetime"])
        if np.datetime64(t_truth.replace(" ", "T"), "h") != \
                np.datetime64(str(hrrr["init_datetime"]).replace(" ", "T"), "h"):
            print(f"WARNING: truth valid {t_truth} != analysis {hrrr['init_datetime']}")
        # distance (grid cells) to nearest assimilated obs location
        H, W = x_b.shape[1:3]
        yy, xx = np.mgrid[0:H, 0:W]
        pts = {(int(a), int(b)) for d in train.values()
               for a, b in zip(d["iy"].numpy(), d["ix"].numpy())}
        dist = np.full((H, W), np.inf)
        for a, b in pts:
            dist = np.minimum(dist, np.hypot(yy - a, xx - b))
        near = dist <= args.near_radius
        is_land = (np.asarray(hrrr["LAND_raw"]) > 0.5) if "LAND_raw" in hrrr.files \
            else np.ones_like(near)
        print(f"\ntruth: {args.truth_npz}  ({t_truth})")
        print(f"  near-obs region (<= {args.near_radius} cells ~ {args.near_radius * dx_km:.0f} km): "
              f"{100 * near.mean():.1f}% of grid   far-land {100 * (~near & is_land).mean():.1f}%   "
              f"far-sea {100 * (~near & ~is_land).mean():.1f}%")
        bg_minus_truth = {}

    # ------------------------------------------------------------------
    # pseudo-observations / insertion from the reference (fake OSSE)
    # ------------------------------------------------------------------
    po_specs, po_box, ins_names, ins_w = pseudo_settings(args, cfg)
    refc_src, mrms_dir, mrms_field = refc_source_settings(args, cfg)
    ch_refc = names.index(resolve_name(names, "REFC")) if "REFC" in names else None
    used = {resolve_name(names, n) for n, *_ in po_specs} | {resolve_name(names, n) for n in ins_names}
    need_truth = bool(used - ({"REFC"} if refc_src == "mrms" else set()))
    if need_truth and truth is None:
        raise SystemExit("pseudo_obs / insert need --truth-npz (reference at the analysis time)")
    src, src_label = truth, args.truth_npz
    if used and refc_src == "mrms":
        mr, mp = load_mrms_refc(mrms_dir, hrrr["init_datetime"], mrms_field, x_b.shape[1:3])
        mu_r, sd_r = stats.get("REFC")
        src = (truth if truth is not None else x_b).copy()
        src[0, ..., ch_refc] = (mr - mu_r) / sd_r                     # NaN where no coverage
        src_label = f"REFC from MRMS {mp} [{mrms_field}]" + (f", other channels {args.truth_npz}"
                                                             if truth is not None else "")
        print(f"\nREFC source: MRMS {mp}  field {mrms_field}  mean {np.nanmean(mr):.3f} dBZ  "
              f">=20 dBZ {100 * np.nanmean(mr >= 20):.2f}%  no coverage {100 * np.isnan(mr).mean():.1f}%")
    pseudo, inserts = {}, []
    if po_specs:
        banner(f"PSEUDO-OBS from {src_label}  (superob box {po_box} cells ~ {po_box * dx_km:.0f} km)")
        H_, W_ = x_b.shape[1:3]
        by, bx = superob_boxes(H_, W_, po_box)
        truth_tf = tf.constant(src)
        for name, err, floor in po_specs:
            nm = resolve_name(names, name)
            ch = names.index(nm)
            mu, sd = stats.get(nm)
            op_ = make_superob_operator(ch, mu, sd, po_box, floor)
            y_all = op_(truth_tf, tf.constant(by), tf.constant(bx)).numpy()
            ok = np.isfinite(y_all)
            iy_, ix_ = tf.constant(by[ok]), tf.constant(bx[ok])
            y_ = tf.constant(y_all[ok].astype(np.float32))
            hb = op_(x_b_tf, iy_, ix_).numpy()
            pseudo[f"po_{nm}"] = dict(y=y_, s=tf.constant(np.full(int(ok.sum()), err, np.float32)),
                                      iy=iy_, ix=ix_, op=op_, n=int(ok.sum()), dz=None)
            yv = y_.numpy()
            ref_raw = np.nan_to_num(src[0, ..., ch] * sd + mu, nan=0.0)
            msg = (f"  {nm:10s} ch {ch:3d}  n={int(ok.sum()):6d}  error={err:g} {unit_of(nm)}  "
                   f"floor={floor}  ref mean={yv.mean():.3f}  bg mean={hb.mean():.3f}  "
                   f"O-B rms={rms(hb - yv):.3f}  (raw ref min {ref_raw.min():.2f} max {ref_raw.max():.2f})")
            if floor is not None:
                thr = floor + 5.0
                msg += (f"\n             tiles > floor+5: ref {100 * (yv > thr).mean():.1f}%  "
                        f"bg {100 * (hb > thr).mean():.1f}%")
            print(msg)
            if not any(resolve_name(names, b[0]) == nm for b in block_specs):
                print(f"  NOTE: no --block on {nm}: only the latent part can move it (weak for REFC)")
        train.update(pseudo)
    for name in ins_names:
        nm = resolve_name(names, name)
        inserts.append((nm, names.index(nm)))
    if inserts:
        print(f"\nINSERT after the solve: {', '.join(n for n, _ in inserts)}  weight {ins_w:g}  "
              f"(from {src_label})")

    TRUTH_VARS = [("T2M", T2M_CHANNEL, T2M_STD, "K"), ("D2M", D2M_CHANNEL, D2M_STD, "K"),
                  ("PRES", PRES_CHANNEL, PRES_STD, "Pa")]
    for key in ("U10", "V10", "TMP_850", "REFC", "APCP"):
        try:
            nm = resolve_name(names, key)
        except KeyError:
            continue
        try:
            sd_, un_ = stats.get(nm)[1], unit_of(nm)
        except KeyError:
            sd_, un_ = 1.0, "sigma"
        TRUTH_VARS.append((key, names.index(nm), sd_, un_))

    bmats = {}
    combos = [(L, sb, Lps) for L in lengths for sb in sigmas for Lps in ps_lengths]

    for L, sigma_b, Lps in combos:
        bmat = None
        if ps_on:
            blocks = [(BLOCK_STD[n][0], sg / BLOCK_STD[n][1], Lb, n + ("@land" if mk else ""))
                      + ((land,) if mk else ()) for n, sg, Lb, mk in block_specs]
            if args.ps_sigma > 0:
                blocks.append((PRES_CHANNEL, args.ps_sigma / PRES_STD, Lps, "PRES"))
            cs = HybridControlRuntime(runtime, bmat, latent_b, sigma_b, blocks,
                                      x_b.shape[1:3], x_b.shape[-1])
            ctrl_b = tf.zeros(cs.control_shape, tf.float32)
            std_of = lambda n: BLOCK_STD[n.split("@")[0]][1]
            btag = "_".join(f"{n.replace('@', '-')}{sg * std_of(n):g}L{Lb:g}"
                            for _, sg, Lb, n, *_ in blocks)
            tag = btag
            print(f"\nHYBRID: latent block={cs.n_z:,}  state blocks={len(blocks)} x {cs.n_blk:,}")
            for _, sg, Lb, n, *_ in blocks:
                print(f"  {n:10s} sigma={sg * std_of(n):g} {unit_of(resolve_name(names, n.split('@')[0]))}  "
                      f"L={Lb:g} cells (~{Lb * dx_km:.0f} km)")
        else:
            cs = runtime
            ctrl_b = latent_b
            tag = f"sb{sigma_b:g}"

        solver = LS3DVARMultiSolver(cs)
        ctrl_sigma = sigma_b if args.control == "latent" else 1.0
        to_latent = cs.to_latent if hasattr(cs, "to_latent") else (lambda c: c)
        sb_dir = outdir / tag
        sb_dir.mkdir(parents=True, exist_ok=True)
        first_guess = ctrl_b
        j_ref = None

        for outer in range(args.outer):
            banner(f"{tag}   OUTER LOOP {outer}")
            stops = {}
            if args.stop_rel is not None:
                stops.update(stop_rel=args.stop_rel, stop_window=args.stop_window)
            if args.stop_reduction is not None:
                stops.update(stop_reduction=args.stop_reduction, j_ref=j_ref)
            ctrl_a, result, inner = solver.solve(
                latent_background=ctrl_b, latent_first_guess=first_guess, state=state,
                groups=train, sigma_b=ctrl_sigma, max_iters=args.maxiter, **stops,
            )
            if j_ref is None and inner:
                j_ref = inner[0]["J"]
            pd.DataFrame(inner).to_csv(sb_dir / f"lbfgs_history_outer_{outer:02d}.csv", index=False)
            try:
                plot_convergence(inner, sb_dir, outer)
            except Exception as e:                    # noqa: BLE001 (old diagnostics expect JT2M ...)
                print(f"  (convergence plot skipped: {e})")
            first_guess = ctrl_a
            if getattr(result, "message", "") == "reduction" and outer + 1 < args.outer:
                print(f"  (J reduction reached: outer loops {outer + 1}..{args.outer - 1} skipped)")
                break

        latent_a = to_latent(ctrl_a)
        x_a = np.array(cs.forecast_only(ctrl_a, state).numpy(), copy=True)   # includes blocks
        if inserts:
            x_pre = x_a.copy()
            for nm_, ch_ in inserts:
                new_ = src[0, ..., ch_]
                x_a[0, ..., ch_] = np.where(np.isfinite(new_),
                                            (1.0 - ins_w) * x_a[0, ..., ch_] + ins_w * np.nan_to_num(new_),
                                            x_a[0, ..., ch_])
        inc = x_a - x_b
        dz = (latent_a - latent_b).numpy()
        fit_a_tr = evaluate(x_a, train)
        fit_a_te = evaluate(x_a, test) if test else {}

        banner(f"{tag}: O-B -> O-A rms")
        dps = inc[..., PRES_CHANNEL] * PRES_STD
        row = {"control": args.control + ("+blocks" if ps_on and args.control != "blocks" else ""),
               "blocks": ";".join(args.block) + (f";PRES:{args.ps_sigma:g}:{Lps:g}" if args.ps_sigma > 0 else ""),
               "corr_length": L, "sigma_b": sigma_b,
               "ps_sigma": args.ps_sigma if ps_on else 0.0, "ps_corr_length": Lps,
               "ps_increment_rms_pa": rms(dps), "ps_increment_max_pa": float(np.abs(dps).max()), "latent_increment_rms": rms(dz),
               "state_increment_rms": rms(inc), "state_increment_max": float(np.abs(inc).max()),
               "Jb": inner[-1]["Jb"], "Jobs": inner[-1]["Jobs"]}
        for k in data:
            imp_tr = 100 * (fit_b_tr[k] - fit_a_tr[k]) / fit_b_tr[k]
            line = (f"  {k:4s} assimilated: {fit_b_tr[k]:9.4f} -> {fit_a_tr[k]:9.4f} "
                    f"({imp_tr:+6.2f}%)")
            row[f"{k}_train_impr_pct"] = imp_tr
            if test and test[k]["n"]:
                imp_te = 100 * (fit_b_te[k] - fit_a_te[k]) / fit_b_te[k]
                line += f"   held-out: {fit_b_te[k]:9.4f} -> {fit_a_te[k]:9.4f} ({imp_te:+6.2f}%)"
                row[f"{k}_heldout_impr_pct"] = imp_te
            print(line)
        for k, d_ in pseudo.items():
            ob_ = rms((d_["op"](tf.constant(x_b), d_["iy"], d_["ix"]) - d_["y"]).numpy())
            oa_ = rms((d_["op"](tf.constant(x_a), d_["iy"], d_["ix"]) - d_["y"]).numpy())
            imp_ = 100 * (ob_ - oa_) / ob_ if ob_ else 0.0
            row[f"{k}_impr_pct"] = imp_
            print(f"  {k:10s} pseudo-obs: {ob_:9.4f} -> {oa_:9.4f} ({imp_:+6.2f}%)"
                  + ("   (after insertion)" if inserts else ""))
        if inserts:
            for nm_, ch_ in inserts:
                sd_ = stats.get(nm_)[1]
                print(f"  insert {nm_}: rms change from insertion "
                      f"{rms((x_a[0, ..., ch_] - x_pre[0, ..., ch_]) * sd_):.3f} {unit_of(nm_)}")
        row["pseudo_obs"] = ";".join(f"{n}:{e:g}" + (f":{f:g}" if f is not None else "")
                                     for n, e, f in po_specs)
        row["insert"] = ";".join(n for n, _ in inserts)
        print(f"  latent increment rms = {rms(dz):.4e}   state increment rms = {rms(inc):.4e} "
              f"sigma   max = {np.abs(inc).max():.4e}")
        print(f"  PRES increment rms = {rms(dps):.1f} Pa   max = {np.abs(dps).max():.1f} Pa")

        # domain-mean O-A bias (does the analysis remove the background bias?)
        for k in data:
            d_ = data[k]
            ob = (d_["op"](tf.constant(x_b), d_["iy"], d_["ix"]) - d_["y"]).numpy()
            oa = (d_["op"](tf.constant(x_a), d_["iy"], d_["ix"]) - d_["y"]).numpy()
            print(f"  {k:4s} mean(H x - y): background {ob.mean():+8.3f}   analysis {oa.mean():+8.3f}")

        # per-station report (O-B, O-A, held-out flag) -> station_report.csv
        srows = []
        for k, d_ in data.items():
            ob = (d_["op"](tf.constant(x_b), d_["iy"], d_["ix"]) - d_["y"]).numpy()
            oa = (d_["op"](tf.constant(x_a), d_["iy"], d_["ix"]) - d_["y"]).numpy()
            for j, o in enumerate(d_["obs"]):
                loc = (int(d_["iy"].numpy()[j]), int(d_["ix"].numpy()[j]))
                srows.append(dict(variable=k, station=o.station, network=o.source, lat=o.lat, lon=o.lon,
                                  iy=loc[0], ix=loc[1], held_out=held_flag(o, *loc), sigma_o=float(d_["s"].numpy()[j]),
                                  omb=float(-ob[j]), oma=float(-oa[j]), fold=args.fold or "", seed=args.seed))
        for r in obs_qc.rejected:
            srows.append(dict(r, held_out=False, omb=r["omb"], oma=np.nan, rejected=r["reason"],
                              fold=args.fold or "", seed=args.seed))
        sfx = (f"stn{args.holdout_stations.replace('/', 'of')}" if args.holdout_stations else
               f"fold{args.fold.replace('/', 'of')}" if args.fold else f"seed{args.seed}")
        srep = sb_dir / f"station_report_{sfx}.csv"
        pd.DataFrame(srows).to_csv(srep, index=False)
        print(f"  station report -> {srep}")

        if truth is not None:
            print(f"  vs TRUTH   rms(bg-truth) -> rms(an-truth)   "
                  f"[all | near obs | far-land | far-sea]")
            for name, ch, sd, unit in TRUTH_VARS:
                eb = (x_b[0, ..., ch] - truth[0, ..., ch]) * sd
                ea = (x_a[0, ..., ch] - truth[0, ..., ch]) * sd
                parts = []
                for m in (np.ones_like(near), near, ~near & is_land, ~near & ~is_land):
                    if not m.any():
                        parts.append((np.nan, np.nan, np.nan))
                        continue
                    rb, ra = rms(eb[m]), rms(ea[m])
                    parts.append((rb, ra, 100 * (rb - ra) / rb if rb else 0.0))
                print(f"    {name:4s} " + "  |  ".join(
                    f"{rb:7.3f}->{ra:7.3f} ({imp:+5.2f}%)" for rb, ra, imp in parts) + f"  {unit}")
                row[f"truth_{name}_all_impr_pct"] = parts[0][2]
                row[f"truth_{name}_near_impr_pct"] = parts[1][2]
                row[f"truth_{name}_farland_impr_pct"] = parts[2][2]
                row[f"truth_{name}_farsea_impr_pct"] = parts[3][2]
        sweep.append(row)

        out = {k: hrrr[k] for k in hrrr.files}
        mi = out["model_input"].copy()
        mi[:1, ..., :x_b.shape[-1]] = x_a.astype(mi.dtype)
        out["model_input"] = mi
        out["latent_analysis"] = latent_a.numpy()
        out["latent_background"] = latent_b.numpy()
        np.savez(sb_dir / "analysis_t2m_d2m_ps.npz", **out)
        if args.analysis_out:
            Path(args.analysis_out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sb_dir / "analysis_t2m_d2m_ps.npz", args.analysis_out)
            print(f"analysis -> {args.analysis_out}")
        np.savez(sb_dir / "increment_t2m_d2m_ps.npz", increment=inc, x_b=x_b, x_a=x_a,
                 latent_b=latent_b.numpy(), latent_a=latent_a.numpy())

    df = pd.DataFrame(sweep)
    df.to_csv(outdir / "sigma_b_sweep.csv", index=False)
    banner("SIGMA_B SWEEP  (choose by HELD-OUT improvement)")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {outdir / 'sigma_b_sweep.csv'} and {outdir}/sb*/analysis_t2m_d2m_ps.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
