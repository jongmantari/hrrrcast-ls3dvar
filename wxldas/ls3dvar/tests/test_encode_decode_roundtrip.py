#!/usr/bin/env python3
# ls3dvar/tests/test_encode_decode_roundtrip.py  (extracted from mantari/workflows/test_encode_decode_roundtrip.py)
"""
Encode -> decode round trip, set up so HRRRCast acts as a denoiser of the
input state instead of a forecaster.

Why the previous version could not reproduce the input
-------------------------------------------------------
HRRRCast is a diffusion x0-predictor (see src HRRRCast.forward):

    out = refined + (hrrr - noised) * skip_scale + noised

where `noised` is the diffusion iterate x_t for the *target* state and the
step channel carries t / NUM_DIFFUSION_STEPS. build_operational_input fills
noised, date, step and lead with ONES placeholders (fcst.py overwrites them
during the rollout), so the network was asked to denoise a constant field at
diffusion step 200 -> it hallucinates a large increment.

Round-trip configuration used here
----------------------------------
    hrrr slot   = x                       (state to reproduce)
    noised slot = x            (clean, default)   or
                = sqrt(ab_t) x + sqrt(1-ab_t) eps (with --t T)
    step slot   = t / NUM_DIFFUSION_STEPS (0 by default: "almost clean")
    date slot   = real cyclic features from the npz init_datetime
    lead slot   = lead_hours / 6

At t = 0 the network was trained to return ~x_t, so out ~= x and
refined ~= 0. The script runs two paths and compares both to the input:

    bypass : encoder -> decoder(latent in place of proc_out) -> tail
    full   : encoder -> processor -> decoder -> tail        (reference)

and writes the chosen one (--write) to a new hrrr npz.

Run:
    python -m ls3dvar.tests.test_encode_decode_roundtrip
    python -m ls3dvar.tests.test_encode_decode_roundtrip --write full
    python -m ls3dvar.tests.test_encode_decode_roundtrip --t 8 --seed 0
    python -m ls3dvar.tests.test_encode_decode_roundtrip --legacy-placeholders
    python -m ls3dvar.tests.test_encode_decode_roundtrip --anchor --tl-check
"""
import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from ls3dvar.tests.test_decode_parity import (
    CUSTOM_OBJECTS,
    SKIP0_LAYER,
    SKIP1_LAYER,
    PROCESSED_LAYER,
    TIME_VEC_LAYER,
    MODEL_PATH,
    banner,
    build_operational_input,
    compare,
    load_config,
    to_np,
)
import resnet

from ls3dvar.hrrrcast.encoder import EncoderRuntime
from ls3dvar.hrrrcast.decoder_numpy import DecoderRuntime
from ls3dvar.hrrrcast.tail import TailModelRuntimeTF

REFINE_LAYER = "output_refine_recomp"
NUM_DIFFUSION_STEPS = 200
N_DATE, N_STEP, N_LEAD = 7, 1, 1


# =============================================================================
# Diffusion schedule (copy of src diffusion_params: cosine, T=200)
# =============================================================================

def alpha_bar_schedule(T=NUM_DIFFUSION_STEPS, s=0.008):
    f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    ab = np.array([f(t) for t in range(T + 1)])
    ab = ab / ab[0]
    beta = np.clip(1 - ab[1:] / ab[:-1], 1e-4, 0.9999)
    return np.cumprod(1.0 - beta).astype(np.float32)


ALPHA_BAR = alpha_bar_schedule()


# =============================================================================
# Date features (copy of fcst.WeatherForecaster.compute_time_features)
# =============================================================================

def time_features(init_dt, lead_hours):
    tc = pd.DatetimeIndex([pd.Timestamp(init_dt) + pd.Timedelta(hours=lead_hours)])
    hours = tc.hour.astype(np.float32)
    doy = tc.dayofyear.astype(np.float32)
    v4 = (tc >= np.datetime64("2021-03-23T00")).astype(np.float32)
    v3 = ((tc >= np.datetime64("2018-07-12T00"))
          & (tc < np.datetime64("2021-03-23T00"))).astype(np.float32)
    gfs15 = ((tc >= np.datetime64("2019-06-01T00"))
             & (tc < np.datetime64("2021-03-23T00"))).astype(np.float32)
    return np.stack([
        np.sin(2 * np.pi * hours / 24.0), np.cos(2 * np.pi * hours / 24.0),
        np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0),
        v4, v3, gfs15,
    ], axis=-1).astype(np.float32)[0]            # (7,)


# =============================================================================
# Input assembly
# =============================================================================

def build_roundtrip_input(hrrr_npz, gfs_npz, t, seed, clean, lead_hours):
    """Layout (TF, 329 ch):
    [x 138][gfs 42][noised 138][hrrr static 2][date 7][step 1][lead 1]"""
    X, n_state = build_operational_input(hrrr_npz, gfs_npz)   # placeholders
    hrrr = np.load(hrrr_npz, allow_pickle=True)
    x = X[..., :n_state]
    n_gfs = np.load(gfs_npz, allow_pickle=True)["model_input"].shape[-1]
    s0 = n_state + n_gfs                                       # noised slot start

    if clean:
        noised = x.copy()
    else:
        ab = ALPHA_BAR[t]
        eps = np.random.default_rng(seed).standard_normal(x.shape).astype(np.float32)
        noised = np.sqrt(ab) * x + np.sqrt(1.0 - ab) * eps

    X = X.copy()
    X[..., s0:s0 + n_state] = noised
    init_dt = datetime.fromisoformat(str(hrrr["init_datetime"]))
    X[..., -(N_DATE + N_STEP + N_LEAD):-(N_STEP + N_LEAD)] = time_features(init_dt, lead_hours)
    X[..., -2] = t / NUM_DIFFUSION_STEPS
    X[..., -1] = lead_hours / 6.0
    info = dict(init=init_dt, t=t, ab=float(ALPHA_BAR[t]), clean=clean,
                lead=lead_hours, noised_slot=(s0, s0 + n_state))
    return X.astype(np.float32), n_state, info


# =============================================================================
# Tail discovery (shortcut = input of UnpadLayer)
# =============================================================================

def as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def find_shortcut_tensor(model, name=None):
    if name:
        return model.get_layer(name).output
    unpads = [l for l in model.layers if isinstance(l, resnet.UnpadLayer)]
    if not unpads:
        raise RuntimeError("No UnpadLayer found; pass --shortcut-layer NAME")
    return as_list(unpads[-1].input)[0]


# =============================================================================
# npz + metrics
# =============================================================================

def channel_names(hrrr):
    """fcst.py ordering: for v in pl_vars: for lev in levels, then sfc_vars."""
    try:
        pl = [str(v) for v in hrrr["pl_vars"]]
        lv = [str(int(float(l))) for l in hrrr["levels"]]
        return [f"{v}_{l}" for v in pl for l in lv] + [str(v) for v in hrrr["sfc_vars"]]
    except Exception:
        return None


def write_roundtrip_npz(src_path, out_path, recon, n_state):
    src = np.load(src_path, allow_pickle=True)
    out = {k: src[k] for k in src.files}
    mi = out["model_input"].copy()
    mi[:1, ..., :n_state] = recon.astype(mi.dtype)
    out["model_input"] = mi
    np.savez(out_path, **out)


def stats(ref, got):
    err = got - ref
    return (float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err))),
            float(np.mean(err)),
            float(np.linalg.norm(err.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12)))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def per_channel_report(ref, got, names, tol_rmse, show):
    """Pass/fail on per-channel rmse. model_input is normalized, so rmse is in
    units of climatological std (sigma); rel_l2 is shown but not judged because
    it blows up for near-zero-mean fields (VVEL, clouds)."""
    print(f"{'channel':<20}{'rmse[sig]':>11}{'max_abs':>11}{'bias':>11}{'rel_l2':>11}")
    rows = [(names[c] if names else f"ch{c:03d}", *stats(ref[..., c], got[..., c]))
            for c in range(ref.shape[-1])]
    order = sorted(rows, key=lambda r: -r[1]) if show else rows
    for r in order[: show or None]:
        flag = "" if r[1] <= tol_rmse else "  FAIL"
        print(f"{r[0]:<20}{r[1]:>11.3e}{r[2]:>11.3e}{r[3]:>11.3e}{r[4]:>11.3e}{flag}")
    if show and len(rows) > show:
        print(f"... ({len(rows) - show} more, worst rmse first; --show 0 for all)")
    g = stats(ref, got)
    print("-" * 64)
    print(f"{'ALL':<20}{g[0]:>11.3e}{g[1]:>11.3e}{g[2]:>11.3e}{g[3]:>11.3e}")
    n_fail = sum(r[1] > tol_rmse for r in rows)
    print(f"failing channels: {n_fail}/{len(rows)}  (rmse tol {tol_rmse:g} sigma)")
    return n_fail == 0


def latent_tl_check(decode, z_b, refine_b, unpad, eps_list, seed):
    """Tangent-linear sanity for latent 3DVar.

    dx(eps) = unpad(refine(z_b + eps*sz*n) - refine(z_b))
    linearity = ||dx(2eps) - 2 dx(eps)|| / ||dx(2eps)||   (-> 0 if linear)
    symmetry  = ||dx(eps) + dx(-eps)|| / ||dx(eps)||      (-> 0 if linear)
    """
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(z_b.shape).astype(np.float32)
    sz = float(np.std(z_b))
    dx = lambda e: unpad(decode(z_b + e * sz * n) - refine_b)
    nrm = lambda a: float(np.linalg.norm(a.ravel()))
    print(f"latent std = {sz:.3e}   perturbation = eps * std * N(0,1)")
    print(f"{'eps':>8}{'rms(dx)':>12}{'gain':>12}{'linearity':>12}{'symmetry':>12}")
    for e in eps_list:
        d1, d2, dm = dx(e), dx(2 * e), dx(-e)
        gain = rms(d1) / (e * sz)          # state-rms per latent-rms
        lin = nrm(d2 - 2 * d1) / (nrm(d2) + 1e-30)
        sym = nrm(d1 + dm) / (nrm(d1) + 1e-30)
        print(f"{e:>8.0e}{rms(d1):>12.3e}{gain:>12.3e}{lin:>12.3e}{sym:>12.3e}")


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrrr-npz", default=None)
    ap.add_argument("--gfs-npz", default=None)
    ap.add_argument("--out", default="/tmp/hrrr_roundtrip.npz")
    ap.add_argument("--write", choices=["bypass", "full"], default="bypass",
                    help="which path's reconstruction goes into the new npz")
    ap.add_argument("--t", type=int, default=0, help="diffusion step (0..199)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noisy", action="store_true",
                    help="noised slot = q(x_t|x) sample instead of clean x")
    ap.add_argument("--lead-hours", type=int, default=1)
    ap.add_argument("--legacy-placeholders", action="store_true",
                    help="use build_operational_input as-is (ones) for comparison")
    ap.add_argument("--tol-rmse", type=float, default=0.05,
                    help="per-channel rmse tolerance in sigma (normalized) units")
    ap.add_argument("--anchor", action="store_true",
                    help="write x + refine(z) - refine(z_b) (exact at z=z_b)")
    ap.add_argument("--tl-check", action="store_true",
                    help="latent tangent-linear / sensitivity check (bypass path)")
    ap.add_argument("--tl-eps", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--shortcut-layer", default=None)
    args = ap.parse_args()

    banner("LOAD MODEL")
    model = tf.keras.models.load_model(str(MODEL_PATH), custom_objects=CUSTOM_OBJECTS,
                                       compile=False, safe_mode=False)
    cfg = load_config()
    hrrr_path = args.hrrr_npz or cfg["hrrr_npz"]
    gfs_path = args.gfs_npz or cfg["gfs_npz"]

    if args.legacy_placeholders:
        X, n_state = build_operational_input(hrrr_path, gfs_path)
        print("input: legacy placeholders (noised/date/step/lead = 1)")
    else:
        X, n_state, info = build_roundtrip_input(
            hrrr_path, gfs_path, args.t, args.seed, not args.noisy, args.lead_hours)
        print(f"input: denoise mode  init={info['init']}  t={info['t']}  "
              f"alpha_bar={info['ab']:.6f}  noise_std={np.sqrt(1 - info['ab']):.4f}  "
              f"clean={info['clean']}  lead={info['lead']}h")
    print("X:", X.shape, " predicted channels:", n_state, " hrrr npz:", hrrr_path)

    L = model.get_layer
    taps = {
        "skip0": L(SKIP0_LAYER).output,
        "skip1": L(SKIP1_LAYER).output,
        "processed": L(PROCESSED_LAYER).output,
        "time_vec": L(TIME_VEC_LAYER).output,
        "refine_full": L(REFINE_LAYER).output,
        "shortcut_full": find_shortcut_tensor(model, args.shortcut_layer),
    }
    banner("REFERENCE TAPS (full model, same input)")
    tap_model = tf.keras.Model(inputs=model.input, outputs=list(taps.values()))
    ref = dict(zip(taps, [to_np(t) for t in tap_model(X, training=False)]))
    for k, v in ref.items():
        print(f"  {k:16s} {v.shape}")

    # residual = (hrrr - noised) * skip_scale + noised  (input-only, no processor)
    residual = ref["shortcut_full"] - ref["refine_full"]
    tail = TailModelRuntimeTF()
    zeros = np.zeros_like(residual)
    up = lambda a: to_np(tail.unpad(tf.constant(a)))
    results = []

    banner("TAIL SANITY (runtime tail vs model output)")
    full_recon = to_np(tail.forecast(tf.constant(ref["refine_full"]),
                                     tf.constant(residual), tf.constant(zeros)))
    model_out = [to_np(o) for o in as_list(model(X, training=False))]
    match = [o for o in model_out if o.shape == full_recon.shape]
    if match:
        results.append(compare("tail", full_recon, match[0], args.atol, args.rtol))

    banner("BYPASS PATH: encoder -> decoder(latent) -> tail")
    latent = to_np(EncoderRuntime(model).encode(X))
    decoder = DecoderRuntime(model)
    skips = [tf.constant(ref[k]) for k in ("skip1", "skip0", "processed", "time_vec")]
    decode = lambda z: to_np(decoder.decode_features(tf.constant(z), *skips)["refine"])
    refine_bp = decode(latent)
    bypass_recon = to_np(tail.forecast(tf.constant(refine_bp),
                                       tf.constant(residual), tf.constant(zeros)))
    print("latent:", latent.shape, " recon:", bypass_recon.shape)

    recon = bypass_recon if args.write == "bypass" else full_recon
    if args.anchor:
        # x(z) = x_b + unpad(refine(z) - refine(z_b)); evaluated at z = z_b
        x_b = X[..., :n_state]
        recon = x_b + up(decode(latent) - refine_bp)
        print("anchored decode: x_b + refine(z) - refine(z_b) at z = z_b")
    tag = f"{args.write}{'+anchor' if args.anchor else ''}"
    banner(f"WRITE NPZ ({tag})")
    write_roundtrip_npz(hrrr_path, args.out, recon, n_state)
    print("wrote", args.out)

    orig = np.load(hrrr_path, allow_pickle=True)
    new = np.load(args.out, allow_pickle=True)
    x_orig = orig["model_input"][:1, ..., :n_state].astype(np.float32)
    x_new = new["model_input"][:1, ..., :n_state].astype(np.float32)
    names = channel_names(orig)
    untouched = np.array_equal(orig["model_input"][..., n_state:],
                               new["model_input"][..., n_state:])
    print("non-state channels unchanged:", untouched)
    results.append(untouched)

    banner(f"ROUND TRIP ({tag}): new npz vs original npz")
    results.append(per_channel_report(x_orig, x_new, names, args.tol_rmse, args.show))

    if args.tl_check:
        banner("LATENT TANGENT-LINEAR CHECK (bypass decoder, skips frozen)")
        latent_tl_check(decode, latent, refine_bp, up, args.tl_eps, args.seed)

    banner("PATH SUMMARY (unpadded, normalized units)")
    print(f"  rms(x_orig)                    = {rms(x_orig):.3e}")
    print(f"  rms(residual - x_orig)         = {rms(up(residual) - x_orig):.3e}"
          "   (direct skip path)")
    print(f"  rms(refine)  bypass            = {rms(up(refine_bp)):.3e}")
    print(f"  rms(refine)  full              = {rms(up(ref['refine_full'])):.3e}")
    for name, r in [("bypass", bypass_recon), ("full", full_recon)]:
        g = stats(x_orig, r)
        print(f"  {name:6s} recon vs x: rmse={g[0]:.3e} max_abs={g[1]:.3e} "
              f"bias={g[2]:.3e} rel_l2={g[3]:.3e}")
    print(f"  bypass vs full: rms diff       = {rms(bypass_recon - full_recon):.3e}")

    banner("SUMMARY")
    ok = all(results)
    print("ROUND TRIP PASS" if ok else "ROUND TRIP FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
