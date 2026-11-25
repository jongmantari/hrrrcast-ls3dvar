#!/usr/bin/env python3
# ls3dvar/cli/make_forecast_background.py  (extracted from mantari/workflows/make_forecast_background.py)
"""
Build an LS-3DVar BACKGROUND npz from a short HRRRCast forecast.

    hrrr/gfs npz at INIT  --HRRRCast diffusion forecast, lead L h-->  x_f (normalized)
                          --> background npz valid at INIT + L

The output npz is a copy of the INIT hrrr npz with
    model_input[..., :138]  = forecast state (normalized, same units as model_input)
    init_datetime / init_*  = VALID time  (so the LS-3DVar driver treats it as x_b)
    static channels, lats/lons, OROG_raw, LAND_raw, metadata unchanged
    + bg_source_init, bg_lead_hours, bg_member, bg_sampler

Mirrors fcst.py WeatherForecaster.autoregressive_rollout() + predict() for
lead <= 6 h (all leads come straight from the INIT state):

    X_base   = [ X[pred+gfs : -9], date(valid), X[-2:-1], lead/6 ]
    X_member = [ state_init, gfs_model_input[hour-1], X_base ]
    noise    : anchor N(0,1) seed [member, 0], advanced hourly with make_noise(rho)
    sampler  : DPM-Solver++(2M) over INFERENCE_STEPS (fcst.py default)

Not reproduced: output clipping to physical bounds (negligible for
T2M / D2M / PRES; matters for bounded fields such as APCP, clouds).

Run:
    python -m ls3dvar.cli.make_forecast_background \
        --hrrr-npz .../20260810/23/hrrr_20260810_23.npz \
        --gfs-npz  .../20260810/23/gfs_20260810_23.npz \
        --lead 1 2 3 \
        --out .../fcst_f{lead:02d}.npz \
        [--ref-nc .../20260810/23/hrrrcast_m00_f01.nc]   # check vs operational output
"""
import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ls3dvar.paths import add_hrrrcast_src  # noqa: E402

add_hrrrcast_src()

import resnet  # noqa: E402
from diffusion_params import (  # noqa: E402
    NUM_DIFFUSION_STEPS,
    NUM_INFERENCE_STEPS,
    INFERENCE_STEPS,
    compute_epsilon,
    ddim,
    dpmpp_2m,
)
from fcst import make_noise  # noqa: E402

from ls3dvar.paths import MODEL_PATH  # noqa: E402
from ls3dvar.hrrrcast.time_features import make_date_encoding  # noqa: E402
from ls3dvar.obs.operators import (  # noqa: E402
    T2M_CHANNEL, T2M_MEAN, T2M_STD, PRES_CHANNEL, PRES_MEAN, PRES_STD,
)

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

ROLLOUT_HOUR = 6


def banner(t):
    print()
    print("=" * 80)
    print(t)
    print("=" * 80)


def predicted_channels(hrrr):
    return len(hrrr["pl_vars"]) * len(hrrr["levels"]) + len(hrrr["sfc_vars"])


def initial_input(hrrr, gfs, n_pred):
    """fcst.py main(): placeholders (ones) for noise/date/step/lead."""
    mi_h = hrrr["model_input"].astype(np.float32)
    mi_g = gfs["model_input"].astype(np.float32)
    nlat, nlon = mi_h.shape[1:3]
    ones = lambda c: np.ones((1, nlat, nlon, c), np.float32)
    return np.concatenate(
        [mi_h[..., :n_pred], mi_g[0:1], ones(n_pred), mi_h[..., n_pred:],
         ones(7), ones(1), ones(1)], axis=-1)


def member_input(X, gfs_input, n_pred, n_gfs, init_dt, hour):
    """fcst.py autoregressive_rollout(), hour <= 6, member phase shift 0."""
    nlat, nlon = X.shape[1:3]
    start = n_pred + n_gfs
    step = hour - ((hour - 1) // ROLLOUT_HOUR) * ROLLOUT_HOUR
    valid = pd.Timestamp(init_dt) + pd.Timedelta(hours=hour)
    date_enc = make_date_encoding(valid.to_pydatetime(), nlat, nlon)
    lead_enc = np.full((1, nlat, nlon, 1), step / 6.0, np.float32)
    X_base = np.concatenate([X[..., start:-(7 + 2)], date_enc, X[..., -2:-1], lead_enc], axis=-1)
    idx = int(np.clip(hour - 1, 0, gfs_input.shape[0] - 1))
    return np.concatenate([X[..., :n_pred], gfs_input[idx:idx + 1], X_base], axis=-1)


def diffusion_predict(model, X, Xn, n_pred, n_gfs, sampler):
    """fcst.py WeatherForecaster.predict(), diffusion branch."""
    start = n_pred + n_gfs
    X = tf.convert_to_tensor(X, tf.float32)

    def build(X_noisy, t):
        step_enc = tf.fill(tf.concat([tf.shape(X)[:-1], [1]], 0),
                           tf.cast(t / NUM_DIFFUSION_STEPS, X.dtype))
        return tf.concat([X[..., :start], X_noisy, X[..., start + n_pred:-2],
                          step_enc, X[..., -1:]], axis=-1)

    prev_x0 = prev_h = None
    for t_ in range(NUM_INFERENCE_STEPS):
        ti = NUM_INFERENCE_STEPS - 1 - t_
        t = INFERENCE_STEPS[ti]
        x0_t = model(build(Xn, t), training=False)
        if ti == 0:
            Xn = x0_t
        elif sampler == "dpmpp-2m":
            Xn, prev_x0, prev_h = dpmpp_2m(Xn, x0_t, ti, prev_x0=prev_x0, prev_h=prev_h)
        else:
            eps = compute_epsilon(Xn, x0_t, t)
            Xn = ddim(Xn, eps, ti, seed=[0], eta=0.0)
        print(f"  diffusion step {t_ + 1:2d}/{NUM_INFERENCE_STEPS}  t={int(t):3d}")
    return Xn.numpy()


def nc_field(path, var):
    import xarray as xr
    ds = xr.open_dataset(path)
    x = ds[var]
    for d in ("lead_time", "time"):
        if d in x.dims:
            x = x.isel({d: 0})
    return np.asarray(x).squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrrr-npz", required=True, help="hrrr npz at the forecast INIT time")
    ap.add_argument("--gfs-npz", required=True, help="gfs npz at the forecast INIT time")
    ap.add_argument("--lead", type=int, nargs="+", default=[1],
                    help="lead hour(s) 1..6; several leads share one model load")
    ap.add_argument("--out", required=True,
                    help="output npz; with several leads use {lead} in the name, e.g. f{lead:02d}.npz")
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--noise-rho", type=float, default=0.9)
    ap.add_argument("--sampler", choices=["dpmpp-2m", "ddim"], default="dpmpp-2m")
    ap.add_argument("--model-path", default=str(MODEL_PATH))
    ap.add_argument("--ref-nc", default=None,
                    help="operational hrrrcast_mXX_fYY.nc for the same init/lead (check)")
    args = ap.parse_args()

    leads = sorted(set(args.lead))
    if not all(1 <= L <= ROLLOUT_HOUR for L in leads):
        raise SystemExit("--lead must be 1..6 (state is only refreshed every 6 h)")
    if len(leads) > 1 and "{lead" not in args.out:
        raise SystemExit("several --lead values need {lead} in --out")

    hrrr = np.load(args.hrrr_npz, allow_pickle=True)
    gfs = np.load(args.gfs_npz, allow_pickle=True)
    n_pred = predicted_channels(hrrr)
    gfs_input = gfs["model_input"].astype(np.float32)
    n_gfs = gfs_input.shape[-1]
    init_dt = pd.to_datetime(str(hrrr["init_datetime"]))
    banner("FORECAST")
    print(f"init  {init_dt}   leads {leads}")
    print(f"hrrr model_input {hrrr['model_input'].shape}   gfs model_input {gfs_input.shape}  "
          f"(forcing for hour h = gfs[h-1])")
    print(f"member {args.member}   rho {args.noise_rho}   sampler {args.sampler}")

    model = tf.keras.models.load_model(args.model_path, custom_objects=CUSTOM_OBJECTS,
                                       compile=False, safe_mode=False)

    nlat, nlon = hrrr["model_input"].shape[1:3]
    X = initial_input(hrrr, gfs, n_pred)

    # member noise: anchor, then one make_noise() advance per hour (fcst.py order)
    noise = tf.expand_dims(tf.random.stateless_normal(
        (nlat, nlon, n_pred), dtype=tf.float32, seed=[args.member, 0]), 0)
    noise_by_lead = {}
    for hour in range(1, leads[-1] + 1):
        noise = make_noise(noise, hour, args.member, args.noise_rho)
        noise_by_lead[hour] = noise

    for lead in leads:
        valid_dt = init_dt + timedelta(hours=lead)
        out_path = args.out.format(lead=lead)
        banner(f"HRRRCast F{lead:02d}  valid {valid_dt}")
        Xm = member_input(X, gfs_input, n_pred, n_gfs, init_dt, lead)
        print("member input:", Xm.shape)
        x_f = diffusion_predict(model, Xm, noise_by_lead[lead], n_pred, n_gfs, args.sampler)
        print("forecast:", x_f.shape)

        x_init = hrrr["model_input"][..., :n_pred].astype(np.float32)
        d = x_f - x_init
        banner("FORECAST - INIT (normalized)")
        print(f"rms={np.sqrt(np.mean(d ** 2)):.4f}   "
              f"T2M rms={np.sqrt(np.mean(d[..., T2M_CHANNEL] ** 2)) * T2M_STD:.3f} K   "
              f"PRES rms={np.sqrt(np.mean(d[..., PRES_CHANNEL] ** 2)) * PRES_STD:.1f} Pa")

        if args.ref_nc:
            banner(f"CHECK vs operational {args.ref_nc}")
            for var, ch, mu, sd in (("T2M", T2M_CHANNEL, T2M_MEAN, T2M_STD),
                                    ("PRES", PRES_CHANNEL, PRES_MEAN, PRES_STD)):
                try:
                    ref = nc_field(args.ref_nc, var)
                    ours = x_f[0, ..., ch] * sd + mu
                    if ref.shape != ours.shape:
                        print(f"  {var}: shape {ref.shape} vs {ours.shape} (skip)")
                        continue
                    print(f"  {var}: rms diff={np.sqrt(np.nanmean((ours - ref) ** 2)):.4f}   "
                          f"max={np.nanmax(np.abs(ours - ref)):.4f}")
                except Exception as e:
                    print(f"  {var}: {e}")

        out = {k: hrrr[k] for k in hrrr.files}
        mi = out["model_input"].copy()
        mi[..., :n_pred] = x_f.astype(mi.dtype)
        out["model_input"] = mi
        out["init_datetime"] = np.array(valid_dt.strftime("%Y-%m-%d %H:%M:%S"))
        out["init_year"] = np.array(valid_dt.strftime("%Y"))
        out["init_month"] = np.array(valid_dt.strftime("%m"))
        out["init_day"] = np.array(valid_dt.strftime("%d"))
        out["init_hh"] = np.array(valid_dt.strftime("%H"))
        out["bg_source_init"] = np.array(init_dt.strftime("%Y-%m-%d %H:%M:%S"))
        out["bg_lead_hours"] = np.array(lead)
        out["bg_member"] = np.array(args.member)
        out["bg_sampler"] = np.array(args.sampler)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **out)

        banner("WRITTEN")
        print(out_path)
        print(f"valid {valid_dt}  -> set hrrr_npz to this file in test.yaml "
              f"(gfs_npz: the {valid_dt:%Y%m%d %H}Z gfs npz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
