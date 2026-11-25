# ls3dvar/hrrrcast/analysis_input.py  (extracted from mantari/ls3dvar/analysis_input_builder.py)
"""
mantari/ls3dvar/analysis_input_builder.py

HRRRCast input for ANALYSIS-TIME LS-3DVar (no forecast).

HRRRCast is a diffusion x0-predictor:

    out = refine + (hrrr - noised) * skip_scale + noised

Fed with  noised = x_b  and  step = 0  (last diffusion step, ~clean), the
network returns ~x_b, so the decoder acts as a map latent -> analysis-time
state.  Verified in test_encode_decode_roundtrip: rmse(D(E(x_b)) - x_b) =
0.017 sigma, and exactly 0 with the anchored decode.

Layout (329):
    predicted_state 138 | forcing 42 | noised 138 | static 2 |
    date 7 | step 1 | lead 1

Contrast with input_builder.build_operational_input, which puts N(0,1) noise
in the noised slot and step = 1.0 (diffusion step 200): that is the first
step of a forecast sampler, not the background.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from ls3dvar.hrrrcast.time_features import (
    make_date_encoding,
    make_lead_encoding,
    make_step_encoding,
)

NUM_DIFFUSION_STEPS = 200


def predicted_channel_count(hrrr):
    return len(hrrr["pl_vars"]) * len(hrrr["levels"]) + len(hrrr["sfc_vars"])


def build_analysis_input(hrrr, gfs, lead_hour=1, diffusion_step=0):
    """
    Returns
    -------
    X   : (1, H, W, 329) float32
    x_b : (1, H, W, 138) float32   background state (normalized model_input)
    """
    model_input_hrrr = hrrr["model_input"]
    model_input_gfs = gfs["model_input"]
    n_pred = predicted_channel_count(hrrr)
    nlat, nlon = model_input_hrrr.shape[1:3]

    x_b = np.asarray(model_input_hrrr[..., :n_pred], np.float32)

    init_time = pd.to_datetime(str(hrrr["init_datetime"]))
    valid_time = init_time + timedelta(hours=lead_hour)

    X = np.concatenate(
        [
            x_b,                                   # predicted state
            model_input_gfs[0:1],                  # forcing
            x_b,                                   # noised slot = background
            model_input_hrrr[..., n_pred:],        # static
            make_date_encoding(valid_time, nlat, nlon),
            make_step_encoding(diffusion_step / NUM_DIFFUSION_STEPS, nlat, nlon),
            make_lead_encoding(lead_hour, nlat, nlon),
        ],
        axis=-1,
    ).astype(np.float32)

    return X, x_b
