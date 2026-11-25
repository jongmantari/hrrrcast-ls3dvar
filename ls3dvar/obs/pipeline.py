# ls3dvar/obs/pipeline.py  (extracted from mantari/ls3dvar/observation_pipeline.py)
import numpy as np
import tensorflow as tf

from ls3dvar.obs.ioda import (
    ObservationProvider,
)

from ls3dvar.obs.grid import (
    build_station_indices,
)

from ls3dvar.obs.gross_filter import (
    ObservationFilter,
)

from ls3dvar.obs.time_window import (
    analysis_time_from_npz,
    apply_time_window,
)

VARIABLES = {
    "t2m": "airTemperatureAt2M",
    "d2m": "dewPointTemperatureAt2M",
    "ps": "stationPressure",
    "u10": "windEastwardAt10M",
    "v10": "windNorthwardAt10M",
}


def load_observations(
    cfg,
):
    """
    Load observations, apply the analysis-time window / nearest-report
    thinning, and organize by variable.

    NOTE:
    Innovation-based filtering is NOT applied here, because H(xb) does not
    exist yet; the driver applies it after computing innovations.
    """

    provider = ObservationProvider()
    obs_set = provider.from_config(cfg)
    obs_filter = ObservationFilter(cfg)

    hrrr = np.load(cfg["hrrr_npz"], allow_pickle=True)
    t_ana = analysis_time_from_npz(hrrr)

    observations = apply_time_window(obs_set.observations, t_ana, cfg)

    out = {"filter": obs_filter, "analysis_time": t_ana}

    for key, variable in VARIABLES.items():
        obs_list = [o for o in observations if o.variable == variable]
        iy, ix = build_station_indices(type(obs_set)(obs_list), cfg["hrrr_npz"])
        out[f"{key}_obs"] = obs_list
        out[f"{key}_y"] = tf.constant(iy, dtype=tf.int32)
        out[f"{key}_x"] = tf.constant(ix, dtype=tf.int32)

    return out
