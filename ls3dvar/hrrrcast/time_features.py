# ls3dvar/hrrrcast/time_features.py  (extracted from mantari/ls3dvar/time_features.py)
from datetime import datetime

import numpy as np


def compute_time_features(valid_time):
    """
    Exact copy of fcst.py time feature logic.

    Returns
    -------
    np.ndarray
        Shape (7,)
    """

    hour = float(valid_time.hour)
    doy = float(valid_time.timetuple().tm_yday)

    v4 = float(
        valid_time >= datetime(2021, 3, 23)
    )

    v3 = float(
        datetime(2018, 7, 12)
        <= valid_time
        < datetime(2021, 3, 23)
    )

    gfs_v15 = float(
        datetime(2019, 6, 1)
        <= valid_time
        < datetime(2021, 3, 23)
    )

    return np.array(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * doy / 365.0),
            np.cos(2.0 * np.pi * doy / 365.0),
            v4,
            v3,
            gfs_v15,
        ],
        dtype=np.float32,
    )


def make_date_encoding(
    valid_time,
    nlat,
    nlon,
):
    """
    Returns
    -------
    (1, nlat, nlon, 7)
    """

    features = compute_time_features(
        valid_time
    )

    return np.broadcast_to(
        features.reshape(1, 1, 1, 7),
        (1, nlat, nlon, 7),
    ).astype(np.float32)


def make_step_encoding(
    value,
    nlat,
    nlon,
):
    """
    Returns
    -------
    (1, nlat, nlon, 1)
    """

    return np.full(
        (1, nlat, nlon, 1),
        float(value),
        dtype=np.float32,
    )


def make_lead_encoding(
    lead_hour,
    nlat,
    nlon,
):
    """
    Operational fcst.py uses:

        lead_encoding = step / 6.0

    Returns
    -------
    (1, nlat, nlon, 1)
    """

    return np.full(
        (1, nlat, nlon, 1),
        float(lead_hour) / 6.0,
        dtype=np.float32,
    )


def build_temporal_encodings(
    valid_time,
    lead_hour,
    nlat,
    nlon,
):
    """
    Convenience wrapper.

    Returns
    -------
    date_encoding
    step_encoding
    lead_encoding
    """

    date_encoding = make_date_encoding(
        valid_time,
        nlat,
        nlon,
    )

    step_encoding = make_step_encoding(
        1.0,
        nlat,
        nlon,
    )

    lead_encoding = make_lead_encoding(
        lead_hour,
        nlat,
        nlon,
    )

    return (
        date_encoding,
        step_encoding,
        lead_encoding,
    )
