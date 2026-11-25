# ls3dvar/obs/grid.py  (extracted from mantari/station_mapper.py)
import numpy as np


def station_to_grid(
    lat,
    lon,
    grid_lat,
    grid_lon,
):

    dist = (
        (grid_lat - lat) ** 2
        +
        (grid_lon - lon) ** 2
    )

    iy, ix = np.unravel_index(
        np.argmin(dist),
        dist.shape,
    )

    return iy, ix


def build_station_indices(
    obs_set,
    hrrr_npz,
):

    hrrr = np.load(
        hrrr_npz,
        allow_pickle=True,
    )

    grid_lat = hrrr["lats"]
    grid_lon = hrrr["lons"]

    obs_y = []
    obs_x = []

    for obs in obs_set.observations:

        iy, ix = station_to_grid(
            obs.lat,
            obs.lon,
            grid_lat,
            grid_lon,
        )

        obs_y.append(iy)
        obs_x.append(ix)

    return (

        np.asarray(
            obs_y,
            dtype=np.int32,
        ),

        np.asarray(
            obs_x,
            dtype=np.int32,
        ),
    )
