# ls3dvar/obs/ioda.py  (extracted from mantari/observation_provider.py)
from pathlib import Path
import glob
import json

import numpy as np
import xarray as xr

from ls3dvar.obs.types import (
    SurfaceObservation,
    SurfaceObservationSet,
)

ELEVATION_KEYS = ("stationElevation", "height", "elevation", "heightOfStation")

WIND_VARIABLES = ("windEastwardAt10M", "windNorthwardAt10M")


def _read_times(meta, n):
    """
    MetaData/dateTime -> datetime64[s] array (NaT where missing).

    Handles both xarray-decoded datetime64 and raw integer seconds since
    1970-01-01 (IODA convention, units attribute 'seconds since ...').
    """
    if "dateTime" not in meta:
        return np.full(n, np.datetime64("NaT"), dtype="datetime64[s]")

    t = meta["dateTime"].values
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[s]")

    units = str(meta["dateTime"].attrs.get("units", "seconds since 1970-01-01T00:00:00Z"))
    epoch = np.datetime64(units.split("since")[-1].strip().rstrip("Z").replace(" ", "T"), "s")
    t = np.asarray(t, dtype=np.float64)
    out = np.full(n, np.datetime64("NaT"), dtype="datetime64[s]")
    ok = np.isfinite(t) & (np.abs(t) < 1e11)
    out[ok] = epoch + t[ok].astype(np.int64).astype("timedelta64[s]")
    return out


def _read_elevation(meta, n):
    for k in ELEVATION_KEYS:
        if k in meta:
            e = np.asarray(meta[k].values, dtype=np.float64)
            return np.where(np.isfinite(e) & (np.abs(e) < 9000.0), e, np.nan)
    return np.full(n, np.nan)


class ObservationProvider:

    # =================================================
    # MANUAL
    # =================================================

    def from_manual(
        self,
        observations,
    ):

        obs = []

        for row in observations:
            t = row.get("time")
            obs.append(
                SurfaceObservation(
                    station=row["station"],
                    lat=row["lat"],
                    lon=row["lon"],
                    variable=row.get("variable", "airTemperatureAt2M"),
                    value=row["value"],
                    error=row.get("error", 1.0),
                    source=row.get("source", "manual"),
                    time=None if t is None else np.datetime64(t, "s"),
                    elevation=row.get("elevation"),
                )
            )

        return SurfaceObservationSet(obs)

    # =================================================
    # JSON
    # =================================================

    def from_json(
        self,
        filename,
    ):

        with open(Path(filename), "r") as f:
            data = json.load(f)

        return self.from_manual(data)

    # =================================================
    # IODA
    # =================================================

    def from_ioda_directory(
        self,
        path,
        source_name,
    ):

        observations = []

        files = sorted(glob.glob(f"{path}/*.nc4"))

        for ncfile in files:

            # decode_times=False: read dateTime as raw seconds; _read_times decodes
            meta = xr.open_dataset(ncfile, group="MetaData", decode_times=False)
            obsv = xr.open_dataset(ncfile, group="ObsValue")

            try:
                derived = xr.open_dataset(ncfile, group="DerivedObsValue")
            except Exception:
                derived = None

            try:
                errs = xr.open_dataset(ncfile, group="ObsError")
            except Exception:
                errs = None

            lat = meta["latitude"].values
            lon = meta["longitude"].values
            station = meta["stationIdentification"].values
            n = len(lat)
            times = _read_times(meta, n)
            elev = _read_elevation(meta, n)

            def make(i, variable, value, error):
                return SurfaceObservation(
                    station=str(station[i]),
                    lat=float(lat[i]),
                    lon=float(lon[i]),
                    variable=variable,
                    value=float(value),
                    error=float(error),
                    source=source_name,
                    time=None if np.isnat(times[i]) else times[i],
                    elevation=None if np.isnan(elev[i]) else float(elev[i]),
                )

            # -------------------------------------------
            # T2M / D2M
            # -------------------------------------------

            for variable in ["airTemperatureAt2M", "dewPointTemperatureAt2M"]:

                if variable not in obsv:
                    continue

                value = obsv[variable].values

                if errs is not None and variable in errs:
                    error = errs[variable].values
                else:
                    error = np.ones_like(value, dtype=np.float32)

                good = (
                    np.isfinite(value)
                    & np.isfinite(lat)
                    & np.isfinite(lon)
                    & (value > 180.0)
                    & (value < 340.0)
                )

                for i in np.where(good)[0]:
                    observations.append(make(i, variable, value[i], error[i]))

            # -------------------------------------------
            # 10-m wind (earth-relative components, m/s)
            # -------------------------------------------

            for variable in WIND_VARIABLES:

                if variable not in obsv:
                    continue

                value = np.asarray(obsv[variable].values, np.float64)

                if errs is not None and variable in errs:
                    error = errs[variable].values
                else:
                    error = np.ones_like(value, dtype=np.float32)

                good = (
                    np.isfinite(value)
                    & np.isfinite(lat)
                    & np.isfinite(lon)
                    & (np.abs(value) < 75.0)
                )

                for i in np.where(good)[0]:
                    observations.append(make(i, variable, value[i], error[i]))

            # -------------------------------------------
            # Station pressure
            # -------------------------------------------

            if derived is not None and "stationPressure" in derived:

                value = derived["stationPressure"].values
                error = np.ones_like(value, dtype=np.float32)

                good = (
                    np.isfinite(value)
                    & np.isfinite(lat)
                    & np.isfinite(lon)
                    & (value > 30000.0)
                    & (value < 110000.0)
                )

                for i in np.where(good)[0]:
                    observations.append(make(i, "stationPressure", value[i], error[i]))

            meta.close()
            obsv.close()
            if derived is not None:
                derived.close()
            if errs is not None:
                errs.close()

        return SurfaceObservationSet(observations)

    # =================================================
    # CONFIG
    # =================================================

    def from_config(
        self,
        cfg,
    ):

        providers = cfg.get("observations", {}).get("providers", [])

        observations = []

        for provider in providers:
            obs_set = self.from_ioda_directory(provider["path"], provider["type"])
            observations.extend(obs_set.observations)

        return SurfaceObservationSet(observations)
