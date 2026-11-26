#!/usr/bin/env python3
"""
iem_to_ioda.py: convert the local IEM ASOS mirror into IODA v3 observation files.

Input:  /Users/flampouris/Data/datasets/iem-asos/YYYY/MM/iem-asos-{STATION}-{YYYYMM}.csv
Output: /Users/flampouris/Data/datasets-etl/ioda/iem_asos_obs_{YYYYMMDDTHHMMSS}Z.nc4
        One file per data-assimilation window (an IODA "ObsGroup").

Conventions
-----------
Written to follow the JEDI data conventions:
  https://jcsda-jedi-docs.readthedocs-hosted.com/en/latest/inside/conventions/index.html
and validated against the machine-readable schema those pages describe,
JCSDA/ioda:share/ioda/yaml/validation/ObsSpace.yaml (develop).

What that implies, concretely:

* Dimension ``Location`` is required and is an int64 dimension scale at the root
  group. Only dimension scales live at the root; everything else is in a group.
* Group ``MetaData`` is required and must contain ``dateTime``, ``latitude``,
  ``longitude``. ``dateTime`` is int64 epoch seconds with units
  "seconds since 1970-01-01T00:00:00Z".
* Variable names are camelCase, adjective-before-noun, taken from the
  conventions tables (which derive from WMO BUFR element table 36). Note the
  capital P in ``dewPointTemperature``.
* Units are SI and UDUNITS-parsable. Dimensionless physical quantities use "1",
  so relative humidity is a 0-1 fraction, not a percentage.
* Every variable carries ``_FillValue``; every non-enumerated variable carries
  ``units``. ``long_name`` is optional and is written here.
* Measured quantities go in ``ObsValue``; quantities this script computes from
  them go in ``DerivedObsValue``.
* Missing values are the IODA defaults documented under "Storage type and
  missing values" (float -3.3688e+38, int64 -9223372036854775801).

Two deliberate deviations, both flagged here because they are judgement calls:

1. ``Conventions`` is *not* set as a global attribute. It is not in the list of
   attributes the root group permits, even though some IODA files in the wild
   carry it. Similarly ``r2d2*`` window attributes are omitted: the conventions
   assign those to R2D2 at ingest, not to the converter. Window bounds are
   recorded in the permitted ``datetimeRange`` / ``datetimeReference`` instead.
2. ``windEastward`` / ``windNorthward`` are derived from the reported speed and
   direction, so strictly they belong in ``DerivedObsValue``. They are written
   to ``ObsValue`` by default because that is where UFO operator yaml normally
   looks for them. Use ``--uv-group DerivedObsValue`` for the strict reading.

Variables not carried over, and why
-----------------------------------
* ``feel`` (apparent temperature / heat index / wind chill): no conventions name.
* ``wxcodes``: ``presentWeather`` is an enumerated WMO code-table 020003 value.
  IEM reports METAR abbreviation strings ("-RA BR"), and the string-to-code-figure
  mapping is a lookup table this script does not implement. Dropped rather than
  guessed at.
* ``ice_accretion_{1,3,6}hr``: no conventions name.
* ``peak_wind_gust`` / ``peak_wind_drct``: these would collide with
  ``maximumWindGustSpeed`` / ``maximumWindGustDirection``, which are taken from
  ``gust`` / ``drct`` over the standard reporting period. Enable
  ``--use-peak-wind`` to fill the gust variables from the peak-wind columns
  instead (they cover a different accumulation period, so do not mix them).

Usage:
  # One window centred on a cycle time (default half-width 1 hour):
  uv run --project . iem_to_ioda.py --cycle 2026-06-15T12

  # 6-hour window, 3-hour half-width, as PREPBUFR-style cycles:
  uv run --project . iem_to_ioda.py --cycle 2026-06-15T12 --half-width 3h

  # Every 6-hourly cycle in a date range:
  uv run --project . iem_to_ioda.py --start 2026-06-01 --end 2026-06-07 --step 6h

  # Restrict the station set / region:
  uv run --project . iem_to_ioda.py --cycle 2026-06-15T12 --station KBOS --station KJFK
  uv run --project . iem_to_ioda.py --cycle 2026-06-15T12 --bbox 24 -125 50 -66
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ETL_ROOT, RAW_ROOT, setup_logging

RAW_DIR  = RAW_ROOT / "iem-asos"
OUT_DIR  = ETL_ROOT / "ioda"
LOG_FILE = ETL_ROOT / "iem-asos-ioda.log"

# ── IODA missing values ───────────────────────────────────────────────────────
#
# From the conventions' "Storage type and missing values" table. int32 is not
# listed there; the value below is oops' util::missingValue<int>(), used for the
# enumerated PreQC group.

FLOAT_MISSING = np.float32(-3.3688e38)
INT64_MISSING = np.int64(-9223372036854775801)
INT32_MISSING = np.int32(-2147483645)
STR_MISSING   = ""

EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"
EPOCH       = pd.Timestamp("1970-01-01", tz="UTC")

# ASOS sensor heights above ground (WMO/NWS siting standard).
THERMO_HEIGHT_M   = 2.0
MOMENTUM_HEIGHT_M = 10.0

# ── Unit conversions to SI ────────────────────────────────────────────────────

F_TO_K_OFFSET = 273.15
KNOT_TO_MPS   = 0.514444
INHG_TO_PA    = 3386.389
HPA_TO_PA     = 100.0
MILE_TO_M     = 1609.344
INCH_TO_KGM2  = 25.4      # 1 mm of liquid water = 1 kg m-2
INCH_TO_M     = 0.0254
FOOT_TO_M     = 0.3048
PCT_TO_FRAC   = 0.01


def f_to_k(x: np.ndarray) -> np.ndarray:
    return (x - 32.0) * 5.0 / 9.0 + F_TO_K_OFFSET


def scale(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: x * factor


# ── Variable table ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObsVar:
    """One IODA variable derived from a single IEM CSV column."""

    name: str                       # IODA conventions variable name
    column: str                     # IEM CSV column
    convert: Callable[[np.ndarray], np.ndarray]
    units: str                      # UDUNITS string, SI
    long_name: str
    obs_error: float                # placeholder, in IODA units; only used with --obs-error
    expected_range: tuple[float, float] = (np.nan, np.nan)
    """Physically plausible bounds in IODA units, written as the conventions'
    optional ``ExpectedRange`` attribute. Values outside it are passed through
    unaltered: gross-error rejection belongs to UFO's QC filters, not here. IEM
    does emit implausible values, mostly mean-sea-level pressure reduced from
    high-elevation stations."""


# Names and units verified against ObsSpace.yaml. "At2M"/"At10M" spellings are
# the level-suffixed alternatives selected by --at-level-names.
OBS_VARS: list[ObsVar] = [
    ObsVar("airTemperature", "tmpf", f_to_k, "K",
           "air temperature", 1.0, (180.0, 340.0)),
    ObsVar("dewPointTemperature", "dwpf", f_to_k, "K",
           "dew point temperature", 1.5, (180.0, 320.0)),
    ObsVar("relativeHumidity", "relh", scale(PCT_TO_FRAC), "1",
           "relative humidity", 0.1, (0.0, 1.0)),
    ObsVar("windDirection", "drct", scale(1.0), "degree",
           "wind from direction", 5.0, (0.0, 360.0)),
    ObsVar("windSpeed", "sknt", scale(KNOT_TO_MPS), "m s-1",
           "wind speed", 1.5, (0.0, 120.0)),
    ObsVar("maximumWindGustSpeed", "gust", scale(KNOT_TO_MPS), "m s-1",
           "maximum wind gust speed", 2.5, (0.0, 150.0)),
    ObsVar("altimeterSetting", "alti", scale(INHG_TO_PA), "Pa",
           "altimeter setting", 100.0, (85000.0, 110000.0)),
    ObsVar("pressureReducedToMeanSeaLevel", "mslp", scale(HPA_TO_PA), "Pa",
           "pressure reduced to mean sea level", 100.0, (85000.0, 110000.0)),
    ObsVar("horizontalVisibility", "vsby", scale(MILE_TO_M), "m",
           "horizontal visibility", 1000.0, (0.0, 100000.0)),
    ObsVar("totalPrecipitationPast1Hour", "p01i", scale(INCH_TO_KGM2), "kg m-2",
           "total precipitation past 1 hour", 1.0, (0.0, 500.0)),
    ObsVar("totalSnowDepth", "snowdepth", scale(INCH_TO_M), "m",
           "total snow depth", 0.05, (0.0, 15.0)),
    ObsVar("heightOfBaseOfCloud", "skyl1", scale(FOOT_TO_M), "m",
           "height of base of lowest cloud layer", 100.0, (0.0, 25000.0)),
]

# --at-level-names: generic name -> level-suffixed conventions name.
AT_LEVEL_NAMES = {
    "airTemperature":      "airTemperatureAt2M",
    "dewPointTemperature": "dewPointTemperatureAt2M",
    "relativeHumidity":    "relativeHumidityAt2M",
    "windDirection":       "windDirectionAt10M",
    "windSpeed":           "windSpeedAt10M",
    "windEastward":        "windEastwardAt10M",
    "windNorthward":       "windNorthwardAt10M",
}

# --use-peak-wind: take the gust variables from the peak-wind columns instead.
PEAK_WIND_VARS: list[ObsVar] = [
    ObsVar("maximumWindGustSpeed", "peak_wind_gust", scale(KNOT_TO_MPS), "m s-1",
           "peak wind gust speed", 2.5, (0.0, 150.0)),
    ObsVar("maximumWindGustDirection", "peak_wind_drct", scale(1.0), "degree",
           "direction of peak wind gust", 10.0, (0.0, 360.0)),
]

# METAR sky-cover abbreviation -> cloud amount fraction (midpoint of the okta
# range each abbreviation covers). VV (vertical visibility) is treated as overcast.
SKY_COVER_FRACTION = {
    "CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0,
    "FEW": 1.5 / 8.0,
    "SCT": 3.5 / 8.0,
    "BKN": 6.0 / 8.0,
    "OVC": 1.0,
    "VV":  1.0,
}

# All CSV columns the converter may read.
NEEDED_COLUMNS = sorted(
    {"station", "valid", "lon", "lat", "elevation", "skyc1"}
    | {v.column for v in OBS_VARS}
    | {v.column for v in PEAK_WIND_VARS}
)

# Columns where IEM uses 'T' for a trace amount, which becomes 0.
TRACE_COLUMNS = ("p01i", "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr")

FNAME_RE = re.compile(r"^iem-asos-(?P<station>.+)-(?P<yyyymm>\d{6})\.csv$")


# ── Derived quantities ────────────────────────────────────────────────────────

def uv_from_speed_direction(
    speed_mps: np.ndarray, direction_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological direction (where the wind comes FROM) to u/v components."""
    rad = np.deg2rad(direction_deg)
    return -speed_mps * np.sin(rad), -speed_mps * np.cos(rad)


def station_pressure_from_altimeter(
    altimeter_pa: np.ndarray, elevation_m: np.ndarray
) -> np.ndarray:
    """Invert the altimeter-setting reduction to recover station pressure.

    p = (A**k - c * z) ** (1/k) with A and p in hPa, z in m, using the US
    Standard Atmosphere constants: k = 0.190284 (= R*L/g) and
    c = 1013.25**k * L / T0 with L = 0.0065 K m-1 and T0 = 288.15 K.
    Exact at z = 0 and within ~0.1 hPa of the standard atmosphere at 1 km.
    """
    k = 0.190284
    c = 1013.25 ** k * 0.0065 / 288.15
    a_hpa = altimeter_pa / HPA_TO_PA
    with np.errstate(invalid="ignore"):
        base = np.power(a_hpa, k) - c * elevation_m
        p_hpa = np.where(base > 0.0, np.power(np.abs(base), 1.0 / k), np.nan)
    return p_hpa * HPA_TO_PA


# CONUS bounding box, used only to decide whether a 3-character IEM id can be
# safely expanded to a K-prefixed ICAO identifier.
CONUS_BOX = (24.0, -125.0, 50.0, -66.0)   # lat_min, lon_min, lat_max, lon_max


def icao_ids(ids: list[str], lat: np.ndarray, lon: np.ndarray) -> list[str]:
    """Best-effort ICAO identifier for each IEM station id.

    IEM uses the 3-character local id inside CONUS ("BOS") and the full
    4-character ICAO id elsewhere ("PANC", "TJSJ", "PHNL"). A 3-character id
    inside the CONUS box expands to "K" + id, which holds for every CONUS
    airport. Anything else is left missing rather than guessed at: the
    non-CONUS prefixes (P, PA, PH, TJ, NST, ...) are not recoverable from the
    id alone.
    """
    lat_min, lon_min, lat_max, lon_max = CONUS_BOX
    out: list[str] = []
    for i, s in enumerate(ids):
        if len(s) == 4:
            out.append(s)
        elif (
            len(s) == 3
            and np.isfinite(lat[i]) and np.isfinite(lon[i])
            and lat_min <= lat[i] <= lat_max
            and lon_min <= lon[i] <= lon_max
        ):
            out.append("K" + s)
        else:
            out.append(STR_MISSING)
    return out


def cloud_amount_from_sky_cover(codes: pd.Series) -> np.ndarray:
    """METAR sky-cover abbreviations to a 0-1 cloud amount fraction."""
    cleaned = codes.astype(str).str.strip().str.upper()
    return cleaned.map(SKY_COVER_FRACTION).to_numpy(dtype="float64")


# ── CSV reading ───────────────────────────────────────────────────────────────

def month_dirs(start: datetime, end: datetime) -> list[Path]:
    """Mirror directories covering [start, end], inclusive of both endpoints."""
    out: list[Path] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        d = RAW_DIR / str(year) / f"{month:02d}"
        if d.is_dir():
            out.append(d)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def station_filter(requested: Optional[list[str]]) -> Optional[set[str]]:
    """Expand requested ids into the forms the mirror actually uses.

    The mirror is keyed on the IEM id, which is the 3-character local id inside
    CONUS. Accept the ICAO spelling too, so both --station BOS and
    --station KBOS select iem-asos-BOS-YYYYMM.csv.
    """
    if not requested:
        return None
    out: set[str] = set()
    for s in requested:
        s = s.strip().upper()
        out.add(s)
        if len(s) == 4 and s.startswith("K"):
            out.add(s[1:])
    return out


def csv_files(start: datetime, end: datetime, stations: Optional[set[str]]) -> list[Path]:
    files: list[Path] = []
    for d in month_dirs(start, end):
        for path in d.glob("iem-asos-*.csv"):
            m = FNAME_RE.match(path.name)
            if m and (stations is None or m.group("station") in stations):
                files.append(path)
    return sorted(files)


def read_window(path: Path, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Read one station-month CSV and return only the rows inside [start, end).

    Returns None when the file has no usable rows in the window.
    """
    text = path.read_text(errors="replace")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return None

    header = [h.strip() for h in lines[0].split(",")]
    usecols = [c for c in header if c in NEEDED_COLUMNS]
    if "valid" not in usecols:
        return None

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        dtype=str,
        usecols=usecols,
        na_values=["M", ""],
        keep_default_na=False,
    )
    if df.empty:
        return None

    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df[df["valid"].notna()]
    # Half-open window so that adjacent cycles cannot claim the same observation.
    df = df[(df["valid"] >= start) & (df["valid"] < end)]
    if df.empty:
        return None

    for col in TRACE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace("T", "0")

    if "station" not in df.columns:
        m = FNAME_RE.match(path.name)
        df["station"] = m.group("station") if m else ""

    return df


def collect(
    files: list[Path], start: datetime, end: datetime, workers: int, log
) -> pd.DataFrame:
    """Read every station-month file in parallel and concatenate the window rows."""
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for df in pool.map(lambda p: _safe_read(p, start, end, log), files):
            if df is not None:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _safe_read(path: Path, start: datetime, end: datetime, log) -> Optional[pd.DataFrame]:
    try:
        return read_window(path, start, end)
    except Exception as exc:                       # one bad file must not kill the run
        log.warning("skipping %s: %s", path.name, exc)
        return None


def thin_to_nearest(df: pd.DataFrame, cycle: datetime) -> pd.DataFrame:
    """Keep the single observation per station closest to the cycle time."""
    offset = (df["valid"] - pd.Timestamp(cycle)).abs()
    order = offset.argsort(kind="stable")
    return df.iloc[order].drop_duplicates(subset="station", keep="first")


# ── Assembly ──────────────────────────────────────────────────────────────────

def numeric(df: pd.DataFrame, column: str) -> Optional[np.ndarray]:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype="float64")


def build_variables(
    df: pd.DataFrame,
    *,
    at_level_names: bool,
    use_peak_wind: bool,
    uv_group: str,
    derive_station_pressure: bool,
    derive_cloud_amount: bool,
) -> tuple[dict[str, tuple[ObsVar, np.ndarray]], dict[str, tuple[ObsVar, np.ndarray]]]:
    """Return ({ObsValue vars}, {DerivedObsValue vars}), values in IODA units."""
    obs: dict[str, tuple[ObsVar, np.ndarray]] = {}
    derived: dict[str, tuple[ObsVar, np.ndarray]] = {}

    table = list(OBS_VARS)
    if use_peak_wind:
        table = [v for v in table if v.name != "maximumWindGustSpeed"] + PEAK_WIND_VARS

    def rename(spec: ObsVar) -> ObsVar:
        if at_level_names and spec.name in AT_LEVEL_NAMES:
            return ObsVar(AT_LEVEL_NAMES[spec.name], spec.column, spec.convert,
                          spec.units, spec.long_name, spec.obs_error,
                          spec.expected_range)
        return spec

    for spec in table:
        raw = numeric(df, spec.column)
        if raw is None or np.all(np.isnan(raw)):
            continue
        spec = rename(spec)
        obs[spec.name] = (spec, spec.convert(raw))

    # Wind components from speed and direction.
    speed_key = "windSpeedAt10M" if at_level_names else "windSpeed"
    dir_key   = "windDirectionAt10M" if at_level_names else "windDirection"
    if speed_key in obs and dir_key in obs:
        u, v = uv_from_speed_direction(obs[speed_key][1], obs[dir_key][1])
        target = obs if uv_group == "ObsValue" else derived
        for name, data, long_name in (
            ("windEastward", u, "eastward wind component"),
            ("windNorthward", v, "northward wind component"),
        ):
            spec = rename(ObsVar(name, "", scale(1.0), "m s-1", long_name, 1.5,
                                 (-120.0, 120.0)))
            target[spec.name] = (spec, data)

    # Station pressure from the altimeter setting and station elevation.
    if derive_station_pressure and "altimeterSetting" in obs:
        elev = numeric(df, "elevation")
        if elev is not None:
            p = station_pressure_from_altimeter(obs["altimeterSetting"][1], elev)
            derived["stationPressure"] = (
                ObsVar("stationPressure", "", scale(1.0), "Pa",
                       "pressure at station, reduced from altimeter setting", 100.0,
                       (30000.0, 110000.0)),
                p,
            )

    # Cloud amount of the lowest layer from the METAR sky-cover abbreviation.
    if derive_cloud_amount and "skyc1" in df.columns:
        frac = cloud_amount_from_sky_cover(df["skyc1"])
        if not np.all(np.isnan(frac)):
            derived["cloudAmountOfLowestLayer"] = (
                ObsVar("cloudAmountOfLowestLayer", "", scale(1.0), "1",
                       "cloud amount of lowest layer", 0.2, (0.0, 1.0)),
                frac,
            )

    return obs, derived


# ── NetCDF writing ────────────────────────────────────────────────────────────

def _as_float32(data: np.ndarray) -> np.ndarray:
    out = data.astype("float32")
    out[~np.isfinite(data)] = FLOAT_MISSING
    return out


def _write_float(
    group,
    name: str,
    data: np.ndarray,
    units: str,
    long_name: str,
    expected_range: tuple[float, float] = (np.nan, np.nan),
) -> None:
    var = group.createVariable(name, "f4", ("Location",), fill_value=FLOAT_MISSING)
    var[:] = _as_float32(data)
    var.units = units
    var.long_name = long_name
    if np.all(np.isfinite(expected_range)):
        var.setncattr("ExpectedRange", np.array(expected_range, dtype="float32"))


def write_ioda(
    df: pd.DataFrame,
    path: Path,
    *,
    cycle: datetime,
    window_start: datetime,
    window_end: datetime,
    source_files: int,
    at_level_names: bool,
    use_peak_wind: bool,
    uv_group: str,
    derive_station_pressure: bool,
    derive_cloud_amount: bool,
    write_obs_error: bool,
    write_preqc: bool,
) -> tuple[int, int]:
    """Write one IODA ObsGroup file. Returns (n_locations, n_variables)."""
    nloc = len(df)
    obs, derived = build_variables(
        df,
        at_level_names=at_level_names,
        use_peak_wind=use_peak_wind,
        uv_group=uv_group,
        derive_station_pressure=derive_station_pressure,
        derive_cloud_amount=derive_cloud_amount,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with nc.Dataset(tmp, "w", format="NETCDF4") as ds:
        ds.createDimension("Location", nloc)

        # Root group: only permitted global attributes, only dimension scales.
        ds.setncattr("_ioda_layout", "ObsGroup")
        ds.setncattr("_ioda_layout_version", np.int32(0))
        ds.setncattr("converter", "iem_to_ioda.py")
        ds.setncattr("platformCommonName", "ASOS")
        ds.setncattr(
            "platformLongDescription",
            "Automated Surface Observing System land stations, METAR-based hourly "
            "reports quality controlled and distributed by the Iowa Environmental Mesonet",
        )
        ds.setncattr("processingLevel", "2")
        ds.setncattr("source", "Iowa Environmental Mesonet ASOS archive")
        ds.setncattr("sourceFiles", np.int32(source_files))
        ds.setncattr("description",
                     "IEM ASOS surface observations for a single assimilation window")
        ds.setncattr("datetimeRange", [_iso(window_start), _iso(window_end)])
        ds.setncattr("datetimeReference", _iso(cycle))

        loc = ds.createVariable("Location", "i8", ("Location",))
        loc[:] = np.arange(nloc, dtype="int64")

        meta = ds.createGroup("MetaData")

        # Required MetaData variables.
        _write_float(meta, "latitude", numeric(df, "lat"), "degree_north", "latitude")
        _write_float(meta, "longitude", numeric(df, "lon"), "degree_east", "longitude")

        seconds = (df["valid"] - EPOCH).dt.total_seconds().to_numpy(dtype="float64")
        epoch_s = np.where(np.isfinite(seconds), seconds, INT64_MISSING).astype("int64")
        dtv = meta.createVariable("dateTime", "i8", ("Location",), fill_value=INT64_MISSING)
        dtv[:] = epoch_s
        dtv.units = EPOCH_UNITS
        dtv.long_name = "observation date and time"

        # Station metadata.
        elev = numeric(df, "elevation")
        if elev is not None:
            _write_float(meta, "stationElevation", elev, "m",
                         "station elevation above mean sea level")
            # height is the observation height above mean sea level; the
            # thermometer level is the one the temperature/humidity refer to.
            _write_float(meta, "height", elev + THERMO_HEIGHT_M, "m",
                         "observation height above mean sea level")

        _write_float(meta, "sensorThermoHeight",
                     np.full(nloc, THERMO_HEIGHT_M), "m",
                     "thermometer height above local ground")
        _write_float(meta, "sensorMomentumHeight",
                     np.full(nloc, MOMENTUM_HEIGHT_M), "m",
                     "anemometer height above local ground")

        ids = [str(s) for s in df["station"].tolist()]
        _write_string(meta, "stationIdentification", ids, "station identification")

        icao = icao_ids(ids, numeric(df, "lat"), numeric(df, "lon"))
        if any(icao):
            _write_string(meta, "stationICAO", icao, "ICAO station identifier")

        # The conventions treat sequenceNumber as a record number, so every
        # observation from one station in this window shares a value.
        seq = meta.createVariable("sequenceNumber", "i4", ("Location",),
                                  fill_value=np.int32(INT32_MISSING))
        seq[:] = (pd.factorize(df["station"], sort=True)[0] + 1).astype("int32")
        seq.long_name = "record number, one per station"
        seq.units = "unitless"

        # ObsValue / DerivedObsValue, plus optional ObsError and PreQC.
        for group_name, table in (("ObsValue", obs), ("DerivedObsValue", derived)):
            if not table:
                continue
            grp = ds.createGroup(group_name)
            for name, (spec, data) in sorted(table.items()):
                _write_float(grp, name, data, spec.units, spec.long_name,
                             spec.expected_range)

        if write_obs_error:
            grp = ds.createGroup("ObsError")
            for name, (spec, data) in sorted({**obs, **derived}.items()):
                err = np.where(np.isfinite(data), spec.obs_error, np.nan)
                _write_float(grp, name, err, spec.units,
                             f"{spec.long_name} observation error (placeholder)")

        if write_preqc:
            grp = ds.createGroup("PreQC")
            for name, (spec, data) in sorted({**obs, **derived}.items()):
                # IEM supplies no per-observation flag, so anything present is 0.
                flags = np.where(np.isfinite(data), 0, INT32_MISSING).astype("int32")
                qv = grp.createVariable(name, "i4", ("Location",),
                                        fill_value=INT32_MISSING)
                qv[:] = flags
                qv.long_name = f"{spec.long_name} pre-quality-control flag"

    tmp.replace(path)
    return nloc, len(obs) + len(derived)


def _write_string(group, name: str, values: list[str], long_name: str) -> None:
    # The conventions type these as StringFixedLen, which netCDF4-python cannot
    # create; variable-length strings are what the IODA python API emits too.
    var = group.createVariable(name, str, ("Location",), fill_value=STR_MISSING)
    for i, s in enumerate(values):
        var[i] = s
    # Categorical variables take "unitless" rather than "1", which is reserved
    # for dimensionless physical quantities.
    var.units = "unitless"
    var.long_name = long_name


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── CLI ───────────────────────────────────────────────────────────────────────

DURATION_RE = re.compile(r"^(?P<n>\d+(?:\.\d+)?)(?P<unit>[smhd])$")


def parse_duration(text: str) -> timedelta:
    m = DURATION_RE.match(text.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(
            f"bad duration {text!r}; use e.g. 30m, 1h, 6h, 1d"
        )
    n = float(m.group("n"))
    unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[m.group("unit")]
    return timedelta(**{unit: n})


def parse_time(text: str) -> datetime:
    t = pd.to_datetime(text, utc=True)
    return t.to_pydatetime().replace(tzinfo=timezone.utc)


def cycles(args: argparse.Namespace) -> Iterator[datetime]:
    if args.cycle:
        yield parse_time(args.cycle)
        return
    t, end, step = parse_time(args.start), parse_time(args.end), args.step
    while t <= end:
        yield t
        t += step


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert the local IEM ASOS mirror to IODA v3 observation files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    when = p.add_mutually_exclusive_group(required=True)
    when.add_argument("--cycle", metavar="ISO",
                      help="Single analysis time, e.g. 2026-06-15T12.")
    when.add_argument("--start", metavar="ISO",
                      help="First analysis time of a range (needs --end).")
    p.add_argument("--end", metavar="ISO", help="Last analysis time of a range.")
    p.add_argument("--step", type=parse_duration, default=parse_duration("6h"),
                   metavar="DUR", help="Cycle interval for a range (default: 6h).")
    p.add_argument("--half-width", type=parse_duration, default=parse_duration("1h"),
                   metavar="DUR",
                   help="Half-width of the assimilation window (default: 1h). The "
                        "window is [cycle - half-width, cycle + half-width).")

    p.add_argument("--station", action="append", metavar="ID",
                   help="Restrict to this station id; repeatable. Either the IEM id "
                        "(BOS, PANC) or the ICAO spelling (KBOS) is accepted.")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
                   help="Restrict to a lat/lon box.")
    p.add_argument("--all-obs-in-window", action="store_true",
                   help="Keep every observation in the window. Default keeps only "
                        "the one per station nearest the cycle time.")

    p.add_argument("--at-level-names", action="store_true",
                   help="Use the level-suffixed conventions names (airTemperatureAt2M, "
                        "windSpeedAt10M, ...) instead of the generic ones.")
    p.add_argument("--uv-group", choices=("ObsValue", "DerivedObsValue"),
                   default="ObsValue",
                   help="Group for the derived wind components (default: ObsValue, "
                        "which is where UFO yaml normally looks).")
    p.add_argument("--use-peak-wind", action="store_true",
                   help="Fill the gust variables from peak_wind_* instead of gust/drct.")
    p.add_argument("--no-station-pressure", action="store_true",
                   help="Do not derive DerivedObsValue/stationPressure from the "
                        "altimeter setting and station elevation.")
    p.add_argument("--no-cloud-amount", action="store_true",
                   help="Do not derive DerivedObsValue/cloudAmountOfLowestLayer "
                        "from the METAR sky-cover abbreviation.")
    p.add_argument("--obs-error", action="store_true",
                   help="Write an ObsError group of placeholder constants. Off by "
                        "default: IEM supplies no error estimates.")
    p.add_argument("--no-preqc", action="store_true",
                   help="Do not write the PreQC group.")

    p.add_argument("--outdir", type=Path, default=OUT_DIR,
                   help=f"Output directory (default: {OUT_DIR}).")
    p.add_argument("--workers", type=int, default=8,
                   help="Threads used to read the mirror (default: 8).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written without writing it.")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)
    if args.start and not args.end:
        p.error("--start requires --end")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = setup_logging("iem-ioda", LOG_FILE, args.verbose)
    log.info("=== IEM ASOS -> IODA started ===")

    stations = station_filter(args.station)
    written = 0

    for cycle in cycles(args):
        start = cycle - args.half_width
        end   = cycle + args.half_width

        files = csv_files(start, end, stations)
        if not files:
            log.warning("%s: no mirror files for the window; is the month synced?",
                        _iso(cycle))
            continue

        df = collect(files, start, end, args.workers, log)
        if df.empty:
            log.warning("%s: no observations in window from %d file(s)",
                        _iso(cycle), len(files))
            continue

        if not args.all_obs_in_window:
            df = thin_to_nearest(df, cycle)

        if args.bbox:
            lat_min, lon_min, lat_max, lon_max = args.bbox
            lat, lon = numeric(df, "lat"), numeric(df, "lon")
            keep = ((lat >= lat_min) & (lat <= lat_max)
                    & (lon >= lon_min) & (lon <= lon_max))
            df = df[keep]

        # MetaData/latitude, longitude and dateTime are required, so rows
        # missing any of them cannot be represented.
        df = df[np.isfinite(numeric(df, "lat")) & np.isfinite(numeric(df, "lon"))]
        if df.empty:
            log.warning("%s: no rows with usable position", _iso(cycle))
            continue

        df = df.sort_values(["station", "valid"]).reset_index(drop=True)
        out = args.outdir / f"iem_asos_obs_{cycle:%Y%m%dT%H%M%S}Z.nc4"

        if args.dry_run:
            log.info("%s: would write %s with %d location(s) from %d file(s)",
                     _iso(cycle), out.name, len(df), len(files))
            continue

        nloc, nvar = write_ioda(
            df, out,
            cycle=cycle,
            window_start=start,
            window_end=end,
            source_files=len(files),
            at_level_names=args.at_level_names,
            use_peak_wind=args.use_peak_wind,
            uv_group=args.uv_group,
            derive_station_pressure=not args.no_station_pressure,
            derive_cloud_amount=not args.no_cloud_amount,
            write_obs_error=args.obs_error,
            write_preqc=not args.no_preqc,
        )
        written += 1
        log.info("%s: wrote %s, %d location(s), %d variable(s)",
                 _iso(cycle), out.name, nloc, nvar)

    log.info("=== IEM ASOS -> IODA finished: %d file(s) ===", written)
    return 0 if written or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
