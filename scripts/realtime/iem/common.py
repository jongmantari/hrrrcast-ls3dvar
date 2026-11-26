"""
Shared configuration and utilities for all obs-etl scripts.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────────

#RAW_ROOT = Path("/home/jonggyunkim/Data/datasets")
#ETL_ROOT = Path("/home/jonggyunkim/Data/datasets-etl")
#ETL_ROOT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    os.environ.get(
        "WXBOX_DATA_ROOT",
        str(Path.home() / "Data"),
    )
)

RAW_ROOT = DATA_ROOT / "datasets"
ETL_ROOT = DATA_ROOT / "datasets-etl"

ETL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# ── Compression ───────────────────────────────────────────────────────────────

COMPRESSOR  = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
DTYPE       = "float32"
ZARR_FORMAT = 2   # write zarr v2 format; zarr 3 library writes it but append_dim works reliably

# ── Variable selection ────────────────────────────────────────────────────────

# Surface variables: cfgrib shortName → output name
SURFACE_VARS: dict[str, str] = {
    "t2m":  "t2m",   # 2-m temperature            K
    "d2m":  "d2m",   # 2-m dewpoint temperature   K
    "u10":  "u10",   # 10-m u-wind                m s-1
    "v10":  "v10",   # 10-m v-wind                m s-1
    "msl":  "msl",   # mean sea-level pressure    Pa
    "sp":   "sp",    # surface pressure           Pa
    "tcc":  "tcc",   # total cloud cover          0-1
}

# Pressure-level variables: cfgrib shortName → output prefix
# Stored as {prefix}_{level} (e.g. z_500, t_850).
# Uses geopotential height (gh, m) rather than geopotential (z, m2 s-2)
# for cross-model consistency; tag units in zarr attrs.
PLEV_VARS: dict[str, str] = {
    "gh": "z",   # geopotential height  m (gpm)
    "t":  "t",   # temperature          K
    "u":  "u",   # u-wind               m s-1
    "v":  "v",   # v-wind               m s-1
    "q":  "q",   # specific humidity    kg kg-1
}

# Pressure levels to extract (hPa)
TARGET_LEVELS: list[int] = [500, 700, 850, 925]

# Derived canonical variable list (surface + pressure-level combinations)
ALL_VARS: list[str] = list(SURFACE_VARS.values()) + [
    f"{prefix}_{lev}"
    for prefix in PLEV_VARS.values()
    for lev in TARGET_LEVELS
]

# ── GRIB extraction ───────────────────────────────────────────────────────────

def extract_fields(filepath: Path) -> xr.Dataset:
    """
    Read one GRIB2 file and return a flat xr.Dataset with target variables.

    Dimensions: (latitude, longitude) — caller adds time/lead_time.
    Missing variables are silently omitted; missing pressure levels are NaN.
    """
    import cfgrib  # noqa: PLC0415

    sub_datasets = cfgrib.open_datasets(str(filepath))

    arrays: dict[str, np.ndarray] = {}
    lats = lons = None

    for ds in sub_datasets:
        if lats is None and "latitude" in ds.coords:
            lats = ds["latitude"].values
            lons = ds["longitude"].values

        if "isobaricInhPa" in ds.dims:
            levels = ds["isobaricInhPa"].values.tolist()
            for grib_name, out_prefix in PLEV_VARS.items():
                if grib_name not in ds:
                    continue
                for lev in TARGET_LEVELS:
                    if lev not in levels:
                        continue
                    arrays[f"{out_prefix}_{lev}"] = (
                        ds[grib_name].sel(isobaricInhPa=lev).values.astype(np.float32)
                    )
        else:
            for grib_name, out_name in SURFACE_VARS.items():
                if grib_name in ds:
                    arrays[out_name] = ds[grib_name].values.astype(np.float32)

        ds.close()

    if lats is None:
        raise ValueError(f"No usable data in {filepath}")

    return xr.Dataset(
        {
            name: xr.DataArray(arr, dims=("latitude", "longitude"))
            for name, arr in arrays.items()
        },
        coords={"latitude": lats, "longitude": lons},
    )


# ── Zarr encoding ─────────────────────────────────────────────────────────────

def encoding_for(ds: xr.Dataset, chunks: tuple[int, ...]) -> dict[str, Any]:
    """Return per-variable zarr encoding with consistent chunk shape and compressor."""
    return {
        var: {
            "dtype":      DTYPE,
            "compressor": COMPRESSOR,   # numcodecs style, used by zarr v2 format
            "chunks":     chunks,
            "_FillValue": np.float32(np.nan),
        }
        for var in ds.data_vars
    }


# ── Manifest ──────────────────────────────────────────────────────────────────

class Manifest:
    """
    Tracks which raw files have been successfully processed.
    Stored as a JSON list of string keys next to the zarr store.
    """

    def __init__(self, zarr_path: Path) -> None:
        self._path = zarr_path.parent / (zarr_path.name + ".manifest.json")
        self._done: set[str] = set()
        if self._path.exists():
            self._done = set(json.loads(self._path.read_text()))

    def done(self, key: str) -> bool:
        return key in self._done

    def mark(self, key: str) -> None:
        self._done.add(key)
        self._path.write_text(json.dumps(sorted(self._done), indent=2))

    def __len__(self) -> int:
        return len(self._done)


# ── Zarr first-write / append ─────────────────────────────────────────────────

def zarr_write(
    ds: xr.Dataset,
    store: Path,
    append_dim: str,
    chunks: tuple[int, ...],
    first: bool,
) -> None:
    """Write ds to a zarr store, creating on first call and appending thereafter.

    zarr_format=2 is used throughout so that xarray's append_dim logic (which
    reads dimension metadata from the existing store) works reliably.  zarr 3
    stores dimension info differently and breaks the append lookup.
    """
    if first:
        enc = encoding_for(ds, chunks)
        ds.to_zarr(str(store), mode="w", encoding=enc, consolidated=True, zarr_format=2)
    else:
        ds.to_zarr(str(store), mode="a", append_dim=append_dim, consolidated=True, zarr_format=2)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(name: str, log_file: Path, verbose: bool = False) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log
