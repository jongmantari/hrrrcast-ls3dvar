# ls3dvar/hrrrcast/channels.py  (extracted from mantari/ls3dvar/channels.py)
"""
mantari/ls3dvar/channels.py

Channel lookup, normalization statistics and wind-grid geometry for the
HRRRCast state (hrrr npz: pl_vars x levels, then sfc_vars; normalized).

    names = channel_names(npz)              # ['UGRD_200', ..., 'PRES', 'MSLMA', ..., 'T2M', ...]
    ch    = resolve_channel(names, "U10")   # alias -> UGRD10M / U10 / ...
    stats = ChannelStats(cfg, names)        # stats.get("UGRD10M") -> (mean, std)
    cos_t, sin_t = grid_rotation(lats, lons)

Statistics come from, in order:
    1. cfg["channel_stats"]            {NAME: {mean: .., std: ..}}  or  {NAME: [mean, std]}
    2. cfg["normalize_stats_nc"]       net-diffusion/normalize-stats.nc (several layouts tried;
                                       validated against the known T2M stats before use)
    3. built-ins for PRES / MSLMA / T2M / D2M (operators.py)
"""
from __future__ import annotations

import numpy as np

# same values as operators.py (kept here so numpy-only tools need no TensorFlow import)
PRES_MEAN, PRES_STD = 96069.5, 571.6757
MSLMA_MEAN, MSLMA_STD = 99621.234, 617.72125
T2M_MEAN, T2M_STD = 288.07220458984375, 8.021387
D2M_MEAN, D2M_STD = 281.51956, 8.593886

BUILTIN_STATS = {
    "PRES": (PRES_MEAN, PRES_STD),
    "MSLMA": (MSLMA_MEAN, MSLMA_STD),
    "T2M": (T2M_MEAN, T2M_STD),
    "D2M": (D2M_MEAN, D2M_STD),
}

ALIASES = {
    "U10": ["UGRD10M", "UGRD_10M", "U10", "UGRD10", "U10M", "10U", "UGRD_10"],
    "V10": ["VGRD10M", "VGRD_10M", "V10", "VGRD10", "V10M", "10V", "VGRD_10"],
    "PS": ["PRES"],
}

UNITS = {"T2M": "K", "D2M": "K", "PRES": "Pa", "MSLMA": "Pa", "TMP": "K",
         "UGRD": "m/s", "VGRD": "m/s", "HGT": "m", "SPFH": "kg/kg", "VVEL": "Pa/s",
         "REFC": "dBZ", "APCP": "log1p(mm)"}


def channel_names(npz):
    pl = [str(v) for v in npz["pl_vars"]]
    lv = [str(int(float(l))) for l in npz["levels"]]
    return [f"{v}_{l}" for v in pl for l in lv] + [str(v) for v in npz["sfc_vars"]]


def resolve_name(names, key):
    """exact channel name, or first alias found (U10 -> UGRD10M ...)"""
    if key in names:
        return key
    for cand in ALIASES.get(key.upper(), []):
        if cand in names:
            return cand
    sfc = [n for n in names if "_" not in n or n.endswith("M")]
    raise KeyError(f"channel '{key}' not found. surface channels: {sfc}")


def resolve_channel(names, key):
    return names.index(resolve_name(names, key))


def unit_of(name):
    if name in UNITS:
        return UNITS[name]
    if name.startswith(("UGRD", "VGRD")):
        return "m/s"
    return UNITS.get(name.split("_")[0], "")


# =============================================================================
# normalization statistics
# =============================================================================

def _split(name):
    """'TMP_850' -> ('TMP', 850.0); 'UGRD10M' -> ('UGRD10M', None)"""
    if "_" in name:
        v, lev = name.rsplit("_", 1)
        try:
            return v, float(lev)
        except ValueError:
            pass
    return name, None


def stats_from_dataset(ds, name, prefix=""):
    """
    (mean, std) for channel `name` from an HRRRCast normalize-stats dataset:
        VAR(stat[, level])   stat = ['mean','std','min','max'], level / level_gfs coords
    prefix="GFS-" for the GFS forcing statistics. Returns None if not found.
    """
    var, lev = _split(name)
    key = prefix + (var if lev is not None else name)
    if key not in ds.data_vars:
        return None
    da = ds[key]
    if lev is not None:
        ldims = [d for d in da.dims if "lev" in d.lower() or "plev" in d.lower()]
        if not ldims:
            return None
        lv = np.asarray(ds[ldims[0]].values, float)
        i = int(np.argmin(np.abs(lv - lev)))
        if abs(lv[i] - lev) > 0.5:
            return None
        da = da.isel({ldims[0]: i})
    for d in da.dims:
        if d in ds.coords:
            lab = [str(x).lower() for x in np.ravel(ds[d].values)]
            if "mean" in lab and "std" in lab:
                a = np.ravel(np.asarray(da.values, float))
                return float(a[lab.index("mean")]), float(a[lab.index("std")])
    a = np.ravel(np.asarray(da.values, float))
    if a.size == 2:
        return float(a[0]), float(a[1])
    if "mean" in da.attrs and "std" in da.attrs:
        return float(da.attrs["mean"]), float(da.attrs["std"])
    return None


def open_stats(path):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return ds.load()


class ChannelStats:
    """
    stats.get("UGRD10M")            -> (mean, std) of the HRRR state channel
    stats.get("TMP_850", gfs=True)  -> (mean, std) of the GFS forcing channel (GFS-TMP @ 850)
    """

    def __init__(self, cfg, names=None):
        self.names = names
        self.explicit = {}
        for k, v in (cfg.get("channel_stats") or {}).items():
            self.explicit[k] = (float(v["mean"]), float(v["std"])) if isinstance(v, dict) \
                else (float(v[0]), float(v[1]))
        self.nc = cfg.get("normalize_stats_nc")
        self.ds = None
        self.cache = {}

    def _dataset(self):
        if self.ds is None:
            self.ds = False
            if self.nc:
                try:
                    ds = open_stats(self.nc)
                    r = stats_from_dataset(ds, "T2M")
                    if r and abs(r[0] - T2M_MEAN) < 1e-2 and abs(r[1] - T2M_STD) < 1e-3:
                        self.ds = ds
                    else:
                        print(f"WARNING normalize_stats_nc: T2M read as {r}, expected "
                              f"({T2M_MEAN:.4f}, {T2M_STD:.4f}) -> layout not understood, not used")
                except Exception as e:                         # noqa: BLE001
                    print(f"WARNING normalize_stats_nc unreadable ({e}) -> not used")
        return self.ds

    def get(self, name, gfs=False):
        key = ("gfs:" if gfs else "") + name
        if key in self.cache:
            return self.cache[key]
        r = None if gfs else self.explicit.get(name)
        if r is None and self._dataset() is not False:
            r = stats_from_dataset(self.ds, name, prefix="GFS-" if gfs else "")
        if r is None and not gfs:
            r = BUILTIN_STATS.get(name)
        if r is None:
            raise KeyError(
                f"no normalization stats for '{key}'. Add to the config:\n"
                f"  channel_stats:\n    {name}: {{mean: ..., std: ...}}\n"
                f"(values from net-diffusion/normalize-stats.nc), or set normalize_stats_nc.")
        self.cache[key] = (float(r[0]), float(r[1]))
        return self.cache[key]


# =============================================================================
# wind geometry
# =============================================================================

def grid_rotation(lats, lons):
    """
    cos/sin of the angle theta between the grid x-axis and local east
    (counter-clockwise), from the lat/lon arrays themselves (projection-agnostic).

        u_earth = u_grid cos(theta) - v_grid sin(theta)
        v_earth = u_grid sin(theta) + v_grid cos(theta)
    """
    lat = np.deg2rad(np.asarray(lats, np.float64))
    lon = np.unwrap(np.deg2rad(np.asarray(lons, np.float64)), axis=1)
    dlon = np.gradient(lon, axis=1)
    dlat = np.gradient(lat, axis=1)
    theta = np.arctan2(dlat, dlon * np.cos(lat))
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
