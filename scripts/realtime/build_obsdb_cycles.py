#!/usr/bin/env python3
"""
build_obsdb_cycles.py  (standalone, cycling)

Standalone: needs only numpy, netCDF4, PyYAML. No package install.

    python build_obsdb_cycles.py obsdb_cycles.yaml
    python build_obsdb_cycles.py obsdb_cycles.yaml --start 20260811T00Z --end 20260811T00Z
    python build_obsdb_cycles.py obsdb_cycles.yaml --dry-run      # list input files only

Build one IODA file per (obs type, analysis cycle) for LS-3DVar / LETKF:

    raw IODA files (hourly windows, overlapping)
        -> read every file whose name time is within +/- search_hours
        -> keep locations with ANY assimilated variable valid      (was: ALL)
        -> keep |dateTime - t_cycle| <= half_width_minutes         (was: whole files)
        -> keep inside the model domain (bbox from the hrrr npz)   (was: disabled)
        -> drop exact duplicates (station, dateTime) from overlapping files
        -> optional: keep the report nearest to t_cycle per station
        -> fill stationElevation where missing (per-type rule)
        -> write <output.root>/<obs_type>/<YYYYMMDDTHHZ>/<output_pattern>
"""

import argparse
import glob
import yaml
import pathlib
import datetime as dt

import numpy as np
import netCDF4 as nc

KEEP_GROUPS = ["MetaData", "ObsValue", "DerivedObsValue", "PreQC"]

FILL = {
    "f": np.float32(-3.3688e38),
    "i4": np.int32(-2147483645),
    "i8": np.int64(-9223372036854775801),
}

DEFAULT_ASSIM_VARS = [
    "airTemperatureAt2M",
    "dewPointTemperatureAt2M",
    "relativeHumidityAt2M",
    "windEastwardAt10M",
    "windNorthwardAt10M",
    "stationPressure",              # DerivedObsValue
]


# =====================================================
# helpers
# =====================================================

def fill_for(dtype):
    dtype = np.dtype(dtype) if not isinstance(dtype, type) else dtype
    if dtype is str or dtype == np.dtype(object):
        return ""
    if np.issubdtype(dtype, np.floating):
        return FILL["f"].astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return FILL["i8"] if dtype.itemsize == 8 else FILL["i4"]
    return None


def is_str_var(var):
    return var.dtype is str or (isinstance(var.dtype, type) and var.dtype is str)


def read_file(path):
    """-> dict[group][var] = (np.ndarray raw values, attrs dict, dtype), nloc, global attrs"""
    out = {}
    with nc.Dataset(path) as ds:
        ds.set_auto_mask(False)
        nloc = len(ds.dimensions["Location"])
        gattrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
        for g in KEEP_GROUPS:
            if g not in ds.groups:
                continue
            out[g] = {}
            for vname, var in ds.groups[g].variables.items():
                attrs = {a: var.getncattr(a) for a in var.ncattrs()}
                vals = var[:]
                if is_str_var(var):
                    vals = np.asarray(vals, dtype=object)
                    dtype = str
                else:
                    vals = np.asarray(vals)
                    dtype = vals.dtype
                out[g][vname] = (vals, attrs, dtype)
    return out, nloc, gattrs


def valid_mask(vals, attrs):
    if not np.issubdtype(vals.dtype, np.floating):
        return np.ones(vals.shape, bool)
    ok = np.isfinite(vals) & (np.abs(vals) < 1.0e30)
    if "_FillValue" in attrs:
        ok &= vals != attrs["_FillValue"]
    return ok


def decode_times(vals, attrs):
    units = str(attrs.get("units", "seconds since 1970-01-01T00:00:00Z"))
    epoch = np.datetime64(units.split("since")[-1].strip().rstrip("Z").replace(" ", "T"), "s")
    v = np.asarray(vals, np.float64)
    ok = np.isfinite(v) & (np.abs(v) < 1e11)
    t = np.full(v.shape, np.datetime64("NaT"), dtype="datetime64[s]")
    t[ok] = epoch + v[ok].astype(np.int64).astype("timedelta64[s]")
    return t


def domain_from_npz(npz_path, margin_deg):
    d = np.load(npz_path, allow_pickle=True)
    lat = np.asarray(d["lats"], np.float64)
    lon = ((np.asarray(d["lons"], np.float64) + 180.0) % 360.0) - 180.0
    return (lon.min() - margin_deg, lon.max() + margin_deg,
            lat.min() - margin_deg, lat.max() + margin_deg)


def candidate_files(input_dir, pattern, t_cycle, search_hours):
    files = set()
    for dh in range(-search_hours, search_hours + 1):
        t = t_cycle + dt.timedelta(hours=dh)
        for f in glob.glob(str(pathlib.Path(input_dir) / t.strftime(pattern))):
            files.add(f)
    return sorted(files)


# =====================================================
# per-cycle build
# =====================================================

def build_cycle(obs_type, ocfg, gcfg, t_cycle, bounds, dry_run=False):
    win = gcfg["window"]
    half = np.timedelta64(int(win.get("half_width_minutes", 30) * 60), "s")
    t64 = np.datetime64(t_cycle.strftime("%Y-%m-%dT%H:%M:%S"), "s")
    assim_vars = gcfg.get("assim_vars", DEFAULT_ASSIM_VARS)
    thinning = gcfg.get("thinning", "none")

    files = candidate_files(ocfg["input_dir"], ocfg["file_pattern"], t_cycle,
                            int(win.get("search_hours", 2)))
    if not files:
        print(f"  {obs_type:10s} {t_cycle:%Y%m%dT%HZ}: no input files "
              f"({ocfg['input_dir']}/{ocfg['file_pattern']})")
        return None
    if dry_run:
        print(f"  {obs_type:10s} {t_cycle:%Y%m%dT%HZ}: {len(files)} files")
        for f in files:
            print(f"      {f}")
        return None

    # ---- read all files, union of variables ----------------------------
    parts, gattrs0 = [], None
    schema = {g: {} for g in KEEP_GROUPS}          # g -> vname -> (attrs, dtype)
    for f in files:
        data, nloc, gattrs = read_file(f)
        gattrs0 = gattrs0 or gattrs
        parts.append((data, nloc))
        for g, vars_ in data.items():
            for vname, (_, attrs, dtype) in vars_.items():
                schema[g].setdefault(vname, (attrs, dtype))

    def column(g, vname):
        attrs, dtype = schema[g][vname]
        cols = []
        for data, nloc in parts:
            if g in data and vname in data[g]:
                cols.append(data[g][vname][0])
            else:
                cols.append(np.full(nloc, fill_for(dtype),
                                    dtype=object if dtype is str else dtype))
        return np.concatenate(cols)

    n_raw = sum(n for _, n in parts)
    cols = {g: {v: column(g, v) for v in schema[g]} for g in KEEP_GROUPS if schema[g]}
    meta = cols.get("MetaData", {})

    # ---- masks ----------------------------------------------------------
    any_valid = np.zeros(n_raw, bool)
    for g in ("ObsValue", "DerivedObsValue"):
        for v in assim_vars:
            if g in cols and v in cols[g]:
                any_valid |= valid_mask(cols[g][v], schema[g][v][0])
    keep = any_valid.copy()
    n_valid = int(keep.sum())

    if "dateTime" in meta:
        t_obs = decode_times(meta["dateTime"], schema["MetaData"]["dateTime"][0])
        in_win = ~np.isnat(t_obs) & (np.abs(t_obs - t64) <= half)
        keep &= in_win
    else:
        t_obs = np.full(n_raw, np.datetime64("NaT"), dtype="datetime64[s]")
        print(f"  WARNING {obs_type}: no MetaData/dateTime -> time window not applied")
    n_win = int(keep.sum())

    if bounds is not None:
        lat = np.asarray(meta["latitude"], np.float64)
        lon = ((np.asarray(meta["longitude"], np.float64) + 180.0) % 360.0) - 180.0
        lonmin, lonmax, latmin, latmax = bounds
        keep &= (lat >= latmin) & (lat <= latmax) & (lon >= lonmin) & (lon <= lonmax)
    n_dom = int(keep.sum())

    # ---- exact duplicates from overlapping files ------------------------
    stn = meta.get("stationIdentification", np.arange(n_raw).astype(str))
    tsec = np.where(np.isnat(t_obs), -1, t_obs.astype(np.int64))
    seen = set()
    for i in np.where(keep)[0]:
        key = (str(stn[i]), int(tsec[i]))
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    n_dedup = int(keep.sum())

    # ---- optional: nearest report per station ---------------------------
    if thinning == "nearest":
        best = {}
        dt_abs = np.abs((t_obs - t64).astype(np.int64))
        for i in np.where(keep)[0]:
            s = str(stn[i])
            if s not in best or dt_abs[i] < dt_abs[best[s]]:
                best[s] = i
        keep[:] = False
        keep[list(best.values())] = True
    idx = np.where(keep)[0]
    n_out = len(idx)

    # ---- stationElevation fill ------------------------------------------
    erule = ocfg.get("elevation", {"source": "file"})
    n_elev_filled = 0
    if "MetaData" in cols:
        m = cols["MetaData"]
        if "stationElevation" not in m:
            m["stationElevation"] = np.full(n_raw, FILL["f"], np.float32)
            schema["MetaData"]["stationElevation"] = (
                {"units": "m", "long_name": "station elevation above mean sea level",
                 "_FillValue": FILL["f"]}, np.dtype(np.float32))
        e = np.asarray(m["stationElevation"], np.float32)
        missing = ~valid_mask(e, schema["MetaData"]["stationElevation"][0])
        src = erule.get("source", "file")
        if src == "height_minus" and "height" in m:
            h = np.asarray(m["height"], np.float32)
            hok = valid_mask(h, schema["MetaData"]["height"][0])
            fillv = h - float(erule.get("value", 2.0))
            e = np.where(missing & hok, fillv, e)
            n_elev_filled = int((missing & hok)[idx].sum())
        elif src == "constant":
            e = np.where(missing, np.float32(erule["value"]), e)
            n_elev_filled = int(missing[idx].sum())
        m["stationElevation"] = e
        n_elev_missing = int((~valid_mask(e, schema["MetaData"]["stationElevation"][0]))[idx].sum())
    else:
        n_elev_missing = n_out

    # ---- write ----------------------------------------------------------
    outdir = pathlib.Path(gcfg["output"]["root"]) / obs_type / t_cycle.strftime("%Y%m%dT%HZ")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / t_cycle.strftime(ocfg["output_pattern"])

    with nc.Dataset(outfile, "w") as out:
        for a, v in (gattrs0 or {}).items():
            out.setncattr(a, v)
        t0 = (t_cycle - dt.timedelta(seconds=int(half / np.timedelta64(1, "s"))))
        t1 = (t_cycle + dt.timedelta(seconds=int(half / np.timedelta64(1, "s"))))
        out.setncattr("datetimeReference", t_cycle.strftime("%Y-%m-%dT%H:%M:%SZ"))
        out.setncattr("datetimeWindowStart", t0.strftime("%Y-%m-%dT%H:%M:%SZ"))
        out.setncattr("datetimeWindowEnd", t1.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if "datetimeRange" in out.ncattrs():
            out.delncattr("datetimeRange")
        out.setncattr("sourceFiles", len(files))
        out.setncattr("builder", "build_obsdb_cycles.py (standalone)")

        out.createDimension("Location", n_out)
        out.createVariable("Location", "i8", ("Location",))[:] = np.arange(n_out)

        for g in KEEP_GROUPS:
            if g not in cols:
                continue
            gout = out.createGroup(g)
            for vname, vals in cols[g].items():
                attrs, dtype = schema[g][vname]
                if dtype is str:
                    ov = gout.createVariable(vname, str, ("Location",))
                    ov[:] = np.asarray(vals[idx], dtype=object)
                else:
                    fv = attrs.get("_FillValue", fill_for(dtype))
                    ov = gout.createVariable(vname, dtype, ("Location",), fill_value=fv)
                    ov[:] = vals[idx]
                for a, v in attrs.items():
                    if a != "_FillValue":
                        ov.setncattr(a, v)

        # ObsError: constant per variable from config (as before)
        gerr = out.createGroup("ObsError")
        for vname, err in ocfg.get("obserror", {}).items():
            for g in ("ObsValue", "DerivedObsValue"):
                if g in cols and vname in cols[g]:
                    v = gerr.createVariable(vname, np.float32, ("Location",))
                    v[:] = np.full(n_out, err, np.float32)
                    units = schema[g][vname][0].get("units")
                    if units:
                        v.units = units
                    v.long_name = f"observation error for {vname}"
                    break

    # ---- report ---------------------------------------------------------
    counts = []
    for g in ("ObsValue", "DerivedObsValue"):
        for v in assim_vars:
            if g in cols and v in cols[g]:
                n = int(valid_mask(cols[g][v][idx], schema[g][v][0]).sum())
                counts.append(f"{v.replace('AtTwoM', '')}={n}")
    dtm = ((t_obs[idx] - t64) / np.timedelta64(1, "m")) if n_out else np.array([0.0])
    print(f"  {obs_type:10s} {t_cycle:%Y%m%dT%HZ}: files={len(files)}  raw={n_raw}  "
          f"any-valid={n_valid}  window={n_win}  domain={n_dom}  dedup={n_dedup}  "
          f"out={n_out}  stations={len({str(stn[i]) for i in idx})}")
    print(f"      dt[min] {np.nanmin(dtm):+.0f}..{np.nanmax(dtm):+.0f}   "
          f"elev filled={n_elev_filled} missing={n_elev_missing}   " + "  ".join(counts))
    print(f"      -> {outfile}")
    return outfile


# =====================================================
# main
# =====================================================

def main():
    ap = argparse.ArgumentParser(description="Build per-cycle IODA obs files")
    ap.add_argument("config", nargs="?", default="obsdb_cycles.yaml")
    ap.add_argument("--start", help="override cycles.start, e.g. 20260811T00Z")
    ap.add_argument("--end", help="override cycles.end")
    ap.add_argument("--types", nargs="+", help="override obs_types, e.g. asos tempest")
    ap.add_argument("--output-root", help="override output.root")
    ap.add_argument("--dry-run", action="store_true", help="list input files, write nothing")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.start:
        cfg["cycles"]["start"] = args.start
    if args.end:
        cfg["cycles"]["end"] = args.end
    if args.types:
        cfg["obs_types"] = args.types
    if args.output_root:
        cfg["output"]["root"] = args.output_root

    bounds = None
    dcfg = cfg.get("domain", {})
    if dcfg.get("enabled", False):
        if "npz" in dcfg:
            bounds = domain_from_npz(dcfg["npz"], float(dcfg.get("margin_deg", 0.05)))
        else:
            bounds = (dcfg["lon_min"], dcfg["lon_max"], dcfg["lat_min"], dcfg["lat_max"])
        print(f"Domain: lon=[{bounds[0]:.3f}, {bounds[1]:.3f}]  lat=[{bounds[2]:.3f}, {bounds[3]:.3f}]")

    start = dt.datetime.strptime(cfg["cycles"]["start"], "%Y%m%dT%HZ")
    end = dt.datetime.strptime(cfg["cycles"]["end"], "%Y%m%dT%HZ")
    step = dt.timedelta(hours=cfg["cycles"]["interval_hours"])
    print(f"Cycles {start:%Y%m%dT%HZ} .. {end:%Y%m%dT%HZ} every {step}   "
          f"window +/-{cfg['window'].get('half_width_minutes', 30)} min   "
          f"thinning={cfg.get('thinning', 'none')}")
    print(f"Output root: {cfg['output']['root']}" + ("   (dry run)" if args.dry_run else ""))

    t = start
    while t <= end:
        print(f"\n=== {t:%Y-%m-%d %HZ} ===")
        for obs_type in cfg["obs_types"]:
            build_cycle(obs_type, cfg["obs"][obs_type], cfg, t, bounds, args.dry_run)
        t += step

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
