#!/usr/bin/env python3
"""
mrms_stage.py - fetch MRMS composite reflectivity and put it on the HRRRCast (HRRR) grid.

MRMS MergedReflectivityQCComposite: 2-min, 0.01 deg CONUS mosaic of the column-max reflectivity,
files  MRMS_MergedReflectivityQCComposite_00.50_YYYYMMDD-HHMMSS.grib2.gz
from the NOAA open-data bucket (anonymous HTTPS, no AWS account needed):
    https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MergedReflectivityQCComposite_00.50/YYYYMMDD/...

For every analysis time t in [--start, --end] (every --step-min, default 60) it keeps the frames in
(t - window, t] (default 30 min, i.e. ~15 frames), maps each to the target grid and writes

    OUT/YYYYMMDDTHHZ/mrms_YYYYMMDD_HH.npz        (t on the hour)
    OUT/YYYYMMDDTHHZ/mrms_YYYYMMDD_HHMM.npz      (sub-hourly t, e.g. --step-min 15; dir = hour of t)
        refc_seq    (T, H, W) float32  dBZ on the target grid, "max" aggregation, no echo -> 0,
                                       no radar coverage -> NaN
        refc_mean   (T, H, W) float32  same with "mean" aggregation (mean of linear Z, back to dBZ)
        coverage    (T, H, W) float32  fraction of MRMS pixels in the cell with radar coverage
        times       (T,) str           frame valid times (UTC, ISO)
        refc_t      (H, W)    float32  frame nearest to t (the "analysis-time" field)
        lats, lons  (H, W)             target grid (copied from the grid npz)

Target grid: lats/lons of an HRRRCast npz (the reference hrrr npz of that hour, or --grid-npz).
Aggregation: every MRMS pixel goes to its nearest target cell (KD-tree on the sphere) if it is
within --max-dist-km; the cell takes the max (and the mean in linear Z) of its pixels. At 0.01 deg
(~1 km) a 3 km cell gets ~9 pixels, so "max" keeps cores and "mean" is closer to a model field.

--compare-ref prints, per hour, MRMS vs HRRR-analysis REFC (domain mean, coverage >= 20 dBZ,
correlation) and writes a 3-panel png - a first look at how "HRRR-like" raw MRMS is.

Needs: numpy, scipy, xarray + cfgrib (eccodes) or pygrib.
    conda install -c conda-forge cfgrib        # if neither is installed

Example (8/12-13 case):
    python mrms_stage.py --start 2026-08-12T19 --end 2026-08-13T03 \
        --grid-npz reference/hrrr_ana/20260811T00Z/hrrr_20260811_00.npz \
        --ref-dir reference/hrrr_ana --out mrms --compare-ref
"""
import argparse
import gzip
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "MergedReflectivityQCComposite_00.50"
FNAME_RE = re.compile(r"MRMS_" + re.escape(PRODUCT) + r"_(\d{8})-(\d{6})\.grib2\.gz$")
MISSING = -900.0        # MRMS: -999 = no radar coverage, -99 = no echo / below threshold


# ----------------------------------------------------------------------------- listing / download
def http_get(url, dest=None, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                if dest is None:
                    return r.read()
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
                return dest
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"    retry {k + 1}: {e}")
            time.sleep(2 * (k + 1))


def list_day(day, base=BUCKET, product=PRODUCT):
    """{datetime: url} of all files of one UTC day (S3 ListObjectsV2, paginated)."""
    prefix = f"CONUS/{product}/{day:%Y%m%d}/"
    out, token = {}, None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = http_get(f"{base}/?{urllib.parse.urlencode(q)}")
        root = ET.fromstring(xml)
        ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        pfx = "s:" if ns else ""
        for c in root.findall(f"{pfx}Contents", ns):
            key = c.find(f"{pfx}Key", ns).text
            m = FNAME_RE.search(key)
            if m:
                out[datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")] = f"{base}/{key}"
        trunc = root.find(f"{pfx}IsTruncated", ns)
        if trunc is not None and trunc.text == "true":
            token = root.find(f"{pfx}NextContinuationToken", ns).text
        else:
            return out


# ----------------------------------------------------------------------------- GRIB reading
def read_grib(path):
    """(values 2-D, lats 1-D, lons 1-D in [-180,180)) from an MRMS grib2 file.
    Tries pygrib, then grib2io, then xarray+cfgrib."""
    v = la = lo = None
    errs = []
    try:
        import pygrib
        with pygrib.open(str(path)) as f:
            g = f.message(1)
            v = np.ma.filled(g.values, np.nan).astype(np.float32)
            la2, lo2 = g.latlons()
        la, lo = la2[:, 0].astype(np.float64), lo2[0, :].astype(np.float64)
    except Exception as e:  # noqa: BLE001
        errs.append(f"pygrib: {e}")
    if v is None:
        try:
            import grib2io
            with grib2io.open(str(path)) as f:
                m = f[0]
                v = np.asarray(m.data, np.float32)
                la2, lo2 = m.latlons()
            la, lo = np.asarray(la2)[:, 0].astype(np.float64), np.asarray(lo2)[0, :].astype(np.float64)
        except Exception as e:  # noqa: BLE001
            errs.append(f"grib2io: {e}")
    if v is None:
        try:
            import xarray as xr
            ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
            v = np.asarray(ds[list(ds.data_vars)[0]].values, np.float32)
            la = np.asarray(ds["latitude"].values, np.float64)
            lo = np.asarray(ds["longitude"].values, np.float64)
        except Exception as e:  # noqa: BLE001
            errs.append(f"cfgrib: {e}")
    if v is None:
        raise RuntimeError("cannot read GRIB2 (install pygrib, grib2io or cfgrib):\n  " + "\n  ".join(errs))
    lo = np.where(lo >= 180.0, lo - 360.0, lo)
    return v, la, lo


# ----------------------------------------------------------------------------- regridding
def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class Regridder:
    """MRMS regular lat/lon (1-D axes) -> curvilinear target grid, by nearest-cell binning."""

    def __init__(self, tlats, tlons, max_dist_km=2.5, margin_deg=0.3):
        from scipy.spatial import cKDTree
        self.shape = tlats.shape
        self.tree = cKDTree(_xyz(tlats.ravel(), tlons.ravel()))
        self.box = (tlats.min() - margin_deg, tlats.max() + margin_deg,
                    tlons.min() - margin_deg, tlons.max() + margin_deg)
        self.maxd = max_dist_km / 6371.0
        self.key = None

    def _prepare(self, la, lo):
        key = (la.size, lo.size, float(la[0]), float(lo[0]))
        if key == self.key:
            return
        la0, la1, lo0, lo1 = self.box
        self.iy = np.where((la >= la0) & (la <= la1))[0]
        self.ix = np.where((lo >= lo0) & (lo <= lo1))[0]
        LA, LO = np.meshgrid(la[self.iy], lo[self.ix], indexing="ij")
        d, idx = self.tree.query(_xyz(LA.ravel(), LO.ravel()))
        self.ok = d <= self.maxd
        self.cell = idx[self.ok]
        self.key = key
        n = np.bincount(self.cell, minlength=int(np.prod(self.shape)))
        self.npix = n

    def __call__(self, v, la, lo):
        self._prepare(la, lo)
        sub = v[np.ix_(self.iy, self.ix)].ravel()[self.ok]
        ncell = int(np.prod(self.shape))
        cov = sub > MISSING
        dbz = np.where(cov, np.maximum(sub, 0.0), 0.0)          # no echo (-99 etc.) -> 0 dBZ
        # max
        mx = np.full(ncell, -np.inf)
        np.maximum.at(mx, self.cell[cov], dbz[cov])
        # mean in linear Z
        z = np.where(cov, 10.0 ** (dbz / 10.0), 0.0)
        zs = np.bincount(self.cell, weights=z, minlength=ncell)
        nc = np.bincount(self.cell, weights=cov.astype(float), minlength=ncell)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = 10.0 * np.log10(zs / nc)
            frac = nc / np.maximum(self.npix, 1)
        mean = np.where(nc > 0, np.maximum(mean, 0.0), np.nan)
        mx = np.where(nc > 0, mx, np.nan)
        frac = np.where(self.npix > 0, frac, 0.0)
        return (mx.reshape(self.shape).astype(np.float32), mean.reshape(self.shape).astype(np.float32),
                frac.reshape(self.shape).astype(np.float32))


# ----------------------------------------------------------------------------- helpers
def parse_time(s):
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise SystemExit(f"bad time {s!r}: use YYYY-MM-DDTHH or YYYY-MM-DDTHH:MM")


def mrms_path(out, t):
    """OUT/YYYYMMDDTHHZ/mrms_YYYYMMDD_HH.npz on the hour, ..._HHMM.npz otherwise"""
    name = f"mrms_{t:%Y%m%d_%H}.npz" if t.minute == 0 else f"mrms_{t:%Y%m%d_%H%M}.npz"
    return Path(out) / f"{t:%Y%m%dT%HZ}" / name


def grid_from_npz(path):
    z = np.load(path, allow_pickle=True)
    return np.asarray(z["lats"], np.float64), np.asarray(z["lons"], np.float64)


def ref_refc(path, stats_nc, repo):
    """HRRR-analysis REFC (dBZ) from a reference hrrr npz."""
    sys.path.insert(0, str(Path(repo).resolve()))
    import xarray as xr
    from mantari.ls3dvar.channels import channel_names, stats_from_dataset
    z = np.load(path, allow_pickle=True)
    n = channel_names(z)
    c = n.index("REFC")
    mu, sd = stats_from_dataset(xr.open_dataset(stats_nc), "REFC")[:2]
    return z["model_input"][0, ..., c] * sd + mu


def compare_plot(png, lats, lons, mrms, ref, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    lv = np.arange(5, 65, 5)
    cmap = plt.get_cmap("turbo")
    norm = BoundaryNorm(lv, cmap.N, extend="max")
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for a, f, t in ((ax[0], ref, "HRRR analysis REFC"), (ax[1], mrms, "MRMS on HRRRCast grid")):
        m = a.pcolormesh(lons, lats, np.ma.masked_less(f, 5), cmap=cmap, norm=norm, shading="auto")
        a.set_title(f"{t}\nmean {np.nanmean(f):.3g}  max {np.nanmax(f):.1f} dBZ")
    fig.colorbar(m, ax=ax[:2], shrink=0.8, label="dBZ")
    d = np.where(np.isfinite(mrms), mrms - ref, np.nan)
    m2 = ax[2].pcolormesh(lons, lats, d, cmap="RdBu_r", vmin=-30, vmax=30, shading="auto")
    ax[2].set_title(f"MRMS - HRRR  rms {np.sqrt(np.nanmean(d ** 2)):.2f} dBZ")
    fig.colorbar(m2, ax=ax[2], shrink=0.8, label="dBZ")
    fig.suptitle(title, fontweight="bold")
    fig.savefig(png, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first analysis time, YYYY-MM-DDTHH[:MM] (UTC)")
    ap.add_argument("--end", required=True, help="last analysis time")
    ap.add_argument("--step-min", type=int, default=60, help="analysis interval (15 for sub-hourly)")
    ap.add_argument("--window-min", type=int, default=30, help="frames in (t - window, t]")
    ap.add_argument("--out", default="mrms")
    ap.add_argument("--grid-npz", default=None, help="npz with lats/lons of the target grid")
    ap.add_argument("--ref-dir", default=None,
                    help="reference dir (TAG/hrrr_YYYYMMDD_HH.npz); grid + --compare-ref")
    ap.add_argument("--stats", default=None, help="normalize-stats.nc (for --compare-ref; "
                    "default <repo>/net-diffusion/normalize-stats.nc via --repo)")
    ap.add_argument("--repo", default=".", help="HRRRCast-live repo (mantari imports, stats)")
    ap.add_argument("--compare-ref", action="store_true")
    ap.add_argument("--max-dist-km", type=float, default=2.5)
    ap.add_argument("--base-url", default=BUCKET)
    ap.add_argument("--keep-grib", action="store_true", help="keep downloaded files in OUT/raw")
    a = ap.parse_args()

    t0, t1 = parse_time(a.start), parse_time(a.end)
    win = timedelta(minutes=a.window_min)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    if a.keep_grib:
        raw.mkdir(exist_ok=True)
    stats = a.stats or str(Path(a.repo) / "net-diffusion/normalize-stats.nc")

    hours = []
    t = t0
    while t <= t1:
        hours.append(t)
        t += timedelta(minutes=a.step_min)

    # listing for all days touched (window reaches into the previous day at 00Z)
    days = sorted({(h - win).date() for h in hours} | {h.date() for h in hours})
    files = {}
    for d in days:
        lst = list_day(datetime(d.year, d.month, d.day), a.base_url)
        print(f"listing {d}: {len(lst)} files")
        files.update(lst)
    if not files:
        raise SystemExit("no MRMS files found - check the date, or --base-url")

    regrid, grid = None, None
    for h in hours:
        tag = f"{h:%Y%m%dT%H%MZ}" if h.minute else f"{h:%Y%m%dT%HZ}"
        dest = mrms_path(out, h)
        frames = sorted(ft for ft in files if h - win < ft <= h + timedelta(seconds=59))
        print(f"\n{tag}: {len(frames)} frames in ({(h - win):%H:%M}, {h:%H:%M}]")
        if not frames:
            print("  no frames - skipped")
            continue
        refp = Path(a.ref_dir) / f"{h:%Y%m%dT%HZ}" / f"hrrr_{h:%Y%m%d_%H}.npz" if a.ref_dir else None
        if regrid is None:
            gp = a.grid_npz or (refp if refp and refp.exists() else None)
            if gp is None:
                raise SystemExit("need --grid-npz (or --ref-dir with the reference of the first hour)")
            grid = grid_from_npz(gp)
            regrid = Regridder(*grid, max_dist_km=a.max_dist_km)
            print(f"target grid {grid[0].shape} from {gp}  "
                  f"lat {grid[0].min():.2f}..{grid[0].max():.2f}  lon {grid[1].min():.2f}..{grid[1].max():.2f}")
        if dest.exists():
            print(f"  exists: {dest}")
        else:
            seq_max, seq_mean, cov, times = [], [], [], []
            with tempfile.TemporaryDirectory() as td:
                for ft in frames:
                    url = files[ft]
                    gz = (raw / Path(url).name) if a.keep_grib else Path(td) / Path(url).name
                    if not gz.exists():
                        http_get(url, gz)
                    g2 = Path(td) / gz.name[:-3]
                    with gzip.open(gz, "rb") as fi, open(g2, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    v, la, lo = read_grib(g2)
                    if la[0] > la[-1]:                       # make latitude ascending
                        la, v = la[::-1], v[::-1]
                    mx, mn, fr = regrid(v, la, lo)
                    seq_max.append(mx); seq_mean.append(mn); cov.append(fr); times.append(ft.isoformat())
                    g2.unlink()
                    print(f"  {ft:%H:%M:%S}  max {np.nanmax(mx):5.1f} dBZ  >=20: "
                          f"{100 * np.nanmean(mx >= 20):5.2f}%  coverage {100 * np.mean(fr > 0):5.1f}%")
            k = int(np.argmin([abs((datetime.fromisoformat(s) - h).total_seconds()) for s in times]))
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, refc_seq=np.stack(seq_max), refc_mean=np.stack(seq_mean),
                                coverage=np.stack(cov), times=np.array(times), refc_t=seq_max[k],
                                refc_t_mean=seq_mean[k], lats=grid[0], lons=grid[1],
                                product=PRODUCT, window_min=a.window_min)
            print(f"  -> {dest}  (analysis-time frame {times[k]})")

        if h.minute:                                  # the reference exists on the hour only
            continue
        if a.compare_ref and (refp is None or not refp.exists()):
            print(f"  (no reference {refp} - comparison skipped; stage it with stage_reference.sh)")
        if a.compare_ref and refp is not None and refp.exists():
            z = np.load(dest, allow_pickle=True)
            ref = ref_refc(refp, stats, a.repo)
            for lab, f in (("max ", z["refc_t"]), ("mean", z["refc_t_mean"])):
                ok = np.isfinite(f)
                c = np.corrcoef(f[ok], ref[ok])[0, 1]
                print(f"  vs HRRR [{lab}]  mean MRMS {np.nanmean(f):.3f}  HRRR {ref.mean():.3f}   "
                      f">=20 dBZ MRMS {100 * np.nanmean(f >= 20):.2f}%  HRRR {100 * np.mean(ref >= 20):.2f}%"
                      f"   corr {c:.3f}   no-coverage {100 * (~ok).mean():.1f}%")
            compare_plot(dest.with_suffix(".png"), z["lats"], z["lons"], z["refc_t_mean"], ref,
                         f"REFC {tag}: HRRR analysis vs MRMS (mean aggregation)")
            print(f"  plot -> {dest.with_suffix('.png')}")


if __name__ == "__main__":
    main()