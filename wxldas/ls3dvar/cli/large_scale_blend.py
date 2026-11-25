#!/usr/bin/env python3
# ls3dvar/cli/large_scale_blend.py  (extracted from mantari/ls3dvar/large_scale_blend.py)
"""
ls3dvar/cli/large_scale_blend.py

Large-scale blending of the cycled HRRRCast background toward GFS (RRFS-style),
to stop upper-air drift when only surface observations are assimilated.

    D_k   = LowPass_lambda( gfs_phys(p_k) - bg_phys(p_k) )      at the GFS levels p_k
    D(p)  = linear-in-log(p) interpolation of D_k to every HRRR level p <= pmax
    x_b  += w * D(p) / std_hrrr(var, p)                         (normalized state)

  - GFS = the forcing npz HRRRCast already uses (HRRR grid, GFS-normalized, levels
    250/500/850/1000). Both states go to physical units with their own statistics
    (normalize-stats.nc: VAR vs GFS-VAR), bg is interpolated in log p to 250 hPa.
  - LowPass = Gaussian, half-power wavelength --cutoff-km (default 1000 km).
  - Levels p <= --pmax (850) only; boundary layer and surface untouched.
  - GFS time index: automatic (closest to the background) unless --gfs-index.
  - Safety: per (var, GFS level) mean difference and rms difference in physical units vs
    the climatological std (normalize-stats); abort if |mean diff| > 0.5 std or
    rms diff > 1 std (normalization / time mix-up) unless --force.

    python -m ls3dvar.cli.large_scale_blend --hrrr background.npz --gfs gfs_20260811_02.npz \
        --out background_blend.npz [--truth hrrr_ana.npz] [--check-only]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from ls3dvar.paths import REPO_ROOT as PROJECT_ROOT  # noqa: E402  (repo root, as before)

from ls3dvar.hrrrcast.channels import ChannelStats, open_stats  # noqa: E402

DEFAULT_VARS = ["TMP", "UGRD", "VGRD", "HGT"]
DEFAULT_STATS = PROJECT_ROOT / "net-diffusion" / "normalize-stats.nc"
UNIT = {"TMP": "K", "UGRD": "m/s", "VGRD": "m/s", "HGT": "m", "SPFH": "kg/kg", "VVEL": "Pa/s"}


def _strip(n):
    for pre in ("GFS-", "GFS_", "gfs-", "gfs_", "gfs."):
        if n.startswith(pre):
            return n[len(pre):]
    return n


def names_of(npz, label, nchan=None):
    """channel names of a hrrr/gfs npz (GFS-/gfs_ prefixes stripped)"""
    names = None
    if all(k in npz.files for k in ("pl_vars", "levels")):
        pl = [_strip(str(v)) for v in npz["pl_vars"]]
        lv = [str(int(float(l))) for l in np.ravel(npz["levels"])]
        sfc = [_strip(str(v)) for v in npz["sfc_vars"]] if "sfc_vars" in npz.files else []
        names = [f"{v}_{l}" for v in pl for l in lv] + sfc
    else:
        for k in ("channel_names", "variables", "var_names", "channels", "names"):
            if k in npz.files:
                names = [_strip(str(v)) for v in np.ravel(npz[k])]
                break
    if names is None:
        raise SystemExit(f"{label}: cannot find channel names. keys: {npz.files}")
    names = [_strip(n) for n in names]
    if nchan is not None and len(names) != nchan:
        print(f"WARNING {label}: {len(names)} names for {nchan} channels")
    return names


def levels_of(names, var):
    out = []
    for n in names:
        if n.startswith(var + "_"):
            try:
                out.append(float(n.rsplit("_", 1)[1]))
            except ValueError:
                pass
    return sorted(out)


def rms(a):
    return float(np.sqrt(np.nanmean(np.asarray(a, np.float64) ** 2)))


def logp_weights(p, plist):
    """indices/weights for linear interpolation in log p (clamped at the ends)"""
    lp, lpl = np.log(p), np.log(np.asarray(plist, float))
    if lp <= lpl[0]:
        return 0, 0, 1.0
    if lp >= lpl[-1]:
        n = len(plist) - 1
        return n, n, 1.0
    j = int(np.searchsorted(lpl, lp)) - 1
    w = (lpl[j + 1] - lp) / (lpl[j + 1] - lpl[j])
    return j, j + 1, float(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrrr", required=True, help="background npz (hrrr format, normalized)")
    ap.add_argument("--gfs", required=True, help="gfs npz on the HRRR grid (HRRRCast forcing)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", default=str(DEFAULT_STATS), help="net-diffusion/normalize-stats.nc")
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    ap.add_argument("--pmax", type=float, default=850.0,
                    help="blend HRRR levels with p <= pmax hPa (boundary layer stays HRRRCast)")
    ap.add_argument("--cutoff-km", type=float, default=1000.0)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--gfs-index", default="auto")
    ap.add_argument("--truth", default=None, help="reference npz: rms before/after (diagnostic only)")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    st = ChannelStats({"normalize_stats_nc": a.stats})
    if st._dataset() is False:
        raise SystemExit(f"cannot use stats file {a.stats}")

    h = np.load(a.hrrr, allow_pickle=True)
    g = np.load(a.gfs, allow_pickle=True)
    gx = g["model_input"]
    hn, gn = names_of(h, "hrrr"), names_of(g, "gfs", gx.shape[-1])
    xb_all = np.asarray(h["model_input"][0], np.float64)
    if gx.shape[1:3] != xb_all.shape[:2]:
        raise SystemExit(f"grid mismatch: gfs {gx.shape} vs hrrr {h['model_input'].shape}")
    print(f"background {a.hrrr}  {xb_all.shape}\ngfs        {a.gfs}  {gx.shape}")
    print(f"gfs npz keys : {g.files}")
    print(f"gfs channels : {gn}")

    def h_phys(var, p, x=xb_all):
        """HRRR field in physical units at p (log-p interpolated if p is not a model level)"""
        hl = levels_of(hn, var)
        j0, j1, w = logp_weights(p, hl)
        out = 0.0
        for j, ww in ((j0, w), (j1, 1.0 - w)):
            if ww == 0.0:
                continue
            nm = f"{var}_{int(hl[j])}"
            mu, sd = st.get(nm)
            out = out + ww * (x[..., hn.index(nm)] * sd + mu)
        return out

    def g_phys(var, p, t):
        nm = f"{var}_{int(p)}"
        mu, sd = st.get(nm, gfs=True)
        return np.asarray(gx[t][..., gn.index(nm)], np.float64) * sd + mu

    # GFS levels usable for each var: within the HRRR column and up to the blend top/bottom
    plan = {}
    for v in a.vars:
        hl, gl = levels_of(hn, v), levels_of(gn, v)
        if not hl or not gl:
            print(f"  skip {v}: hrrr levels {hl}, gfs levels {gl}")
            continue
        targets = [p for p in hl if p <= a.pmax]
        glev = [p for p in gl if hl[0] <= p <= hl[-1] and p <= max(a.pmax, min(gl))]
        if targets and glev:
            plan[v] = (glev, targets)
    if not plan:
        raise SystemExit("nothing to blend (check --vars / level names)")
    for v, (glev, targ) in plan.items():
        print(f"  {v:5s} GFS levels {[int(p) for p in glev]} -> HRRR levels {[int(p) for p in targ]}")

    # ---- GFS time index -----------------------------------------------------
    def score(t):
        return np.sqrt(np.mean([np.mean(((g_phys(v, p, t) - h_phys(v, p)) / st.get(f"{v}_{int(p)}", gfs=True)[1]) ** 2)
                                for v, (glev, _) in plan.items() for p in glev]))
    if a.gfs_index == "auto":
        sc = [score(t) for t in range(gx.shape[0])]
        t_g = int(np.argmin(sc))
        print("gfs index -> normalized rms(gfs - bg): "
              + "  ".join(f"[{t}] {s:.3f}" for t, s in enumerate(sc)) + f"   -> {t_g}")
    else:
        t_g = int(a.gfs_index)

    # ---- consistency (physical units) --------------------------------------------
    print(f"\n{'var@p':10s} {'unit':5s} {'mean(g-b)':>10s} {'std g/b':>8s} {'corr':>6s} {'rms(g-b)':>9s} {'clim std':>9s}")
    bad, diff = [], {}
    for v, (glev, _) in plan.items():
        for p in glev:
            b_, g_ = h_phys(v, p), g_phys(v, p, t_g)
            md = float(np.mean(g_ - b_))
            sr = float(np.std(g_) / max(np.std(b_), 1e-12))
            cr = float(np.corrcoef(g_.ravel(), b_.ravel())[0, 1])
            sd_ref = st.get(f"{v}_{int(p)}", gfs=True)[1]
            print(f"{v + '@' + str(int(p)):10s} {UNIT.get(v, ''):5s} {md:+10.3f} {sr:8.3f} {cr:6.3f} "
                  f"{rms(g_ - b_):9.3f} {sd_ref:9.3f}")
            # normalization / time mix-ups give differences of order the climatological std;
            # std ratio and correlation are informational only (noisy on a small domain)
            if abs(md) > 0.5 * sd_ref or rms(g_ - b_) > 1.0 * sd_ref:
                bad.append(f"{v}@{int(p)}")
            diff[(v, p)] = g_ - b_
    if bad and not a.force:
        raise SystemExit(f"\nABORT: {bad} inconsistent (normalization / time index?). --force to override.")

    # ---- low-pass + vertical interpolation ----------------------------------------
    lat, lon = np.deg2rad(h["lats"]), np.deg2rad(((np.asarray(h["lons"]) + 180) % 360) - 180)
    dx_km = float(np.median(6371.0 * np.hypot(np.diff(lat, axis=1)[:-1, :],
                                              np.cos(lat[:-1, :-1]) * np.diff(lon, axis=1)[:-1, :])))
    sig = 0.1874 * a.cutoff_km / dx_km           # Gaussian response 0.5 at the cutoff wavelength
    ny, nx = xb_all.shape[:2]
    print(f"\ngrid {dx_km:.2f} km  ({ny}x{nx} = {ny * dx_km:.0f}x{nx * dx_km:.0f} km)  cutoff {a.cutoff_km:g} km"
          f" -> gaussian sigma {sig:.1f} cells   weight {a.weight:g}   gfs index {t_g}")
    if sig > 0.5 * min(ny, nx):
        print("  NOTE: cutoff larger than the domain -> the blend is essentially the domain-mean "
              "(plus gradient) difference")
    D = {k: gaussian_filter(d, sig, mode="nearest") for k, d in diff.items()}

    xa_all = xb_all.copy()
    truth = None
    if a.truth:
        truth = np.asarray(np.load(a.truth, allow_pickle=True)["model_input"][0], np.float64)
    print(f"\n{'channel':10s} {'unit':5s} {'inc rms':>8s}"
          + (f" {'bg-truth':>9s} {'blend-truth':>11s} {'impr%':>7s}" if truth is not None else ""))
    for v, (glev, targ) in plan.items():
        for p in targ:
            j0, j1, w = logp_weights(p, glev)
            dp = w * D[(v, glev[j0])] + (1.0 - w) * D[(v, glev[j1])]
            nm = f"{v}_{int(p)}"
            mu, sd = st.get(nm)
            ch = hn.index(nm)
            xa_all[..., ch] += a.weight * dp / sd
            line = f"{nm:10s} {UNIT.get(v, ''):5s} {rms(a.weight * dp):8.3f}"
            if truth is not None:
                eb = rms((xb_all[..., ch] - truth[..., ch]) * sd)
                ea = rms((xa_all[..., ch] - truth[..., ch]) * sd)
                line += f" {eb:9.3f} {ea:11.3f} {100 * (eb - ea) / max(eb, 1e-12):+7.2f}"
            print(line)

    if a.check_only or not a.out:
        print("\n(check only, nothing written)")
        return 0
    out = {k: h[k] for k in h.files}
    mi = out["model_input"].copy()
    mi[0] = xa_all.astype(mi.dtype)
    out["model_input"] = mi
    out["blend_info"] = np.array(f"gfs={a.gfs} idx={t_g} cutoff_km={a.cutoff_km} w={a.weight} "
                                 f"vars={list(plan)} pmax={a.pmax}")
    np.savez(a.out, **out)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
