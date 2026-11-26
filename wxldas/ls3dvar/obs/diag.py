# ls3dvar/obs/diag.py
"""
Per-observation diagnostics for one LS-3DVar analysis: O-B and O-A for every observation
that reached the analysis (after the time window), including held-out and rejected ones.

Files (JEDI-style group names, one Location per station-variable report):

  obs_diag_<cycle>.nc / .csv
    MetaData/   stationIdentification provider variableName variableKey latitude longitude
                stationElevation dateTime (s since 1970-01-01T00:00:00Z) gridRow gridColumn qcReason
    ObsValue/value        observation y (physical units; station pressure at station height)
    ObsError/value        assigned error (observation_errors)
    EffectiveError/value  error used (after obs_qc network error_scale)
    hofx0/value hofx1/value   H(x_b), H(x_a)  (same operator as the analysis, height corrections incl.)
    ombg/value oman/value     y - H(x_b), y - H(x_a)
    EffectiveQC/value     0 assimilated, 1 held out, 2 blacklist, 3 network check, 4 buddy check,
                          5 innovation (background) check

  refc_diag_<cycle>.nc   MRMS REFC pseudo-obs (superob tiles)
    Location: tileRow tileColumn latitude longitude ObsValue ObsError hofx0 hofx1 ombg oman
    grid (tile_y, tile_x): obs hofx0 hofx1 ombg oman, NaN where no MRMS coverage
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

QC_ASSIMILATED, QC_HELDOUT, QC_BLACKLIST, QC_NETWORK, QC_BUDDY, QC_GROSS = range(6)
QC_MEANING = "0 assimilated, 1 held out, 2 blacklist, 3 network check, 4 buddy check, 5 innovation check"


def _qc_code(reason):
    r = str(reason)
    if r.startswith("blacklist"):
        return QC_BLACKLIST
    if r.startswith("network"):
        return QC_NETWORK
    if r.startswith("buddy"):
        return QC_BUDDY
    return QC_GROSS


def _epoch(t):
    if t is None:
        return np.iinfo(np.int64).min + 7             # IODA int64 missing (-9223372036854775801)
    return int(np.datetime64(t, "s").astype("int64"))


class ObsDiag:
    """collects each variable's full report list once, then the rejections; writes after the solve"""

    def __init__(self):
        self.vars = {}                                # key -> dict

    # ------------------------------------------------------------------
    def add(self, key, var, obs_list, iy, ix, op, err, sigma):
        """full list (after the time window, before any QC) and the operator built for it"""
        n = len(obs_list)
        s = np.asarray(sigma, np.float32)
        self.vars[key] = dict(var=var, obs=list(obs_list), iy=tf.identity(iy), ix=tf.identity(ix), op=op,
                              err=float(err), sigma=np.full(n, s, np.float32) if s.ndim == 0 else s.copy(),
                              qc=np.full(n, -1, np.int32), reason=[""] * n)

    def _mark(self, key, before, kept, reasons):
        v = self.vars.get(key)
        if v is None:
            return
        kept_ids = {id(o) for o in kept}
        pos = {id(o): i for i, o in enumerate(v["obs"])}
        for o in before:
            if id(o) in kept_ids or id(o) not in pos:
                continue
            i = pos[id(o)]
            if v["qc"][i] < 0:
                r = reasons(o)
                v["qc"][i], v["reason"][i] = _qc_code(r), r

    def reject_gross(self, key, before, kept, threshold=None):
        r = f"innovation |O-B|>{threshold:g}" if threshold is not None else "innovation check"
        self._mark(key, before, kept, lambda o: r)

    def reject_qc(self, key, before, kept, rejected_rows):
        """rejected_rows: ObsQC.rejected entries of this call (station, lat, lon, reason)"""
        by = {(str(r["station"]), round(float(r["lat"]), 5), round(float(r["lon"]), 5)): r["reason"]
              for r in rejected_rows}
        self._mark(key, before, kept,
                   lambda o: by.get((str(o.station), round(float(o.lat), 5), round(float(o.lon), 5)), "obs_qc"))

    # ------------------------------------------------------------------
    def table(self, x_b, x_a, held_flag):
        xb = tf.convert_to_tensor(x_b, tf.float32)
        xa = tf.convert_to_tensor(x_a, tf.float32)
        frames = []
        for key, v in self.vars.items():
            n = len(v["obs"])
            if n == 0:
                continue
            hb = v["op"](xb, v["iy"], v["ix"]).numpy().astype(np.float64)
            ha = v["op"](xa, v["iy"], v["ix"]).numpy().astype(np.float64)
            iy, ix = v["iy"].numpy(), v["ix"].numpy()
            y = np.array([o.value for o in v["obs"]], np.float64)
            qc = v["qc"].copy()
            for i, o in enumerate(v["obs"]):
                if qc[i] < 0:
                    qc[i] = QC_HELDOUT if held_flag(o, int(iy[i]), int(ix[i])) else QC_ASSIMILATED
            reason = [r or ("held out" if q == QC_HELDOUT else "") for r, q in zip(v["reason"], qc)]
            frames.append(pd.DataFrame(dict(
                variableKey=key, variableName=v["var"],
                stationIdentification=[str(o.station) for o in v["obs"]],
                provider=[str(o.source) for o in v["obs"]],
                latitude=[float(o.lat) for o in v["obs"]], longitude=[float(o.lon) for o in v["obs"]],
                stationElevation=[np.nan if o.elevation is None else float(o.elevation) for o in v["obs"]],
                dateTime=[_epoch(o.time) for o in v["obs"]],
                gridRow=iy.astype(np.int32), gridColumn=ix.astype(np.int32),
                ObsValue=y, ObsError=v["err"], EffectiveError=v["sigma"].astype(np.float64),
                hofx0=hb, hofx1=ha, ombg=y - hb, oman=y - ha,
                EffectiveQC=qc, qcReason=reason)))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------
    def write(self, path_nc, x_b, x_a, held_flag, attrs=None, csv=True):
        import netCDF4 as nc
        df = self.table(x_b, x_a, held_flag)
        path_nc = Path(path_nc)
        path_nc.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(path_nc, "w") as ds:
            ds.createDimension("Location", len(df))
            loc = ds.createVariable("Location", "i8", ("Location",))
            loc[:] = np.arange(len(df), dtype=np.int64)
            for k, v in (attrs or {}).items():
                ds.setncattr(k, str(v))
            ds.setncattr("EffectiveQC_meaning", QC_MEANING)
            meta = ds.createGroup("MetaData")
            for col in ("stationIdentification", "provider", "variableName", "variableKey", "qcReason"):
                var = meta.createVariable(col, str, ("Location",))
                var[:] = np.array(df[col].astype(str) if len(df) else [], dtype=object)
            for col, typ, fill, units in (("latitude", "f4", -3.3687953e38, "degrees_north"),
                                          ("longitude", "f4", -3.3687953e38, "degrees_east"),
                                          ("stationElevation", "f4", -3.3687953e38, "m"),
                                          ("dateTime", "i8", -9223372036854775801,
                                           "seconds since 1970-01-01T00:00:00Z"),
                                          ("gridRow", "i4", -2147483643, "1"),
                                          ("gridColumn", "i4", -2147483643, "1")):
                var = meta.createVariable(col, typ, ("Location",), fill_value=fill)
                var.units = units
                a = df[col].to_numpy() if len(df) else np.array([])
                var[:] = np.where(np.isnan(a), fill, a) if typ == "f4" else a
            for grp in ("ObsValue", "ObsError", "EffectiveError", "hofx0", "hofx1", "ombg", "oman"):
                g = ds.createGroup(grp)
                var = g.createVariable("value", "f4", ("Location",), fill_value=-3.3687953e38)
                var.long_name = f"{grp} (units of MetaData/variableName)"
                var[:] = df[grp].to_numpy(np.float32) if len(df) else []
            g = ds.createGroup("EffectiveQC")
            var = g.createVariable("value", "i4", ("Location",))
            var.flag_meanings = QC_MEANING
            var[:] = df["EffectiveQC"].to_numpy(np.int32) if len(df) else []
        if csv:
            df.to_csv(path_nc.with_suffix(".csv"), index=False)
        return df


def write_refc_diag(path_nc, pseudo, x_b, x_a, box, lats, lons, grid_shape, attrs=None):
    """pseudo: run_analysis pseudo-obs groups {name: dict(y, s, iy, ix, op)} (superob tiles)"""
    import netCDF4 as nc
    if not pseudo:
        return None
    xb = tf.convert_to_tensor(x_b, tf.float32)
    xa = tf.convert_to_tensor(x_a, tf.float32)
    H, W = grid_shape
    ny, nx = H // int(box), W // int(box)
    path_nc = Path(path_nc)
    path_nc.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with nc.Dataset(path_nc, "w") as ds:
        for k, v in (attrs or {}).items():
            ds.setncattr(k, str(v))
        ds.setncattr("superob_box_cells", int(box))
        ds.createDimension("tile_y", ny)
        ds.createDimension("tile_x", nx)
        for name, d in pseudo.items():
            ty, tx = d["iy"].numpy().astype(np.int64), d["ix"].numpy().astype(np.int64)
            y = d["y"].numpy().astype(np.float64)
            hb = d["op"](xb, d["iy"], d["ix"]).numpy().astype(np.float64)
            ha = d["op"](xa, d["iy"], d["ix"]).numpy().astype(np.float64)
            cy = np.clip(ty * box + box // 2, 0, H - 1)
            cx = np.clip(tx * box + box // 2, 0, W - 1)
            g = ds.createGroup(name)
            dim = f"Location_{name}"
            g.createDimension(dim, len(y))
            cols = dict(tileRow=ty.astype(np.int32), tileColumn=tx.astype(np.int32),
                        latitude=np.asarray(lats)[cy, cx].astype(np.float32),
                        longitude=np.asarray(lons)[cy, cx].astype(np.float32),
                        ObsValue=y, ObsError=d["s"].numpy(), hofx0=hb, hofx1=ha, ombg=y - hb, oman=y - ha)
            for c, a in cols.items():
                typ = "i4" if a.dtype.kind == "i" else "f4"
                var = g.createVariable(c, typ, (dim,))
                var[:] = a.astype(typ)
            for c in ("ObsValue", "hofx0", "hofx1", "ombg", "oman"):
                grid = np.full((ny, nx), np.nan, np.float32)
                grid[ty, tx] = cols[c]
                var = g.createVariable(f"{c}_grid", "f4", ("tile_y", "tile_x"), fill_value=np.float32(np.nan))
                var[:] = grid
            summary[name] = dict(n=len(y), omb_rms=float(np.sqrt(np.mean((y - hb) ** 2))) if len(y) else np.nan,
                                 oma_rms=float(np.sqrt(np.mean((y - ha) ** 2))) if len(y) else np.nan)
    return summary
