# ls3dvar/obs/qc.py  (extracted from mantari/ls3dvar/obs_qc.py)
"""
mantari/ls3dvar/obs_qc.py

Tighter observation QC for LS-3DVar, applied after the background check
(ObservationFilter). All tests use the O-B innovations d = H(x_b) - y.

  1. blacklist     station ids, or whole networks per variable
  2. network check per-network background-check threshold (|d - median_net| > k_net * sigma)
  3. buddy check   spatial consistency: compare d_i with the median innovation of
                   same-variable neighbours within radius_km; reject if
                       |d_i - median_nb| > k * max(sigma_i, 1.4826 * MAD_nb)
                   worst offender removed first, then re-evaluated (a bad station
                   cannot get its good neighbours rejected). Isolated obs are kept.
  4. error scale   per-network (optionally per-variable) multiplier on sigma_o

config (all optional; absent block = off):

ls3dvar:
  obs_qc:
    enabled: true
    blacklist:
      stations: []                       # stationIdentification strings
      networks: {}                       # e.g. {dewPointTemperatureAt2M: [tempest]}
    network_check: {k: 3.0}
    buddy: {radius_km: 50.0, min_buddies: 2, k: 2.5, max_reject_fraction: 0.2}
    error_scale:                         # network -> scale  or  network -> {variable: scale}
      asos: 1.0
      tempest: 1.5
      grenade: 1.5
"""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0


def _hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class ObsQC:

    def __init__(self, cfg):
        q = (cfg.get("ls3dvar", {}) or {}).get("obs_qc", {}) or {}
        self.enabled = bool(q.get("enabled", False))
        bl = q.get("blacklist", {}) or {}
        self.bl_stations = {str(s) for s in (bl.get("stations") or [])}
        self.bl_networks = {k: set(v or []) for k, v in (bl.get("networks") or {}).items()}
        self.net_k = (q.get("network_check") or {}).get("k")
        b = q.get("buddy") or {}
        self.buddy = dict(radius_km=float(b.get("radius_km", 50.0)),
                          min_buddies=int(b.get("min_buddies", 2)),
                          k=float(b.get("k", 2.5)),
                          max_frac=float(b.get("max_reject_fraction", 0.2))) if b else None
        self.err_scale = q.get("error_scale") or {}
        self.rejected = []                       # rows for the station report

    # ------------------------------------------------------------------
    def scale(self, var, source):
        v = self.err_scale.get(source, 1.0)
        if isinstance(v, dict):
            v = v.get(var, 1.0)
        return float(v)

    def sigmas(self, var, obs_list, sigma):
        return np.array([sigma * self.scale(var, o.source) for o in obs_list], np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def network_summary(var, obs_list, innov):
        src = np.array([o.source for o in obs_list])
        print(f"  {var} O-B by network:  " + "   ".join(
            f"{s}: n={int((src == s).sum())} mean={innov[src == s].mean():+.2f} "
            f"rms={np.sqrt(np.mean(innov[src == s] ** 2)):.2f}"
            for s in sorted(set(src))))

    # ------------------------------------------------------------------
    def apply(self, var, obs_list, innov, sigma):
        """returns kept obs list; prints and records rejections"""
        n = len(obs_list)
        if not self.enabled or n == 0:
            return obs_list
        innov = np.asarray(innov, float)
        sig = self.sigmas(var, obs_list, sigma).astype(float)
        keep = np.ones(n, bool)
        reason = [""] * n
        detail = [np.nan] * n

        # 1. blacklist
        bad_net = self.bl_networks.get(var, set()) | self.bl_networks.get("all", set())
        for i, o in enumerate(obs_list):
            if str(o.station) in self.bl_stations or o.source in bad_net:
                keep[i], reason[i] = False, "blacklist"

        # 2. per-network background check (relative to the network median: removes
        #    gross station errors without penalising a network-wide bias)
        if self.net_k:
            src = np.array([o.source for o in obs_list])
            for s in set(src):
                m = (src == s) & keep
                if m.sum() < 3:
                    continue
                med = np.median(innov[m])
                bad = m & (np.abs(innov - med) > self.net_k * sig)
                for i in np.where(bad)[0]:
                    keep[i], reason[i], detail[i] = False, f"network>{self.net_k:g}sig", innov[i] - med

        # 3. buddy check, worst first
        if self.buddy:
            lat = np.array([o.lat for o in obs_list])
            lon = np.array([o.lon for o in obs_list])
            D = _hav_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
            np.fill_diagonal(D, np.inf)
            near = D <= self.buddy["radius_km"]
            max_rej = int(np.floor(self.buddy["max_frac"] * keep.sum()))
            nrej = 0
            while nrej < max_rej:
                score = np.zeros(n)
                for i in np.where(keep)[0]:
                    nb = near[i] & keep
                    if nb.sum() < self.buddy["min_buddies"]:
                        continue
                    dn = innov[nb]
                    med = np.median(dn)
                    spread = 1.4826 * np.median(np.abs(dn - med))
                    tol = self.buddy["k"] * max(sig[i], spread)
                    score[i] = abs(innov[i] - med) / tol
                    detail[i] = innov[i] - med
                worst = int(np.argmax(score))
                if score[worst] <= 1.0:
                    break
                keep[worst], reason[worst] = False, f"buddy {score[worst]:.2f}xtol"
                nrej += 1

        for i in np.where(~keep)[0]:
            o = obs_list[i]
            self.rejected.append(dict(variable=var, station=o.station, network=o.source,
                                      lat=o.lat, lon=o.lon, omb=-innov[i],
                                      vs_buddies=-detail[i] if np.isfinite(detail[i]) else np.nan,
                                      reason=reason[i]))
        if (~keep).any():
            print(f"  [obs_qc] {var}: {n} -> {int(keep.sum())}")
            for r in self.rejected[-int((~keep).sum()):]:
                print(f"     reject {r['network']:8s} {r['station']:>14s} ({r['lat']:7.3f},{r['lon']:8.3f})"
                      f"  O-B={r['omb']:+7.2f}  vs buddies={r['vs_buddies']:+7.2f}  {r['reason']}")
        else:
            print(f"  [obs_qc] {var}: {n} -> {n}")
        return [o for o, k in zip(obs_list, keep) if k]
