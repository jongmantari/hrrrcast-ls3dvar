# ls3dvar/obs/holdout.py  (extracted from mantari/ls3dvar/station_split.py)
"""
mantari/ls3dvar/station_split.py

Stable station-based hold-out: a station belongs to fold crc32(station_id) % K,
independent of which other stations report, the cycle, or the grid. Fold k of K
is never assimilated in any cycle and is used for independent verification.

    spec = parse_spec("0/5")          # (0, 5)
    is_heldout("KLAX", spec)          # True / False
"""
import zlib


def parse_spec(s):
    if not s:
        return None
    k, K = (int(v) for v in str(s).split("/"))
    if not 0 <= k < K:
        raise ValueError(f"bad hold-out spec {s!r}: need 0 <= k < K")
    return k, K


def station_fold(station, K):
    return zlib.crc32(str(station).strip().encode()) % K


def is_heldout(station, spec):
    return spec is not None and station_fold(station, spec[1]) == spec[0]
