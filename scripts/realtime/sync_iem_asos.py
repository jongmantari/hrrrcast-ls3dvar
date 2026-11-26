#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.32",
#   "urllib3>=2.2",
#   "boto3>=1.34",
# ]
# ///
"""
sync_iem_asos.py - Mirror IEM ASOS hourly observations to a local directory.

Remote source: Iowa Environmental Mesonet (IEM)
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

IEM stores QC'd hourly METAR-based observations, distinct from the raw 5-minute
NCEI data in asos-5min/. Variables are in US customary units (°F, inHg, knots).

Local mirror structure:
  {LOCAL_ROOT}/YYYY/MM/iem-asos-{STATION}-{YYYYMM}.csv  (~5-50 KB each)

LOCAL_ROOT defaults to ~/Data/datasets/iem-asos regardless of
where this script is invoked from or where a copy of it lives, because that is
the one location the downstream ETL and IODA converters read. Use --local-root
for a scratch mirror.

Download strategy:
- Historic months (> 2 months ago): download once, skip on re-run.
- Recent months (last 2): always refresh; IEM QC runs on a rolling basis.
- Daily cron: run with no flags; re-fetches only current and previous month.

Bulk backfill (--full / --year / --start-year):
- IEM rate-limits by request count, not bytes, and its CGI takes an arbitrary
  date range, so one request per station-month is the wrong unit. Backfills
  request each station's missing range in one call and split the response into
  the same per-month files, measured 12x fewer requests over a year and ~80x
  over 2020-present. --no-bulk restores the old one-request-per-month behaviour.
- Only missing months are requested, coalesced into contiguous runs, capped at
  MAX_BULK_MONTHS per request.
- Months IEM confirms have no data are recorded under .empty-months/{STATION}.json
  so they are not re-requested on every future run. Without that memo a station
  that opened after the start year re-costs its empty months forever, which is
  why the backfill never used to converge.

Known-issue history:
- Before 2026-08-05 the per-month path sent day2=last-day-of-month. IEM's range
  end is EXCLUSIVE at 00:00, so every file written before then is missing its
  final calendar day (~3% of observations). Both paths now request through the
  first of the following month. Re-fetch affected files with --force, and clear
  the downstream ETL manifests so the corrected months are reprocessed.

S3 push:
- Pass --s3-bucket (and optionally --s3-prefix) to upload each file to S3
  immediately after it's downloaded, instead of relying on a separate
  bulk `aws s3 sync` afterwards. This matters because this script commonly
  runs on ephemeral instances (see aws_run_iem_pipeline.sh): if the instance
  dies mid-run, anything already pushed is safe in S3 even though nothing
  synced back yet via a trailing bulk sync. A failed individual upload is
  logged and counted, not fatal — recent months get retried automatically
  next run since they're always re-fetched regardless of local state.

Concurrency note:
  With bulk backfill, 2020-present for all ~2900 US ASOS stations is about
  2,900 requests (one per station), measured ~12 s each for an 80-month span,
  so roughly 5 hours at --workers 2. Keep workers low: the limit is IEM's, not
  the machine's.

Usage:
  # Daily update (last 2 months, same as scheduled job):
  uv run sync_iem_asos.py

  # Initial sync from 2020 to present (~100k requests):
  uv run sync_iem_asos.py --start-year 2020

  # Full historical mirror from 1928 (~3M requests, multi-day):
  uv run sync_iem_asos.py --full

  # Single year:
  uv run sync_iem_asos.py --year 2024

  # Single IEM network only:
  uv run sync_iem_asos.py --network FL_ASOS

  # Scratch mirror somewhere else (downstream scripts will not see it):
  uv run sync_iem_asos.py --network FL_ASOS --local-root /tmp/iem-test

  # Push each file to S3 as it's downloaded (recommended on ephemeral hosts):
  uv run sync_iem_asos.py --s3-bucket mantari-observations --s3-prefix iem/csv

  # One-time repair of files missing their last day (see Known-issue history):
  uv run sync_iem_asos.py --start-year 2020 --force \
      --s3-bucket mantari-observations --s3-prefix iem/csv

  # Options:
  #   --local-root DIR  mirror into DIR instead of the canonical location
  #   --months N      re-sync last N months (default: 2)
  #   --workers N     concurrent threads (default: 2)
  #   --force         re-download even if local file exists
  #   --no-bulk       backfill one request per station-month (slow; comparison)
  #   --s3-bucket NAME   upload each downloaded file to this bucket immediately
  #   --s3-prefix PREFIX key prefix under the bucket (default: iem/csv)
  #   --verbose       debug-level logging
"""

from __future__ import annotations

import os
import argparse
import calendar
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Iterator

import time

import boto3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Configuration ─────────────────────────────────────────────────────────────

# The mirror location is fixed rather than derived from __file__, so that a copy
# of this script placed elsewhere still writes to the one canonical mirror.
# Override with --local-root. Matches obs-etl/common.py RAW_ROOT / "iem-asos".
DEFAULT_LOCAL_ROOT = Path(os.environ.get("IEM_ASOS_ROOT", str(Path.home() / "Data/datasets/iem-asos")))

# Set from DEFAULT_LOCAL_ROOT or --local-root in main(); the download helpers
# read them as globals.
LOCAL_ROOT = DEFAULT_LOCAL_ROOT
LOG_FILE   = LOCAL_ROOT / "sync.log"
FIRST_YEAR = 1928                        # IEM ASOS archive goes back to ~1928

# Set from --s3-bucket / --s3-prefix in main(); None disables the per-file
# push. Read as globals by download_station_month, same pattern as LOCAL_ROOT.
S3_BUCKET: str | None = None
S3_PREFIX = "iem/csv"

WORKERS      = 2      # IEM rate-limits aggressively; keep concurrency low
REQUEST_DELAY = 0.5   # seconds between requests per worker thread
TIMEOUT      = 120    # seconds; monthly pulls can be slow for active stations
MAX_RETRY    = 4
CHUNK_SIZE   = 1 << 17   # 128 KB

DATA_API     = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
STATIONS_API = "https://mesonet.agron.iastate.edu/sites/networks.php"

# All US ASOS state and territory network codes
US_ASOS_NETWORKS = [
    "AL_ASOS", "AK_ASOS", "AZ_ASOS", "AR_ASOS", "CA_ASOS",
    "CO_ASOS", "CT_ASOS", "DE_ASOS", "FL_ASOS", "GA_ASOS",
    "HI_ASOS", "ID_ASOS", "IL_ASOS", "IN_ASOS", "IA_ASOS",
    "KS_ASOS", "KY_ASOS", "LA_ASOS", "ME_ASOS", "MD_ASOS",
    "MA_ASOS", "MI_ASOS", "MN_ASOS", "MS_ASOS", "MO_ASOS",
    "MT_ASOS", "NE_ASOS", "NV_ASOS", "NH_ASOS", "NJ_ASOS",
    "NM_ASOS", "NY_ASOS", "NC_ASOS", "ND_ASOS", "OH_ASOS",
    "OK_ASOS", "OR_ASOS", "PA_ASOS", "RI_ASOS", "SC_ASOS",
    "SD_ASOS", "TN_ASOS", "TX_ASOS", "UT_ASOS", "VT_ASOS",
    "VA_ASOS", "WA_ASOS", "WV_ASOS", "WI_ASOS", "WY_ASOS",
    "DC_ASOS", "PR_ASOS", "VI_ASOS", "GU_ASOS", "AS_ASOS",
]

# Variables to request from the IEM API (all useful met variables)
IEM_VARS = ",".join([
    "tmpf",               # temperature (°F)
    "dwpf",               # dewpoint (°F)
    "relh",               # relative humidity (%)
    "drct",               # wind direction (deg)
    "sknt",               # wind speed (knots)
    "gust",               # wind gust (knots)
    "p01i",               # 1-hr precipitation (inches)
    "alti",               # altimeter (inHg)
    "mslp",               # sea-level pressure (mb)
    "vsby",               # visibility (miles)
    "feel",               # heat index / wind chill (°F)
    "skyl1", "skyl2", "skyl3",  # sky layer heights AGL (feet)
    "skyc1", "skyc2", "skyc3",  # sky cover codes (CLR/FEW/SCT/BKN/OVC)
    "wxcodes",            # present weather codes
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
    "peak_wind_gust",     # peak gust during obs period (knots)
    "peak_wind_drct",     # direction of peak gust (deg)
    "snowdepth",          # snow depth (inches)
])

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s %(levelname)-8s %(message)s"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

# ── HTTP session ──────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRY,
        backoff_factor=4,       # waits 4, 8, 16, 32 s on successive 429s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers["User-Agent"] = "iem-mirror/1.0 (stelios@mantari.com)"
    return session


_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = make_session()
    return _local.session


def get_s3_client():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client("s3")
    return _local.s3


def push_to_s3(path: Path, station: str, year: int, month: int) -> None:
    """Upload one downloaded file to S3 under S3_PREFIX/YYYY/MM/. Raises on failure."""
    key = f"{S3_PREFIX}/{year:04d}/{month:02d}/{path.name}"
    get_s3_client().upload_file(str(path), S3_BUCKET, key)

# ── Station list ──────────────────────────────────────────────────────────────

def fetch_stations(network: str) -> list[str]:
    """Return list of station IDs for a given IEM network code."""
    url = f"{STATIONS_API}?network={network}&format=csv&nohtml=on"
    try:
        resp = get_session().get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not fetch stations for %s: %s", network, exc)
        return []

    lines = resp.text.strip().splitlines()
    if not lines:
        log.warning("Network %s returned empty station list", network)
        return []

    header = [h.strip().lower() for h in lines[0].split(",")]
    if "stid" not in header:
        log.warning("Network %s: unexpected CSV header: %s", network, lines[0])
        return []
    idx = header.index("stid")

    stations = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > idx:
            stid = parts[idx].strip()
            if stid:
                stations.append(stid)
    return stations


def fetch_all_stations(networks: list[str]) -> list[str]:
    """Fetch and deduplicate station IDs across all given networks."""
    seen: set[str] = set()
    for net in networks:
        stids = fetch_stations(net)
        log.info("Network %-12s  %d stations", net, len(stids))
        seen.update(stids)
    return sorted(seen)

# ── Download logic ────────────────────────────────────────────────────────────

def _months_ago(year: int, month: int) -> int:
    now = datetime.now(tz=timezone.utc)
    return (now.year * 12 + now.month) - (year * 12 + month)


def local_path(station: str, year: int, month: int) -> Path:
    return LOCAL_ROOT / str(year) / f"{month:02d}" / f"iem-asos-{station}-{year}{month:02d}.csv"


def download_station_month(
    station: str,
    year: int,
    month: int,
    force: bool,
    recent_months: int,
) -> str:
    """
    Fetch one station/month from IEM.
    Returns 'downloaded', 'skipped', 'nodata', 'error', or 'upload_error'.

    Historic files (older than recent_months) are only downloaded if absent.
    Recent files are always re-fetched because IEM QC is applied continuously.

    When S3_BUCKET is set, a freshly downloaded file is pushed to S3
    immediately. An upload failure is logged and counted separately
    ('upload_error') but does not fail the download itself — the local file
    is still written, and since recent months are always re-fetched, a failed
    upload for one of those self-heals on the next run.
    """
    path = local_path(station, year, month)
    is_recent = _months_ago(year, month) < recent_months

    if path.exists() and not force and not is_recent:
        return "skipped"

    # Delegated to the range fetcher so this path gets the exclusive-end fix
    # too: requesting day2=last-day-of-month drops that entire final day, which
    # is how every file written before 2026-08-05 came to be short one day.
    # A single-month span keeps the semantics of this function unchanged.
    written, _empty, upload_errors, status = download_station_range(
        station, (year, month), (year, month), force, recent_months
    )

    if status == "error":
        return "error"
    if written == 0:
        return "nodata"
    if upload_errors:
        return "upload_error"
    return "downloaded"

# ── Bulk range download ───────────────────────────────────────────────────────
#
# IEM rate-limits by request count, not by bytes, and the CGI accepts an
# arbitrary date range in one call. One request per station-month is therefore
# the wrong unit: a whole station-decade costs about the same as a single month
# (measured 2020-2026 for one station: 1 request, 12 s, 10.8 MB, 80 months).
# Bulk mode asks for each station's missing range in one request and splits the
# response into the same per-month files the per-month path writes, so nothing
# downstream changes.
#
# Two things make the backfill actually converge rather than restart:
#   1. Only *missing* months are requested, coalesced into contiguous runs.
#   2. Months IEM confirms it has no data for are remembered (see EMPTY_DIR).
#      Without that memo, stations that opened after the start year re-cost
#      their empty months on every single run, forever.

MAX_BULK_MONTHS = 120      # cap one request at 10 years to bound response size
MAX_RUNS_PER_STATION = 3   # above this, one span beats many small requests
EMPTY_DIR_NAME = ".empty-months"

_empty_lock = threading.Lock()


def empty_memo_path(station: str) -> Path:
    return LOCAL_ROOT / EMPTY_DIR_NAME / f"{station}.json"


def load_empty_months(station: str) -> set[tuple[int, int]]:
    """Months this station is known to have no data for, from a previous run."""
    path = empty_memo_path(station)
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable empty-month memo %s: %s", path, exc)
        return set()
    out: set[tuple[int, int]] = set()
    for ym in raw:
        try:
            out.add((int(str(ym)[:4]), int(str(ym)[4:6])))
        except ValueError:
            continue
    return out


def record_empty_months(station: str, months: set[tuple[int, int]]) -> None:
    """Merge newly confirmed empty months into the station's memo, and push it."""
    if not months:
        return
    path = empty_memo_path(station)
    with _empty_lock:
        merged = load_empty_months(station) | months
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(sorted(f"{y:04d}{m:02d}" for y, m in merged), indent=0)
        )
        tmp.replace(path)
    if S3_BUCKET:
        try:
            get_s3_client().upload_file(
                str(path), S3_BUCKET, f"{S3_PREFIX}/{EMPTY_DIR_NAME}/{path.name}"
            )
        except Exception as exc:                # noqa: BLE001 - any boto3/S3 failure
            log.warning("S3 push failed for empty-month memo %s: %s", station, exc)


def missing_months(
    station: str,
    months: list[tuple[int, int]],
    force: bool,
    recent_months: int,
) -> list[tuple[int, int]]:
    """Months to fetch in bulk for one station.

    Excludes what is already on disk, what is known-empty, and the recent
    months: those are still accumulating and are handled by the per-month
    refresh path, which always re-fetches them.
    """
    known_empty = set() if force else load_empty_months(station)
    out = []
    for year, month in months:
        if _months_ago(year, month) < recent_months:
            continue
        if (year, month) in known_empty:
            continue
        if not force and local_path(station, year, month).exists():
            continue
        out.append((year, month))
    return out


def coalesce_runs(
    months: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Group a sorted month list into contiguous (start, end) runs.

    A run longer than MAX_BULK_MONTHS is chunked. If the result is more runs
    than MAX_RUNS_PER_STATION, it collapses to a single span instead: one
    request that re-fetches a few already-present months is cheaper than many
    requests, because the rate limit counts requests.
    """
    if not months:
        return []

    runs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    start = prev = months[0]
    for cur in months[1:]:
        prev_idx = prev[0] * 12 + prev[1]
        cur_idx = cur[0] * 12 + cur[1]
        span = cur_idx - (start[0] * 12 + start[1]) + 1
        if cur_idx == prev_idx + 1 and span <= MAX_BULK_MONTHS:
            prev = cur
            continue
        runs.append((start, prev))
        start = prev = cur
    runs.append((start, prev))

    if len(runs) > MAX_RUNS_PER_STATION:
        span = (months[0][0] * 12 + months[0][1], months[-1][0] * 12 + months[-1][1])
        if span[1] - span[0] + 1 <= MAX_BULK_MONTHS:
            return [(months[0], months[-1])]
    return runs


def split_bulk_csv(content: bytes) -> tuple[list[bytes], bytes, dict[tuple[int, int], list[bytes]]]:
    """Split a multi-month IEM response into per-month row groups.

    Returns (comment_lines, header_line, {(year, month): rows}). The `valid`
    column (index 1) carries "YYYY-MM-DD HH:MM".
    """
    comments: list[bytes] = []
    header = b""
    groups: dict[tuple[int, int], list[bytes]] = {}

    for line in content.split(b"\n"):
        line = line.rstrip(b"\r")
        if not line:
            continue
        if line.startswith(b"#"):
            comments.append(line)
            continue
        if not header:
            header = line
            continue
        fields = line.split(b",", 3)
        if len(fields) < 3:
            continue
        valid = fields[1]
        if len(valid) < 7:
            continue
        try:
            key = (int(valid[0:4]), int(valid[5:7]))
        except ValueError:
            continue
        groups.setdefault(key, []).append(line)

    return comments, header, groups


def _month_preamble(comments: list[bytes], year: int, month: int) -> list[bytes]:
    """Rewrite the Time Period debug line so a split file looks like a direct pull."""
    last_day = calendar.monthrange(year, month)[1]
    period = (
        f"#DEBUG: Time Period   -> {year:04d}-{month:02d}-01 00:00:00+00:00 "
        f"{year:04d}-{month:02d}-{last_day:02d} 00:00:00+00:00"
    ).encode()
    return [period if c.startswith(b"#DEBUG: Time Period") else c for c in comments]


def next_month(ym: tuple[int, int]) -> tuple[int, int]:
    year, month = ym
    return (year + 1, 1) if month == 12 else (year, month + 1)


def download_station_range(
    station: str,
    start: tuple[int, int],
    end: tuple[int, int],
    force: bool,
    recent_months: int,
) -> tuple[int, int, int, str]:
    """Fetch one station over [start, end] in a single request and split it.

    Returns (n_written, n_empty, n_upload_errors, status) where status is
    'ok', 'nodata', or 'error'.

    The IEM range end is EXCLUSIVE at 00:00 of (year2, month2, day2), so the
    request runs to the first day of the month after `end`. Asking for
    day2=last_day-of-month silently drops that whole final day; splitting by
    each row's own month means the extra boundary rows land in the right file
    rather than being duplicated into this span's last month.
    """
    y1, m1 = start
    y2, m2 = next_month(end)

    params = {
        "station":     station,
        "data":        IEM_VARS,
        "year1":       y1, "month1": m1, "day1": 1,
        "year2":       y2, "month2": m2, "day2": 1,
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "yes",
        "elev":        "yes",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "3,4",
    }

    span = [(y, m) for y, m in iter_month_span(start, end)]

    try:
        time.sleep(REQUEST_DELAY)
        resp = get_session().get(DATA_API, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        content = resp.content
    except requests.RequestException as exc:
        log.error("FAIL %s %04d/%02d-%04d/%02d: %s", station, y1, m1, y2, m2, exc)
        return 0, 0, 0, "error"

    if b"No data found" in content or len(content) < 80:
        # Only memoize months old enough to be final; a current month that is
        # simply not populated yet must stay eligible for a later fetch.
        record_empty_months(
            station, {ym for ym in span if _months_ago(*ym) >= recent_months}
        )
        return 0, len(span), 0, "nodata"

    comments, header, groups = split_bulk_csv(content)
    if not header:
        log.warning("No header in response for %s %04d/%02d-%04d/%02d",
                    station, y1, m1, y2, m2)
        return 0, 0, 0, "error"

    written = upload_errors = 0
    for year, month in span:
        rows = groups.get((year, month))
        if not rows:
            continue
        path = local_path(station, year, month)
        if path.exists() and not force and _months_ago(year, month) >= recent_months:
            # Present already (a coalesced span can cover a few of these).
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        body = b"\n".join(_month_preamble(comments, year, month) + [header] + rows) + b"\n"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
        written += 1

        if S3_BUCKET:
            try:
                push_to_s3(path, station, year, month)
            except Exception as exc:            # noqa: BLE001 - any boto3/S3 failure
                log.warning("S3 push failed for %s %04d/%02d: %s", station, year, month, exc)
                upload_errors += 1

    empty = {(y, m) for y, m in span if not groups.get((y, m))}
    record_empty_months(
        station, {ym for ym in empty if _months_ago(*ym) >= recent_months}
    )
    return written, len(empty), upload_errors, "ok"


def bulk_sync(
    stations: list[str],
    months: list[tuple[int, int]],
    workers: int,
    force: bool,
    recent_months: int,
) -> None:
    """Backfill by requesting each station's missing range in as few calls as possible."""
    jobs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    n_stations_with_work = 0
    for station in stations:
        gaps = missing_months(station, months, force, recent_months)
        if not gaps:
            continue
        n_stations_with_work += 1
        for start, end in coalesce_runs(gaps):
            jobs.append((station, start, end))

    naive = sum(
        len(missing_months(s, months, force, recent_months)) for s in stations
    )
    log.info(
        "Bulk backfill: %d request(s) across %d station(s) with gaps "
        "(%d station-months; %.0fx fewer requests than one-per-month)",
        len(jobs), n_stations_with_work, naive,
        (naive / len(jobs)) if jobs else 0.0,
    )
    if not jobs:
        log.info("Nothing missing in the requested range; bulk backfill is a no-op.")
        return

    n_written = n_empty = n_err = n_upload_err = 0
    done = 0
    report_every = max(1, len(jobs) // 20)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_station_range, sta, start, end, force, recent_months): (sta, start, end)
            for sta, start, end in jobs
        }
        for fut in as_completed(futures):
            written, empty, upload_errors, status = fut.result()
            done += 1
            n_written += written
            n_empty += empty
            n_upload_err += upload_errors
            if status == "error":
                n_err += 1
            if done % report_every == 0 or done == len(jobs):
                log.info(
                    "  %d/%d requests  files: %d  empty-months: %d  "
                    "failed-requests: %d  upload-errors: %d",
                    done, len(jobs), n_written, n_empty, n_err, n_upload_err,
                )

    log.info(
        "Bulk backfill complete: %d file(s) written, %d empty month(s) recorded, "
        "%d failed request(s), %d upload error(s)",
        n_written, n_empty, n_err, n_upload_err,
    )

# ── Month iteration ───────────────────────────────────────────────────────────

def iter_month_span(
    start: tuple[int, int], end: tuple[int, int]
) -> Iterator[tuple[int, int]]:
    """Every (year, month) from start to end inclusive."""
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def iter_months(start_year: int, end_year: int) -> Iterator[tuple[int, int]]:
    now = datetime.now(tz=timezone.utc)
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year, month) > (now.year, now.month):
                return
            yield year, month

# ── Main sync ─────────────────────────────────────────────────────────────────

def sync(
    stations: list[str],
    months: list[tuple[int, int]],
    workers: int,
    force: bool,
    recent_months: int,
) -> None:
    tasks = list(product(stations, months))
    total = len(tasks)
    log.info(
        "Queuing %d tasks (%d stations x %d months)",
        total, len(stations), len(months),
    )

    n_dl = n_skip = n_err = n_nodata = n_upload_err = 0
    done = 0
    report_every = max(1, total // 20)   # log at every ~5% increment

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_station_month, sta, yr, mo, force, recent_months): (sta, yr, mo)
            for sta, (yr, mo) in tasks
        }
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result == "downloaded":
                n_dl += 1
            elif result == "skipped":
                n_skip += 1
            elif result == "nodata":
                n_nodata += 1
            elif result == "upload_error":
                n_upload_err += 1
            else:
                n_err += 1

            if done % report_every == 0 or done == total:
                log.info(
                    "  %d/%d  downloaded: %d  skipped: %d  no-data: %d  "
                    "errors: %d  upload-errors: %d",
                    done, total, n_dl, n_skip, n_nodata, n_err, n_upload_err,
                )

    log.info(
        "Sync complete — downloaded: %d  skipped: %d  no-data: %d  "
        "errors: %d  upload-errors: %d",
        n_dl, n_skip, n_nodata, n_err, n_upload_err,
    )

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror IEM ASOS hourly observations locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full", action="store_true",
        help=f"Sync all US ASOS stations from {FIRST_YEAR} to present (multi-day run).",
    )
    mode.add_argument(
        "--year", type=int, metavar="YYYY",
        help="Sync all months of a specific year.",
    )
    mode.add_argument(
        "--start-year", type=int, metavar="YYYY",
        help="Sync from YYYY through the current month (e.g. --start-year 2020).",
    )
    parser.add_argument(
        "--months", type=int, default=2, metavar="N",
        help="Daily update mode: re-sync last N months (default: 2).",
    )
    parser.add_argument(
        "--network", metavar="CODE",
        help="Sync only stations from this IEM network code (e.g. FL_ASOS).",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS,
        help=f"Concurrent download threads (default: {WORKERS}). IEM rate-limits "
             "aggressively; keep this low (2-4).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download all files, even if local copies exist.",
    )
    parser.add_argument(
        "--no-bulk", action="store_true",
        help="Backfill one request per station-month instead of one per missing "
             "range. Far slower against IEM's per-request rate limit; kept for "
             "comparison and as a fallback.",
    )
    parser.add_argument(
        "--local-root", type=Path, default=DEFAULT_LOCAL_ROOT, metavar="DIR",
        help=f"Directory to mirror into (default: {DEFAULT_LOCAL_ROOT}). The "
             "downstream ETL and IODA scripts read the default location, so "
             "override this only for a scratch copy.",
    )
    parser.add_argument(
        "--s3-bucket", metavar="NAME",
        help="Upload each downloaded file to this S3 bucket immediately, "
             "instead of relying on a separate bulk sync afterwards.",
    )
    parser.add_argument(
        "--s3-prefix", default="iem/csv", metavar="PREFIX",
        help="S3 key prefix under --s3-bucket (default: iem/csv). Ignored "
             "if --s3-bucket is not set.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Rebind the module globals the download helpers read.
    global LOCAL_ROOT, LOG_FILE, S3_BUCKET, S3_PREFIX
    LOCAL_ROOT = args.local_root.expanduser().resolve()
    LOG_FILE   = LOCAL_ROOT / "sync.log"
    S3_BUCKET  = args.s3_bucket
    S3_PREFIX  = args.s3_prefix.rstrip("/")

    setup_logging(args.verbose)
    log.info("=== IEM ASOS sync started ===")
    log.info("Local root: %s", LOCAL_ROOT)
    if S3_BUCKET:
        log.info("S3 push enabled: s3://%s/%s/", S3_BUCKET, S3_PREFIX)

    networks = [args.network] if args.network else US_ASOS_NETWORKS
    log.info("Fetching station list for %d network(s) ...", len(networks))
    stations = fetch_all_stations(networks)
    log.info("Total unique stations: %d", len(stations))
    if not stations:
        log.error("No stations found. Check network codes.")
        sys.exit(1)

    now = datetime.now(tz=timezone.utc)
    is_backfill = bool(args.full or args.year or args.start_year)

    if args.full:
        months = list(iter_months(FIRST_YEAR, now.year))
        log.info("Full sync: %d year-month pairs", len(months))
    elif args.year:
        months = list(iter_months(args.year, args.year))
        log.info("Year sync: %04d (%d months)", args.year, len(months))
    elif args.start_year:
        if args.start_year < FIRST_YEAR or args.start_year > now.year:
            parser.error(f"--start-year must be between {FIRST_YEAR} and {now.year}")
        months = list(iter_months(args.start_year, now.year))
        log.info("Sync from %04d: %d year-month pairs", args.start_year, len(months))
    else:
        all_months = list(iter_months(FIRST_YEAR, now.year))
        months = all_months[-args.months:]
        log.info(
            "Daily update: last %d months (%s - %s)",
            args.months,
            f"{months[0][0]:04d}/{months[0][1]:02d}",
            f"{months[-1][0]:04d}/{months[-1][1]:02d}",
        )

    log.info(
        "Month range: %s - %s (%d months)",
        f"{months[0][0]:04d}/{months[0][1]:02d}",
        f"{months[-1][0]:04d}/{months[-1][1]:02d}",
        len(months),
    )

    if is_backfill and not args.no_bulk:
        # Historic months in as few requests as possible, then the recent months
        # through the per-month path, which always re-fetches them.
        bulk_sync(stations, months, args.workers, args.force, args.months)
        recent = [m for m in months if _months_ago(*m) < args.months]
        if recent:
            log.info("Refreshing %d recent month(s) per-month", len(recent))
            sync(stations, recent, args.workers, args.force, args.months)
    else:
        sync(stations, months, args.workers, args.force, args.months)

    log.info("=== IEM ASOS sync finished ===")


if __name__ == "__main__":
    main()
