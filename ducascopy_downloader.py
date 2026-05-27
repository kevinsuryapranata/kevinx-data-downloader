#!/usr/bin/env python3
"""
Dukascopy raw tick data downloader.

Downloads historical tick data from Dukascopy's public feed and writes it
to a CSV file. Interactive CLI: pick a ticker, enter a date, get a file.

Dukascopy serves one LZMA-compressed (.bi5) file per hour. Each tick is a
20-byte big-endian record:
    uint32  ms offset from the hour
    uint32  ask price (integer, scaled by point factor)
    uint32  bid price (integer, scaled by point factor)
    float32 ask volume (in millions)
    float32 bid volume (in millions)
"""

import calendar
import io
import lzma
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Root output directory. Downloads land under:
#   {OUTPUT_DIR}/raw/{symbol}/{YYYY_MM}/{filename}
# A processed/ sibling is created ready for future use.
OUTPUT_DIR = "data"

# Output filename template.
# Available fields: {symbol} {date}
FILENAME_TEMPLATE = "{symbol}_ticks_{date}.csv"

# --- Network tuning -------------------------------------------------------
# If you're on a VPN or a slow connection, raise these. Read timeouts mean
# the connection works but data arrives too slowly -- bigger timeouts and
# more retries usually fix it. If downloads are reliably fast, you can
# lower them so genuine failures are detected sooner.
CONNECT_TIMEOUT = 15      # seconds to establish the connection
READ_TIMEOUT    = 45      # seconds to wait for data once connected
MAX_ATTEMPTS    = 6       # total tries per hour before giving up
RETRY_BACKOFF   = 3       # base backoff seconds; grows each retry
MAX_WORKERS     = max(1, int(os.cpu_count() * 0.7))  # 70% of available cores
# --------------------------------------------------------------------------

# A selection of commonly used Dukascopy instruments.
# 'point' is the divisor that turns the integer price into a real price.
# This is not the full Dukascopy catalogue (which has thousands of symbols),
# but covers the popular ones. Add more as you need them.
INSTRUMENTS = {
    # symbol        point     description
    "XAUUSD":     (1000,    "Gold vs US Dollar"),
    "XAGUSD":     (1000,    "Silver vs US Dollar"),
    "EURUSD":     (100000,  "Euro vs US Dollar"),
    "GBPUSD":     (100000,  "British Pound vs US Dollar"),
    "USDJPY":     (1000,    "US Dollar vs Japanese Yen"),
    "USDCHF":     (100000,  "US Dollar vs Swiss Franc"),
    "AUDUSD":     (100000,  "Australian Dollar vs US Dollar"),
    "USDCAD":     (100000,  "US Dollar vs Canadian Dollar"),
    "NZDUSD":     (100000,  "New Zealand Dollar vs US Dollar"),
    "EURGBP":     (100000,  "Euro vs British Pound"),
    "EURJPY":     (1000,    "Euro vs Japanese Yen"),
    "GBPJPY":     (1000,    "British Pound vs Japanese Yen"),
    "BTCUSD":     (1000,    "Bitcoin vs US Dollar"),
    "ETHUSD":     (1000,    "Ethereum vs US Dollar"),
    "USA500IDXUSD": (1000,  "S&P 500 Index"),
    "USATECHIDXUSD": (1000, "Nasdaq 100 Index"),
    "DEUIDXEUR":  (1000,    "DAX 40 Index"),
    "USDOLLARIDXUSD": (1000, "US Dollar Index"),
    "LIGHTCMDUSD": (1000,   "WTI Crude Oil"),
    "BRENTCMDUSD": (1000,   "Brent Crude Oil"),
}

# Dukascopy only serves *tick* data through this feed. Bar timeframes
# (1m, 5m, 1h, etc.) are not separate downloads -- you build them by
# aggregating ticks yourself. Listed here for reference / future use.
TIMEFRAMES = {
    "TICK": "Raw ticks (what this script downloads)",
    "M1":   "1-minute bars   (aggregate from ticks)",
    "M5":   "5-minute bars   (aggregate from ticks)",
    "M15":  "15-minute bars  (aggregate from ticks)",
    "M30":  "30-minute bars  (aggregate from ticks)",
    "H1":   "1-hour bars     (aggregate from ticks)",
    "H4":   "4-hour bars     (aggregate from ticks)",
    "D1":   "Daily bars      (aggregate from ticks)",
}

TICK_STRUCT = struct.Struct(">IIIff")  # 20 bytes per tick


# ---------------------------------------------------------------------------
# Download / parse
# ---------------------------------------------------------------------------

def hour_url(symbol: str, dt: datetime) -> str:
    """Build the .bi5 URL for one symbol-hour. Dukascopy month is 0-indexed."""
    return (
        f"{BASE_URL}/{symbol}/{dt.year:04d}/{dt.month - 1:02d}/"
        f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )


# Sentinel returned by fetch_hour when Dukascopy has no data for that hour
# (404). Distinguishable from a genuine empty list (which shouldn't occur).
EMPTY_HOUR = object()


def fetch_hour(symbol: str, dt: datetime, on_retry=None, on_download=None):
    """
    Download and decode one hour of ticks.

    Returns:
      EMPTY_HOUR sentinel  -- Dukascopy returned 404 (market closed / no data)
      list of tuples       -- (timestamp, ask, bid, ask_vol, bid_vol)

    on_retry:    optional callable(attempt, max_attempts)
    on_download: optional callable() -- called once response received
    """
    url = hour_url(symbol, dt)
    point = INSTRUMENTS[symbol][0]

    resp = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            break
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
            if on_retry:
                on_retry(attempt, MAX_ATTEMPTS)
            time.sleep(RETRY_BACKOFF * attempt)

    if resp.status_code == 404:
        return EMPTY_HOUR       # market closed / no data for this hour
    resp.raise_for_status()

    if on_download:
        on_download()

    raw = resp.content
    if not raw:
        return EMPTY_HOUR

    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return EMPTY_HOUR

    ticks = []
    for ms, ask, bid, ask_vol, bid_vol in TICK_STRUCT.iter_unpack(data):
        ts = dt.replace(minute=0, second=0, microsecond=0) + timedelta(milliseconds=ms)
        ticks.append((ts, ask / point, bid / point, ask_vol, bid_vol))
    return ticks


def _fmt_elapsed(seconds: float) -> str:
    """Format a duration as MM:SS (or HH:MM:SS if over an hour)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


BOARD_LINES = MAX_WORKERS + 2   # header + separator + one line per slot


def _render_board(symbol: str, day: datetime, slots: dict,
                  done: int, total: int, ticks: int,
                  elapsed: float, first_draw: bool) -> str:
    """
    Build the full worker-board string.

    slots: dict of  slot_index -> {hour, state, detail, since}
      state: 'idle' | 'connecting' | 'downloading' | 'retrying' | 'done' | 'failed'
    """
    ICON = {
        "idle":        "  ",
        "connecting":  "↓ ",
        "downloading": "↓ ",
        "retrying":    "⟳ ",
        "done":        "✓ ",
        "failed":      "✗ ",
    }

    lines = []

    # ── header ──────────────────────────────────────────────────────────────
    pct = 100 * done / total
    lines.append(
        f"  {symbol}  {day:%Y-%m-%d}  "
        f"elapsed: {_fmt_elapsed(elapsed)}   "
        f"done: {done}/{total} ({pct:.0f}%)   "
        f"ticks: {ticks:,}"
    )
    lines.append("  " + "─" * 54)

    # ── one line per worker slot ─────────────────────────────────────────────
    for i in range(MAX_WORKERS):
        s = slots.get(i)
        if s is None:
            lines.append(f"  slot {i+1}  –  idle")
            continue

        icon  = ICON.get(s["state"], "  ")
        state = s["state"]

        if state == "done":
            tc = s.get("ticks", 0)
            lines.append(
                f"  slot {i+1}  {icon} {s['hour']:02d}:00  "
                f"{tc:>6,} ticks"
            )
        elif state == "failed":
            lines.append(
                f"  slot {i+1}  {icon} {s['hour']:02d}:00  FAILED"
            )
        elif state == "retrying":
            waited = int(time.monotonic() - s["since"])
            lines.append(
                f"  slot {i+1}  {icon} {s['hour']:02d}:00  "
                f"{s['detail']}  ({waited}s)"
            )
        else:
            lines.append(
                f"  slot {i+1}  {icon} {s['hour']:02d}:00  {state}..."
            )

    # Move cursor up to overwrite previous board (after first draw).
    move_up = "" if first_draw else f"\033[{len(lines)}A"
    return move_up + "\n".join(lines) + "\n"


def download_day(symbol: str, day: datetime):
    """
    Download all 24 hours for a given calendar day (UTC) using a thread pool.

    Up to MAX_WORKERS hours are fetched in parallel. The main thread drives
    a fixed worker-board display: one line per slot showing state, hour,
    retry count, retry timer, and tick count. Redraws in place every second.
    """
    total    = 24
    start    = time.monotonic()
    lock     = threading.Lock()

    # ── shared state ────────────────────────────────────────────────────────
    completed  = [0]
    tick_count = [0]
    failed_hours = []
    results    = {}            # hour -> ticks list or Exception

    # slots: slot_index -> state dict
    # hour_to_slot: hour -> slot_index (assigned when worker starts)
    slots        = {}
    hour_to_slot = {}
    next_slot    = [0]         # simple round-robin slot assignment

    def fetch_one(hour: int):
        dt = day.replace(hour=hour, minute=0, second=0, microsecond=0)

        with lock:
            slot = next_slot[0] % MAX_WORKERS
            next_slot[0] += 1
            hour_to_slot[hour] = slot
            slots[slot] = {"hour": hour, "state": "connecting",
                           "detail": "", "since": time.monotonic(), "ticks": 0}

        def on_retry(attempt, max_attempts):
            with lock:
                slots[slot]["state"]  = "retrying"
                slots[slot]["detail"] = f"retry {attempt}/{max_attempts-1}"
                slots[slot]["since"]  = time.monotonic()

        def on_download():
            with lock:
                slots[slot]["state"] = "downloading"
                slots[slot]["since"] = time.monotonic()

        try:
            outcome = fetch_hour(symbol, dt, on_retry=on_retry,
                                 on_download=on_download)
            with lock:
                results[hour] = outcome
                n = 0 if outcome is EMPTY_HOUR else len(outcome)
                tick_count[0]       += n
                completed[0]        += 1
                slots[slot]["state"] = "done"
                slots[slot]["ticks"] = n
        except Exception as exc:                        # noqa: BLE001
            with lock:
                results[hour]        = exc
                completed[0]        += 1
                slots[slot]["state"] = "failed"

    # ── run pool + display loop ──────────────────────────────────────────────
    first_draw = True
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for hour in range(total):
            pool.submit(fetch_one, hour)

        while completed[0] < total:
            time.sleep(0.5)
            with lock:
                snap_slots = {k: dict(v) for k, v in slots.items()}
                done       = completed[0]
                ticks      = tick_count[0]
            elapsed = time.monotonic() - start
            board = _render_board(symbol, day, snap_slots,
                                  done, total, ticks, elapsed, first_draw)
            sys.stdout.write(board)
            sys.stdout.flush()
            first_draw = False

    # ── final board draw ─────────────────────────────────────────────────────
    with lock:
        snap_slots = {k: dict(v) for k, v in slots.items()}
        ticks      = tick_count[0]
    elapsed = time.monotonic() - start
    sys.stdout.write(
        _render_board(symbol, day, snap_slots,
                      total, total, ticks, elapsed, first_draw)
    )
    sys.stdout.flush()

    # ── collect results in hour order ────────────────────────────────────────
    # hour_status: hour -> "ok" | "empty" | "failed"
    all_ticks   = []
    hour_status = {}
    for hour in range(total):
        outcome = results.get(hour)
        if isinstance(outcome, Exception):
            failed_hours.append(hour)
            hour_status[hour] = "failed"
        elif outcome is EMPTY_HOUR:
            hour_status[hour] = "empty"
        else:
            all_ticks.extend(outcome)
            hour_status[hour] = "ok"

    return all_ticks, failed_hours, hour_status


def write_csv(ticks, path):
    """Write ticks to a CSV file."""
    with open(path, "w") as f:
        f.write("timestamp,ask,bid,ask_volume,bid_volume\n")
        for ts, ask, bid, ask_vol, bid_vol in ticks:
            f.write(
                f"{ts:%Y-%m-%d %H:%M:%S.%f},"
                f"{ask},{bid},{ask_vol},{bid_vol}\n"
            )


def _meta_path(csv_path: Path) -> Path:
    """Return the .meta.json path for a given CSV path."""
    return csv_path.with_suffix(".meta.json")


def write_meta(csv_path: Path, symbol: str, day: datetime,
               ticks: list, hour_status: dict):
    """
    Write a sidecar .meta.json alongside the CSV recording per-hour status.

    hour_status: {hour_int -> "ok" | "empty" | "failed"}
    """
    import json

    tick_by_hour = {}
    for ts, *_ in ticks:
        tick_by_hour[ts.hour] = tick_by_hour.get(ts.hour, 0) + 1

    hours = {}
    for h in range(24):
        status = hour_status.get(h, "failed")
        entry  = {"status": status}
        if status == "ok":
            entry["ticks"] = tick_by_hour.get(h, 0)
        hours[f"{h:02d}"] = entry

    meta = {
        "symbol":        symbol,
        "date":          day.strftime("%Y-%m-%d"),
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hours":         hours,
        "summary": {
            "ok":     sum(1 for v in hours.values() if v["status"] == "ok"),
            "empty":  sum(1 for v in hours.values() if v["status"] == "empty"),
            "failed": sum(1 for v in hours.values() if v["status"] == "failed"),
            "total_ticks": len(ticks),
        },
    }

    with open(_meta_path(csv_path), "w") as f:
        json.dump(meta, f, indent=2)


def check_integrity(csv_path: Path) -> dict | None:
    """
    Read the sidecar .meta.json for a CSV and return a integrity report dict,
    or None if no meta file exists.

    Report keys: symbol, date, ok, empty, failed, total_ticks, failed_hours
    """
    import json

    mp = _meta_path(csv_path)
    if not mp.exists():
        return None

    with open(mp) as f:
        meta = json.load(f)

    failed_hours = [
        int(h) for h, v in meta["hours"].items()
        if v["status"] == "failed"
    ]
    return {
        "symbol":       meta["symbol"],
        "date":         meta["date"],
        "ok":           meta["summary"]["ok"],
        "empty":        meta["summary"]["empty"],
        "failed":       meta["summary"]["failed"],
        "total_ticks":  meta["summary"]["total_ticks"],
        "failed_hours": failed_hours,
    }


def print_integrity(report: dict):
    """Print a formatted integrity report for one day."""
    ok_str     = f"{report['ok']:2d} hours with data"
    empty_str  = f"{report['empty']:2d} hours empty (market closed)"
    failed_str = f"{report['failed']:2d} hours MISSING"

    status = "✓" if report["failed"] == 0 else "⚠"
    print(f"  {status}  {report['date']}  ticks: {report['total_ticks']:>8,}")
    print(f"       ✓  {ok_str}")
    print(f"       ○  {empty_str}")
    if report["failed"] > 0:
        hrs = "  ".join(f"{h:02d}:00" for h in report["failed_hours"])
        print(f"       ✗  {failed_str}  →  {hrs}")


# ---------------------------------------------------------------------------
# CLI menus
# ---------------------------------------------------------------------------

def show_timeframes():
    print("\nAvailable timeframes on Dukascopy:")
    print("-" * 55)
    for code, desc in TIMEFRAMES.items():
        print(f"  {code:6s}  {desc}")
    print("\nNote: Dukascopy's free feed only serves raw TICK data.")
    print("Bar timeframes are built by aggregating ticks locally.\n")


def choose_ticker() -> str:
    """Menu 1: pick which instrument to download."""
    symbols = list(INSTRUMENTS.keys())
    print("\n" + "=" * 55)
    print("  MENU 1 -- choose a ticker")
    print("=" * 55)
    for i, sym in enumerate(symbols, 1):
        desc = INSTRUMENTS[sym][1]
        print(f"  {i:2d}. {sym:16s} {desc}")
    print("-" * 55)

    while True:
        choice = input(f"Select ticker [1-{len(symbols)}] (default 1=XAUUSD): ").strip()
        if choice == "":
            return symbols[0]
        if choice.isdigit() and 1 <= int(choice) <= len(symbols):
            return symbols[int(choice) - 1]
        if choice.upper() in INSTRUMENTS:
            return choice.upper()
        print("  Invalid choice, try again.")


def choose_mode() -> str:
    """Menu 2: single day or full month."""
    print("\n" + "=" * 55)
    print("  MENU 2 -- download mode")
    print("=" * 55)
    print("  1. Single day   (YYYY-MM-DD)")
    print("  2. Full month   (YYYY-MM)")
    print("-" * 55)

    while True:
        choice = input("Select mode [1-2] (default 1): ").strip()
        if choice in ("", "1"):
            return "day"
        if choice == "2":
            return "month"
        print("  Invalid choice, try again.")


def choose_date() -> datetime:
    """Menu 3a: enter a single date."""
    print("\n" + "=" * 55)
    print("  MENU 3 -- enter date")
    print("=" * 55)
    print("  Format : YYYY-MM-DD   e.g. 2024-03-15")
    print("  Note   : dates are UTC; weekends usually have no data.")
    print("-" * 55)

    while True:
        raw = input("Enter date: ").strip()
        try:
            day = datetime.strptime(raw, "%Y-%m-%d")
            return day.replace(tzinfo=timezone.utc)
        except ValueError:
            print("  Invalid format. Use YYYY-MM-DD, e.g. 2024-03-15")


def choose_month() -> tuple[int, int]:
    """Menu 3b: enter a year-month, return (year, month)."""
    print("\n" + "=" * 55)
    print("  MENU 3 -- enter month")
    print("=" * 55)
    print("  Format : YYYY-MM   e.g. 2024-03")
    print("  Note   : all days in the month will be downloaded.")
    print("-" * 55)

    while True:
        raw = input("Enter month: ").strip()
        try:
            dt = datetime.strptime(raw, "%Y-%m")
            return dt.year, dt.month
        except ValueError:
            print("  Invalid format. Use YYYY-MM, e.g. 2024-03")


def _out_path(symbol: str, day: datetime) -> Path:
    """
    Return the full output filepath for one day's CSV.
    Structure: {OUTPUT_DIR}/raw/{symbol}/{YYYY_MM}/{filename}
    """
    month_folder = day.strftime("%Y_%m")
    filename     = FILENAME_TEMPLATE.format(
        symbol=symbol,
        date=day.strftime("%Y-%m-%d"),
    )
    out_dir = Path(OUTPUT_DIR) / "raw" / symbol / month_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def _already_exists(symbol: str, day: datetime) -> bool:
    """
    Return True if a non-empty CSV already exists for this symbol/day.
    Used to skip re-downloading files from interrupted runs.
    """
    p = _out_path(symbol, day)
    return p.exists() and p.stat().st_size > 0


def _save_day(symbol: str, day: datetime, ticks: list,
              failed_hours: list, hour_status: dict):
    """Write one day's CSV + meta, then print an integrity report."""
    filepath = _out_path(symbol, day)
    write_csv(ticks, filepath)
    write_meta(filepath, symbol, day, ticks, hour_status)

    report = check_integrity(filepath)
    print_integrity(report)


def main():
    print("\n" + "#" * 55)
    print("#  Dukascopy raw tick data downloader")
    print(f"#  workers: {MAX_WORKERS}  ({os.cpu_count()} cores × 70%)")
    print("#" * 55)

    show_timeframes()

    symbol = choose_ticker()
    mode   = choose_mode()

    # ── single day ────────────────────────────────────────────────────────
    if mode == "day":
        day = choose_date()

        if _already_exists(symbol, day):
            p = _out_path(symbol, day)
            print(f"\n  Already exists: {p}")
            print("  Delete the file first if you want to re-download.")
            return

        print(f"\nDownloading {symbol}  {day:%Y-%m-%d} ...\n")
        ticks, failed_hours, hour_status = download_day(symbol, day)

        if not ticks and not failed_hours:
            print("\nNo ticks found. Check the date (weekend/holiday?) or symbol.")
            return

        _save_day(symbol, day, ticks, failed_hours, hour_status)

        if ticks:
            print(f"\n  First tick : {ticks[0][0]:%Y-%m-%d %H:%M:%S.%f}")
            print(f"  Last  tick : {ticks[-1][0]:%Y-%m-%d %H:%M:%S.%f}")

        if failed_hours:
            print(f"\n  WARNING: {len(failed_hours)} hour(s) missing from file.")
            print("  Re-run this date or raise CONNECT/READ_TIMEOUT at top of script.")
            print("  If on a VPN, try turning it off.")
        else:
            print("\n  All 24 hours downloaded successfully.")

    # ── full month ────────────────────────────────────────────────────────
    else:
        year, month = choose_month()
        _, n_days   = calendar.monthrange(year, month)
        days        = [
            datetime(year, month, d, tzinfo=timezone.utc)
            for d in range(1, n_days + 1)
        ]

        month_name = datetime(year, month, 1).strftime("%B %Y")

        # pre-flight: check which days already exist
        existing = [d for d in days if _already_exists(symbol, d)]
        pending  = [d for d in days if not _already_exists(symbol, d)]

        print(f"\n  {month_name}  →  {n_days} days  "
              f"({days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d})")
        print(f"  Symbol   : {symbol}")
        print(f"  Workers  : {MAX_WORKERS}")
        print(f"  Output   : {Path(OUTPUT_DIR) / 'raw' / symbol / f'{year:04d}_{month:02d}'}/")
        print(f"  To fetch : {len(pending)}  /  Already done : {len(existing)}")
        print()

        if not pending:
            print("  All days already downloaded. Nothing to do.")
            return

        confirm = input("  Start download? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

        print()
        total_ticks   = 0
        total_failed  = []
        skipped_days  = []

        for i, day in enumerate(pending, 1):
            print(f"  [{i:2d}/{len(pending)}]  {day:%Y-%m-%d}")
            ticks, failed_hours, hour_status = download_day(symbol, day)

            if not ticks and not failed_hours:
                skipped_days.append(day)
                print(f"         no data (weekend / holiday)")
                continue

            _save_day(symbol, day, ticks, failed_hours, hour_status)
            total_ticks += len(ticks)
            if failed_hours:
                total_failed.append((day, failed_hours))
            print()

        # ── month summary ─────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"  {month_name} download complete")
        print("=" * 55)
        print(f"  Total ticks     : {total_ticks:,}")
        print(f"  Days downloaded : {len(pending) - len(skipped_days)}")
        print(f"  Days skipped    : {len(skipped_days)}  (no market data)")
        print(f"  Already existed : {len(existing)}")

        if total_failed:
            print(f"  Days with gaps  : {len(total_failed)}")
            for day, hours in total_failed:
                hrs = ", ".join(f"{h:02d}:00" for h in hours)
                print(f"    {day:%Y-%m-%d}  missing hours: {hrs}")
            print("\n  Re-run to retry gaps (existing complete days will be skipped).")
        else:
            print("  All days complete  ✓")
        print()


def run_check(folder: str):
    """
    Scan a data folder for .meta.json files and print an integrity report.
    Usage:  python dukascopy_downloader.py --check data/raw/XAUUSD/2025_09
    """
    import json

    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"  Folder not found: {folder_path}")
        return

    meta_files = sorted(folder_path.glob("*.meta.json"))
    if not meta_files:
        print(f"  No .meta.json files found in {folder_path}")
        print("  Run a download first to generate them.")
        return

    print(f"\n  Integrity check: {folder_path}")
    print("  " + "─" * 53)

    total_ok = total_empty = total_failed = total_ticks = 0
    days_with_gaps = []

    for mp in meta_files:
        # Strip .meta.json -> .csv  (mp.stem is e.g. "XAUUSD_ticks_2024-03-15.meta")
        csv_path = mp.parent / (mp.stem.replace(".meta", "") + ".csv")
        report   = check_integrity(csv_path)
        if report is None:
            continue
        print_integrity(report)
        total_ok     += report["ok"]
        total_empty  += report["empty"]
        total_failed += report["failed"]
        total_ticks  += report["total_ticks"]
        if report["failed"] > 0:
            days_with_gaps.append(report["date"])

    print("\n  " + "─" * 53)
    print(f"  Days checked    : {len(meta_files)}")
    print(f"  Total ticks     : {total_ticks:,}")
    print(f"  Hours OK        : {total_ok}")
    print(f"  Hours empty     : {total_empty}  (market closed, normal)")
    print(f"  Hours missing   : {total_failed}")

    if days_with_gaps:
        print(f"\n  ⚠  {len(days_with_gaps)} day(s) have gaps:")
        for d in days_with_gaps:
            print(f"     {d}")
        print("\n  Re-download those dates to fill gaps.")
    else:
        print("\n  All hours accounted for  ✓")
    print()


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--check":
        if len(_sys.argv) < 3:
            print("Usage: python dukascopy_downloader.py --check <folder>")
            print("  e.g. python dukascopy_downloader.py --check data/raw/XAUUSD/2025_09")
        else:
            try:
                run_check(_sys.argv[2])
            except KeyboardInterrupt:
                print("\n\nAborted.")
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n\nAborted.")