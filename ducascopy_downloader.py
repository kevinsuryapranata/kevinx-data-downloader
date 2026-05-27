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

import io
import lzma
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

# Output directory -- created automatically if it doesn't exist.
# Change this to any absolute or relative path you prefer.
OUTPUT_DIR = "data"

# Output filename template. Iterate on this as needed.
# Available fields: {symbol} {date} {start} {end}
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
MAX_WORKERS     = 6       # parallel download threads (cap: 6)
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


def fetch_hour(symbol: str, dt: datetime, on_retry=None):
    """
    Download and decode one hour of ticks.

    Returns a list of (timestamp, ask, bid, ask_vol, bid_vol) tuples.
    An empty list means no ticks for that hour (weekend, holiday, etc.).

    on_retry: optional callable(attempt, max_attempts) invoked before each
    retry, so callers can show retry activity in a progress display.
    """
    url = hour_url(symbol, dt)
    point = INSTRUMENTS[symbol][0]

    # Retry the request a few times with growing backoff. Dukascopy
    # occasionally responds slowly to rapid sequential requests, causing
    # read timeouts -- a retry usually succeeds where the first try failed.
    # VPNs make this worse; tune the constants at the top of the file.
    resp = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            break
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise                       # give up -- caller logs it
            if on_retry:
                on_retry(attempt, MAX_ATTEMPTS)
            time.sleep(RETRY_BACKOFF * attempt)

    if resp.status_code == 404:
        return []           # no data this hour -- normal for closed markets
    resp.raise_for_status()

    raw = resp.content
    if not raw:
        return []

    # The file is LZMA-compressed. Empty/closed hours sometimes return tiny
    # files that fail to decompress -- treat those as "no data".
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return []

    ticks = []
    for ms, ask, bid, ask_vol, bid_vol in TICK_STRUCT.iter_unpack(data):
        ts = dt.replace(minute=0, second=0, microsecond=0) + timedelta(milliseconds=ms)
        ticks.append((ts, ask / point, bid / point, ask_vol, bid_vol))
    return ticks


def _fmt_elapsed(seconds: float) -> str:
    """Format a duration as MM:SS (or HH:MM:SS if long)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _progress_line(done: int, total: int, ticks: int,
                   elapsed: float, status: str) -> str:
    """Build a single-line progress display."""
    width = 24
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100 * done / total
    return (
        f"\r  [{bar}] {done:2d}/{total} ({pct:3.0f}%)  "
        f"ticks={ticks:<7d}  elapsed={_fmt_elapsed(elapsed)}  {status}   "
    )


def download_day(symbol: str, day: datetime):
    """
    Download all 24 hours for a given calendar day (UTC) using a thread pool.

    Up to MAX_WORKERS hours are fetched in parallel. The main thread drives
    a live progress line showing active slots, completed count, tick count,
    and elapsed time -- updated every second.
    """
    total      = 24
    start      = time.monotonic()
    lock       = threading.Lock()

    # Shared state written by workers, read by the display loop.
    completed  = [0]           # hours finished (success or fail)
    tick_count = [0]           # ticks collected so far
    failed_hours = []
    results    = {}            # hour -> list-of-ticks (ordered later)
    active     = {}            # hour -> status string (what it's doing now)

    def fetch_one(hour: int):
        dt = day.replace(hour=hour, minute=0, second=0, microsecond=0)

        def on_retry(attempt, max_attempts):
            with lock:
                active[hour] = f"{hour:02d}:00 retry {attempt}/{max_attempts-1}"

        with lock:
            active[hour] = f"{hour:02d}:00 connecting..."

        try:
            ticks = fetch_hour(symbol, dt, on_retry=on_retry)
            with lock:
                results[hour] = ticks
                tick_count[0] += len(ticks)
                completed[0]  += 1
                active.pop(hour, None)
        except Exception as exc:                   # noqa: BLE001
            with lock:
                results[hour] = exc
                completed[0] += 1
                active.pop(hour, None)

    # Submit all 24 hours; the pool caps concurrency at MAX_WORKERS.
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for hour in range(total):
            futures[pool.submit(fetch_one, hour)] = hour

        # Drive the progress display while workers run.
        while completed[0] < total:
            time.sleep(0.5)
            elapsed = time.monotonic() - start
            with lock:
                done   = completed[0]
                ticks  = tick_count[0]
                slots  = sorted(active.values())

            # Show up to 3 active slots so the line stays one-liner width.
            if slots:
                slot_str = "  [" + "  |  ".join(slots[:3])
                if len(slots) > 3:
                    slot_str += f"  +{len(slots)-3} more"
                slot_str += "]"
            else:
                slot_str = "  [waiting...]"

            sys.stdout.write(
                _progress_line(done, total, ticks, elapsed, slot_str)
            )
            sys.stdout.flush()

    # All futures done -- collect errors and sort ticks by hour.
    all_ticks = []
    for hour in range(total):
        outcome = results.get(hour, [])
        if isinstance(outcome, Exception):
            sys.stdout.write("\r" + " " * 120 + "\r")
            print(f"  ! hour {hour:02d} failed: {outcome}")
            failed_hours.append(hour)
        else:
            all_ticks.extend(outcome)

    elapsed = time.monotonic() - start
    sys.stdout.write(
        _progress_line(total, total, len(all_ticks), elapsed, "  [complete]")
    )
    sys.stdout.write("\n")
    sys.stdout.flush()
    return all_ticks, failed_hours


def write_csv(ticks, path: str):
    """Write ticks to a CSV file."""
    with open(path, "w") as f:
        f.write("timestamp,ask,bid,ask_volume,bid_volume\n")
        for ts, ask, bid, ask_vol, bid_vol in ticks:
            f.write(
                f"{ts:%Y-%m-%d %H:%M:%S.%f},"
                f"{ask},{bid},{ask_vol},{bid_vol}\n"
            )


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
        # also allow typing the symbol directly
        if choice.upper() in INSTRUMENTS:
            return choice.upper()
        print("  Invalid choice, try again.")


def choose_date() -> datetime:
    """Menu 2: enter the date to download."""
    print("\n" + "=" * 55)
    print("  MENU 2 -- enter the date")
    print("=" * 55)
    print("  Format: YYYY-MM-DD   (example: 2024-03-15)")
    print("  Note: dates are UTC; weekends usually have no data.")
    print("-" * 55)

    while True:
        raw = input("Enter date: ").strip()
        try:
            day = datetime.strptime(raw, "%Y-%m-%d")
            return day.replace(tzinfo=timezone.utc)
        except ValueError:
            print("  Invalid format. Use YYYY-MM-DD, e.g. 2024-03-15")


def main():
    print("\n" + "#" * 55)
    print("#  Dukascopy raw tick data downloader")
    print("#" * 55)

    show_timeframes()

    symbol = choose_ticker()
    day = choose_date()

    print(f"\nDownloading tick data for {symbol} on {day:%Y-%m-%d} ...")
    ticks, failed_hours = download_day(symbol, day)

    if not ticks and not failed_hours:
        print("\nNo ticks found. Check the date (weekend/holiday?) or symbol.")
        return

    filename = FILENAME_TEMPLATE.format(
        symbol=symbol,
        date=day.strftime("%Y-%m-%d"),
        start=day.strftime("%Y%m%d"),
        end=day.strftime("%Y%m%d"),
    )
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    write_csv(ticks, filepath)
    print(f"\nSaved {len(ticks)} ticks to: {filepath}")
    if ticks:
        print(f"First tick: {ticks[0][0]:%Y-%m-%d %H:%M:%S.%f}")
        print(f"Last  tick: {ticks[-1][0]:%Y-%m-%d %H:%M:%S.%f}")

    # Tell the user exactly which hours failed (vs. were just empty), so
    # they know whether to re-run. A failed hour means a download error;
    # an empty hour with no failures means the market was simply closed.
    if failed_hours:
        hrs = ", ".join(f"{h:02d}:00" for h in failed_hours)
        print(f"\n  WARNING: {len(failed_hours)} hour(s) failed to download: {hrs}")
        print("  These are NOT in the file. The data is incomplete.")
        print("  Re-run for the same date to retry, or raise the timeout")
        print("  values at the top of the script (CONNECT/READ_TIMEOUT).")
        print("  If you are on a VPN, try turning it off for the download.")
    else:
        print("\n  All 24 hours downloaded successfully.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")