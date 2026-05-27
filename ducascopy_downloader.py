#!/usr/bin/env python3
"""
Dukascopy raw tick data downloader.

Downloads historical tick data from Dukascopy's public feed and writes it
to CSV files. Interactive CLI: pick a ticker, choose single day or full month.

Dukascopy serves one LZMA-compressed (.bi5) file per hour. Each tick is a
20-byte big-endian record:
    uint32  ms offset from the hour
    uint32  ask price (integer, scaled by point factor)
    uint32  bid price (integer, scaled by point factor)
    float32 ask volume (in millions)
    float32 bid volume (in millions)

Folder layout:
    data/
      raw/
        XAUUSD/
          2026_05/
            XAUUSD_ticks_2026-05-01.csv
            XAUUSD_ticks_2026-05-02.csv
            ...
      processed/   <- empty, ready for future use
"""

import calendar
import lzma
import os
import queue
import struct
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL        = "https://datafeed.dukascopy.com/datafeed"
OUTPUT_DIR      = "data"
FILENAME_TEMPLATE = "{symbol}_ticks_{date}.csv"

MAX_WORKERS     = max(1, int(os.cpu_count() * 0.7))  # 70% of cores
MAX_RETRIES     = 3       # attempts per hour before permanently failed
CONNECT_TIMEOUT = 10      # seconds
READ_TIMEOUT    = 30      # seconds

# ---------------------------------------------------------------------------
# Instrument catalogue
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    # Metals
    "XAUUSD": (100,    "Gold / US Dollar"),
    "XAGUSD": (1000,   "Silver / US Dollar"),
    "XPTUSD": (100,    "Platinum / US Dollar"),
    "XPDUSD": (100,    "Palladium / US Dollar"),
    # Major FX
    "EURUSD": (100000, "Euro / US Dollar"),
    "GBPUSD": (100000, "British Pound / US Dollar"),
    "USDJPY": (1000,   "US Dollar / Japanese Yen"),
    "USDCHF": (100000, "US Dollar / Swiss Franc"),
    "AUDUSD": (100000, "Australian Dollar / US Dollar"),
    "NZDUSD": (100000, "New Zealand Dollar / US Dollar"),
    "USDCAD": (100000, "US Dollar / Canadian Dollar"),
    # Minor FX
    "EURGBP": (100000, "Euro / British Pound"),
    "EURJPY": (1000,   "Euro / Japanese Yen"),
    "GBPJPY": (1000,   "British Pound / Japanese Yen"),
    "AUDJPY": (1000,   "Australian Dollar / Japanese Yen"),
    "EURCHF": (100000, "Euro / Swiss Franc"),
    "GBPCHF": (100000, "British Pound / Swiss Franc"),
    "CADJPY": (1000,   "Canadian Dollar / Japanese Yen"),
    "NZDJPY": (1000,   "New Zealand Dollar / Japanese Yen"),
    # Indices / Commodities
    "WTIUSD": (100,    "WTI Crude Oil / US Dollar"),
    "BRNUSD": (100,    "Brent Crude Oil / US Dollar"),
    "SPXUSD": (100,    "S&P 500 Index"),
    "NSXUSD": (100,    "NASDAQ 100 Index"),
    "DJUSD":  (100,    "Dow Jones Industrial Average"),
    "FTSUSD": (100,    "FTSE 100 Index"),
    "FDXEUR": (100,    "DAX 40 Index"),
    "JP225":  (100,    "Nikkei 225 Index"),
    # Crypto (historical)
    "BTCUSD": (1,      "Bitcoin / US Dollar"),
    "ETHUSD": (100,    "Ethereum / US Dollar"),
}

TIMEFRAMES = [
    ("Tick", "Raw feed — every bid/ask change"),
    ("M1",   "1-minute bars  (aggregate ticks)"),
    ("M5",   "5-minute bars  (aggregate ticks)"),
    ("M15",  "15-minute bars (aggregate ticks)"),
    ("M30",  "30-minute bars (aggregate ticks)"),
    ("H1",   "1-hour bars    (aggregate ticks)"),
    ("H4",   "4-hour bars    (aggregate ticks)"),
    ("D1",   "Daily bars     (aggregate ticks)"),
]

# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------

def _bi5_url(symbol: str, dt: datetime) -> str:
    return (
        f"{BASE_URL}/{symbol}/"
        f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/"
        f"{dt.hour:02d}h_ticks.bi5"
    )


def _parse_bi5(data: bytes, hour_dt: datetime, point: int) -> list:
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return []
    ticks = []
    for off in range(0, len(raw) - 19, 20):
        ms, ask_i, bid_i, av_raw, bv_raw = struct.unpack_from(">IIIff", raw, off)
        ts = hour_dt + timedelta(milliseconds=ms)
        ticks.append((ts, ask_i / point, bid_i / point,
                       round(av_raw, 6), round(bv_raw, 6)))
    return ticks


def _fetch_hour(symbol: str, hour_dt: datetime, point: int):
    """
    Returns (ticks, error_string_or_None).
    404 → empty ticks, no error (market closed).
    """
    url = _bi5_url(symbol, hour_dt)
    try:
        r = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code == 404:
            return [], None
        r.raise_for_status()
        return _parse_bi5(r.content, hour_dt, point), None
    except Exception as exc:
        return [], str(exc)


# ---------------------------------------------------------------------------
# Queue-based parallel engine
# ---------------------------------------------------------------------------

class HourTask:
    """One unit of work: a single hour to download, with retry tracking."""
    __slots__ = ("symbol", "hour_dt", "point", "attempt")

    def __init__(self, symbol, hour_dt, point, attempt=0):
        self.symbol   = symbol
        self.hour_dt  = hour_dt
        self.point    = point
        self.attempt  = attempt

    # PriorityQueue sorts by the first element; use datetime as sort key
    def __lt__(self, other):
        return self.hour_dt < other.hour_dt


class DownloadEngine:
    """
    Shared sorted queue of HourTask items.
    Workers pull tasks, attempt download, requeue on transient failure,
    discard after MAX_RETRIES permanent failures.

    Per-day tick buffers are flushed to CSV once all hours for that day
    are resolved (success or permanently failed).
    """

    def __init__(self, symbol: str, days: list):
        self.symbol      = symbol
        self.point       = INSTRUMENTS[symbol][0]
        self.days        = sorted(days)          # list of date-only datetimes
        self.total_hours = len(days) * 24

        # PriorityQueue gives us sorted order for free
        self._q           = queue.PriorityQueue()
        self._lock        = threading.Lock()

        # per-day state  {date_key: {"ticks": [], "done": int, "failed": []}}
        self._day_state   = {
            d.date(): {"ticks": [], "done": 0, "failed": []}
            for d in days
        }

        # progress counters (read by display thread, written under _lock)
        self.hours_done   = 0        # resolved (success or perm-failed)
        self.hours_ok     = 0        # successful (including empty 404s)
        self.hours_perm_failed = 0
        self.days_done    = 0
        self.current_day  = days[0] if days else None  # earliest in-flight day

        # results
        self.perm_failed_hours = []  # list of hour_dt that exhausted retries

        # seed the queue
        for d in days:
            for h in range(24):
                hour_dt = d.replace(hour=h)
                self._q.put(HourTask(symbol, hour_dt, self.point, attempt=0))

    # ------------------------------------------------------------------
    def _resolve_hour(self, task: HourTask, ticks: list, error):
        """
        Called after each attempt. Updates day buffer; flushes day to CSV
        when all 24 hours are resolved.
        """
        dk = task.hour_dt.date()
        with self._lock:
            ds = self._day_state[dk]
            ds["ticks"].extend(ticks)
            ds["done"] += 1
            if error:
                ds["failed"].append(task.hour_dt)
            self.hours_done += 1
            if ticks or not error:
                self.hours_ok += 1

            # update current_day for display: earliest day still in flight
            in_flight = [d for d, s in self._day_state.items() if s["done"] < 24]
            self.current_day = (
                datetime.combine(min(in_flight), datetime.min.time())
                .replace(tzinfo=timezone.utc)
                if in_flight else None
            )

            # flush day if all 24 hours resolved
            if ds["done"] == 24:
                self.days_done += 1
                self._flush_day(dk, ds)

    def _flush_day(self, dk, ds):
        """Write day's ticks to CSV. Called under _lock."""
        day_dt = datetime(dk.year, dk.month, dk.day, tzinfo=timezone.utc)
        ticks  = sorted(ds["ticks"], key=lambda t: t[0])
        fp     = output_path(self.symbol, day_dt)
        write_csv(ticks, fp)

    # ------------------------------------------------------------------
    def worker(self):
        while True:
            try:
                task = self._q.get(timeout=2)
            except queue.Empty:
                # queue empty and all hours accounted for → done
                if self.hours_done + self.hours_perm_failed >= self.total_hours:
                    break
                continue

            ticks, error = _fetch_hour(task.symbol, task.hour_dt, task.point)

            if error and task.attempt < MAX_RETRIES - 1:
                # transient failure — requeue with incremented attempt
                requeued = HourTask(
                    task.symbol, task.hour_dt, task.point,
                    attempt=task.attempt + 1
                )
                self._q.put(requeued)
                self._q.task_done()
                continue

            if error:
                # permanently failed after MAX_RETRIES attempts
                with self._lock:
                    self.hours_perm_failed += 1
                    self.perm_failed_hours.append(task.hour_dt)
                self._resolve_hour(task, [], error=True)
            else:
                self._resolve_hour(task, ticks, error=None)

            self._q.task_done()

    # ------------------------------------------------------------------
    def run(self, start_time: float):
        threads = [threading.Thread(target=self.worker, daemon=True)
                   for _ in range(MAX_WORKERS)]
        for t in threads:
            t.start()

        # display loop runs on main thread
        _display_loop(self, start_time)

        for t in threads:
            t.join()


# ---------------------------------------------------------------------------
# Live progress display
# ---------------------------------------------------------------------------

def _bar(filled: int, total: int, width: int = 24) -> str:
    if total == 0:
        return "░" * width
    n = int(width * filled / total)
    return "█" * n + "░" * (width - n)


def _elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {sec:02d}s"
    return f"{m:02d}m {sec:02d}s"


def _display_loop(engine: DownloadEngine, start_time: float):
    total_days  = len(engine.days)
    total_hours = engine.total_hours
    sym         = engine.symbol

    # move cursor up N lines to overwrite in place
    LINES = 4
    first = True

    while True:
        elapsed = time.time() - start_time

        with engine._lock:
            h_done  = engine.hours_done
            d_done  = engine.days_done
            perm_f  = engine.hours_perm_failed
            cur_day = engine.current_day

        # hours within the current day
        if cur_day:
            dk = cur_day.date()
            with engine._lock:
                h_in_day = engine._day_state[dk]["done"]
        else:
            h_in_day = 24

        day_label = cur_day.strftime("%Y-%m-%d") if cur_day else "finishing…"

        line1 = (f"  Symbol  : {sym}"
                 f"   workers: {MAX_WORKERS}"
                 f"   elapsed: {_elapsed(elapsed)}")
        line2 = (f"  Days    : [{_bar(d_done, total_days)}]"
                 f"  {d_done:3d} / {total_days}")
        line3 = (f"  Hours   : [{_bar(h_done, total_hours)}]"
                 f"  {h_done:4d} / {total_hours}"
                 f"   perm-failed: {perm_f}")
        line4 = (f"  Current : {day_label}"
                 f"   {h_in_day:2d}/24 hours resolved")

        if not first:
            # move cursor up LINES lines
            sys.stdout.write(f"\033[{LINES}A")
        sys.stdout.write(
            f"{line1}\n{line2}\n{line3}\n{line4}\n"
        )
        sys.stdout.flush()
        first = False

        # done?
        if h_done + perm_f >= total_hours:
            break

        time.sleep(0.4)


# ---------------------------------------------------------------------------
# CSV writer & path helper
# ---------------------------------------------------------------------------

def write_csv(ticks: list, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("timestamp,ask,bid,ask_volume,bid_volume\n")
        for ts, ask, bid, av, bv in ticks:
            f.write(
                f"{ts.strftime('%Y-%m-%d %H:%M:%S.%f')},"
                f"{ask},{bid},{av},{bv}\n"
            )


def output_path(symbol: str, day: datetime) -> Path:
    month_folder = day.strftime("%Y_%m")
    filename = FILENAME_TEMPLATE.format(
        symbol=symbol,
        date=day.strftime("%Y-%m-%d"),
        start=day.strftime("%Y%m%d"),
        end=day.strftime("%Y%m%d"),
    )
    return Path(OUTPUT_DIR) / "raw" / symbol / month_folder / filename


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------

def show_timeframes() -> None:
    print("\nAvailable timeframes (informational — this script downloads raw ticks):")
    for tf, desc in TIMEFRAMES:
        print(f"  {tf:6s}  {desc}")
    print("  Bar timeframes are built by aggregating ticks locally.\n")


def choose_ticker() -> str:
    symbols = list(INSTRUMENTS.keys())
    print("\n" + "=" * 58)
    print("  MENU 1 -- choose a ticker")
    print("=" * 58)
    for i, sym in enumerate(symbols, 1):
        print(f"  {i:2d}. {sym:10s} {INSTRUMENTS[sym][1]}")
    print("-" * 58)
    while True:
        choice = input(f"  Select [1-{len(symbols)}] (default 1=XAUUSD): ").strip()
        if choice == "":
            return symbols[0]
        if choice.isdigit() and 1 <= int(choice) <= len(symbols):
            return symbols[int(choice) - 1]
        if choice.upper() in INSTRUMENTS:
            return choice.upper()
        print("  Invalid choice, try again.")


def choose_mode() -> str:
    print("\n" + "=" * 58)
    print("  MENU 2 -- download mode")
    print("=" * 58)
    print("  1. Single day   (YYYY-MM-DD)")
    print("  2. Full month   (YYYY-MM)")
    print("-" * 58)
    while True:
        choice = input("  Select [1/2] (default 1): ").strip()
        if choice in ("", "1"):
            return "day"
        if choice == "2":
            return "month"
        print("  Invalid choice. Enter 1 or 2.")


def choose_date() -> datetime:
    print("\n  Format : YYYY-MM-DD   e.g. 2024-03-15")
    print("  Note   : UTC; weekends usually have no data.")
    while True:
        raw = input("  Enter date: ").strip()
        try:
            return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("  Invalid — use YYYY-MM-DD")


def choose_month() -> list:
    """Returns list of all day datetimes in the chosen month."""
    print("\n  Format : YYYY-MM   e.g. 2024-03")
    while True:
        raw = input("  Enter month: ").strip()
        try:
            first = datetime.strptime(raw, "%Y-%m").replace(
                day=1, tzinfo=timezone.utc
            )
            _, n_days = calendar.monthrange(first.year, first.month)
            return [first.replace(day=d) for d in range(1, n_days + 1)]
        except ValueError:
            print("  Invalid — use YYYY-MM")


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run(symbol: str, days: list) -> None:
    """Fire up the engine for any list of days."""
    Path(OUTPUT_DIR, "processed").mkdir(parents=True, exist_ok=True)

    print(f"\n  Queueing {len(days)} day(s) × 24 hours "
          f"= {len(days) * 24} tasks   "
          f"({MAX_WORKERS} workers, max {MAX_RETRIES} retries)\n")

    start = time.time()
    engine = DownloadEngine(symbol, days)
    engine.run(start)

    # final summary
    elapsed = time.time() - start
    print(f"\n\n{'=' * 58}")
    print(f"  Done — {symbol}   {_elapsed(elapsed)}")
    print(f"  Days completed     : {engine.days_done} / {len(days)}")
    print(f"  Hours resolved     : {engine.hours_ok} / {engine.total_hours}")
    if engine.perm_failed_hours:
        print(f"  Permanently failed : {engine.hours_perm_failed} hour(s)")
        for hdt in sorted(engine.perm_failed_hours):
            print(f"    ✗  {hdt:%Y-%m-%d %H:%M}")
    else:
        print("  Permanently failed : 0  ✓")
    print(f"  Output             : {(Path(OUTPUT_DIR) / 'raw' / symbol).resolve()}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#" * 58)
    print("#  Dukascopy raw tick data downloader")
    print(f"#  workers : {MAX_WORKERS}  ({os.cpu_count()} cores × 70%)")
    print(f"#  retries : {MAX_RETRIES} per hour")
    print(f"#  output  : {(Path(OUTPUT_DIR) / 'raw').resolve()}")
    print("#" * 58)

    show_timeframes()

    symbol = choose_ticker()
    mode   = choose_mode()

    if mode == "day":
        day = choose_date()
        run(symbol, [day])

    else:
        days = choose_month()
        print(f"\n  Will download: {days[0]:%B %Y}  "
              f"({days[0]:%Y-%m-%d} → {days[-1]:%Y-%m-%d}, "
              f"{len(days)} days)")
        confirm = input("  Proceed? [Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("  Aborted.")
            return
        run(symbol, days)

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")