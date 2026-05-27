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
import heapq
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

# Per-hour retry limit. Exhausted hours are marked failed and skipped.
# The original MAX_ATTEMPTS (network-level retries inside fetch_hour) is
# kept at 2 so the queue-level MAX_RETRIES controls the visible retry count.
MAX_ATTEMPTS    = 2       # low-level retries inside a single fetch_hour call
MAX_RETRIES     = 3       # queue-level requeues before an hour is abandoned
RETRY_BACKOFF   = 3       # base backoff seconds; grows each queue-level retry

MAX_WORKERS     = max(1, int(os.cpu_count() * 0.7))  # 70% of available cores
# --------------------------------------------------------------------------

# A selection of commonly used Dukascopy instruments.
# 'point' is the divisor that turns the integer price into a real price.
# 'type' drives the weekend/holiday skip logic (see dead_hours below).
# This is not the full Dukascopy catalogue (which has thousands of symbols),
# but covers the popular ones. Add more as you need them.
INSTRUMENTS = {
    # symbol            point     description                  type
    "XAUUSD":     (1000,    "Gold vs US Dollar",         "metals"),
    "XAGUSD":     (1000,    "Silver vs US Dollar",       "metals"),
    "EURUSD":     (100000,  "Euro vs US Dollar",         "fx"),
    "GBPUSD":     (100000,  "British Pound vs US Dollar","fx"),
    "USDJPY":     (1000,    "US Dollar vs Japanese Yen", "fx"),
    "USDCHF":     (100000,  "US Dollar vs Swiss Franc",  "fx"),
    "AUDUSD":     (100000,  "Australian Dollar vs USD",  "fx"),
    "USDCAD":     (100000,  "US Dollar vs Canadian Dollar","fx"),
    "NZDUSD":     (100000,  "New Zealand Dollar vs USD", "fx"),
    "EURGBP":     (100000,  "Euro vs British Pound",     "fx"),
    "EURJPY":     (1000,    "Euro vs Japanese Yen",      "fx"),
    "GBPJPY":     (1000,    "British Pound vs Japanese Yen","fx"),
    "BTCUSD":     (1000,    "Bitcoin vs US Dollar",      "crypto"),
    "ETHUSD":     (1000,    "Ethereum vs US Dollar",     "crypto"),
    "USA500IDXUSD": (1000,  "S&P 500 Index",             "index_us"),
    "USATECHIDXUSD": (1000, "Nasdaq 100 Index",          "index_us"),
    "DEUIDXEUR":  (1000,    "DAX 40 Index",              "index_eu"),
    "USDOLLARIDXUSD": (1000,"US Dollar Index",           "fx"),
    "LIGHTCMDUSD": (1000,   "WTI Crude Oil",             "metals"),
    "BRENTCMDUSD": (1000,   "Brent Crude Oil",           "metals"),
}


def dead_hours(symbol: str, day: datetime) -> set:
    """
    Return the set of hours (0-23 UTC) that are guaranteed to have no data
    for this symbol on this day. These are skipped without hitting the network.

    Philosophy: conservative. Only skip hours that are *always* dead.
    When in doubt, let Dukascopy's 404 be the answer.

    Schedule reference (all times UTC):
      FX / metals / oil : Sun 22:00 open → Fri 22:00 close
      US indices        : Mon-Fri 13:30-20:00 (core); wider pre/post ~11:00-21:00
      EU indices        : Mon-Fri 07:00-21:00 roughly
      Crypto            : 24/7, never skip
    """
    mtype   = INSTRUMENTS[symbol][2]
    weekday = day.weekday()   # 0=Mon … 6=Sun

    # ── crypto: never skip ───────────────────────────────────────────────────
    if mtype == "crypto":
        return set()

    # ── Saturday: always dead for everything non-crypto ──────────────────────
    if weekday == 5:   # Saturday
        return set(range(24))

    # ── FX / metals / oil ────────────────────────────────────────────────────
    if mtype in ("fx", "metals"):
        # Sunday: only 22:00-23:00 UTC is live; hours 00-21 are dead
        if weekday == 6:
            return set(range(0, 22))
        # Friday: market closes at 22:00 UTC; hour 22-23 is dead
        if weekday == 4:
            return {22, 23}
        # Mon-Thu: fully live
        return set()

    # ── US indices (S&P, Nasdaq) ──────────────────────────────────────────────
    # Core session 13:30-20:00 UTC; pre-market from ~11:00; after ~21:00 dead.
    # Be conservative: only skip the clearly-dead overnight hours.
    if mtype == "index_us":
        if weekday == 6:   # Sunday: dead all day
            return set(range(24))
        # Mon-Fri: skip 21:00-10:00 UTC (overnight dead zone, 13 hours)
        dead = set(range(0, 10)) | {21, 22, 23}
        return dead

    # ── EU indices (DAX etc.) ─────────────────────────────────────────────────
    # Roughly 07:00-21:00 UTC Mon-Fri.
    if mtype == "index_eu":
        if weekday == 6:
            return set(range(24))
        dead = set(range(0, 7)) | set(range(21, 24))
        return dead

    return set()   # unknown type: don't skip anything


def is_weekend_day(symbol: str, day: datetime) -> bool:
    """
    Return True if this entire day is dead for the given symbol.
    Used to pre-filter days before seeding the queue.

    Saturday is always fully dead for non-crypto.
    Sunday is fully dead for index_us; for FX/metals only 22:00-23:00 live
    so we don't skip the whole day at the day level.
    """
    mtype   = INSTRUMENTS[symbol][2]
    weekday = day.weekday()

    if mtype == "crypto":
        return False
    if weekday == 5:
        return True
    if mtype == "index_us" and weekday == 6:
        return True
    if mtype == "index_eu" and weekday == 6:
        return True
    return False


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


def _bar(done: int, total: int, width: int = 28) -> str:
    """Render a compact ASCII progress bar: [████░░░░]  42%"""
    if total == 0:
        pct = 1.0
    else:
        pct = done / total
    filled = round(pct * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:3.0f}%"


# ---------------------------------------------------------------------------
# Priority Queue Engine
# ---------------------------------------------------------------------------

class HourQueue:
    """
    Thread-safe min-heap priority queue for (day, hour) work items.

    Each item is a tuple: (priority_key, day, hour, attempt)
      priority_key = (day_ordinal, hour, attempt)   — sorts chronologically,
      retries sort after fresh items for the same slot.
    """

    def __init__(self):
        self._heap  = []
        self._lock  = threading.Lock()
        self._cv    = threading.Condition(self._lock)
        self._done  = False

    def push(self, day: datetime, hour: int, attempt: int = 0):
        key = (day.toordinal(), hour, attempt)
        with self._cv:
            heapq.heappush(self._heap, (key, day, hour, attempt))
            self._cv.notify()

    def pop(self):
        """
        Block until an item is available or the queue is marked done.
        Returns (day, hour, attempt) or None when drained and done.
        """
        with self._cv:
            while not self._heap:
                if self._done:
                    return None
                self._cv.wait(timeout=0.2)
            _, day, hour, attempt = heapq.heappop(self._heap)
            return day, hour, attempt

    def mark_done(self):
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def __len__(self):
        with self._lock:
            return len(self._heap)


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------

class LiveDisplay:
    """
    Draws a live in-place status board. Bars shown depend on mode:

      Year mode  : Months / Days / Hours  (all epoch-style, reset per level)
      Month mode : Days / Hours
      Day mode   : Hours only

    The board is redrawn in-place using ANSI cursor-up escapes.
    """

    ICON = {
        "idle":        "   ",
        "connecting":  "↓  ",
        "downloading": "↓  ",
        "retrying":    "⟳  ",
        "done":        "✓  ",
        "failed":      "✗  ",
        "waiting":     "…  ",
    }

    def __init__(self, symbol: str, n_workers: int,
                 total_days: int, total_hours: int,
                 label: str = "",
                 total_months: int = 0):
        self.symbol        = symbol
        self.n_workers     = n_workers
        self.total_days    = total_days
        self.total_hours   = total_hours
        self.total_months  = total_months  # >0 only in year mode
        self.label         = label

        self._n_lines = 5 + n_workers
        self._first   = True
        self._lock    = threading.Lock()

    def draw(self, slots: dict,
             done_hours: int, done_days: int,
             total_ticks: int, elapsed: float,
             done_months: int = 0):
        """Render and print the board, overwriting the previous frame."""
        lines = self._build(slots, done_hours, done_days,
                            total_ticks, elapsed, done_months)
        with self._lock:
            if not self._first:
                sys.stdout.write(f"\033[{self._n_lines}A")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            self._first = False

    def _build(self, slots: dict,
               done_hours: int, done_days: int,
               total_ticks: int, elapsed: float,
               done_months: int = 0) -> list[str]:

        W = 62

        def box(text: str) -> str:
            return f"  {text:<{W}}"

        lines = []

        # ── title ────────────────────────────────────────────────────────────
        title = (f"{self.symbol}  {self.label}"
                 f"  elapsed: {_fmt_elapsed(elapsed)}"
                 f"  ticks: {total_ticks:,}")
        lines.append(box(title))
        lines.append("  " + "─" * W)

        # ── months bar (year mode only) ───────────────────────────────────────
        if self.total_months > 0:
            mb = _bar(done_months, self.total_months, width=24)
            lines.append(box(f"  Months {mb}  {done_months}/{self.total_months}"))

        # ── days bar (multi-day runs) ─────────────────────────────────────────
        if self.total_days > 1:
            db = _bar(done_days, self.total_days, width=24)
            lines.append(box(f"  Days   {db}  {done_days}/{self.total_days}"))

        # ── hours bar ────────────────────────────────────────────────────────
        hb = _bar(done_hours, self.total_hours, width=24)
        lines.append(box(f"  Hours  {hb}  {done_hours}/{self.total_hours}"))
        lines.append("  " + "─" * W)

        # recompute line count based on which bars are visible
        n_bars = 1  # hours always shown
        if self.total_days > 1:
            n_bars += 1
        if self.total_months > 0:
            n_bars += 1
        self._n_lines = 2 + n_bars + 1 + self.n_workers  # title+sep + bars + sep + workers

        # ── one line per worker ───────────────────────────────────────────────
        for i in range(self.n_workers):
            s = slots.get(i)
            if s is None or s["state"] == "idle":
                lines.append(box(f"  worker {i+1:2d}  –  idle"))
                continue

            icon  = self.ICON.get(s["state"], "   ")
            state = s["state"]
            day_s = s["day"].strftime("%Y-%m-%d") if s.get("day") else "???"
            hr    = s.get("hour", 0)
            att   = s.get("attempt", 1)

            if state == "done":
                tc = s.get("ticks", 0)
                lines.append(box(
                    f"  worker {i+1:2d}  {icon} {day_s} {hr:02d}:00"
                    f"  {tc:>6,} ticks"
                ))
            elif state == "failed":
                lines.append(box(
                    f"  worker {i+1:2d}  {icon} {day_s} {hr:02d}:00"
                    f"  FAILED (gave up after {MAX_RETRIES} attempts)"
                ))
            elif state == "waiting":
                waited = int(time.monotonic() - s["since"])
                lines.append(box(
                    f"  worker {i+1:2d}  {icon} {day_s} {hr:02d}:00"
                    f"  backing off  ({waited}s)"
                ))
            elif state == "retrying":
                lines.append(box(
                    f"  worker {i+1:2d}  {icon} {day_s} {hr:02d}:00"
                    f"  attempt {att}/{MAX_RETRIES}"
                ))
            else:
                lines.append(box(
                    f"  worker {i+1:2d}  {icon} {day_s} {hr:02d}:00"
                    f"  {state}…  attempt {att}/{MAX_RETRIES}"
                ))

        return lines


# ---------------------------------------------------------------------------
# Queue-driven downloader (replaces download_day for multi-day runs too)
# ---------------------------------------------------------------------------

def _seed_queue(queue: HourQueue, symbol: str,
                days: list[datetime]) -> tuple[int, int]:
    """
    Push all live (day, hour) pairs onto the queue, weekends already excluded.

    Returns (n_days_queued, n_hours_queued).
    """
    n_days  = 0
    n_hours = 0
    for day in days:
        skip = dead_hours(symbol, day)
        live = [h for h in range(24) if h not in skip]
        if live:
            n_days += 1
            n_hours += len(live)
            for h in live:
                queue.push(day, h, attempt=0)
    return n_days, n_hours


def download_queued(symbol: str, days: list[datetime],
                    label: str = "",
                    total_months: int = 0,
                    done_months: int = 0) -> tuple[list, list, dict]:
    """
    Download all (day, hour) pairs in *days* using a shared priority queue.

    Workers pull independently — no waiting for a full day to finish.
    Failed hours are requeued in sorted order with an incremented attempt
    counter; once attempt >= MAX_RETRIES the hour is abandoned.

    total_months / done_months: passed in by year mode to show the months bar.

    Returns:
      all_ticks    : list of tick tuples, in chronological order
      failed_hours : list of (day, hour) pairs that were abandoned
      hour_status  : dict of {(day, hour) -> "ok" | "empty" | "failed"}
    """
    start        = time.monotonic()
    lock         = threading.Lock()
    done_months  = done_months   # shadow param so display loop can read it

    # ── shared accumulators ──────────────────────────────────────────────────
    results      = {}          # (day, hour) -> list | EMPTY_HOUR | Exception
    tick_count   = [0]
    done_hours   = [0]
    done_days    = [0]         # days where *all* live hours finished
    day_pending  = {}          # day -> remaining live-hour count

    # ── slots: worker_id -> state dict ──────────────────────────────────────
    slots         = {i: {"state": "idle"} for i in range(MAX_WORKERS)}
    worker_ids    = {}         # thread_ident -> slot index

    # ── seed the queue ───────────────────────────────────────────────────────
    queue           = HourQueue()
    n_days_q, n_hours_q = _seed_queue(queue, symbol, days)

    # pre-count live hours per day for day-completion tracking
    for day in days:
        skip = dead_hours(symbol, day)
        live = len([h for h in range(24) if h not in skip])
        if live:
            day_pending[day] = live

    if not label:
        if len(days) == 1:
            label = days[0].strftime("%Y-%m-%d")
        else:
            label = f"{days[0]:%Y-%m-%d} → {days[-1]:%Y-%m-%d}"

    display = LiveDisplay(
        symbol        = symbol,
        n_workers     = MAX_WORKERS,
        total_days    = n_days_q,
        total_hours   = n_hours_q,
        label         = label,
        total_months  = total_months,
    )

    # ── worker function ──────────────────────────────────────────────────────
    def worker_loop():
        tid = threading.get_ident()

        # Claim a slot index for this worker
        with lock:
            # assign next free slot
            used = set(worker_ids.values())
            slot = next(i for i in range(MAX_WORKERS) if i not in used)
            worker_ids[tid] = slot

        while True:
            item = queue.pop()
            if item is None:
                break
            day, hour, attempt = item
            dt = day.replace(hour=hour, minute=0, second=0, microsecond=0)

            # ── backoff for retries ──────────────────────────────────────────
            if attempt > 0:
                backoff = RETRY_BACKOFF * attempt
                with lock:
                    slots[slot] = {
                        "state": "waiting",
                        "day":   day,
                        "hour":  hour,
                        "attempt": attempt,
                        "since": time.monotonic(),
                    }
                time.sleep(backoff)

            # ── mark connecting ──────────────────────────────────────────────
            with lock:
                slots[slot] = {
                    "state":   "connecting",
                    "day":     day,
                    "hour":    hour,
                    "attempt": attempt + 1,
                    "since":   time.monotonic(),
                }

            def on_retry(a, ma):
                with lock:
                    slots[slot]["state"] = "retrying"
                    slots[slot]["since"] = time.monotonic()

            def on_download():
                with lock:
                    slots[slot]["state"] = "downloading"
                    slots[slot]["since"] = time.monotonic()

            # ── fetch ────────────────────────────────────────────────────────
            try:
                outcome = fetch_hour(symbol, dt,
                                     on_retry=on_retry,
                                     on_download=on_download)
                n = 0 if outcome is EMPTY_HOUR else len(outcome)

                with lock:
                    results[(day, hour)] = outcome
                    tick_count[0]       += n
                    done_hours[0]       += 1
                    slots[slot] = {
                        "state":   "done",
                        "day":     day,
                        "hour":    hour,
                        "attempt": attempt + 1,
                        "ticks":   n,
                    }
                    # check day completion
                    day_pending[day] -= 1
                    if day_pending[day] == 0:
                        done_days[0] += 1

            except Exception:
                if attempt + 1 < MAX_RETRIES:
                    # requeue with incremented attempt (stays sorted)
                    queue.push(day, hour, attempt + 1)
                    with lock:
                        slots[slot] = {
                            "state":   "retrying",
                            "day":     day,
                            "hour":    hour,
                            "attempt": attempt + 1,
                            "since":   time.monotonic(),
                        }
                else:
                    # give up
                    with lock:
                        results[(day, hour)] = Exception("max retries exceeded")
                        done_hours[0]       += 1
                        slots[slot] = {
                            "state":   "failed",
                            "day":     day,
                            "hour":    hour,
                            "attempt": attempt + 1,
                        }
                        day_pending[day] -= 1
                        if day_pending[day] == 0:
                            done_days[0] += 1

    # ── launch workers and display loop ─────────────────────────────────────
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker_loop) for _ in range(MAX_WORKERS)]

        # signal queue done once all workers have been submitted
        # (queue drains naturally; mark_done lets pop() return None)
        def _drain_watcher():
            # wait until all futures are done, then mark queue done
            for f in futures:
                f.result()
            queue.mark_done()

        # we actually just mark done after submitting; workers will drain
        # and queue.pop() will unblock via the done flag
        # Mark done so that once heap is empty workers exit
        queue.mark_done()

        while True:
            with lock:
                snap   = {k: dict(v) for k, v in slots.items()}
                dh     = done_hours[0]
                dd     = done_days[0]
                ticks  = tick_count[0]
            elapsed = time.monotonic() - start
            display.draw(snap, dh, dd, ticks, elapsed,
                         done_months=done_months)

            # check if all workers finished
            if all(f.done() for f in futures):
                break
            time.sleep(0.4)

    # ── final draw ───────────────────────────────────────────────────────────
    with lock:
        snap  = {k: dict(v) for k, v in slots.items()}
        ticks = tick_count[0]
    elapsed = time.monotonic() - start
    display.draw(snap, n_hours_q, n_days_q, ticks, elapsed,
                 done_months=done_months)

    # ── collect results ──────────────────────────────────────────────────────
    all_ticks    = []
    failed_hours = []
    hour_status  = {}

    # first handle skipped (dead) hours
    for day in days:
        skip = dead_hours(symbol, day)
        for h in skip:
            hour_status[(day, h)] = "empty"

    # then fetched hours, in chronological order
    for day in sorted(day_pending.keys()):
        for hour in range(24):
            key = (day, hour)
            if key not in results:
                continue
            outcome = results[key]
            if isinstance(outcome, Exception):
                failed_hours.append((day, hour))
                hour_status[key] = "failed"
            elif outcome is EMPTY_HOUR:
                hour_status[key] = "empty"
            else:
                all_ticks.extend(outcome)
                hour_status[key] = "ok"

    # sort ticks just in case concurrent workers delivered them out of order
    all_ticks.sort(key=lambda t: t[0])

    return all_ticks, failed_hours, hour_status


# ---------------------------------------------------------------------------
# Single-day shim (keeps the rest of the code unchanged)
# ---------------------------------------------------------------------------

def download_day(symbol: str, day: datetime):
    """
    Download all live hours for a single calendar day using the queue engine.

    Returns the same (all_ticks, failed_hours, hour_status) triple as before,
    but failed_hours is now a plain list of int hours (not (day, hour) pairs)
    and hour_status keys are plain ints, to keep _save_day / write_meta happy.
    """
    all_ticks, failed_pairs, hs_pairs = download_queued(
        symbol, [day], label=day.strftime("%Y-%m-%d")
    )

    failed_hours = [h for (d, h) in failed_pairs]
    hour_status  = {h: status for (d, h), status in hs_pairs.items()}

    return all_ticks, failed_hours, hour_status


# ---------------------------------------------------------------------------
# CSV / meta / integrity
# ---------------------------------------------------------------------------

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
    """Print a compact one-line integrity report for one day."""
    status = "✓" if report["failed"] == 0 else "⚠"
    ok     = report["ok"]
    empty  = report["empty"]
    ticks  = report["total_ticks"]
    date   = report["date"]

    if report["failed"] == 0:
        print(f"  {status}  {date}   {ok:2d}h data  {empty:2d}h empty"
              f"   {ticks:>9,} ticks")
    else:
        failed = report["failed"]
        hrs    = " ".join(f"{h:02d}:00" for h in report["failed_hours"])
        print(f"  {status}  {date}   {ok:2d}h data  {empty:2d}h empty"
              f"   {ticks:>9,} ticks   MISSING {failed}h → {hrs}")


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

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


def _month_complete(symbol: str, year: int, month: int,
                    cutoff: datetime) -> bool:
    """
    Return True if every tradeable day in this month (up to cutoff) already
    has a non-empty CSV on disk. Used by the year downloader to skip months
    that are fully done so we never re-queue them.

    A month is considered complete when:
      - all non-weekend days up to (and including) cutoff exist on disk, AND
      - there is at least one such day (i.e. not an entirely-future month)
    """
    _, n_days = calendar.monthrange(year, month)
    tradeable = [
        datetime(year, month, d, tzinfo=timezone.utc)
        for d in range(1, n_days + 1)
        if not is_weekend_day(
            symbol, datetime(year, month, d, tzinfo=timezone.utc)
        )
        and datetime(year, month, d, tzinfo=timezone.utc) <= cutoff
    ]
    if not tradeable:
        return False   # nothing to check yet (future month)
    return all(_already_exists(symbol, d) for d in tradeable)


def _save_day(symbol: str, day: datetime, ticks: list,
              failed_hours: list, hour_status: dict):
    """Write one day's CSV + meta, then print an integrity report."""
    filepath = _out_path(symbol, day)
    write_csv(ticks, filepath)
    write_meta(filepath, symbol, day, ticks, hour_status)

    report = check_integrity(filepath)
    print_integrity(report)


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
    """Menu 2: single day, full month, or full year."""
    print("\n" + "=" * 55)
    print("  MENU 2 -- download mode")
    print("=" * 55)
    print("  1. Single day   (YYYY-MM-DD)")
    print("  2. Full month   (YYYY-MM)")
    print("  3. Full year    (YYYY)")
    print("-" * 55)

    while True:
        choice = input("Select mode [1-3] (default 1): ").strip()
        if choice in ("", "1"):
            return "day"
        if choice == "2":
            return "month"
        if choice == "3":
            return "year"
        print("  Invalid choice, try again.")


def choose_year() -> int:
    """Menu 3c: enter a year, return int."""
    print("\n" + "=" * 55)
    print("  MENU 3 -- enter year")
    print("=" * 55)
    print("  Format : YYYY   e.g. 2024")
    print("  Note   : all months up to yesterday will be downloaded.")
    print("           Months already fully on disk are skipped.")
    print("-" * 55)

    while True:
        raw = input("Enter year: ").strip()
        if raw.isdigit() and len(raw) == 4:
            y = int(raw)
            if 2000 <= y <= datetime.now().year:
                return y
        print("  Invalid year. Use a 4-digit year, e.g. 2024.")


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#" * 55)
    print("#  Dukascopy raw tick data downloader")
    print(f"#  workers: {MAX_WORKERS}  ({os.cpu_count()} cores × 70%)")
    print(f"#  max retries per hour: {MAX_RETRIES}")
    print("#" * 55)

    show_timeframes()

    symbol = choose_ticker()
    mode   = choose_mode()

    # ── full year ────────────────────────────────────────────────────
    if mode == "year":
        year = choose_year()

        # cutoff = yesterday UTC (today's data not yet complete)
        today_utc = datetime.now(timezone.utc).date()
        cutoff    = datetime(
            today_utc.year, today_utc.month, today_utc.day,
            tzinfo=timezone.utc
        ) - timedelta(days=1)

        # build list of months in this year that have at least one day <= cutoff
        months_in_year = [m for m in range(1, 13)
                          if datetime(year, m, 1, tzinfo=timezone.utc) <= cutoff]

        if not months_in_year:
            print(f"\n  No data available yet for {year} (all months are in the future).")
            return

        # classify each month: fully on disk vs still needs work
        complete_months = []
        pending_months  = []
        for m in months_in_year:
            if _month_complete(symbol, year, m, cutoff):
                complete_months.append(m)
            else:
                pending_months.append(m)

        print(f"\n  {year}  →  {len(months_in_year)} months available"
              f"  (up to {cutoff:%Y-%m-%d})")
        print(f"  Symbol     : {symbol}")
        print(f"  Workers    : {MAX_WORKERS}")
        print(f"  Max retries: {MAX_RETRIES} per hour")
        print(f"  Output     : {Path(OUTPUT_DIR) / 'raw' / symbol}/")
        if complete_months:
            names = ", ".join(datetime(year, m, 1).strftime("%b")
                              for m in complete_months)
            print(f"  Complete (skip) : {len(complete_months)} months  [{names}]")
        print(f"  To download     : {len(pending_months)} months")
        print()

        if not pending_months:
            print("  All months already complete. Nothing to do.")
            return

        confirm = input("  Start download? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

        print()

        year_total_ticks  = 0
        year_total_failed = []

        done_months = [0]   # counter for months bar

        for m in pending_months:
            _, n_days  = calendar.monthrange(year, m)
            month_name = datetime(year, m, 1).strftime("%B %Y")

            # days in this month up to (and including) cutoff
            all_days = [
                datetime(year, m, d, tzinfo=timezone.utc)
                for d in range(1, n_days + 1)
                if datetime(year, m, d, tzinfo=timezone.utc) <= cutoff
            ]

            existing     = [d for d in all_days if _already_exists(symbol, d)]
            weekend_days = [d for d in all_days
                            if is_weekend_day(symbol, d)
                            and not _already_exists(symbol, d)]
            pending_days = [d for d in all_days
                            if not _already_exists(symbol, d)
                            and not is_weekend_day(symbol, d)]

            total_live_hours = sum(
                24 - len(dead_hours(symbol, d)) for d in pending_days
            )

            print("─" * 55)
            print(f"  {month_name}"
                  f"  |  {len(pending_days)} days / {total_live_hours} hours to fetch"
                  f"  |  {len(existing)} already done")
            print()

            if not pending_days:
                print(f"  {month_name}: nothing to fetch, skipping.\n")
                done_months[0] += 1
                continue

            label = f"{year:04d}-{m:02d}"
            all_ticks, failed_pairs, hs_pairs = download_queued(
                symbol, pending_days, label=label,
                total_months=len(pending_months),
                done_months=done_months[0],
            )

            done_months[0] += 1

            print()
            ticks_by_day = {}
            for tick in all_ticks:
                ticks_by_day.setdefault(tick[0].date(), []).append(tick)

            for day in pending_days:
                day_ticks  = ticks_by_day.get(day.date(), [])
                day_hs     = {}
                day_failed = []
                for h in range(24):
                    key    = (day, h)
                    status = hs_pairs.get(key, "empty")
                    day_hs[h] = status
                    if status == "failed":
                        day_failed.append(h)

                if not day_ticks and not day_failed:
                    continue

                _save_day(symbol, day, day_ticks, day_failed, day_hs)
                year_total_ticks += len(day_ticks)
                if day_failed:
                    year_total_failed.append((day, day_failed))

        # ── year summary ──────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"  {year} download complete")
        print("=" * 55)
        print(f"  Total ticks     : {year_total_ticks:,}")
        print(f"  Months fetched  : {len(pending_months)}")
        print(f"  Months skipped  : {len(complete_months)}  (already complete)")

        if year_total_failed:
            print(f"  Days with gaps  : {len(year_total_failed)}")
            for day, hours in year_total_failed:
                hrs = ", ".join(f"{h:02d}:00" for h in hours)
                print(f"    {day:%Y-%m-%d}  missing hours: {hrs}")
            print("\n  Re-run to retry gaps.")
        else:
            print("  All days complete  ✓")
        print()
        return

    # ── single day ────────────────────────────────────────────────────────
    if mode == "day":
        day = choose_date()

        if _already_exists(symbol, day):
            p = _out_path(symbol, day)
            print(f"\n  Already exists: {p}")
            print("  Delete the file first if you want to re-download.")
            return

        if is_weekend_day(symbol, day):
            print(f"\n  {day:%Y-%m-%d} is a weekend/non-trading day for {symbol}.")
            print("  No data will be available.")
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
            print("\n  All hours downloaded successfully.")

    # ── full month ────────────────────────────────────────────────────────
    else:
        year, month = choose_month()
        _, n_days   = calendar.monthrange(year, month)
        all_days    = [
            datetime(year, month, d, tzinfo=timezone.utc)
            for d in range(1, n_days + 1)
        ]

        month_name = datetime(year, month, 1).strftime("%B %Y")

        # pre-flight: check which days already exist / are weekends
        existing     = [d for d in all_days if _already_exists(symbol, d)]
        weekend_days = [d for d in all_days
                        if is_weekend_day(symbol, d)
                        and not _already_exists(symbol, d)]
        pending      = [d for d in all_days
                        if not _already_exists(symbol, d)
                        and not is_weekend_day(symbol, d)]

        # count live hours across pending days
        total_live_hours = sum(
            24 - len(dead_hours(symbol, d)) for d in pending
        )

        print(f"\n  {month_name}  →  {n_days} days  "
              f"({all_days[0]:%Y-%m-%d} to {all_days[-1]:%Y-%m-%d})")
        print(f"  Symbol     : {symbol}")
        print(f"  Workers    : {MAX_WORKERS}")
        print(f"  Max retries: {MAX_RETRIES} per hour")
        print(f"  Output     : "
              f"{Path(OUTPUT_DIR) / 'raw' / symbol / f'{year:04d}_{month:02d}'}/")
        print(f"  Weekend days filtered : {len(weekend_days)}")
        print(f"  To fetch   : {len(pending)} days / {total_live_hours} hours")
        print(f"  Already done: {len(existing)}")
        print()

        if not pending:
            print("  All days already downloaded. Nothing to do.")
            return

        confirm = input("  Start download? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

        print()

        # ── run the unified queue engine across all pending days ──────────
        label = f"{year:04d}-{month:02d}"
        all_ticks, failed_pairs, hs_pairs = download_queued(
            symbol, pending, label=label
        )

        # ── write one CSV per day ─────────────────────────────────────────
        print()
        total_ticks  = 0
        total_failed = []

        # partition ticks by day
        ticks_by_day = {}
        for tick in all_ticks:
            d = tick[0].date()
            ticks_by_day.setdefault(d, []).append(tick)

        for day in pending:
            day_ticks = ticks_by_day.get(day.date(), [])
            # rebuild per-day hour_status (int keys) and failed_hours
            day_hs     = {}
            day_failed = []
            for h in range(24):
                key    = (day, h)
                status = hs_pairs.get(key, "empty")
                day_hs[h] = status
                if status == "failed":
                    day_failed.append(h)

            if not day_ticks and not day_failed:
                continue    # fully empty day (all dead hours)

            _save_day(symbol, day, day_ticks, day_failed, day_hs)
            total_ticks  += len(day_ticks)
            if day_failed:
                total_failed.append((day, day_failed))
            print()

        # ── month summary ──────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print(f"  {month_name} download complete")
        print("=" * 55)
        print(f"  Total ticks     : {total_ticks:,}")
        print(f"  Days downloaded : {len(pending)}")
        print(f"  Days skipped    : {len(weekend_days)}  (weekend / no market)")
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