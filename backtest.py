#!/usr/bin/env python3
"""
H1 Trend Following Backtest — interactive menu
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters (tune these)
# ---------------------------------------------------------------------------
EMA_FAST      = 20      # fast EMA period (H1 bars)
EMA_SLOW      = 50      # slow EMA period (H1 bars)
SL_PIPS       = 200     # stop loss in pips  (1 pip XAUUSD = $0.10 = 0.10 price)
TP_PIPS       = 400     # take profit in pips
PIP_SIZE      = 0.10    # XAUUSD: 1 pip = $0.10
LOT_SIZE      = 1.0     # lots per trade (1 lot XAUUSD = 100 oz)
PIP_VALUE     = 10.0    # $ per pip per lot (100 oz × $0.10)
SPREAD_PIPS   = 2       # assumed constant spread in pips
COMMISSION    = 7.0     # $ round-trip per lot

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ticks(data_dir: Path, symbol: str,
               date_from: datetime | None,
               date_to:   datetime | None) -> pd.DataFrame:
    """Load all parquet files for symbol, return sorted tick dataframe."""
    raw_dir = data_dir / "raw" / symbol
    if not raw_dir.exists():
        print(f"  ERROR: data folder not found: {raw_dir}")
        sys.exit(1)

    files = sorted(raw_dir.rglob("*.parquet"))
    if not files:
        print(f"  ERROR: no .parquet files found under {raw_dir}")
        sys.exit(1)

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)

    ticks = pd.concat(frames, ignore_index=True)

    # timestamp may be int (ms epoch) or datetime — normalise
    if pd.api.types.is_integer_dtype(ticks["timestamp"]):
        ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], unit="ms", utc=True)
    else:
        ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)

    ticks.sort_values("timestamp", inplace=True)
    ticks.reset_index(drop=True, inplace=True)

    if date_from:
        ticks = ticks[ticks["timestamp"] >= pd.Timestamp(date_from)]
    if date_to:
        ticks = ticks[ticks["timestamp"] <= pd.Timestamp(date_to)]

    print(f"  Loaded {len(ticks):,} ticks | "
          f"{ticks['timestamp'].iloc[0].date()} → "
          f"{ticks['timestamp'].iloc[-1].date()}")
    return ticks


# ---------------------------------------------------------------------------
# Resample ticks → H1 OHLC
# ---------------------------------------------------------------------------

def build_h1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Resample tick mid prices into H1 OHLC bars."""
    ticks = ticks.copy()
    ticks["mid"] = (ticks["ask"] + ticks["bid"]) / 2.0
    ticks = ticks.set_index("timestamp")

    h1 = ticks["mid"].resample("1h").ohlc()
    h1.dropna(inplace=True)
    h1.columns = ["open", "high", "low", "close"]

    # also track volume
    vol = ticks["ask_volume"].resample("1h").sum()
    h1["volume"] = vol

    print(f"  Built {len(h1):,} H1 bars")
    return h1


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def add_indicators(h1: pd.DataFrame) -> pd.DataFrame:
    h1 = h1.copy()
    h1["ema_fast"] = h1["close"].ewm(span=EMA_FAST, adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # trend: +1 = bullish (fast > slow), -1 = bearish
    h1["trend"] = np.where(h1["ema_fast"] > h1["ema_slow"], 1, -1)

    # crossover: +1 = bullish cross, -1 = bearish cross
    prev_trend = h1["trend"].shift(1)
    h1["cross"] = np.where(
        (h1["trend"] == 1)  & (prev_trend == -1),  1,
        np.where(
        (h1["trend"] == -1) & (prev_trend ==  1), -1, 0)
    )
    return h1


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class Trade:
    __slots__ = ("entry_time", "direction", "entry_price",
                 "sl", "tp", "exit_time", "exit_price", "exit_reason", "pnl")

    def __init__(self, entry_time, direction, entry_price, sl, tp):
        self.entry_time  = entry_time
        self.direction   = direction   # +1 long, -1 short
        self.entry_price = entry_price
        self.sl          = sl
        self.tp          = tp
        self.exit_time   = None
        self.exit_price  = None
        self.exit_reason = None
        self.pnl         = None


def run_backtest(h1: pd.DataFrame) -> list[Trade]:
    """
    Bar-by-bar backtest.

    Entry rule  : On a bullish/bearish EMA cross, enter at next bar open.
    Exit rule   : SL or TP hit intrabar (check high/low), else exit on
                  opposite cross at next open.
    One trade at a time.
    """
    trades: list[Trade] = []
    active: Trade | None = None

    bars = h1.reset_index()   # columns: timestamp, open, high, low, close, ...

    for i in range(1, len(bars)):
        bar = bars.iloc[i]

        # ── check if active trade is stopped out or hits TP this bar ───────
        if active is not None:
            hi = bar["high"]
            lo = bar["low"]
            op = bar["open"]

            hit_sl = (active.direction ==  1 and lo <= active.sl) or \
                     (active.direction == -1 and hi >= active.sl)
            hit_tp = (active.direction ==  1 and hi >= active.tp) or \
                     (active.direction == -1 and lo <= active.tp)

            # gap-open past SL
            if active.direction == 1 and op < active.sl:
                active.exit_price  = op
                active.exit_reason = "SL-gap"
                hit_sl = True
            elif active.direction == -1 and op > active.sl:
                active.exit_price  = op
                active.exit_reason = "SL-gap"
                hit_sl = True

            if hit_tp and not hit_sl:
                active.exit_price  = active.tp
                active.exit_reason = "TP"
                active.exit_time   = bar["timestamp"]
                _close(active, trades)
                active = None
            elif hit_sl:
                if active.exit_price is None:
                    active.exit_price = active.sl
                active.exit_reason = active.exit_reason or "SL"
                active.exit_time   = bar["timestamp"]
                _close(active, trades)
                active = None
            elif bar["cross"] != 0 and bar["cross"] != active.direction:
                # opposite cross → exit at this bar's open
                active.exit_price  = op
                active.exit_reason = "cross-exit"
                active.exit_time   = bar["timestamp"]
                _close(active, trades)
                active = None

        # ── check for new entry signal ───────────────────────────────────────
        if active is None and bar["cross"] != 0:
            direction = int(bar["cross"])
            ep = bar["open"]   # enter at open of signal bar

            if direction == 1:   # long
                entry = ep + SPREAD_PIPS * PIP_SIZE   # pay spread
                sl    = entry - SL_PIPS * PIP_SIZE
                tp    = entry + TP_PIPS * PIP_SIZE
            else:                # short
                entry = ep                             # sell at bid (no extra)
                sl    = entry + SL_PIPS * PIP_SIZE
                tp    = entry - TP_PIPS * PIP_SIZE

            active = Trade(bar["timestamp"], direction, entry, sl, tp)

    # close any open trade at last bar
    if active is not None:
        last = bars.iloc[-1]
        active.exit_price  = last["close"]
        active.exit_reason = "end-of-data"
        active.exit_time   = last["timestamp"]
        _close(active, trades)

    return trades


def _close(t: Trade, trades: list):
    price_diff = (t.exit_price - t.entry_price) * t.direction
    pips       = price_diff / PIP_SIZE
    t.pnl      = pips * PIP_VALUE * LOT_SIZE - COMMISSION
    trades.append(t)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(trades: list[Trade], h1: pd.DataFrame):
    if not trades:
        print("  No trades executed.")
        return

    pnls      = np.array([t.pnl for t in trades])
    wins      = pnls[pnls > 0]
    losses    = pnls[pnls <= 0]
    equity    = np.cumsum(pnls)

    # drawdown
    peak      = np.maximum.accumulate(equity)
    dd        = equity - peak
    max_dd    = dd.min()

    # profit factor
    gross_profit = wins.sum()   if len(wins)   else 0.0
    gross_loss   = abs(losses.sum()) if len(losses) else 1.0
    pf           = gross_profit / gross_loss if gross_loss else float("inf")

    # Sharpe (annualised from H1 returns — rough estimate)
    if len(pnls) > 1:
        sharpe = (pnls.mean() / (pnls.std() + 1e-9)) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0

    # expectancy per trade
    exp = pnls.mean()

    win_rate = len(wins) / len(trades) * 100

    # exit breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    bar_span = (h1.index[-1] - h1.index[0]).days

    w  = 44
    SEP = "─" * w
    print()
    print(f"  Period         : {h1.index[0].date()} → {h1.index[-1].date()} ({bar_span}d)")
    print(f"  H1 bars        : {len(h1):,}")
    print(f"  EMA            : {EMA_FAST} / {EMA_SLOW}")
    print(f"  SL / TP        : {SL_PIPS} / {TP_PIPS} pips")
    print(SEP)
    print(f"  Total trades   : {len(trades)}")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  Profit factor  : {pf:.2f}")
    print(f"  Expectancy     : ${exp:,.2f} / trade")
    print(SEP)
    print(f"  Net P&L        : ${pnls.sum():>10,.2f}")
    print(f"  Gross profit   : ${gross_profit:>10,.2f}")
    print(f"  Gross loss     : ${-gross_loss:>10,.2f}")
    print(f"  Max drawdown   : ${max_dd:>10,.2f}")
    print(f"  Sharpe (est.)  : {sharpe:.2f}")
    print(SEP)
    print(f"  Exit reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<16} : {count}")
    print()

    # per-trade table (last 20)
    print(f"  Last {min(20, len(trades))} trades:")
    print(f"  {'Entry':19} {'Dir':5} {'Entry$':>8} {'Exit$':>8} "
          f"{'Reason':<14} {'P&L':>9}")
    print(f"  {'-'*19} {'-'*5} {'-'*8} {'-'*8} {'-'*14} {'-'*9}")
    for t in trades[-20:]:
        d_str = "LONG " if t.direction == 1 else "SHORT"
        print(f"  {str(t.entry_time)[:19]} {d_str} "
              f"{t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.exit_reason:<14} {t.pnl:>9.2f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def ask_date(prompt: str) -> datetime | None:
    while True:
        raw = ask(prompt, "all")
        if raw.lower() == "all":
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("  ✗  Invalid format — use YYYY-MM-DD or type 'all'")


def discover_symbols(data_dir: Path) -> list[str]:
    raw = data_dir / "raw"
    if not raw.exists():
        return []
    return sorted(p.name for p in raw.iterdir() if p.is_dir())


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("  ✗  Enter a whole number")


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print("  ✗  Enter a number")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def main():
    global EMA_FAST, EMA_SLOW, SL_PIPS, TP_PIPS, LOT_SIZE, SPREAD_PIPS, COMMISSION

    print()
    # ── data directory ────────────────────────────────────────────────────────
    data_dir = Path(ask("  Data directory", "data"))
    if not data_dir.exists():
        print(f"  ✗  Directory not found: {data_dir}")
        sys.exit(1)

    # ── symbol ────────────────────────────────────────────────────────────────
    symbols = discover_symbols(data_dir)
    if not symbols:
        print(f"  ✗  No symbols found under {data_dir / 'raw'}")
        sys.exit(1)

    print()
    print("  Available symbols:")
    for i, s in enumerate(symbols, 1):
        print(f"    {i}) {s}")
    print()

    while True:
        raw = ask("  Pick symbol (number or name)", symbols[0])
        if raw.isdigit() and 1 <= int(raw) <= len(symbols):
            symbol = symbols[int(raw) - 1]
            break
        if raw.upper() in symbols:
            symbol = raw.upper()
            break
        print("  ✗  Not in the list")

    # ── date range ────────────────────────────────────────────────────────────
    print()
    date_from = ask_date("  From (YYYY-MM-DD or 'all')")
    date_to   = ask_date("  To   (YYYY-MM-DD or 'all')")

    # ── strategy params ───────────────────────────────────────────────────────
    print()
    EMA_FAST   = ask_int  ("  EMA fast period  ", EMA_FAST)
    EMA_SLOW   = ask_int  ("  EMA slow period  ", EMA_SLOW)
    SL_PIPS    = ask_int  ("  Stop loss (pips) ", SL_PIPS)
    TP_PIPS    = ask_int  ("  Take profit (pips)", TP_PIPS)
    LOT_SIZE   = ask_float("  Lot size         ", LOT_SIZE)
    SPREAD_PIPS= ask_int  ("  Spread (pips)    ", SPREAD_PIPS)
    COMMISSION = ask_float("  Commission ($/lot)", COMMISSION)

    print()
    ticks  = load_ticks(data_dir, symbol, date_from, date_to)
    h1     = build_h1(ticks)
    h1     = add_indicators(h1)
    trades = run_backtest(h1)

    print_stats(trades, h1)

    # ── run again? ────────────────────────────────────────────────────────────
    print()
    again = ask("  Run again with different params? (y/n)", "n")
    if again.lower() == "y":
        main()


if __name__ == "__main__":
    main()