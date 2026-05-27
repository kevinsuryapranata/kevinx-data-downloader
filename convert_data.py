#!/usr/bin/env python3
"""
convert_data_from_csv_to_parquet.py

Converts Dukascopy tick data files in bulk using parallel workers (70% of cores).

  CSV → Parquet : compress and shrink (~5-10x smaller), delete original CSV
  Parquet → CSV : restore to plain CSV, delete original Parquet

Meta JSON sidecar files are always left untouched.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("  ERROR: pandas is required.  pip install pandas pyarrow")
    sys.exit(1)

try:
    import pyarrow  # noqa: F401
except ImportError:
    print("  ERROR: pyarrow is required.  pip install pyarrow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_ROOT        = "data/raw"
PARQUET_COMPRESSION = "snappy"   # "snappy" = fast r/w; "zstd" = smaller files
MAX_WORKERS         = max(1, int(os.cpu_count() * 0.7))

CSV_DTYPES = {
    "ask":        "float32",
    "bid":        "float32",
    "ask_volume": "float32",
    "bid_volume": "float32",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Single-file converters
# ---------------------------------------------------------------------------

def _to_parquet(src: Path, dry_run: bool) -> tuple[int, int]:
    """
    CSV → Parquet. Returns (src_bytes, dst_bytes).
    dst_bytes == 0 on dry_run.
    """
    src_bytes = src.stat().st_size
    if dry_run:
        return src_bytes, 0

    dst = src.with_suffix(".parquet")
    df  = pd.read_csv(src, parse_dates=["timestamp"], dtype=CSV_DTYPES)
    df.to_parquet(dst, engine="pyarrow",
                  compression=PARQUET_COMPRESSION, index=False)
    dst_bytes = dst.stat().st_size
    src.unlink()
    return src_bytes, dst_bytes


def _to_csv(src: Path, dry_run: bool) -> tuple[int, int]:
    """
    Parquet → CSV. Returns (src_bytes, dst_bytes).
    dst_bytes == 0 on dry_run.
    """
    src_bytes = src.stat().st_size
    if dry_run:
        return src_bytes, 0

    dst = src.with_suffix(".csv")
    df  = pd.read_parquet(src, engine="pyarrow")
    df.to_csv(dst, index=False)
    dst_bytes = dst.stat().st_size
    src.unlink()
    return src_bytes, dst_bytes


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run(root: Path, direction: str, dry_run: bool):
    """
    direction: "csv_to_parquet" | "parquet_to_csv"
    """
    if direction == "csv_to_parquet":
        files    = sorted(root.rglob("*.csv"))
        convert  = _to_parquet
        src_ext  = "CSV"
        dst_ext  = "Parquet"
        arrow    = "→"
    else:
        files    = sorted(root.rglob("*.parquet"))
        convert  = _to_csv
        src_ext  = "Parquet"
        dst_ext  = "CSV"
        arrow    = "→"

    if not files:
        print(f"\n  No {src_ext} files found under {root}")
        print("  Nothing to convert.")
        return

    total_src_bytes = sum(p.stat().st_size for p in files)

    print(f"\n  Root        : {root}")
    print(f"  Direction   : {src_ext} {arrow} {dst_ext}")
    print(f"  Files       : {len(files):,}")
    print(f"  Total size  : {_fmt_size(total_src_bytes)}")
    print(f"  Workers     : {MAX_WORKERS}  ({os.cpu_count()} cores × 70%)")
    if direction == "csv_to_parquet":
        print(f"  Compression : {PARQUET_COMPRESSION}")
    if dry_run:
        print("\n  DRY RUN — no files will be changed")
    print()

    # ── shared counters ──────────────────────────────────────────────────────
    lock            = threading.Lock()
    done            = [0]
    total_src_saved = [0]
    total_dst_saved = [0]
    errors          = []
    print_lock      = threading.Lock()

    def process(path: Path):
        rel = path.relative_to(root)
        try:
            src_b, dst_b = convert(path, dry_run=dry_run)

            with lock:
                done[0]            += 1
                total_src_saved[0] += src_b
                total_dst_saved[0] += dst_b

            with print_lock:
                if dry_run:
                    print(f"  ~  {rel}   {_fmt_size(src_b)}")
                else:
                    if src_b:
                        ratio = abs(1 - dst_b / src_b) * 100
                        sign  = "-" if dst_b < src_b else "+"
                        print(f"  ✓  {rel}"
                              f"   {_fmt_size(src_b)} → {_fmt_size(dst_b)}"
                              f"  ({sign}{ratio:.0f}%)")
                    else:
                        print(f"  ✓  {rel}   (empty)")

        except Exception as exc:
            with lock:
                errors.append((path, exc))
            with print_lock:
                print(f"  ✗  {rel}   ERROR: {exc}")

    # ── run pool ─────────────────────────────────────────────────────────────
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process, f) for f in files]
        for _ in as_completed(futures):
            pass   # results handled inside process()

    elapsed = time.monotonic() - start

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if dry_run:
        print("  DRY RUN complete — nothing was changed")
        print("=" * 55)
        print(f"  Files found : {len(files):,}")
        print(f"  Total size  : {_fmt_size(total_src_saved[0])}")
    else:
        s   = total_src_saved[0]
        d   = total_dst_saved[0]
        diff = s - d
        pct  = (abs(diff) / s * 100) if s else 0
        sign = "-" if diff > 0 else "+"

        print(f"  Conversion complete  ({_fmt_elapsed(elapsed)})")
        print("=" * 55)
        print(f"  Files converted : {done[0]:,} / {len(files):,}")
        print(f"  Before          : {_fmt_size(s)}")
        print(f"  After           : {_fmt_size(d)}")
        print(f"  Difference      : {sign}{_fmt_size(abs(diff))}  ({sign}{pct:.0f}%)")

        if errors:
            print(f"\n  ⚠  {len(errors)} file(s) failed:")
            for path, exc in errors:
                print(f"     {path.relative_to(root)}  →  {exc}")
            print("\n  Those files were NOT deleted. Fix the error and re-run.")
        else:
            print(f"\n  All {src_ext} files converted and removed  ✓")

    print()


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------

def choose_direction() -> str:
    """Menu 1: which direction to convert."""
    print("\n" + "=" * 55)
    print("  MENU 1 -- conversion direction")
    print("=" * 55)
    print("  1. CSV → Parquet   (compress, recommended for storage)")
    print("  2. Parquet → CSV   (restore to plain CSV)")
    print("-" * 55)

    while True:
        choice = input("Select direction [1-2] (default 1): ").strip()
        if choice in ("", "1"):
            return "csv_to_parquet"
        if choice == "2":
            return "parquet_to_csv"
        print("  Invalid choice, try again.")


def choose_mode() -> bool:
    """Menu 2: dry run or real. Returns True = dry run."""
    print("\n" + "=" * 55)
    print("  MENU 2 -- run mode")
    print("=" * 55)
    print("  1. Preview   (dry run, shows files and sizes, nothing changes)")
    print("  2. Convert   (convert and delete originals)")
    print("-" * 55)

    while True:
        choice = input("Select mode [1-2] (default 1): ").strip()
        if choice in ("", "1"):
            return True
        if choice == "2":
            return False
        print("  Invalid choice, try again.")


def choose_root() -> Path:
    """Menu 3: confirm or change the data folder."""
    print("\n" + "=" * 55)
    print("  MENU 3 -- data folder")
    print("=" * 55)
    print(f"  Default: {DEFAULT_ROOT}")
    print("  Press Enter to use default, or type a different path.")
    print("-" * 55)

    while True:
        raw  = input(f"Folder [{DEFAULT_ROOT}]: ").strip()
        root = Path(raw) if raw else Path(DEFAULT_ROOT)
        if root.exists():
            return root
        print(f"  Folder not found: {root}")
        print("  Make sure you run this from your project directory.")


def main():
    print("\n" + "#" * 55)
    print("#  Dukascopy data format converter")
    print(f"#  workers: {MAX_WORKERS}  ({os.cpu_count()} cores × 70%)")
    print("#" * 55)

    direction = choose_direction()
    dry_run   = choose_mode()
    root      = choose_root()

    if not dry_run:
        src_label = "CSVs" if direction == "csv_to_parquet" else "Parquet files"
        print(f"\n  This will convert all {src_label} under {root}")
        print("  and delete the originals. This cannot be undone.")
        confirm = input("\n  Are you sure? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

    print()
    run(root, direction=direction, dry_run=dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")