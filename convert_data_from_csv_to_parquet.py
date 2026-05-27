#!/usr/bin/env python3
"""
convert_data_from_csv_to_parquet.py

Walks the data/raw/ folder, finds every Dukascopy tick CSV,
converts it to Parquet (compressed, ~5-10x smaller), then deletes the CSV.
Meta JSON sidecar files are left untouched.
"""

import sys
import time
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

DEFAULT_ROOT = "data/raw"

# Parquet compression. "snappy" is fast read/write, good ratio.
# "zstd" is slower write but smaller files. "snappy" is fine for backtesting.
PARQUET_COMPRESSION = "snappy"

# CSV dtypes — tells pandas exactly what each column is so it doesn't guess.
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
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def find_csvs(root: Path) -> list[Path]:
    """Return all .csv files under root, sorted by path."""
    return sorted(root.rglob("*.csv"))


def convert_csv(csv_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Convert one CSV to Parquet next to it, then delete the CSV.
    Returns (csv_bytes, parquet_bytes). On dry_run returns (csv_bytes, 0).
    """
    csv_bytes    = csv_path.stat().st_size
    parquet_path = csv_path.with_suffix(".parquet")

    if dry_run:
        return csv_bytes, 0

    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        dtype=CSV_DTYPES,
    )
    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
        index=False,
    )

    parquet_bytes = parquet_path.stat().st_size
    csv_path.unlink()   # delete original CSV only after successful write

    return csv_bytes, parquet_bytes


def run(root: Path, dry_run: bool):
    csvs = find_csvs(root)

    if not csvs:
        print(f"\n  No CSV files found under {root}")
        print("  Nothing to convert.")
        return

    total_csv_bytes = sum(p.stat().st_size for p in csvs)

    print(f"\n  Root       : {root}")
    print(f"  CSV files  : {len(csvs):,}")
    print(f"  Total size : {_fmt_size(total_csv_bytes)}")
    print(f"  Compression: {PARQUET_COMPRESSION}")
    if dry_run:
        print("\n  DRY RUN — no files will be changed")
    print()

    start            = time.monotonic()
    done             = 0
    total_csv_saved  = 0
    total_parq_saved = 0
    errors           = []

    for csv_path in csvs:
        rel = csv_path.relative_to(root)
        try:
            csv_b, parq_b = convert_csv(csv_path, dry_run=dry_run)
            total_csv_saved  += csv_b
            total_parq_saved += parq_b
            done += 1

            if dry_run:
                print(f"  ~  {rel}   {_fmt_size(csv_b)}")
            else:
                ratio = (1 - parq_b / csv_b) * 100 if csv_b else 0
                print(f"  ✓  {rel}"
                      f"   {_fmt_size(csv_b)} → {_fmt_size(parq_b)}"
                      f"  (-{ratio:.0f}%)")

        except Exception as exc:
            errors.append((csv_path, exc))
            print(f"  ✗  {rel}   ERROR: {exc}")

    elapsed = time.monotonic() - start

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if dry_run:
        print("  DRY RUN complete — nothing was changed")
        print("=" * 55)
        print(f"  Files found : {len(csvs):,}")
        print(f"  Total size  : {_fmt_size(total_csv_saved)}")
    else:
        saved = total_csv_saved - total_parq_saved
        ratio = (saved / total_csv_saved * 100) if total_csv_saved else 0
        print(f"  Conversion complete  ({_fmt_elapsed(elapsed)})")
        print("=" * 55)
        print(f"  Files converted : {done:,} / {len(csvs):,}")
        print(f"  Before          : {_fmt_size(total_csv_saved)}")
        print(f"  After           : {_fmt_size(total_parq_saved)}")
        print(f"  Saved           : {_fmt_size(saved)}  (-{ratio:.0f}%)")

        if errors:
            print(f"\n  ⚠  {len(errors)} file(s) failed:")
            for path, exc in errors:
                print(f"     {path.relative_to(root)}  →  {exc}")
            print("\n  Those CSVs were NOT deleted. Fix the error and re-run.")
        else:
            print("\n  All CSVs converted and removed  ✓")

    print()


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------

def choose_mode() -> bool:
    """Returns True = dry run, False = real conversion."""
    print("\n" + "=" * 55)
    print("  MENU 1 -- conversion mode")
    print("=" * 55)
    print("  1. Preview   (dry run, nothing changes)")
    print("  2. Convert   (convert CSVs to Parquet and delete CSVs)")
    print("-" * 55)

    while True:
        choice = input("Select mode [1-2] (default 1): ").strip()
        if choice in ("", "1"):
            return True    # dry run
        if choice == "2":
            return False   # real
        print("  Invalid choice, try again.")


def choose_root() -> Path:
    """Menu 2: confirm or change the data folder."""
    print("\n" + "=" * 55)
    print("  MENU 2 -- data folder")
    print("=" * 55)
    print(f"  Default: {DEFAULT_ROOT}")
    print("  Press Enter to use default, or type a different path.")
    print("-" * 55)

    while True:
        raw = input(f"Folder [{DEFAULT_ROOT}]: ").strip()
        root = Path(raw) if raw else Path(DEFAULT_ROOT)
        if root.exists():
            return root
        print(f"  Folder not found: {root}")
        print("  Make sure you run this from your project directory.")


def main():
    print("\n" + "#" * 55)
    print("#  Dukascopy CSV → Parquet converter")
    print("#" * 55)

    dry_run = choose_mode()
    root    = choose_root()

    if not dry_run:
        print(f"\n  This will convert all CSVs under {root}")
        print("  and delete the originals. This cannot be undone.")
        confirm = input("\n  Are you sure? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("  Aborted.")
            return

    print()
    run(root, dry_run=dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted.")