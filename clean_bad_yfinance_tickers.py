"""
Clean cache/bad_yfinance_tickers.txt by removing symbols that already have DB data.

Why:
  The bad-ticker cache can get "poisoned" during temporary Yahoo issues (rate limits,
  Yahoo lag, etc.). If a symbol already exists in `daily_prices`, it is not truly
  "bad" for our purposes and should not be skipped.

Usage:
  python clean_bad_yfinance_tickers.py
  python clean_bad_yfinance_tickers.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import config


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune bad_yfinance_tickers.txt using DB presence")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes, only print stats")
    args = parser.parse_args()

    bad_file = Path(getattr(config, "CACHE_DIR", Path("cache"))) / "bad_yfinance_tickers.txt"
    if not bad_file.exists():
        print("No bad_yfinance_tickers.txt found.")
        return 0

    raw = bad_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    tickers = [t.strip() for t in raw if t and t.strip()]
    if not tickers:
        print("bad_yfinance_tickers.txt is empty.")
        return 0

    conn = sqlite3.connect(config.DATABASE_FILE)
    cur = conn.cursor()

    has_data: set[str] = set()
    for chunk in _chunked(tickers, 900):
        placeholders = ",".join(["?"] * len(chunk))
        cur.execute(f"SELECT DISTINCT symbol FROM daily_prices WHERE symbol IN ({placeholders})", chunk)
        has_data.update(row[0] for row in cur.fetchall())

    conn.close()

    keep = [t for t in tickers if t not in has_data]
    removed = [t for t in tickers if t in has_data]

    print(f"Bad tickers in file: {len(tickers)}")
    print(f"Removed (already in DB): {len(removed)}")
    print(f"Kept (no DB rows): {len(keep)}")

    if args.dry_run:
        return 0

    backup = bad_file.with_suffix(".bak")
    backup.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    bad_file.write_text(("\n".join(keep) + "\n") if keep else "", encoding="utf-8")
    print(f"Wrote cleaned file: {bad_file}")
    print(f"Backup saved: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

