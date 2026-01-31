"""
Expand `jp_stocks.db` with more Japanese tickers (incremental + resumable).

Why this exists:
- The scanner universe/backtests can only consider symbols present in the DB.
- If a ticker isn't in `daily_prices`, it can't ever be signaled.

Typical usage:
  python expand_db.py --max-new 1000 --start 2024-01-01
  python expand_db.py --max-new 1000 --start 2024-01-01
  ...repeat until coverage is good...

After expanding the DB, rebuild the cache:
  python precompute.py
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import config
import data_manager as dm


BAD_TICKERS_FILE = config.CACHE_DIR / "bad_yfinance_tickers.txt"


def _normalize_ticker_list(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    normalized: list[str] = []
    for p in parts:
        if p.endswith(".T"):
            normalized.append(p)
        elif p.isdigit():
            normalized.append(f"{p}.T")
        else:
            normalized.append(p)
    return normalized


def _load_bad_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except Exception:
        return set()


def _save_bad_tickers(path: Path, tickers: set[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    lines = sorted(tickers)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _symbols_with_any_rows(symbols: list[str]) -> set[str]:
    """
    Returns symbols that exist in `daily_prices` (row count > 0).
    Used to identify yfinance symbols that returned no data at all.
    """
    if not symbols:
        return set()
    import sqlite3

    conn = sqlite3.connect(config.DATABASE_FILE)
    placeholders = ",".join(["?"] * len(symbols))
    query = f"""
        SELECT symbol
        FROM daily_prices
        WHERE symbol IN ({placeholders})
        GROUP BY symbol
        HAVING COUNT(*) > 0
    """
    rows = conn.execute(query, symbols).fetchall()
    conn.close()
    return {r[0] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download more JP tickers into jp_stocks.db")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument(
        "--max-new",
        type=int,
        default=1000,
        help="Target number of NEW symbols to add this run (0 = try all missing once)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Batch size per yfinance loop (smaller is safer, larger is faster)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=60,
        help="Rows required to treat a symbol as already downloaded",
    )
    parser.add_argument(
        "--ticker-cache-hours",
        type=int,
        default=config.CACHE_TICKER_LIST_HOURS,
        help="Reuse cached investpy ticker list for N hours",
    )
    parser.add_argument(
        "--exclude-nikkei",
        action=argparse.BooleanOptionalAction,
        default=config.EXCLUDE_NIKKEI_225,
        help="Exclude Nikkei 225 tickers from download list",
    )
    parser.add_argument("--shuffle", action="store_true", help="Randomize download order")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between yfinance requests (rate limit safety)",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=50,
        help="SQLite commit interval (symbols)",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Redownload even if already present")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument(
        "--ignore-bad-cache",
        action="store_true",
        help="Do not skip tickers listed in cache/bad_yfinance_tickers.txt",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated tickers to download (overrides investpy list). Example: 4596.T,3070.T",
    )
    args = parser.parse_args()

    end = args.end or datetime.now().strftime("%Y-%m-%d")

    dm.setup_database().close()

    if args.symbols:
        symbols = _normalize_ticker_list(args.symbols)
        source = "manual"
    else:
        symbols = dm.get_all_tse_tickers(cache_duration_hours=args.ticker_cache_hours)
        source = "investpy"

    if not symbols:
        print("ERROR: No symbols found.")
        return 1

    if args.exclude_nikkei:
        nikkei = dm.get_nikkei_225_components(cache_duration_hours=args.ticker_cache_hours)
        symbols = [s for s in symbols if s not in nikkei]

    if config.EXCLUDED_SYMBOLS:
        symbols = [s for s in symbols if s not in config.EXCLUDED_SYMBOLS]

    existing = set(dm.get_available_symbols(min_rows=args.min_rows))

    if args.force_refresh:
        to_download = list(symbols)
    else:
        to_download = [s for s in symbols if s not in existing]

    if args.shuffle:
        random.shuffle(to_download)

    bad = _load_bad_tickers(BAD_TICKERS_FILE)
    if bad and not args.ignore_bad_cache:
        to_download = [s for s in to_download if s not in bad]

    print("=" * 70)
    print("DB EXPANSION")
    print("=" * 70)
    print(f"DB: {config.DATABASE_FILE}")
    print(f"Ticker source: {source} ({len(symbols)} candidates)")
    print(f"Already in DB (>= {args.min_rows} rows): {len(existing)}")
    if args.max_new and args.max_new > 0:
        print(f"Target NEW symbols to add: {args.max_new}")
    else:
        print(f"Target NEW symbols to add: all missing (single pass)")
    print(f"Missing candidates after filters: {len(to_download)}")
    print(f"Date range: {args.start} -> {end}")
    print(f"Exclude Nikkei 225: {args.exclude_nikkei}")
    print(f"Sleep seconds/request: {args.sleep}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Bad-ticker cache: {'SKIP' if (bad and not args.ignore_bad_cache) else 'IGNORE'} ({len(bad)} entries)")
    print("=" * 70)

    if not to_download:
        print("Nothing to download.")
        return 0

    target_new = args.max_new if args.max_new and args.max_new > 0 else None

    total_added = 0
    total_attempted = 0
    idx = 0

    # Try until we hit target_new, or run out of candidates
    while idx < len(to_download) and (target_new is None or total_added < target_new):
        chunk = to_download[idx : idx + max(1, args.chunk_size)]
        idx += len(chunk)

        successes = dm.download_price_history(
            chunk,
            start_date=args.start,
            end_date=end,
            commit_interval=args.commit_interval,
            sleep_seconds=args.sleep,
            progress=(not args.no_progress),
        )

        total_attempted += len(chunk)
        total_added += successes

        after_present = _symbols_with_any_rows(chunk)
        still_empty = set(chunk) - after_present
        newly_bad = still_empty - bad
        if newly_bad:
            bad |= newly_bad
            _save_bad_tickers(BAD_TICKERS_FILE, bad)

        if target_new is not None:
            print(f"Progress: added {total_added}/{target_new} (attempted {total_attempted})")
        else:
            print(f"Progress: attempted {total_attempted}/{len(to_download)} (added {total_added})")

    print("\nDone. Next: rebuild cache with `python precompute.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
