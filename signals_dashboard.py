"""
Terminal dashboard to browse latest signal charts (ASCII sparkline).

Usage:
  python signals_dashboard.py
  python signals_dashboard.py --top 30 --days 120
  python signals_dashboard.py --from-latest-signal-date
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import config
import data_manager as dm
from precompute import load_precomputed


SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(series) -> str:
    if series is None or len(series) == 0:
        return ""
    vals = list(series)
    vmin = min(vals)
    vmax = max(vals)
    if vmax == vmin:
        return SPARK_CHARS[0] * len(vals)
    chars = []
    for v in vals:
        idx = int((v - vmin) / (vmax - vmin) * (len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def _get_latest_date_with_signals(precomputed) -> str | None:
    for date in reversed(precomputed.trading_dates):
        if precomputed.signals_by_date.get(date):
            return date
    return None


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal dashboard for latest signals")
    parser.add_argument("--top", type=int, default=20, help="Number of top signals")
    parser.add_argument("--days", type=int, default=180, help="Calendar days to display")
    parser.add_argument(
        "--from-latest-signal-date",
        action="store_true",
        help="Use latest date with signals in cache (fallback if latest date has none)",
    )
    args = parser.parse_args()

    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        print("ERROR: Cache not found. Run: python precompute.py")
        return 1

    precomputed = load_precomputed(cache_path)

    latest_date = precomputed.trading_dates[-1]
    if args.from_latest_signal_date:
        latest_date = _get_latest_date_with_signals(precomputed) or latest_date
    else:
        fallback = _get_latest_date_with_signals(precomputed)
        if fallback and fallback != precomputed.trading_dates[-1]:
            latest_date = fallback

    signals = precomputed.signals_by_date.get(latest_date, [])
    if not signals:
        print(f"No signals found for {latest_date}.")
        return 0

    signals = sorted(signals, key=lambda s: s.get("score", 0), reverse=True)[: args.top]

    idx = 0
    days = args.days
    while True:
        sig = signals[idx]
        symbol = sig["symbol"]
        score = sig.get("score", 0)
        strategy = sig.get("strategy", "")

        end_date = latest_date
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
        df = dm.get_daily_bars(symbol, start_date=start_date, end_date=end_date)

        closes = df["close"].values.tolist() if df is not None and not df.empty else []
        chart = sparkline(closes)

        _clear()
        print("=" * 80)
        print(f"SIGNAL DASHBOARD  |  {idx+1}/{len(signals)}")
        print("=" * 80)
        print(f"Symbol:   {symbol}")
        print(f"Strategy: {strategy}")
        print(f"Score:    {score}")
        print(f"Range:    {start_date} → {end_date} ({days} days)")
        print("-" * 80)
        if chart:
            print(chart)
        else:
            print("(no data)")
        print("-" * 80)
        print("Commands: [n]ext  [p]rev  [#] jump  [d]ays  [q]uit")

        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "n":
            idx = (idx + 1) % len(signals)
            continue
        if cmd == "p":
            idx = (idx - 1) % len(signals)
            continue
        if cmd == "d":
            new_days = input("Days (e.g., 60/120/180): ").strip()
            if new_days.isdigit():
                days = max(5, int(new_days))
            continue
        if cmd.isdigit():
            j = int(cmd) - 1
            if 0 <= j < len(signals):
                idx = j
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
