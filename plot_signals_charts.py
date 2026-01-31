"""
Plot daily charts for the latest signal list (last ~6 months).

Usage:
  python plot_signals_charts.py
  python plot_signals_charts.py --top 20
  python plot_signals_charts.py --days 180
  python plot_signals_charts.py --from-latest-signal-date
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

import config
import data_manager as dm
from precompute import load_precomputed


def _get_latest_date_with_signals(precomputed) -> str | None:
    for date in reversed(precomputed.trading_dates):
        if precomputed.signals_by_date.get(date):
            return date
    return None


def _ensure_output_dir(base: Path, date_str: str) -> Path:
    out_dir = base / "charts" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_symbol(symbol: str, df, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["close"], linewidth=1.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close (JPY)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot charts for latest signal list")
    parser.add_argument("--top", type=int, default=20, help="Number of top signals to plot")
    parser.add_argument("--days", type=int, default=180, help="Number of calendar days to plot")
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
        # Auto-fallback to most recent signal date if latest has none
        fallback = _get_latest_date_with_signals(precomputed)
        if fallback and fallback != precomputed.trading_dates[-1]:
            print(f"Latest date has no signals; using {fallback} instead.")
            latest_date = fallback

    signals = precomputed.signals_by_date.get(latest_date, [])
    if not signals:
        print(f"No signals found for {latest_date}.")
        return 0

    signals = sorted(signals, key=lambda s: s.get("score", 0), reverse=True)[: args.top]
    symbols = [s["symbol"] for s in signals]

    end_date = latest_date
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"Plotting {len(symbols)} symbols from {start_date} to {end_date}")
    out_dir = _ensure_output_dir(config.RESULTS_DIR, latest_date)

    for sig in signals:
        symbol = sig["symbol"]
        df = dm.get_daily_bars(symbol, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            continue
        title = f"{symbol} | Score {sig.get('score', 0):.0f} | {start_date} → {end_date}"
        out_path = out_dir / f"{symbol}.png"
        _plot_symbol(symbol, df, out_path, title)

    print(f"Saved charts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
