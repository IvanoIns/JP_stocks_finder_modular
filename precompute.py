"""
JP Stocks Modular Trading System — Precompute Module v3

ULTRA-OPTIMIZED: Pre-computes EVERYTHING including scanner signals.
Backtest just does lookups - no computation in the hot path!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import config
import data_manager as dm
import technical_analysis as ta
import scanners as sc

logger = logging.getLogger(__name__)


class PrecomputedData:
    """
    Ultra-fast pre-computed data container.
    
    PRE-COMPUTES:
    1. Price data + indicators for each symbol
    2. Scanner signals for each symbol/date combination
    
    Backtest just does O(1) lookups!
    """
    
    def __init__(self):
        # {symbol: DataFrame with OHLCV + indicators}
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        
        # {date: {symbol: bar_dict}} for O(1) bar lookups
        self.bars_by_date: Dict[str, Dict[str, dict]] = {}
        
        # {date: list of signal dicts} - PRE-COMPUTED SIGNALS!
        self.signals_by_date: Dict[str, List[dict]] = {}
        
        # All trading dates
        self.trading_dates: list = []
        
        # JPX short data (optional / legacy)
        self.jpx_data: Dict[str, dict] = {}

        # Metadata
        self.start_date: str = ""
        self.end_date: str = ""
        self.num_symbols: int = 0
        self.num_days: int = 0
        self.scanner_config: Dict[str, float] = {}
        self.min_score: Optional[int] = None
        self.universe_by_date: Dict[str, List[str]] = {}
        self.universe_symbols: List[str] = []
        self.universe_top_n: Optional[int] = None
        self.universe_min_volume: Optional[int] = None
        self.universe_exclude_nikkei: Optional[bool] = None
        self.early_mode_enabled: Optional[bool] = None
        self.early_mode_rsi_max: Optional[int] = None
        self.early_mode_return_max: Optional[float] = None
        self.early_mode_scanners: Optional[List[str]] = None
    
    def get_bar(self, symbol: str, date: str) -> Optional[dict]:
        """O(1) bar lookup."""
        return self.bars_by_date.get(date, {}).get(symbol)
    
    def get_signals(self, date: str) -> List[dict]:
        """O(1) signals lookup - already computed!"""
        return self.signals_by_date.get(date, [])

class _PrecomputedUnpickler(pickle.Unpickler):
    """
    Handle legacy caches created when precompute.py was executed as __main__.
    """
    def find_class(self, module, name):
        if module == "__main__" and name == "PrecomputedData":
            return PrecomputedData
        return super().find_class(module, name)


def load_precomputed(path: Path) -> PrecomputedData:
    """Load precomputed cache with backward compatibility."""
    with open(path, "rb") as f:
        return _PrecomputedUnpickler(f).load()


def precompute_all_data(
    start_date: str,
    end_date: str,
    lookback_days: int = 250,
    top_n: int = None,
    progress: bool = True
) -> PrecomputedData:
    """
    Pre-compute EVERYTHING: indicators AND scanner signals.
    
    After this, backtests just look up pre-computed signals.
    """
    if top_n is None:
        top_n = config.UNIVERSE_TOP_N
    
    precomputed = PrecomputedData()
    precomputed.start_date = start_date
    precomputed.end_date = end_date
    
    print(f"Pre-computing indicators for {start_date} to {end_date}...")
    
    # 1. Get trading dates
    print("  - Getting trading dates...")
    conn = dm.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (start_date, end_date))
    precomputed.trading_dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    precomputed.num_days = len(precomputed.trading_dates)
    print(f"    {precomputed.num_days} trading days")

    # 2. Build daily liquid universe (aligned with run_daily_backtest)
    print("  - Building liquid universe by date...")
    universe_by_date: Dict[str, List[str]] = {}
    date_iterator = precomputed.trading_dates
    if progress:
        date_iterator = tqdm(precomputed.trading_dates, desc="Building universe")

    for date in date_iterator:
        universe_by_date[date] = dm.build_liquid_universe(date, top_n=top_n)

    precomputed.universe_by_date = universe_by_date
    precomputed.universe_top_n = top_n
    precomputed.universe_min_volume = config.MIN_AVG_DAILY_VOLUME
    precomputed.universe_exclude_nikkei = config.EXCLUDE_NIKKEI_225
    precomputed.early_mode_enabled = config.EARLY_MODE_ENABLED
    precomputed.early_mode_rsi_max = config.EARLY_MODE_RSI_MAX
    precomputed.early_mode_return_max = config.EARLY_MODE_10D_RETURN_MAX
    precomputed.early_mode_scanners = list(config.EARLY_MODE_SCANNERS)

    all_symbols = sorted({s for symbols in universe_by_date.values() for s in symbols})
    precomputed.universe_symbols = all_symbols
    print(f"    {len(all_symbols)} unique symbols in universe")

    # 3. Load price data + compute indicators
    print(f"  - Loading price data for {len(all_symbols)} symbols...")
    lookback_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    scanner_config = config.get_scanner_config()
    precomputed.scanner_config = scanner_config.copy()
    
    iterator = all_symbols
    if progress:
        iterator = tqdm(all_symbols, desc="Loading data")
    
    for symbol in iterator:
        try:
            df = dm.get_daily_bars(symbol, lookback_start, end_date)
            if df is None or len(df) < 60:
                continue
            
            # Calculate indicators ONCE
            df_with_indicators = ta.calculate_all_indicators(df, scanner_config)
            precomputed.symbol_data[symbol] = df_with_indicators
            
        except Exception:
            continue
    
    precomputed.num_symbols = len(precomputed.symbol_data)
    print(f"    Loaded {precomputed.num_symbols} symbols")
    
    # 6. Pre-compute scanner signals AND build bar lookups for EACH DATE
    print(f"  - Pre-computing signals for {precomputed.num_days} dates...")
    
    if getattr(config, "CACHE_RAW_SIGNALS", False):
        min_score = 0
        precomputed.raw_signals = True
    else:
        min_score = config.MIN_SCANNER_SCORE
        precomputed.raw_signals = False
    precomputed.min_score = min_score
    
    iterator = precomputed.trading_dates
    if progress:
        iterator = tqdm(precomputed.trading_dates, desc="Computing signals")
    
    for date in iterator:
        precomputed.bars_by_date[date] = {}
        precomputed.signals_by_date[date] = []
        universe_symbols = set(precomputed.universe_by_date.get(date, []))
        
        for symbol, df in precomputed.symbol_data.items():
            # Get data UP TO this date (no lookahead!)
            mask = df.index.strftime('%Y-%m-%d') <= date
            data_up_to = df[mask]
            
            if len(data_up_to) < 60:
                continue
            
            # Store bar for this date (if exists)
            today_mask = df.index.strftime('%Y-%m-%d') == date
            today_data = df[today_mask]
            if len(today_data) > 0:
                bar = today_data.iloc[-1]
                precomputed.bars_by_date[date][symbol] = {
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar['volume'],
                }
            
            if symbol not in universe_symbols:
                continue

            # Run ALL scanners on data up to this date
            signals = sc.get_all_signals(
                symbol, data_up_to, None, scanner_config, min_score
            )
            
            # Store signals
            for sig in signals:
                sig['date'] = date
                precomputed.signals_by_date[date].append(sig)
        
        # Sort signals by score
        precomputed.signals_by_date[date].sort(key=lambda x: x['score'], reverse=True)
    
    print("Pre-computation complete!")
    print(f"  {precomputed.num_symbols} symbols x {precomputed.num_days} days")
    total_signals = sum(len(s) for s in precomputed.signals_by_date.values())
    print(f"  {total_signals:,} total signals detected")
    
    return precomputed


def _default_start_end_dates() -> tuple[str, str]:
    """Default cache range: last ~2 years through today."""
    today = datetime.now()
    start = (today - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')
    return start, end


def _auto_expand_db_if_enabled() -> None:
    """
    Incrementally expand DB coverage before building cache.
    """
    if not getattr(config, "AUTO_EXPAND_DB", False):
        return
    try:
        print("\n[Auto-Expand] Expanding DB coverage (incremental)...")
        added = dm.expand_db_incremental(
            max_new=getattr(config, "AUTO_EXPAND_DB_MAX_NEW", 200),
            start_date=getattr(config, "AUTO_EXPAND_DB_START_DATE", "2024-01-01"),
            end_date=None,
            min_rows=getattr(config, "AUTO_EXPAND_DB_MIN_ROWS", 60),
            exclude_nikkei=getattr(config, "EXCLUDE_NIKKEI_225", True),
            cache_duration_hours=getattr(config, "CACHE_TICKER_LIST_HOURS", 720),
            shuffle=getattr(config, "AUTO_EXPAND_DB_SHUFFLE", True),
            commit_interval=50,
            sleep_seconds=getattr(config, "AUTO_EXPAND_DB_SLEEP_SECONDS", 0.2),
            progress=True,
        )
        print(f"[Auto-Expand] Added {added} new symbols this run.")
    except Exception as e:
        print(f"[Auto-Expand] WARNING: expansion failed: {e}")


def _auto_update_market_caps_if_enabled() -> None:
    """
    Incrementally populate market-cap data before building cache.
    """
    if not getattr(config, "AUTO_UPDATE_MARKET_CAP", False):
        return
    try:
        print("\n[MarketCap] Updating missing market caps (incremental)...")
        updated = dm.update_market_caps_incremental(
            max_symbols=getattr(config, "AUTO_UPDATE_MARKET_CAP_MAX", 300),
            sleep_seconds=getattr(config, "AUTO_UPDATE_MARKET_CAP_SLEEP", 0.2),
            progress=True,
        )
        print(f"[MarketCap] Updated {updated} symbols this run.")
    except Exception as e:
        print(f"[MarketCap] WARNING: update failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/update results/precomputed_cache.pkl")
    default_start, default_end = _default_start_end_dates()
    parser.add_argument("--start", default=default_start, help=f"Start date (YYYY-MM-DD). Default: {default_start}")
    parser.add_argument("--end", default=default_end, help=f"End date (YYYY-MM-DD). Default: {default_end}")
    parser.add_argument("--top", type=int, default=config.UNIVERSE_TOP_N, help="Universe size (top N symbols)")
    parser.add_argument("--lookback", type=int, default=250, help="Indicator lookback days")
    parser.add_argument("--output", default="results/precomputed_cache.pkl", help="Output pickle path")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    _auto_expand_db_if_enabled()
    _auto_update_market_caps_if_enabled()

    precomputed = precompute_all_data(
        start_date=args.start,
        end_date=args.end,
        lookback_days=args.lookback,
        top_n=args.top,
        progress=not args.no_progress,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(precomputed, f)

    print(f"Saved cache to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
