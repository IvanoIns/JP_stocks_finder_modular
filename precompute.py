"""
JP Stocks Modular Trading System — Precompute Module v3

ULTRA-OPTIMIZED: Pre-computes EVERYTHING including scanner signals.
Backtest just does lookups - no computation in the hot path!
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import logging
import argparse
import hashlib
import json
import pickle
from datetime import datetime, timedelta, timezone
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

        # Cache metadata (used for incremental updates)
        self.cache_schema_version: int = 1
        self.cache_fingerprint: str = ""
        self.cache_fingerprint_payload: Dict[str, Any] = {}
        self.lookback_days: int = 0
        self.raw_signals: bool = False
        self.created_utc: str = ""
        self.updated_utc: str = ""

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file. Empty string if file is missing."""
    try:
        return hashlib.sha256(filepath.read_bytes()).hexdigest()
    except Exception:
        return ""


def compute_code_fingerprint() -> Dict[str, str]:
    """Hash files that materially affect cache validity."""
    project_root = Path(__file__).resolve().parent
    return {
        "scanners_py": compute_file_hash(project_root / "scanners.py"),
        "technical_analysis_py": compute_file_hash(project_root / "technical_analysis.py"),
        "data_manager_py": compute_file_hash(project_root / "data_manager.py"),
        "config_py": compute_file_hash(project_root / "config.py"),
        "precompute_py": compute_file_hash(project_root / "precompute.py"),
    }


def _compute_fingerprint_digest(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_cache_payload(
    lookback_days: int,
    top_n: Optional[int],
    scanner_config: Dict[str, float],
    min_score: int,
    raw_signals: bool,
) -> Dict[str, Any]:
    """Build deterministic cache fingerprint payload."""
    return {
        "schema_version": 1,
        "window": {"lookback_days": int(lookback_days)},
        "runtime": {
            "top_n": top_n,
            "min_score": int(min_score),
            "raw_signals": bool(raw_signals),
        },
        "universe": {
            "min_avg_daily_volume": int(config.MIN_AVG_DAILY_VOLUME),
            "exclude_nikkei_225": bool(config.EXCLUDE_NIKKEI_225),
            "enforce_market_cap": bool(getattr(config, "ENFORCE_MARKET_CAP", False)),
            "max_market_cap_jpy": int(getattr(config, "MAX_MARKET_CAP_JPY", 0) or 0),
            "market_cap_missing_policy": str(getattr(config, "MARKET_CAP_MISSING_POLICY", "include")),
        },
        "early_mode": {
            "enabled": bool(getattr(config, "EARLY_MODE_ENABLED", False)),
            "rsi_max": int(getattr(config, "EARLY_MODE_RSI_MAX", 0)),
            "return_10d_max": float(getattr(config, "EARLY_MODE_10D_RETURN_MAX", 0.0)),
            "scanners": list(getattr(config, "EARLY_MODE_SCANNERS", [])),
            "show_both": bool(getattr(config, "EARLY_MODE_SHOW_BOTH", False)),
        },
        "scanner_config": dict(scanner_config),
        "code_hashes": compute_code_fingerprint(),
    }


def print_fingerprint_diff(old_payload: Dict[str, Any], new_payload: Dict[str, Any]) -> None:
    """Print key fingerprint differences to explain forced rebuilds."""
    old_code = (old_payload or {}).get("code_hashes", {})
    new_code = (new_payload or {}).get("code_hashes", {})
    changed = False

    print("Fingerprint diff:")
    for key in sorted(new_code.keys()):
        old_value = old_code.get(key, "")
        new_value = new_code.get(key, "")
        if old_value != new_value:
            changed = True
            print(f"  - code {key}: {old_value[:8] or 'N/A'} -> {new_value[:8] or 'N/A'}")

    for section in ("runtime", "universe", "early_mode"):
        old_section = (old_payload or {}).get(section, {})
        new_section = (new_payload or {}).get(section, {})
        for key in sorted(new_section.keys()):
            old_value = old_section.get(key)
            new_value = new_section.get(key)
            if old_value != new_value:
                changed = True
                print(f"  - {section}.{key}: {old_value} -> {new_value}")

    if not changed:
        print("  - no visible field diff (schema/version mismatch likely)")


def _get_trading_dates(start_date: str, end_date: str) -> List[str]:
    conn = dm.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT date FROM daily_prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """,
        (start_date, end_date),
    )
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def _build_universe_by_date(
    dates: List[str],
    top_n: Optional[int],
    progress: bool,
) -> Dict[str, List[str]]:
    universe_by_date: Dict[str, List[str]] = {}
    iterator = dates
    if progress:
        iterator = tqdm(dates, desc="Building universe")
    for date in iterator:
        universe_by_date[date] = dm.build_liquid_universe(date, top_n=top_n)
    return universe_by_date


def _prepare_symbol_data(
    symbols: List[str],
    lookback_start: str,
    end_date: str,
    scanner_config: Dict[str, float],
    progress: bool,
    desc: str = "Loading data",
) -> Dict[str, pd.DataFrame]:
    symbol_data: Dict[str, pd.DataFrame] = {}
    iterator = symbols
    if progress:
        iterator = tqdm(symbols, desc=desc)

    for symbol in iterator:
        try:
            df = dm.get_daily_bars(symbol, lookback_start, end_date)
            if df is None or len(df) < 60:
                continue
            symbol_data[symbol] = ta.calculate_all_indicators(df, scanner_config)
        except Exception:
            continue
    return symbol_data


def _compute_dates_signals(
    precomputed: PrecomputedData,
    dates: List[str],
    scanner_config: Dict[str, float],
    min_score: int,
    progress: bool,
) -> None:
    iterator = dates
    if progress:
        iterator = tqdm(dates, desc="Computing signals")

    for date in iterator:
        precomputed.bars_by_date[date] = {}
        precomputed.signals_by_date[date] = []
        universe_symbols = set(precomputed.universe_by_date.get(date, []))
        date_ts = pd.Timestamp(date)

        for symbol, df in precomputed.symbol_data.items():
            bar = None
            if date_ts in df.index:
                bar = df.loc[date_ts]
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]
            if bar is not None:
                precomputed.bars_by_date[date][symbol] = {
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["volume"],
                }

            if symbol not in universe_symbols:
                continue

            data_up_to = df.loc[:date_ts]
            if len(data_up_to) < 60:
                continue

            signals = sc.get_all_signals(symbol, data_up_to, None, scanner_config, min_score)
            for sig in signals:
                sig["date"] = date
                precomputed.signals_by_date[date].append(sig)

        precomputed.signals_by_date[date].sort(key=lambda x: x["score"], reverse=True)


def _finalize_metadata(
    precomputed: PrecomputedData,
    start_date: str,
    end_date: str,
    lookback_days: int,
    top_n: Optional[int],
    scanner_config: Dict[str, float],
    min_score: int,
    raw_signals: bool,
) -> None:
    precomputed.start_date = start_date
    precomputed.end_date = end_date
    precomputed.lookback_days = int(lookback_days)
    precomputed.trading_dates = sorted(precomputed.trading_dates)
    precomputed.num_days = len(precomputed.trading_dates)
    precomputed.universe_top_n = top_n
    precomputed.universe_min_volume = config.MIN_AVG_DAILY_VOLUME
    precomputed.universe_exclude_nikkei = config.EXCLUDE_NIKKEI_225
    precomputed.early_mode_enabled = config.EARLY_MODE_ENABLED
    precomputed.early_mode_rsi_max = config.EARLY_MODE_RSI_MAX
    precomputed.early_mode_return_max = config.EARLY_MODE_10D_RETURN_MAX
    precomputed.early_mode_scanners = list(config.EARLY_MODE_SCANNERS)
    precomputed.universe_symbols = sorted({s for v in precomputed.universe_by_date.values() for s in v})
    precomputed.symbol_data = {s: precomputed.symbol_data[s] for s in precomputed.universe_symbols if s in precomputed.symbol_data}
    precomputed.num_symbols = len(precomputed.symbol_data)
    precomputed.scanner_config = scanner_config.copy()
    precomputed.min_score = int(min_score)
    precomputed.raw_signals = bool(raw_signals)

    payload = _compute_cache_payload(
        lookback_days=lookback_days,
        top_n=top_n,
        scanner_config=scanner_config,
        min_score=min_score,
        raw_signals=raw_signals,
    )
    precomputed.cache_fingerprint_payload = payload
    precomputed.cache_fingerprint = _compute_fingerprint_digest(payload)
    precomputed.cache_schema_version = int(payload["schema_version"])
    if not getattr(precomputed, "created_utc", ""):
        precomputed.created_utc = _utc_now_iso()
    precomputed.updated_utc = _utc_now_iso()


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
    
    print(f"Pre-computing indicators for {start_date} to {end_date}...")
    
    # 1. Get trading dates
    print("  - Getting trading dates...")
    precomputed.trading_dates = _get_trading_dates(start_date, end_date)
    precomputed.num_days = len(precomputed.trading_dates)
    print(f"    {precomputed.num_days} trading days")

    # 2. Build daily liquid universe (aligned with run_daily_backtest)
    print("  - Building liquid universe by date...")
    universe_by_date = _build_universe_by_date(precomputed.trading_dates, top_n=top_n, progress=progress)

    precomputed.universe_by_date = universe_by_date
    all_symbols = sorted({s for symbols in universe_by_date.values() for s in symbols})
    precomputed.universe_symbols = all_symbols
    print(f"    {len(all_symbols)} unique symbols in universe")

    # 3. Load price data + compute indicators
    print(f"  - Loading price data for {len(all_symbols)} symbols...")
    lookback_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    scanner_config = config.get_scanner_config()
    precomputed.symbol_data = _prepare_symbol_data(
        symbols=all_symbols,
        lookback_start=lookback_start,
        end_date=end_date,
        scanner_config=scanner_config,
        progress=progress,
        desc="Loading data",
    )
    precomputed.num_symbols = len(precomputed.symbol_data)
    print(f"    Loaded {precomputed.num_symbols} symbols")
    
    # 6. Pre-compute scanner signals AND build bar lookups for EACH DATE
    print(f"  - Pre-computing signals for {precomputed.num_days} dates...")
    
    raw_signals = bool(getattr(config, "CACHE_RAW_SIGNALS", False))
    min_score = 0 if raw_signals else int(config.MIN_SCANNER_SCORE)
    _compute_dates_signals(
        precomputed=precomputed,
        dates=precomputed.trading_dates,
        scanner_config=scanner_config,
        min_score=min_score,
        progress=progress,
    )
    _finalize_metadata(
        precomputed=precomputed,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        top_n=top_n,
        scanner_config=scanner_config,
        min_score=min_score,
        raw_signals=raw_signals,
    )
    
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


def _prune_cache_dates(precomputed: PrecomputedData, keep_dates: List[str]) -> None:
    keep = set(keep_dates)
    precomputed.trading_dates = list(keep_dates)
    precomputed.bars_by_date = {d: v for d, v in precomputed.bars_by_date.items() if d in keep}
    precomputed.signals_by_date = {d: v for d, v in precomputed.signals_by_date.items() if d in keep}
    precomputed.universe_by_date = {d: v for d, v in precomputed.universe_by_date.items() if d in keep}


def update_precomputed_cache_incremental(
    existing: PrecomputedData,
    start_date: str,
    end_date: str,
    lookback_days: int = 250,
    top_n: int = None,
    progress: bool = True,
) -> PrecomputedData:
    """
    Update cache by computing only missing trading dates when fingerprint matches.
    Falls back to full rebuild when settings/code changed.
    """
    if top_n is None:
        top_n = config.UNIVERSE_TOP_N

    scanner_config = config.get_scanner_config()
    raw_signals = bool(getattr(config, "CACHE_RAW_SIGNALS", False))
    min_score = 0 if raw_signals else int(config.MIN_SCANNER_SCORE)
    expected_payload = _compute_cache_payload(
        lookback_days=lookback_days,
        top_n=top_n,
        scanner_config=scanner_config,
        min_score=min_score,
        raw_signals=raw_signals,
    )
    expected_fingerprint = _compute_fingerprint_digest(expected_payload)
    existing_fingerprint = getattr(existing, "cache_fingerprint", "")

    if not existing_fingerprint or existing_fingerprint != expected_fingerprint:
        print("Cache fingerprint mismatch or missing. Running full rebuild.")
        print_fingerprint_diff(getattr(existing, "cache_fingerprint_payload", {}), expected_payload)
        rebuilt = precompute_all_data(
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            top_n=top_n,
            progress=progress,
        )
        return rebuilt

    desired_dates = _get_trading_dates(start_date, end_date)
    if not desired_dates:
        print("No trading dates in requested window. Keeping cache empty for range.")
        _prune_cache_dates(existing, [])
        _finalize_metadata(
            precomputed=existing,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            top_n=top_n,
            scanner_config=scanner_config,
            min_score=min_score,
            raw_signals=raw_signals,
        )
        return existing

    cached_dates = set(getattr(existing, "trading_dates", []))
    missing_dates = [d for d in desired_dates if d not in cached_dates]
    removed_dates = [d for d in getattr(existing, "trading_dates", []) if d not in set(desired_dates)]

    if removed_dates:
        print(f"Pruning {len(removed_dates)} old dates outside requested window.")
    _prune_cache_dates(existing, desired_dates)

    if not missing_dates:
        print("No missing trading dates. Cache is already up to date.")
        _finalize_metadata(
            precomputed=existing,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            top_n=top_n,
            scanner_config=scanner_config,
            min_score=min_score,
            raw_signals=raw_signals,
        )
        return existing

    print(f"Incremental update: computing {len(missing_dates)} missing trading date(s).")
    missing_universe = _build_universe_by_date(missing_dates, top_n=top_n, progress=progress)
    existing.universe_by_date.update(missing_universe)

    all_universe_symbols = sorted({s for v in existing.universe_by_date.values() for s in v})
    lookback_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    print(f"Refreshing indicator data for {len(all_universe_symbols)} universe symbols...")
    existing.symbol_data = _prepare_symbol_data(
        symbols=all_universe_symbols,
        lookback_start=lookback_start,
        end_date=end_date,
        scanner_config=scanner_config,
        progress=progress,
        desc="Refreshing data",
    )

    _compute_dates_signals(
        precomputed=existing,
        dates=missing_dates,
        scanner_config=scanner_config,
        min_score=min_score,
        progress=progress,
    )

    _finalize_metadata(
        precomputed=existing,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        top_n=top_n,
        scanner_config=scanner_config,
        min_score=min_score,
        raw_signals=raw_signals,
    )
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/update results/precomputed_cache.pkl (incremental by default)")
    default_start, default_end = _default_start_end_dates()
    parser.add_argument("--start", default=default_start, help=f"Start date (YYYY-MM-DD). Default: {default_start}")
    parser.add_argument("--end", default=default_end, help=f"End date (YYYY-MM-DD). Default: {default_end}")
    parser.add_argument("--top", type=int, default=config.UNIVERSE_TOP_N, help="Universe size (top N symbols)")
    parser.add_argument("--lookback", type=int, default=250, help="Indicator lookback days")
    parser.add_argument("--output", default="results/precomputed_cache.pkl", help="Output pickle path")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--rebuild", action="store_true", help="Force full cache rebuild")
    args = parser.parse_args()

    _auto_expand_db_if_enabled()
    _auto_update_market_caps_if_enabled()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: Optional[PrecomputedData] = None
    if output_path.exists() and not args.rebuild:
        try:
            existing = load_precomputed(output_path)
        except Exception as e:
            print(f"WARNING: failed to load existing cache. Rebuilding. Error: {e}")
            existing = None

    if existing is None:
        precomputed = precompute_all_data(
            start_date=args.start,
            end_date=args.end,
            lookback_days=args.lookback,
            top_n=args.top,
            progress=not args.no_progress,
        )
    else:
        precomputed = update_precomputed_cache_incremental(
            existing=existing,
            start_date=args.start,
            end_date=args.end,
            lookback_days=args.lookback,
            top_n=args.top,
            progress=not args.no_progress,
        )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(precomputed, f)
    tmp_path.replace(output_path)

    print(f"Saved cache to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
