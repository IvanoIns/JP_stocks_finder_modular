"""
JP Stocks Modular Trading System - Data Manager

Database access, data ingestion from yfinance/investpy, 
JPX short-selling data scraper, and universe building functions.
"""

import sqlite3
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tqdm import tqdm
import logging
import time
import urllib.parse
import random

try:
    import investpy
    INVESTPY_AVAILABLE = True
except ImportError:
    INVESTPY_AVAILABLE = False
    logging.warning("investpy not available. Will use fallback ticker sources.")

import config

# Configure logging - use WARNING level to reduce spam during backtests
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to reduce spam
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# yfinance often logs "possibly delisted" for any empty response (including
# "today's bar not ready yet"). We handle empties ourselves; keep it quiet.
_yf_logger = logging.getLogger("yfinance")
_yf_logger.setLevel(logging.CRITICAL)
# Ensure child loggers don't bubble up to root handlers.
_yf_logger.propagate = False

# In-memory cache for Nikkei 225 (loaded once per session)
_NIKKEI_225_CACHE = None
# In-memory cache for Yahoo "latest daily bar date" guard
_YAHOO_LATEST_DATE_MEMO: dict | None = None


# =============================================================================
# Database Setup
# =============================================================================

def get_connection() -> sqlite3.Connection:
    """Get a database connection with optimized settings."""
    conn = sqlite3.connect(config.DATABASE_FILE)
    # Optimize for read-heavy workloads
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    return conn


def setup_database() -> sqlite3.Connection:
    """
    Create database and all required tables.
    
    Tables:
        - daily_prices: OHLCV data
        - symbol_info: Metadata (name, sector, market cap, Nikkei 225 flag)
        - jpx_short_interest: Short selling data
        - backtest_runs: History of backtest runs
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Daily price data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        UNIQUE(symbol, date)
    )
    """)
    
    # Symbol metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS symbol_info (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        market_cap REAL,
        is_nikkei_225 BOOLEAN DEFAULT FALSE,
        last_updated DATE
    )
    """)
    
    # JPX short interest data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jpx_short_interest (
        symbol TEXT NOT NULL,
        date DATE NOT NULL,
        short_volume INTEGER,
        total_volume INTEGER,
        short_ratio REAL,
        PRIMARY KEY(symbol, date)
    )
    """)
    
    # Backtest run history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        start_date DATE,
        end_date DATE,
        params_json TEXT,
        profit_factor REAL,
        win_rate REAL,
        max_drawdown REAL,
        total_trades INTEGER,
        total_return REAL,
        notes TEXT
    )
    """)
    
    # Create indices for performance
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_daily_symbol_date 
    ON daily_prices(symbol, date)
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_daily_date 
    ON daily_prices(date)
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_jpx_date 
    ON jpx_short_interest(date)
    """)
    
    conn.commit()
    logger.info(f"Database setup complete: {config.DATABASE_FILE}")
    return conn


# =============================================================================
# Ticker List Functions
# =============================================================================

def get_all_tse_tickers(cache_duration_hours: int = None) -> list[str]:
    """
    Fetch all TSE tickers from investpy with caching.
    
    Returns:
        List of ticker symbols in yfinance format (e.g., ['7203.T', '6758.T', ...])
    """
    if cache_duration_hours is None:
        cache_duration_hours = config.CACHE_TICKER_LIST_HOURS
    
    cache_file = config.CACHE_DIR / "tse_tickers_cache.csv"
    
    # Check cache
    if cache_file.exists():
        file_mod_time = cache_file.stat().st_mtime
        if (time.time() - file_mod_time) / 3600 < cache_duration_hours:
            logger.info(f"Using cached TSE ticker list (less than {cache_duration_hours} hours old)")
            df = pd.read_csv(cache_file)
            return df['yfinance_ticker'].tolist()
    
    if not INVESTPY_AVAILABLE:
        logger.warning("investpy not available, using fallback ticker source")
        return _get_fallback_tickers()
    
    logger.info("Fetching fresh list of all TSE stocks from investpy...")
    try:
        stocks_df = investpy.stocks.get_stocks(country='japan')
        stocks_df['yfinance_ticker'] = stocks_df['symbol'].astype(str) + '.T'
        stocks_df.to_csv(cache_file, index=False)
        logger.info(f"Successfully fetched {len(stocks_df)} tickers and updated cache")
        return stocks_df['yfinance_ticker'].tolist()
    except Exception as e:
        logger.error(f"Could not fetch stock list from investpy: {e}")
        return _get_fallback_tickers()


def _get_fallback_tickers() -> list[str]:
    """
    Fallback: Load tickers from existing cache in JP_stocks_rising folder 
    or return a minimal default set.
    """
    fallback_file = Path(config.BASE_DIR).parent / "JP_stocks_rising" / "tse_tickers_cache.csv"
    if fallback_file.exists():
        logger.info(f"Using fallback ticker file: {fallback_file}")
        df = pd.read_csv(fallback_file)
        if 'yfinance_ticker' in df.columns:
            return df['yfinance_ticker'].tolist()
        elif 'symbol' in df.columns:
            return (df['symbol'].astype(str) + '.T').tolist()
    
    # Minimal fallback
    logger.warning("Using minimal default ticker set")
    return ['7203.T', '6758.T', '9984.T', '6920.T', '8058.T']


def get_nikkei_225_components(cache_duration_hours: int = None) -> set[str]:
    """
    Fetch Nikkei 225 components from Wikipedia for exclusion.
    Uses in-memory cache for fast repeated access during backtests.
    
    Returns:
        Set of Nikkei 225 ticker symbols in yfinance format
    """
    global _NIKKEI_225_CACHE
    
    # Return in-memory cache if available (fastest path)
    if _NIKKEI_225_CACHE is not None:
        return _NIKKEI_225_CACHE
    
    if cache_duration_hours is None:
        cache_duration_hours = config.CACHE_TICKER_LIST_HOURS
    
    cache_file = config.CACHE_DIR / "nikkei_225_cache.csv"
    
    # Check file cache
    if cache_file.exists():
        file_mod_time = cache_file.stat().st_mtime
        if (time.time() - file_mod_time) / 3600 < cache_duration_hours:
            df = pd.read_csv(cache_file)
            _NIKKEI_225_CACHE = set(df['yfinance_ticker'])
            return _NIKKEI_225_CACHE
    
    logger.info("Fetching fresh list of Nikkei 225 components from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        tables = pd.read_html(url)
        # Nikkei 225 table is usually the first or second table
        nikkei_df = None
        for table in tables:
            if 'Symbol' in table.columns or 'Code' in table.columns:
                nikkei_df = table
                break
        
        if nikkei_df is None:
            # Try the first table
            nikkei_df = tables[0]
        
        # Handle different column names
        symbol_col = 'Symbol' if 'Symbol' in nikkei_df.columns else 'Code'
        if symbol_col not in nikkei_df.columns:
            symbol_col = nikkei_df.columns[0]
        
        nikkei_df['yfinance_ticker'] = nikkei_df[symbol_col].astype(str).str.extract(r'(\d+)')[0] + '.T'
        nikkei_df = nikkei_df.dropna(subset=['yfinance_ticker'])
        nikkei_df.to_csv(cache_file, index=False)
        logger.info(f"Successfully fetched {len(nikkei_df)} Nikkei 225 components")
        return set(nikkei_df['yfinance_ticker'])
    except Exception as e:
        logger.error(f"Could not fetch Nikkei 225 components: {e}")
        return set()


# =============================================================================
# Price Data Functions
# =============================================================================

def _jst_now() -> datetime:
    """Return current time in JST, with a safe fallback if zoneinfo is unavailable."""
    try:
        return datetime.now(ZoneInfo("Asia/Tokyo"))
    except Exception:
        # Fallback: UTC+9
        return datetime.now(timezone.utc) + timedelta(hours=9)


def _yahoo_latest_date_cache_path() -> Path:
    cache_dir = Path(getattr(config, "CACHE_DIR", Path("cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "yahoo_latest_date.json"


def _get_latest_yahoo_daily_bar_date() -> str | None:
    """
    Best-effort: detect the latest trading day Yahoo has published daily bars for.

    This prevents wasting time requesting a date that Yahoo hasn't published yet
    (which otherwise triggers a flood of misleading "possibly delisted" errors).
    """
    if not getattr(config, "YFINANCE_USE_LATEST_DATE_GUARD", False):
        return None

    sample = getattr(config, "YFINANCE_LATEST_DATE_SAMPLE_TICKER", "7203.T")
    lookback_days = int(getattr(config, "YFINANCE_LATEST_DATE_LOOKBACK_DAYS", 14) or 14)
    max_age_min = int(getattr(config, "YFINANCE_LATEST_DATE_CACHE_MINUTES", 30) or 30)

    cache_path = _yahoo_latest_date_cache_path()
    now_utc = datetime.now(timezone.utc)

    # In-process memo
    global _YAHOO_LATEST_DATE_MEMO
    try:
        memo = _YAHOO_LATEST_DATE_MEMO or {}
        if memo.get("sample_ticker") == sample and memo.get("latest_date") and memo.get("timestamp_utc"):
            ts = datetime.fromisoformat(str(memo["timestamp_utc"]).replace("Z", "+00:00"))
            age_min = (now_utc - ts).total_seconds() / 60.0
            if age_min <= max_age_min:
                return str(memo["latest_date"])
    except Exception:
        pass

    # Read cache
    try:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("sample_ticker") == sample and payload.get("latest_date"):
                ts_raw = payload.get("timestamp_utc")
                if ts_raw:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    age_min = (now_utc - ts).total_seconds() / 60.0
                    if age_min <= max_age_min:
                        _YAHOO_LATEST_DATE_MEMO = payload
                        return str(payload["latest_date"])
    except Exception:
        pass

    # Refresh from Yahoo
    try:
        hist = yf.Ticker(sample).history(period=f"{lookback_days}d")
        if hist is None or hist.empty:
            return None
        max_ts = hist.index.max()
        latest_date = max_ts.date().isoformat()
        payload = {
            "timestamp_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "sample_ticker": sample,
            "latest_date": latest_date,
        }
        _YAHOO_LATEST_DATE_MEMO = payload
        try:
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return latest_date
    except Exception:
        return None


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def download_price_history(
    symbols: list[str],
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    commit_interval: int = 50,
    sleep_seconds: float = 0.0,
    progress: bool = True,
) -> int:
    """
    Download and store daily OHLCV data for symbols.
    
    Args:
        symbols: List of tickers (yfinance format, e.g., '7203.T')
        start_date: Start date (YYYY-MM-DD) or None for period-based
        end_date: End date (YYYY-MM-DD) or None for today
        period: yfinance period string (e.g., '5y', '1y') if start_date not specified
        commit_interval: Commit after this many symbols
        progress: Show progress bar
    
    Returns:
        Number of symbols successfully processed
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if period is None and start_date is None:
        period = config.YFINANCE_HISTORY_PERIOD
    
    if end_date is None:
        end_date = _jst_now().strftime('%Y-%m-%d')

    # Avoid requesting today's data before JP daily bars are typically ready
    jst = _jst_now()
    ready_hour = getattr(config, "YFINANCE_DAILY_READY_HOUR", 23)
    skip_today_always = getattr(config, "YFINANCE_SKIP_TODAY_ALWAYS", False)
    today_str = jst.strftime("%Y-%m-%d")
    if end_date == today_str and (skip_today_always or jst.hour < ready_hour):
        end_date = (jst - timedelta(days=1)).strftime("%Y-%m-%d")

    # If Yahoo is lagging, clamp to the latest daily bar it actually has
    yahoo_latest = _get_latest_yahoo_daily_bar_date()
    if end_date and yahoo_latest and yahoo_latest < end_date:
        logger.warning(f"Yahoo latest daily bar appears to be {yahoo_latest}. Limiting end_date from {end_date} to {yahoo_latest}.")
        end_date = yahoo_latest

    # If clamping made the range invalid, skip quickly
    if start_date and end_date:
        try:
            if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
                logger.warning(f"Skipping download: start_date {start_date} > end_date {end_date} (after clamping).")
                conn.close()
                return 0
        except Exception:
            pass

    # If caller explicitly requested only today's bar before it is ready, skip
    if start_date and end_date and start_date == end_date == today_str and (skip_today_always or jst.hour < ready_hour):
        logger.info(f"Skipping today's download ({today_str}) before {ready_hour}:00 JST")
        conn.close()
        return 0
    
    successful = 0
    empty_count = 0
    new_bad = 0

    requested_range_days = None
    if start_date and end_date:
        try:
            d0 = datetime.strptime(start_date, "%Y-%m-%d")
            d1 = datetime.strptime(end_date, "%Y-%m-%d")
            requested_range_days = (d1 - d0).days
        except Exception:
            requested_range_days = None

    # Load bad tickers cache (to avoid repeated delisted requests)
    bad_file = config.CACHE_DIR / "bad_yfinance_tickers.txt"
    bad_tickers = set()
    if bad_file.exists():
        try:
            bad_tickers = set(pd.read_csv(bad_file, header=None)[0].astype(str).tolist())
        except Exception:
            bad_tickers = set()

    bad_policy = getattr(config, "YFINANCE_BAD_TICKERS_POLICY", "skip_if_no_db_data")
    if bad_tickers and bad_policy == "skip_if_no_db_data":
        # Only skip symbols that are both in the bad list AND have no rows in the DB.
        # This prevents "poisoning" the cache during temporary Yahoo issues.
        bad_no_data: set[str] = set()
        for chunk in _chunked(list(bad_tickers), 900):
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(f"SELECT DISTINCT symbol FROM daily_prices WHERE symbol IN ({placeholders})", chunk)
            has_data = {row[0] for row in cursor.fetchall()}
            bad_no_data.update(set(chunk) - has_data)
        bad_tickers = bad_no_data
    elif bad_policy == "ignore":
        bad_tickers = set()

    symbols = [s for s in symbols if s not in bad_tickers]

    # Use batched multi-ticker downloads for short date ranges (much faster, fewer rate limits)
    bulk_enabled = bool(getattr(config, "YFINANCE_BULK_DOWNLOAD", True))
    bulk_max_range_days = int(getattr(config, "YFINANCE_BULK_MAX_RANGE_DAYS", 14) or 14)
    bulk_batch_size = int(getattr(config, "YFINANCE_BULK_BATCH_SIZE", 200) or 200)
    bulk_min_symbols = int(getattr(config, "YFINANCE_BULK_MIN_SYMBOLS", 50) or 50)
    bulk_threads = bool(getattr(config, "YFINANCE_BULK_THREADS", True))

    use_bulk = (
        bulk_enabled
        and start_date
        and end_date
        and (requested_range_days is not None)
        and (requested_range_days <= bulk_max_range_days)
        and (len(symbols) >= bulk_min_symbols)
    )

    def _end_date_exclusive(end_inclusive: str | None) -> str | None:
        if not end_inclusive:
            return None
        try:
            d1 = datetime.strptime(end_inclusive, "%Y-%m-%d")
            return (d1 + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            return end_inclusive

    def _extract_symbol_frame(df: pd.DataFrame, sym: str) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        # Common shapes:
        # 1) columns: (SYM, Open/High/Low/Close/Adj Close/Volume) when group_by="ticker"
        # 2) columns: (Open/High/..., SYM) depending on yfinance version/options
        lvl0 = set(df.columns.get_level_values(0))
        if sym in lvl0:
            return df[sym]
        lvl1 = set(df.columns.get_level_values(1))
        if sym in lvl1:
            try:
                return df.xs(sym, axis=1, level=1, drop_level=True)
            except Exception:
                return None
        return None

    def _insert_symbol_frame(sym: str, sym_df: pd.DataFrame) -> int:
        if sym_df is None or sym_df.empty:
            return 0

        # Normalize index + columns
        try:
            if getattr(sym_df.index, "tz", None) is not None:
                sym_df = sym_df.copy()
                sym_df.index = sym_df.index.tz_localize(None)
        except Exception:
            pass

        sym_df = sym_df.copy()
        sym_df.columns = [str(c).lower().strip().replace(" ", "_") for c in sym_df.columns]

        required = ["open", "high", "low", "close", "volume"]
        if not all(c in sym_df.columns for c in required):
            return 0

        sym_df = sym_df[required].dropna(subset=["close"])
        if sym_df.empty:
            return 0

        rows = []
        for dt, row in sym_df.iterrows():
            try:
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = str(dt)[:10]

            vol = row.get("volume")
            vol_i = int(vol) if pd.notna(vol) else 0

            rows.append(
                (
                    sym,
                    date_str,
                    float(row["open"]) if pd.notna(row["open"]) else None,
                    float(row["high"]) if pd.notna(row["high"]) else None,
                    float(row["low"]) if pd.notna(row["low"]) else None,
                    float(row["close"]) if pd.notna(row["close"]) else None,
                    vol_i,
                )
            )

        cursor.executemany(
            """
            INSERT OR REPLACE INTO daily_prices
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return 1

    if use_bulk:
        batches = _chunked(symbols, bulk_batch_size)
        iterator = tqdm(batches, desc="Downloading price history (batched)") if progress else batches

        successful = 0
        empty_count = 0
        new_bad = 0

        end_date_yf = _end_date_exclusive(end_date)
        for batch in iterator:
            try:
                df = yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date_yf,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=bulk_threads,
                    progress=False,
                )
            except Exception as e:
                logger.warning(f"yfinance batch download failed for {len(batch)} symbols: {e}")
                empty_count += len(batch)
                continue

            if df is None or df.empty:
                empty_count += len(batch)
                continue

            for sym in batch:
                sym_df = _extract_symbol_frame(df, sym)
                ok_sym = _insert_symbol_frame(sym, sym_df)
                if ok_sym:
                    successful += 1
                else:
                    empty_count += 1

            conn.commit()
            if sleep_seconds and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        conn.close()
        if empty_count or new_bad:
            logger.warning(
                f"Downloaded price history for {successful}/{len(symbols)} symbols "
                f"({empty_count} empty; {new_bad} newly marked bad)"
            )
        else:
            logger.info(f"Downloaded price history for {successful}/{len(symbols)} symbols")
        return successful

    iterator = tqdm(symbols, desc="Downloading price history") if progress else symbols

    for i, symbol in enumerate(iterator):
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date:
                # yfinance end date is exclusive; add +1 day to make end_date inclusive
                hist = ticker.history(start=start_date, end=_end_date_exclusive(end_date))
            else:
                hist = ticker.history(period=period)
            
            if hist.empty:
                empty_count += 1
                # If we requested a meaningful history window and got nothing, stop retrying forever.
                if requested_range_days is not None and requested_range_days >= 365 and symbol not in bad_tickers:
                    # Only mark bad if the symbol has no rows in the DB at all.
                    cursor.execute("SELECT 1 FROM daily_prices WHERE symbol = ? LIMIT 1", (symbol,))
                    has_any = cursor.fetchone() is not None
                    if has_any:
                        continue
                    try:
                        with open(bad_file, "a", encoding="utf-8") as f:
                            f.write(f"{symbol}\n")
                        bad_tickers.add(symbol)
                        new_bad += 1
                    except Exception:
                        pass
                continue
            
            # Prepare data for insertion
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            
            for _, row in hist.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_prices 
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date_str,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']) if pd.notna(row['volume']) else 0
                ))
            
            successful += 1
            
            # Periodic commit for durability
            if (i + 1) % commit_interval == 0:
                conn.commit()
            
            if sleep_seconds and sleep_seconds > 0:
                time.sleep(sleep_seconds)
                 
        except Exception as e:
            msg = str(e).lower()
            # Only mark bad for strong signals. Most "possibly delisted" spam is
            # just Yahoo not having the requested bar (or rate limiting).
            mark_bad = "no timezone found" in msg

            if mark_bad:
                try:
                    with open(bad_file, "a", encoding="utf-8") as f:
                        f.write(f"{symbol}\n")
                except Exception:
                    pass
            logger.debug(f"Error downloading {symbol}: {e}")
            continue
    
    conn.commit()
    conn.close()
    if empty_count or new_bad:
        logger.warning(
            f"Downloaded price history for {successful}/{len(symbols)} symbols "
            f"({empty_count} empty; {new_bad} newly marked bad)"
        )
    else:
        logger.info(f"Downloaded price history for {successful}/{len(symbols)} symbols")
    return successful


def update_recent_data(symbols: list[str] = None, days: int = 5) -> int:
    """
    Incremental update: fetch data from last stored date to today.
    
    Args:
        symbols: List of symbols to update (None = all in DB)
        days: Number of recent days to fetch
    
    Returns:
        Number of symbols updated
    """
    conn = get_connection()
    cursor = conn.cursor()

    if symbols is None:
        # Only update "active" symbols (volume floor) based on latest DB date
        cursor.execute("SELECT MAX(date) FROM daily_prices")
        latest_db_date = cursor.fetchone()[0]
        min_vol = getattr(config, "MIN_AVG_DAILY_VOLUME", 0)
        if latest_db_date:
            cursor.execute(
                "SELECT DISTINCT symbol FROM daily_prices WHERE date = ? AND volume >= ?",
                (latest_db_date, min_vol),
            )
            symbols = [row[0] for row in cursor.fetchall()]
        else:
            cursor.execute("SELECT DISTINCT symbol FROM daily_prices")
            symbols = [row[0] for row in cursor.fetchall()]

    if not symbols:
        logger.warning("No symbols to update")
        conn.close()
        return 0

    # Determine end_date (respect JP daily bar readiness)
    jst = _jst_now()
    ready_hour = getattr(config, "YFINANCE_DAILY_READY_HOUR", 23)
    skip_today_always = getattr(config, "YFINANCE_SKIP_TODAY_ALWAYS", False)
    today_str = jst.strftime("%Y-%m-%d")
    end_date = today_str
    if skip_today_always or jst.hour < ready_hour:
        end_date = (jst - timedelta(days=1)).strftime('%Y-%m-%d')

    yahoo_latest = _get_latest_yahoo_daily_bar_date()
    if end_date and yahoo_latest and yahoo_latest < end_date:
        logger.warning(f"Yahoo latest daily bar appears to be {yahoo_latest}. Limiting update end_date from {end_date} to {yahoo_latest}.")
        end_date = yahoo_latest

    # If DB is already up to yesterday, skip morning updates entirely
    try:
        conn_check = get_connection()
        cur = conn_check.cursor()
        cur.execute("SELECT MAX(date) FROM daily_prices")
        max_date = cur.fetchone()[0]
        conn_check.close()
    except Exception:
        max_date = None

    if (skip_today_always or jst.hour < ready_hour) and max_date and max_date >= end_date:
        logger.info(
            f"Skipping update: latest DB date {max_date} already up to {end_date} "
            f"(before {ready_hour}:00 JST)"
        )
        conn.close()
        return 0

    # Fast path: refresh the last `days` calendar days for all target symbols.
    # This is much faster than per-symbol updates and avoids yfinance rate limits.
    try:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(days or 5))).strftime("%Y-%m-%d")
    except Exception:
        start_date = None
    conn.close()

    updated = download_price_history(
        symbols,
        start_date=start_date,
        end_date=end_date,
        commit_interval=int(getattr(config, "YFINANCE_UPDATE_COMMIT_INTERVAL", 200) or 200),
        sleep_seconds=float(getattr(config, "YFINANCE_UPDATE_SLEEP_SECONDS", 0.0) or 0.0),
        progress=True,
    )

    logger.info(f"Updated recent data for {updated} symbols")
    return int(updated)


def get_daily_bars(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Retrieve daily OHLCV bars for a symbol from database.
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        Indexed by date
    """
    conn = get_connection()
    query = """
        SELECT date, open, high, low, close, volume
        FROM daily_prices
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
    conn.close()
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.set_index("date")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
    
    return df


def get_daily_bars_batch(
    symbols: list[str],
    start_date: str,
    end_date: str
) -> dict[str, pd.DataFrame]:
    """
    Batch fetch daily bars for multiple symbols.
    More efficient than individual calls.
    
    Returns:
        Dict: {symbol: DataFrame}
    """
    conn = get_connection()
    
    placeholders = ','.join(['?' for _ in symbols])
    query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM daily_prices
        WHERE symbol IN ({placeholders}) 
        AND date >= ? AND date <= ?
        ORDER BY symbol, date
    """
    params = symbols + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    result = {}
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(["symbol", "date"])
        for symbol in symbols:
            symbol_df = df[df["symbol"] == symbol].copy()
            if not symbol_df.empty:
                symbol_df = symbol_df.drop("symbol", axis=1)
                symbol_df = symbol_df.drop_duplicates(subset=["date"], keep="last")
                symbol_df = symbol_df.set_index("date")
                symbol_df = symbol_df[~symbol_df.index.duplicated(keep="last")]
                symbol_df = symbol_df.sort_index()
                result[symbol] = symbol_df
    
    return result


def get_available_symbols(min_rows: int = 60) -> list[str]:
    """
    Get list of symbols with sufficient data in the database.
    
    Args:
        min_rows: Minimum number of price rows required
    
    Returns:
        List of symbol strings
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM daily_prices
        GROUP BY symbol
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (min_rows,))
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols


def get_missing_symbols(
    min_rows: int = 60,
    exclude_nikkei: bool = True,
    cache_duration_hours: int = None,
) -> list[str]:
    """
    Return tickers that are NOT yet in the DB (or have fewer than min_rows).
    """
    if cache_duration_hours is None:
        cache_duration_hours = config.CACHE_TICKER_LIST_HOURS

    all_symbols = get_all_tse_tickers(cache_duration_hours=cache_duration_hours)
    if not all_symbols:
        return []

    if exclude_nikkei:
        nikkei = get_nikkei_225_components(cache_duration_hours=cache_duration_hours)
        all_symbols = [s for s in all_symbols if s not in nikkei]

    if config.EXCLUDED_SYMBOLS:
        all_symbols = [s for s in all_symbols if s not in config.EXCLUDED_SYMBOLS]

    existing = set(get_available_symbols(min_rows=min_rows))
    missing = [s for s in all_symbols if s not in existing]
    return missing


def expand_db_incremental(
    max_new: int = 200,
    start_date: str = "2024-01-01",
    end_date: str = None,
    min_rows: int = 60,
    exclude_nikkei: bool = True,
    cache_duration_hours: int = None,
    shuffle: bool = True,
    commit_interval: int = 50,
    sleep_seconds: float = 0.2,
    progress: bool = True,
) -> int:
    """
    Incrementally add missing tickers to the DB (capped per run).
    Returns the number of symbols successfully downloaded.
    """
    if max_new is None or max_new < 0:
        max_new = 0

    missing = get_missing_symbols(
        min_rows=min_rows,
        exclude_nikkei=exclude_nikkei,
        cache_duration_hours=cache_duration_hours,
    )

    if not missing:
        logger.info("No missing symbols to expand.")
        return 0

    if shuffle:
        random.shuffle(missing)

    if max_new > 0:
        missing = missing[:max_new]

    return download_price_history(
        missing,
        start_date=start_date,
        end_date=end_date,
        commit_interval=commit_interval,
        sleep_seconds=sleep_seconds,
        progress=progress,
    )


def update_market_caps_incremental(
    max_symbols: int = 300,
    sleep_seconds: float = 0.2,
    progress: bool = True,
) -> int:
    """
    Populate missing market caps in symbol_info using yfinance.
    Returns number of symbols updated with a market cap.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT symbol
        FROM symbol_info
        WHERE market_cap IS NULL
        OR market_cap <= 0
        """
    )
    existing_rows = {row[0] for row in cursor.fetchall()}

    # Start with symbols that have price data
    symbols_with_data = get_available_symbols(min_rows=60)

    # Ensure symbol_info rows exist for symbols_with_data
    to_seed = [s for s in symbols_with_data if s not in existing_rows]
    for s in to_seed:
        cursor.execute(
            "INSERT OR IGNORE INTO symbol_info (symbol, last_updated) VALUES (?, ?)",
            (s, datetime.now().strftime("%Y-%m-%d")),
        )

    # Now re-select missing caps
    cursor.execute(
        """
        SELECT symbol
        FROM symbol_info
        WHERE market_cap IS NULL
        OR market_cap <= 0
        """
    )
    missing = [row[0] for row in cursor.fetchall()]
    conn.commit()
    conn.close()

    if not missing:
        logger.info("No missing market caps to update.")
        return 0

    if max_symbols and max_symbols > 0:
        missing = missing[:max_symbols]

    updated = 0
    iterator = tqdm(missing, desc="Updating market caps") if progress else missing

    conn = get_connection()
    cursor = conn.cursor()
    for symbol in iterator:
        try:
            ticker = yf.Ticker(symbol)
            # Try fast_info first (lighter)
            mc = None
            name = None
            sector = None

            try:
                fi = getattr(ticker, "fast_info", None)
                if fi and "market_cap" in fi:
                    mc = fi.get("market_cap")
            except Exception:
                pass

            if mc is None:
                info = ticker.info
                mc = info.get("marketCap") or info.get("market_cap")
                name = info.get("shortName") or info.get("longName")
                sector = info.get("sector")

            if mc and mc > 0:
                cursor.execute(
                    """
                    INSERT INTO symbol_info (symbol, name, sector, market_cap, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        name = COALESCE(excluded.name, symbol_info.name),
                        sector = COALESCE(excluded.sector, symbol_info.sector),
                        market_cap = excluded.market_cap,
                        last_updated = excluded.last_updated
                    """,
                    (
                        symbol,
                        name,
                        sector,
                        float(mc),
                        datetime.now().strftime("%Y-%m-%d"),
                    ),
                )
                updated += 1

            if sleep_seconds and sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception:
            continue

    conn.commit()
    conn.close()
    return updated


# =============================================================================
# JPX Short Interest Data
# =============================================================================

def fetch_jpx_short_data(cache_duration_hours: int = None) -> dict[str, dict]:
    """
    Scrape short-selling data from JPX website.
    
    Returns:
        Dict: {symbol: {'short_ratio': float, 'short_volume': int, 'total_volume': int}}
    """
    if cache_duration_hours is None:
        cache_duration_hours = config.CACHE_JPX_DATA_HOURS
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    cache_file = config.CACHE_DIR / f"jpx_data_cache_{today_str}.xlsx"
    
    # Check cache
    if cache_file.exists():
        file_mod_time = cache_file.stat().st_mtime
        if (time.time() - file_mod_time) / 3600 < cache_duration_hours:
            logger.info(f"Using cached JPX data for today")
            try:
                df = pd.read_excel(cache_file)
                return _process_jpx_dataframe(df)
            except Exception as e:
                logger.warning(f"Could not read cached JPX file: {e}")
    
    logger.info("Fetching fresh short-selling data from JPX...")
    
    try:
        main_page_url = "https://www.jpx.co.jp/english/markets/public/short-selling/"
        response = requests.get(main_page_url, timeout=15)
        if response.status_code != 200:
            logger.error(f"Failed to access JPX page. Status: {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.content, 'lxml')
        file_link = None
        
        # Find the Excel download link
        for link in soup.find_all('a'):
            link_text = link.text or ''
            if "Short Selling Positions" in link_text and "Excel" in link_text:
                file_link = link.get('href', '')
                break
        
        if not file_link:
            # Try alternative method
            for link in soup.find_all('a', href=True):
                if '.xlsx' in link['href'] or '.xls' in link['href']:
                    file_link = link['href']
                    break
        
        if not file_link:
            logger.error("Could not find JPX download link")
            return {}
        
        full_url = urllib.parse.urljoin(main_page_url, file_link)
        logger.info(f"Downloading from: {full_url}")
        
        excel_response = requests.get(full_url, timeout=20)
        if excel_response.status_code == 200:
            with open(cache_file, 'wb') as f:
                f.write(excel_response.content)
            df = pd.read_excel(excel_response.content)
            return _process_jpx_dataframe(df)
        else:
            logger.error(f"Failed to download JPX Excel file: {excel_response.status_code}")
            return {}
            
    except Exception as e:
        logger.error(f"Error fetching JPX data: {e}")
        return {}


def _process_jpx_dataframe(df: pd.DataFrame) -> dict[str, dict]:
    """
    Process raw JPX DataFrame into usable format.
    
    The JPX file structure:
    - Rows 0-7: Headers/metadata
    - Row 8-9: Column headers (Japanese/English)
    - Row 10+: Data
    - Column 2 (index 2): Stock Code
    - Column 10 (index 10): Ratio of Short Positions (e.g., 0.0334 = 3.34%)
    - Column 11 (index 11): Number of Short Positions in Shares
    
    Multiple rows exist per symbol (different short sellers), so we aggregate.
    """
    try:
        # The data starts after the header rows
        # Use iloc to access by position since column names are messy
        
        result = {}
        
        for idx in range(len(df)):
            try:
                row = df.iloc[idx]
                
                # Column 2 is the stock code (0-indexed)
                code_raw = row.iloc[2] if len(row) > 2 else None
                if pd.isna(code_raw):
                    continue
                    
                code = str(code_raw).strip()
                
                # Skip non-numeric codes (header rows, etc.)
                if not code.isdigit():
                    continue
                
                symbol = code + '.T'
                
                # Column 10 is the short ratio (e.g., 0.0334)
                short_ratio_raw = row.iloc[10] if len(row) > 10 else None
                short_ratio = float(short_ratio_raw) if pd.notna(short_ratio_raw) else 0.0
                
                # Column 11 is the number of short shares
                short_shares_raw = row.iloc[11] if len(row) > 11 else None
                short_shares = int(float(short_shares_raw)) if pd.notna(short_shares_raw) else 0
                
                # Aggregate: keep the highest short ratio for each symbol
                # (multiple short sellers may be listed, we want the total exposure)
                if symbol in result:
                    # Accumulate short shares, keep max ratio
                    result[symbol]['short_volume'] += short_shares
                    result[symbol]['short_ratio'] = max(result[symbol]['short_ratio'], short_ratio)
                else:
                    result[symbol] = {
                        'short_ratio': short_ratio,
                        'short_volume': short_shares,
                    }
                    
            except Exception:
                continue
        
        # Log summary
        if result:
            avg_ratio = sum(d['short_ratio'] for d in result.values()) / len(result)
            max_ratio = max(d['short_ratio'] for d in result.values())
            logger.warning(f"Processed JPX data: {len(result)} symbols, avg ratio={avg_ratio:.2%}, max={max_ratio:.2%}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing JPX DataFrame: {e}")
        return {}


def get_jpx_short_for_date(date: str = None) -> dict[str, dict]:
    """
    Get short interest data for a specific date.
    Falls back to most recent available <= date (no lookahead).
    
    Returns:
        Dict: {symbol: {'short_ratio': float}}
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Try exact date first
    cursor.execute("""
        SELECT symbol, short_ratio, short_volume, total_volume
        FROM jpx_short_interest
        WHERE date = ?
    """, (date,))
    rows = cursor.fetchall()
    
    # If no data, get most recent
    if not rows:
        cursor.execute("SELECT MAX(date) FROM jpx_short_interest WHERE date <= ?", (date,))
        last_row = cursor.fetchone()
        last_date = last_row[0] if last_row and last_row[0] else None
        if last_date:
            cursor.execute("""
                SELECT symbol, short_ratio, short_volume, total_volume
                FROM jpx_short_interest
                WHERE date = ?
            """, (last_date,))
            rows = cursor.fetchall()
    
    conn.close()
    
    return {
        row[0]: {'short_ratio': row[1], 'short_volume': row[2], 'total_volume': row[3]}
        for row in rows
    }


def store_jpx_short_data(data: dict[str, dict], date: str = None):
    """Store JPX short data to database."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    for symbol, info in data.items():
        cursor.execute("""
            INSERT OR REPLACE INTO jpx_short_interest
            (symbol, date, short_volume, total_volume, short_ratio)
            VALUES (?, ?, ?, ?, ?)
        """, (
            symbol,
            date,
            info.get('short_volume', 0),
            info.get('total_volume', 0),
            info.get('short_ratio', 0.0)
        ))
    
    conn.commit()
    conn.close()
    logger.info(f"Stored JPX short data for {len(data)} symbols")


# =============================================================================
# Universe Building
# =============================================================================

def build_liquid_universe(
    date: str,
    top_n: int = None,
    exclude_nikkei: bool = None,
    min_volume: int = None,
) -> list[str]:
    """
    Build liquid universe for a specific date.
    
    Process:
        1. Get all symbols with data on this date
        2. Calculate notional (close * volume)
        3. Filter by minimum volume/notional
        4. Exclude Nikkei 225 (if enabled)
        5. Return top N symbols by notional
    
    Returns:
        List of qualifying symbol strings
    """
    if top_n is None:
        top_n = config.UNIVERSE_TOP_N
    if exclude_nikkei is None:
        exclude_nikkei = config.EXCLUDE_NIKKEI_225
    if min_volume is None:
        min_volume = config.MIN_AVG_DAILY_VOLUME
    
    conn = get_connection()
    
    # Get symbols with price data for this date (optionally include market cap)
    query = """
        SELECT p.symbol, p.close, p.volume, (p.close * p.volume) as notional, s.market_cap
        FROM daily_prices p
        LEFT JOIN symbol_info s ON p.symbol = s.symbol
        WHERE p.date = ? AND p.volume >= ?
        ORDER BY notional DESC
    """
    df = pd.read_sql_query(query, conn, params=(date, min_volume))
    conn.close()
    
    if df.empty:
        logger.warning(f"No data found for date {date}")
        return []
    
    # Apply market-cap filter if enabled
    if getattr(config, "ENFORCE_MARKET_CAP", False):
        max_cap = getattr(config, "MAX_MARKET_CAP_JPY", None)
        missing_policy = getattr(config, "MARKET_CAP_MISSING_POLICY", "include")
        if max_cap is not None:
            if missing_policy == "exclude":
                df = df[df["market_cap"].notna() & (df["market_cap"] <= max_cap)]
            else:
                df = df[df["market_cap"].isna() | (df["market_cap"] <= max_cap)]

    symbols = df['symbol'].tolist()
    
    # Exclude Nikkei 225
    if exclude_nikkei:
        nikkei_set = get_nikkei_225_components()
        symbols = [s for s in symbols if s not in nikkei_set]
    
    # Exclude configured exclusions
    if config.EXCLUDED_SYMBOLS:
        symbols = [s for s in symbols if s not in config.EXCLUDED_SYMBOLS]
    
    # Return top N (if set). If None/0, return full filtered universe.
    if top_n:
        return symbols[:top_n]
    return symbols


def get_universe_for_backtest(
    start_date: str,
    end_date: str,
    top_n: int = None,
) -> dict[str, list[str]]:
    """
    Get liquid universe for each trading date in the backtest period.
    
    Returns:
        Dict: {date_string: [list of symbols]}
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all unique dates in range
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (start_date, end_date))
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    universe_by_date = {}
    for date in dates:
        universe_by_date[date] = build_liquid_universe(date, top_n=top_n)
    
    return universe_by_date


# =============================================================================
# Backtest Run Logging
# =============================================================================

def log_backtest_run(
    start_date: str,
    end_date: str,
    params: dict,
    metrics: dict,
    notes: str = None
) -> int:
    """
    Log a backtest run to the database for reproducibility.
    
    Returns:
        ID of the inserted run
    """
    import json
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO backtest_runs
        (start_date, end_date, params_json, profit_factor, win_rate, 
         max_drawdown, total_trades, total_return, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        start_date,
        end_date,
        json.dumps(params),
        metrics.get('profit_factor'),
        metrics.get('win_rate'),
        metrics.get('max_drawdown'),
        metrics.get('total_trades'),
        metrics.get('total_return'),
        notes
    ))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Logged backtest run #{run_id}")
    return run_id


def get_recent_backtest_runs(limit: int = 20) -> pd.DataFrame:
    """Get recent backtest runs from the database."""
    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT * FROM backtest_runs
        ORDER BY run_date DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


# =============================================================================
# Data Initialization
# =============================================================================

def ensure_data(
    symbols: list[str] = None,
    start_date: str = None,
    force_refresh: bool = False,
    progress: bool = True
):
    """
    Ensure database has required data.
    Creates database if needed, downloads price history.
    
    Args:
        symbols: List of symbols (None = fetch all TSE)
        start_date: Start date for history (None = use default period)
        force_refresh: Force re-download even if data exists
        progress: Show progress bar
    """
    # Setup database
    setup_database()
    
    # Get symbols if not provided
    if symbols is None:
        symbols = get_all_tse_tickers()
    
    if not symbols:
        logger.error("No symbols to download")
        return
    
    # Check existing data
    existing = set(get_available_symbols(min_rows=10))
    
    if force_refresh:
        to_download = symbols
    else:
        to_download = [s for s in symbols if s not in existing]
    
    if to_download:
        logger.info(f"Downloading data for {len(to_download)} symbols...")
        download_price_history(to_download, start_date=start_date, progress=progress)
    else:
        logger.info("All symbols already have data")
    
    # Update recent data for existing symbols
    if existing:
        logger.info(f"Updating recent data for {len(existing)} existing symbols...")
        update_recent_data(list(existing)[:100])  # Limit for speed


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing data_manager module...")
    
    # Setup database
    conn = setup_database()
    conn.close()
    print(f"✓ Database created: {config.DATABASE_FILE}")
    
    # Test ticker fetching
    tickers = get_all_tse_tickers()
    print(f"✓ Fetched {len(tickers)} TSE tickers")
    
    # Test Nikkei 225
    nikkei = get_nikkei_225_components()
    print(f"✓ Fetched {len(nikkei)} Nikkei 225 components")
    
    # Test JPX short data
    jpx = fetch_jpx_short_data()
    print(f"✓ Fetched JPX short data for {len(jpx)} symbols")
    
    print("\ndata_manager module tests passed!")
