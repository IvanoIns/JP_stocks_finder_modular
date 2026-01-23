"""
JP Stocks Modular Trading System — Code Skeleton

This file provides detailed interfaces for all modules.
Each section describes the module's responsibilities, key functions,
parameters, and notes on porting from existing code.

===============================================================================
                                TABLE OF CONTENTS
===============================================================================
1. config.py           - Centralized configuration
2. data_manager.py     - Database and data ingestion
3. technical_analysis.py - Indicator calculations
4. scanners.py         - Entry signal strategies
5. backtesting.py      - Backtest engine
6. optimizer.py        - Grid search and walk-forward
7. run_backtest.py     - CLI entry point
===============================================================================
"""

# =============================================================================
# 1. CONFIG.PY — Centralized Configuration
# =============================================================================
"""
PURPOSE:
    Single source of truth for all system parameters.
    Allows session overrides in notebooks without editing files.

PORTING SOURCE:
    - Structure from: modular/config.py (crypto bot)
    - Parameters from: JP_stocks_rising/JP stocks py.py CONFIG dict

KEY CHANGES FROM CRYPTO:
    - Remove USDC/USDT quote policy (JPY only)
    - Add Nikkei 225 exclusion setting
    - Daily bars only (no hourly/4h options)
    - Add JPX short data settings
"""

# --- config.py SKELETON ---

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Database Settings
# =============================================================================
DATABASE_FILE = Path(__file__).parent / "jp_stocks.db"
CACHE_DIR = Path(__file__).parent / "cache"

# =============================================================================
# Data Source Settings
# =============================================================================
CACHE_TICKER_LIST_HOURS = 720  # 30 days
CACHE_JPX_DATA_HOURS = 12      # 12 hours

# =============================================================================
# Universe Filters
# =============================================================================
MAX_MARKET_CAP_JPY = 500_000_000_000     # 500B JPY - exclude mega-caps
MIN_AVG_DAILY_VOLUME = 100_000           # Minimum average daily volume
EXCLUDE_NIKKEI_225 = True                # Exclude Nikkei 225 components
UNIVERSE_TOP_N = 100                     # Top N by volume for backtests

# Performance filters for universe (from JP prototype)
PERFORMANCE_LOOKBACK_DAYS = 180
PERFORMANCE_MIN_RISE_PERCENT = 2.0
PERFORMANCE_NEAR_52W_HIGH_THRESHOLD = 0.10

# =============================================================================
# Entry Filters (Scanner Thresholds)
# =============================================================================
ENTRY_FILTERS = {
    'MAX_RSI_ENTRY': 70,          # Maximum RSI for entry
    'MIN_VOLUME_SURGE': 1.5,      # Minimum volume spike factor
    'MIN_SCORE': 50,              # Minimum scanner score
    'REQUIRE_MACD_BULLISH': False,
}

# Individual variables for direct access
MAX_RSI_ENTRY = ENTRY_FILTERS['MAX_RSI_ENTRY']
MIN_VOLUME_SURGE = ENTRY_FILTERS['MIN_VOLUME_SURGE']
MIN_SCORE = ENTRY_FILTERS['MIN_SCORE']

# =============================================================================
# Scanner-Specific Settings (from JP prototype)
# =============================================================================
SCANNER_CONFIG = {
    # Momentum Star
    'MOMENTUM_EMA_FAST': 20,
    'MOMENTUM_EMA_SLOW': 50,
    'MOMENTUM_RSI_PERIOD': 14,
    'MOMENTUM_RSI_MIN': 60,
    'MOMENTUM_RSI_MAX': 80,
    'MOMENTUM_VOLUME_SPIKE_FACTOR': 1.5,
    'MOMENTUM_LOW_SHORT_INTEREST': 0.02,
    'MOMENTUM_HIGH_SHORT_WARNING': 0.08,
    
    # Reversal Rocket
    'REVERSAL_ZSCORE_PERIOD': 30,
    'REVERSAL_ZSCORE_THRESHOLD': -2.0,
    'REVERSAL_VOLUME_SPIKE_FACTOR': 3.0,
    'REVERSAL_HIGH_SHORT_FUEL': 0.10,
    
    # Consolidation Breakout
    'CONSOLIDATION_RANGE_MAX': 0.08,      # 8% max range
    'CONSOLIDATION_VOLATILITY_DROP': 0.20, # 20% ATR contraction
    
    # Burst Detection (from forensic analysis)
    'RSI_MIN': 30, 'RSI_MAX': 70,
    'VOLUME_RATIO_MIN': 80, 'VOLUME_RATIO_MAX': 200,
    'VOLATILITY_MIN': 20, 'VOLATILITY_MAX': 60,
    'SMA_RATIO_MIN': 0.85, 'SMA_RATIO_MAX': 1.15,
    'MIN_RISK_REWARD': 2.0,
}

# =============================================================================
# Exit Strategy Settings
# =============================================================================
EXIT_STRATEGIES = {
    'conservative': {
        'stop_loss_pct': 0.05,
        'target_1_pct': 0.03,
        'target_1_portion': 0.50,
        'target_2_pct': 0.06,
    },
    'aggressive': {
        'stop_loss_pct': 0.07,
        'target_1_pct': 0.05,
        'target_1_portion': 0.33,
        'target_2_pct': 0.10,
    }
}

STOP_LOSS_PCT = 0.05
TARGET_1_PCT = 0.03
TARGET_1_PORTION = 0.50
TARGET_2_PCT = 0.06
EXIT_MODE = 'default'  # 'default', 'trailing', 'breakeven', 'breakeven_trailing'
TRAILING_STOP_PCT = 0.03

# =============================================================================
# Position Sizing
# =============================================================================
MAX_POSITIONS = 5
POSITION_SIZING_METHOD = 'fixed_fractional'  # 'fixed_fractional', 'volatility_adjusted'
FIXED_FRACTIONAL_PCT = 0.20   # 20% per position
MAX_POSITION_PCT = 0.25       # Max 25% in single position
RISK_PER_TRADE_PCT = 0.02     # 2% risk per trade for volatility sizing

# =============================================================================
# Backtest Settings
# =============================================================================
BACKTEST_TIMEFRAME = '1d'     # Daily bars only for JP stocks
BACKTEST_SLIPPAGE_PCT = 0.001 # 0.1% slippage
BACKTEST_COMMISSION_PCT = 0.001 # 0.1% commission

# =============================================================================
# Optimizer Constraints
# =============================================================================
MAX_DRAWDOWN_CAP = 0.25       # Max 25% drawdown
MIN_WIN_RATE = 0.45           # Min 45% win rate
MIN_TRADES = 20               # Minimum trades for validity

# =============================================================================
# API Settings
# =============================================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
API_RATE_LIMIT_DELAY_SECONDS = 15



# =============================================================================
# 2. DATA_MANAGER.PY — Database and Data Ingestion
# =============================================================================
"""
PURPOSE:
    - Create and manage SQLite database
    - Fetch/cache ticker lists from investpy
    - Download price history from yfinance
    - Scrape JPX short-selling data
    - Provide data access functions for backtesting

PORTING SOURCES:
    - DB pattern: modular/data_manager.py (crypto)
    - Ticker fetching: JP_stocks_rising/JP stocks py.py (get_all_tse_tickers, etc.)
    - JPX scraping: JP_stocks_rising/JP stocks py.py (fetch_and_process_jpx_data)

KEY FUNCTIONS:
"""

# --- data_manager.py SKELETON ---

import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import investpy
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import logging
import time

# from config import (DATABASE_FILE, CACHE_DIR, CACHE_TICKER_LIST_HOURS,
#                     CACHE_JPX_DATA_HOURS, EXCLUDE_NIKKEI_225, ...)


def setup_database() -> sqlite3.Connection:
    """
    Create database and all required tables.
    
    Tables:
        - daily_prices: OHLCV data (symbol, date, open, high, low, close, volume)
        - symbol_info: Metadata (symbol, name, sector, market_cap, is_nikkei_225)
        - jpx_short_interest: Short selling data (symbol, date, short_ratio)
        - backtest_runs: History of backtest runs for reproducibility
    
    Returns:
        sqlite3.Connection to the database
    
    PORTING: Similar to modular/data_manager.py create_database()
    """
    pass


def get_all_tse_tickers(cache_duration_hours: int = None) -> list[str]:
    """
    Fetch all TSE tickers from investpy with caching.
    
    Process:
        1. Check if cache file exists and is fresh
        2. If fresh, load from cache
        3. If stale, fetch from investpy.stocks.get_stocks(country='japan')
        4. Convert to yfinance format (add .T suffix)
        5. Save to cache
    
    Returns:
        List of ticker symbols in yfinance format (e.g., ['7203.T', '6758.T', ...])
    
    PORTING: Direct copy from JP_stocks_rising/JP stocks py.py
    """
    pass


def get_nikkei_225_components(cache_duration_hours: int = None) -> set[str]:
    """
    Fetch Nikkei 225 components from Wikipedia for exclusion.
    
    PORTING: Direct copy from JP_stocks_rising/JP stocks py.py
    """
    pass


def filter_investment_universe(
    all_tickers: list[str],
    nikkei_exclusion_set: set[str] = None,
    max_market_cap: float = None,
    min_avg_volume: int = None,
) -> list[str]:
    """
    Filter tickers to eligible investment universe.
    
    Filters applied:
        1. Exclude Nikkei 225 (if enabled)
        2. Market cap <= MAX_MARKET_CAP_JPY
        3. Average volume >= MIN_AVG_DAILY_VOLUME
        4. Has sufficient price history
        5. Performance criteria (from JP prototype)
    
    PORTING: From JP_stocks_rising/JP stocks py.py filter_investment_universe()
    """
    pass


def download_price_history(
    symbols: list[str],
    start_date: str,
    end_date: str,
    commit_interval: int = 50,
) -> int:
    """
    Download and store daily OHLCV data for symbols.
    
    Process:
        1. For each symbol, fetch history via yfinance
        2. Upsert into daily_prices table
        3. Commit periodically for durability
    
    Args:
        symbols: List of tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        commit_interval: Commit after this many symbols
    
    Returns:
        Number of symbols processed
    
    PORTING: Adapted from modular/data_manager.py download_full_history_binance()
             but using yfinance instead of Binance API
    """
    pass


def update_recent_data(symbols: list[str] = None) -> int:
    """
    Incremental update: fetch data from last stored date to today.
    
    PORTING: Adapted from modular/data_manager.py update_recent_data_binance()
    """
    pass


def fetch_jpx_short_data(cache_duration_hours: int = None) -> dict[str, dict]:
    """
    Scrape short-selling data from JPX website.
    
    Process:
        1. Check cache freshness
        2. Fetch Excel file from JPX
        3. Parse and calculate short ratios
        4. Store in jpx_short_interest table
        5. Return dict: {symbol: {'short_ratio': float}}
    
    PORTING: Direct copy from JP_stocks_rising/JP stocks py.py 
             fetch_and_process_jpx_data()
    """
    pass


def get_daily_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve daily OHLCV bars for a symbol from database.
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        Indexed by date
    
    PORTING: Similar to modular/data_manager.py get_historical_prices()
    """
    pass


def get_daily_bars_batch(
    symbols: list[str],
    start_date: str,
    end_date: str
) -> dict[str, pd.DataFrame]:
    """
    Batch fetch daily bars for multiple symbols.
    More efficient than individual calls.
    
    PORTING: Similar to modular/data_manager.py get_hourly_bars_for_symbols()
    """
    pass


def build_liquid_universe(date: str, top_n: int = None) -> list[str]:
    """
    Build liquid universe for a specific date.
    
    Process:
        1. Get all symbols with data on this date
        2. Calculate notional (close * volume)
        3. Filter by minimum notional
        4. Sort by notional descending
        5. Return top N symbols
    
    PORTING: Adapted from modular/backtesting.py universe building logic
    """
    pass


def get_jpx_short_for_date(date: str) -> dict[str, dict]:
    """
    Get short interest data for a specific date.
    Falls back to most recent available if date not found.
    
    Returns:
        Dict: {symbol: {'short_ratio': float}}
    """
    pass


def log_backtest_run(params: dict, metrics: dict) -> int:
    """
    Log a backtest run to the database for reproducibility.
    
    PORTING: From modular/optimizer.py WFA logging
    """
    pass



# =============================================================================
# 3. TECHNICAL_ANALYSIS.PY — Indicator Calculations
# =============================================================================
"""
PURPOSE:
    Calculate all technical indicators used by scanners.
    Pure functions, no side effects.

PORTING SOURCES:
    - RSI, MACD, Bollinger, Volume Surge: modular/technical_analysis.py (crypto)
    - EMA, ATR, Z-Score, Higher Lows: JP_stocks_rising/JP stocks py.py
    - Candlestick patterns: JP_stocks_rising/JP stocks py.py

DESIGN:
    - All functions take pandas Series/DataFrame
    - All functions return pandas Series or scalar
    - No database access, no config access (params passed in)
"""

# --- technical_analysis.py SKELETON ---

import pandas as pd
import numpy as np
from scipy import stats


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    PORTING: Identical in both crypto bot and JP prototype
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.0001)
    return 100 - (100 / (1 + rs))


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    PORTING: From JP prototype
    """
    return prices.ewm(span=span, adjust=False).mean()


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    """
    return prices.rolling(window=period).mean()


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal line, and Histogram.
    
    Returns:
        (macd_line, signal_line, histogram)
    
    PORTING: From modular/technical_analysis.py
    """
    pass


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        (upper_band, middle_band, lower_band)
    
    PORTING: From modular/technical_analysis.py
    """
    pass


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    PORTING: From JP prototype RisingStarsScanner.calculate_atr()
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()


def calculate_z_score(prices: pd.Series, period: int = 30) -> pd.Series:
    """
    Calculate Z-Score (standard deviations from rolling mean).
    
    PORTING: From JP prototype
    """
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    return (prices - rolling_mean) / rolling_std.replace(0, 0.0001)


def calculate_volume_surge(volume: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Calculate volume relative to moving average.
    
    PORTING: From modular/technical_analysis.py
    """
    return volume / volume.rolling(window=lookback).mean()


def detect_higher_lows(lows: np.ndarray, min_improvement: float = 0.001) -> bool:
    """
    Detect higher lows pattern (accumulation signature).
    
    PORTING: From JP prototype RisingStarsScanner.detect_higher_lows()
    """
    pass


def calculate_risk_reward(data: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate risk/reward ratio for current setup.
    
    PORTING: From JP prototype RisingStarsScanner.calculate_risk_reward()
    """
    pass


def calculate_trend_strength(prices: pd.Series, period: int = 20) -> tuple[float, float]:
    """
    Calculate linear regression slope and R-squared.
    
    Returns:
        (slope, r_squared)
    
    PORTING: From JP prototype (used in scan_relative_strength)
    """
    x = np.arange(len(prices[-period:]))
    y = prices.values[-period:]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, r_value ** 2


def check_bullish_reversal_pattern(data: pd.DataFrame) -> tuple[bool, str]:
    """
    Detect bullish reversal candlestick patterns.
    
    Patterns detected:
        - Hammer
        - Bullish Engulfing
        - Morning Star
    
    Returns:
        (pattern_detected, pattern_name)
    
    PORTING: From JP prototype check_for_bullish_reversal_pattern()
    """
    pass


def calculate_all_indicators(data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Calculate all indicators for a price DataFrame.
    Adds columns for each indicator.
    
    Columns added:
        - rsi, ema_fast, ema_slow, sma_20, sma_50
        - macd, macd_signal, macd_hist
        - bb_upper, bb_middle, bb_lower
        - atr, z_score, volume_surge
    
    Returns:
        DataFrame with original + indicator columns
    
    PORTING: New consolidated function
    """
    pass



# =============================================================================
# 4. SCANNERS.PY — Entry Signal Strategies
# =============================================================================
"""
PURPOSE:
    Detect entry signals using various strategies.
    Each scanner returns a score and list of reasons.

PORTING SOURCE:
    - All strategies from JP_stocks_rising/JP stocks py.py RisingStarsScanner class

DESIGN:
    - Each scanner is a standalone function
    - Takes: data (DataFrame with indicators), jpx_data (dict), config (dict)
    - Returns: (score: int, reasons: list[str]) or (0, []) if no signal

SCANNERS INCLUDED:
    1. Momentum Star - Uptrend + RSI zone + volume spike
    2. Reversal Rocket - Oversold + Z-score + reversal patterns
    3. Consolidation Breakout - Tight range + volatility contraction
    4. Relative Strength - Price > MAs + consistent trend
    5. Burst Candidates - Forensic signature from winners
    6. Oversold Bounce - Mean reversion
    7. Volatility Explosion - High volatility + near lows
    8. Power Combinations - Multiple signals together
"""

# --- scanners.py SKELETON ---

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# from technical_analysis import (calculate_rsi, calculate_ema, ...)
# from config import SCANNER_CONFIG


def scan_momentum_star(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Find stocks in strong uptrend with controlled RSI and volume.
    
    Signals:
        +30: EMA Fast > EMA Slow (uptrend)
        +20: RSI in sweet spot (60-80)
        +20: Volume spike > 1.5x average
        +20: Low short interest (< 2%)
        -30: High short interest warning (> 8%)
    
    Threshold: 50 points minimum
    
    PORTING: From JP prototype scan_for_momentum_star()
    """
    pass


def scan_reversal_rocket(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Find oversold stocks with reversal potential.
    
    Signals:
        +20: Price < EMA Slow (downtrend prerequisite)
        +30: Z-Score < -2.0 (oversold)
        +20: Capitulation volume (3x+ average)
        +20: Bullish reversal pattern detected
        +30: High short interest (squeeze fuel > 10%)
    
    Threshold: 70 points minimum
    
    PORTING: From JP prototype scan_for_reversal_rocket()
    """
    pass


def scan_consolidation_breakout(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Find stocks consolidating before potential breakout.
    
    Signals:
        +30: Tight price range (< 8% over 20 days)
        +10: Positioned near highs of range
        +20: ATR contraction (volatility decreasing)
        +15: Volume drying up
        +20: RSI in ready zone (58-72)
        +15: Near 52-week high (within striking distance)
    
    Threshold: 50 points minimum
    
    PORTING: From JP prototype scan_consolidation_breakout()
    """
    pass


def scan_relative_strength(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Find stocks showing relative strength.
    
    Signals:
        +30: Bullish MA alignment (Price > SMA20 > SMA50)
        +25: Short-term momentum (3-15% in 10 days)
        +20: Medium-term momentum (5-25% in 30 days)
        +25: High R-squared on trend (consistent)
    
    Threshold: 50 points minimum
    
    PORTING: From JP prototype scan_relative_strength()
    """
    pass


def scan_burst_candidates(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Find stocks matching "forensic signature" of past winners.
    
    Signals:
        +25: RSI in range (30-70)
        +25: Volume ratio in range (80-200%)
        +25: Volatility in range (20-60%)
        +25: SMA ratio in range (0.85-1.15)
    
    Pre-filter: Risk/reward >= 2.0
    Threshold: 50 points minimum
    
    PORTING: From JP prototype scan_for_burst_candidates()
    """
    pass


def scan_oversold_bounce(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Mean reversion on oversold stocks.
    
    Signals:
        +30: 10-day momentum <= -5%
        +25: Price <= -5% below SMA20
        +25: RSI <= 35
        +15: Volume surge >= 1.5x
        +15: Short interest >= 3%
        +10: Combo: RSI <= 30 AND shorts > 5% (squeeze)
        +5: Monday (reversal day)
    
    Threshold: 50 points minimum
    
    PORTING: From JP prototype detect_oversold_bounce()
    """
    pass


def scan_volatility_explosion(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    High volatility stocks near lows.
    
    Signals:
        +40: High volatility (>= 40% annualized)
        +30: Near lows of range (bottom 30%)
        +20: Volume surge >= 1.5x
        +10: Short interest >= 3% (fuel)
    
    Threshold: 50 points minimum
    
    PORTING: From JP prototype detect_volatility_explosion()
    """
    pass


def scan_power_combinations(
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> tuple[int, list[str]]:
    """
    Multiple signals together = higher probability.
    
    Combos:
        +50: TRIPLE (RSI <= 30 + Volume >= 1.5x + Shorts >= 3%)
        +40: EXTREME (Momentum <= -8% + Volatility >= 40%)
        +30: Friday Oversold (Friday + RSI <= 35)
        +20: Month-end Oversold (day >= 25 + Momentum <= -5%)
    
    Threshold: 50 points minimum
    
    PORTING: From JP prototype detect_power_combinations()
    """
    pass


def get_all_signals(
    symbol: str,
    data: pd.DataFrame,
    jpx_data: dict,
    config: dict
) -> list[dict]:
    """
    Run all scanners and return list of qualifying signals.
    
    Returns:
        List of dicts: [
            {'strategy': 'momentum_star', 'score': 70, 'reasons': [...]},
            {'strategy': 'reversal_rocket', 'score': 80, 'reasons': [...]},
            ...
        ]
    
    A stock can have multiple signals (different strategies).
    """
    signals = []
    
    scanners = [
        ('momentum_star', scan_momentum_star),
        ('reversal_rocket', scan_reversal_rocket),
        ('consolidation_breakout', scan_consolidation_breakout),
        ('relative_strength', scan_relative_strength),
        ('burst_candidates', scan_burst_candidates),
        ('oversold_bounce', scan_oversold_bounce),
        ('volatility_explosion', scan_volatility_explosion),
        ('power_combinations', scan_power_combinations),
    ]
    
    for name, scanner_func in scanners:
        score, reasons = scanner_func(data, jpx_data, config)
        if score >= config.get('MIN_SCORE', 50):
            signals.append({
                'symbol': symbol,
                'strategy': name,
                'score': score,
                'reasons': reasons,
            })
    
    return signals



# =============================================================================
# 5. BACKTESTING.PY — Backtest Engine
# =============================================================================
"""
PURPOSE:
    Run backtests on daily bar data.
    Entry at next-day open, exits checked against daily high/low.

PORTING SOURCE:
    - Engine structure: modular/backtesting.py (crypto)
    - Exit modes: modular/backtesting.py (crypto)
    
KEY ADAPTATIONS FROM CRYPTO:
    - Hourly → Daily bars
    - Next-hour open → Next-day open
    - Hourly exit checks → Daily exit checks
    - Remove 24/7 assumptions (TSE hours)

CLASSES:
    - Position: Track individual positions
    - BacktestEngine: Main engine with state management
    
FUNCTIONS:
    - run_daily_backtest(): Main entry point
"""

# --- backtesting.py SKELETON ---

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from tqdm import tqdm

# from config import (STOP_LOSS_PCT, TARGET_1_PCT, TARGET_1_PORTION, ...)
# from data_manager import (get_daily_bars_batch, build_liquid_universe, ...)
# from technical_analysis import calculate_all_indicators
# from scanners import get_all_signals


@dataclass
class Position:
    """
    Represents an open position.
    
    PORTING: Similar to modular/backtesting.py Position class
    """
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    quantity: float
    stop_price: float
    target_1_price: float
    target_2_price: float
    remaining_quantity: float = None
    target_1_hit: bool = False
    peak_price: float = None  # For trailing stop
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        if self.peak_price is None:
            self.peak_price = self.entry_price


@dataclass
class Trade:
    """
    Represents a closed trade.
    """
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'stop_loss', 'target_1', 'target_2', 'trailing', 'end_of_backtest'


class BacktestEngine:
    """
    Daily backtest engine for JP stocks.
    
    PORTING: Adapted from modular/backtesting.py BacktestEngine
             Major changes: hourly → daily, entry/exit timing
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_positions: int,
        position_size_pct: float,
        stop_loss_pct: float,
        target_1_pct: float,
        target_1_portion: float,
        target_2_pct: float,
        exit_mode: str = 'default',
        trailing_stop_pct: float = 0.03,
        slippage_pct: float = 0.001,
        commission_pct: float = 0.001,
    ):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        
        # Exit parameters
        self.stop_loss_pct = stop_loss_pct
        self.target_1_pct = target_1_pct
        self.target_1_portion = target_1_portion
        self.target_2_pct = target_2_pct
        self.exit_mode = exit_mode
        self.trailing_stop_pct = trailing_stop_pct
        
        # Costs
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        
        # State
        self.positions: dict[str, Position] = {}
        self.trade_history: list[Trade] = []
        self.daily_balances: list[dict] = []
        self.pending_entries: list[dict] = []  # Signals to enter next day
        
    @property
    def equity(self) -> float:
        """Current equity = cash + open positions MTM value."""
        pass
    
    @property
    def open_position_count(self) -> int:
        return len(self.positions)
    
    def can_open_position(self) -> bool:
        return self.open_position_count < self.max_positions
    
    def calculate_position_size(self, entry_price: float) -> float:
        """
        Calculate position size based on sizing method.
        
        PORTING: From modular/backtesting.py
        """
        pass
    
    def queue_entry(self, signal: dict, current_price: float):
        """
        Queue a signal for entry at next day's open.
        This is the key difference from crypto (no same-bar entry).
        """
        self.pending_entries.append({
            'symbol': signal['symbol'],
            'strategy': signal['strategy'],
            'score': signal['score'],
            'current_price': current_price,
        })
    
    def process_pending_entries(self, date: str, open_prices: dict[str, float]):
        """
        Process queued entries at today's open.
        Called at start of each day.
        
        KEY DIFFERENCE FROM CRYPTO:
            - Crypto: Enter at next hour's open (within same day usually)
            - JP Stocks: Enter at next day's open (T+1)
        """
        pass
    
    def check_exits(self, date: str, daily_bars: dict[str, pd.Series]):
        """
        Check all open positions for exit conditions.
        
        Exit logic (checked in order):
            1. Stop loss: low <= stop_price → exit at stop_price
            2. Target 1: high >= target_1_price → partial exit
            3. Target 2: high >= target_2_price → exit remainder
            4. Trailing stop (if enabled): update stop, check vs low
            5. Breakeven stop (if T1 hit): move stop to entry
        
        PORTING: From modular/backtesting.py, adapted for daily bars
        """
        pass
    
    def close_position(
        self,
        symbol: str,
        exit_date: str,
        exit_price: float,
        quantity: float,
        reason: str
    ):
        """
        Close (or partially close) a position.
        """
        pass
    
    def liquidate_all(self, date: str, close_prices: dict[str, float]):
        """
        Close all positions at end of backtest.
        
        PORTING: From modular/backtesting.py
        """
        pass
    
    def record_daily_balance(self, date: str):
        """Record end-of-day balance for equity curve."""
        pass
    
    def calculate_metrics(self) -> dict:
        """
        Calculate backtest performance metrics.
        
        Metrics:
            - profit_factor
            - win_rate
            - total_trades
            - max_drawdown
            - total_return
            - final_balance
            - final_equity
            - entries_count
            - avg_trade_pnl
            - avg_winner, avg_loser
            - sharpe_ratio (if sufficient data)
        
        PORTING: From modular/backtesting.py
        """
        pass


def run_daily_backtest(
    start_date: str,
    end_date: str,
    initial_balance: float,
    top_n: int = 100,
    max_positions: int = 5,
    position_size_pct: float = 0.20,
    stop_loss_pct: float = 0.05,
    target_1_pct: float = 0.03,
    target_1_portion: float = 0.50,
    target_2_pct: float = 0.06,
    exit_mode: str = 'default',
    trailing_stop_pct: float = 0.03,
    min_score: int = 50,
    scanner_config: dict = None,
    liquidate_on_end: bool = True,
    progress: bool = True,
) -> tuple:  # (BacktestEngine, dict)
    """
    Main daily backtest entry point.
    
    FLOW:
        For each trading day T:
            1. Build liquid universe for day T
            2. Load daily bars for universe (with lookback for indicators)
            3. Calculate indicators for each symbol
            4. Run scanners to detect signals
            5. Queue top signals for entry (ranked by score)
            6. Process exits for existing positions (vs day T high/low)
            7. Process pending entries at day T open
            8. Record daily balance
        
        After all days:
            - Liquidate remaining positions (if enabled)
            - Calculate metrics
    
    ENTRY TIMING:
        Signal on day T → Entry at open of day T+1
        (1-day delay, no lookahead)
    
    EXIT TIMING:
        Check daily high/low against stops and targets
    
    Returns:
        (engine: BacktestEngine, metrics: dict)
    
    PORTING: Adapted from modular/backtesting.py run_hourly_backtest()
    """
    pass



# =============================================================================
# 6. OPTIMIZER.PY — Grid Search and Walk-Forward
# =============================================================================
"""
PURPOSE:
    - Grid search over parameter combinations
    - Walk-forward analysis with rolling train/test windows
    - Stability ranking to find robust parameter sets

PORTING SOURCE:
    - All functions from modular/optimizer.py (crypto)
    - Nearly identical, just different parameter names

FUNCTIONS:
    - grid_search_daily(): Run all parameter combinations
    - walk_forward_grid_search(): Rolling train/test
    - summarize_oos(): Aggregate OOS metrics
    - top_params_by_stability(): Find stable picks
"""

# --- optimizer.py SKELETON ---

import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
from tqdm import tqdm

# from backtesting import run_daily_backtest
# from data_manager import log_backtest_run


def grid_search_daily(
    start_date: str,
    end_date: str,
    initial_balance: float,
    # Parameter grids
    rsi_max_values: list[int] = [65, 70, 75],
    volume_surge_values: list[float] = [1.3, 1.5, 2.0],
    min_score_values: list[int] = [50, 60, 70],
    stop_loss_values: list[float] = [0.04, 0.05, 0.06],
    exit_modes: list[str] = ['default', 'trailing'],
    # Fixed params
    top_n: int = 100,
    max_positions: int = 5,
    # Constraints
    min_trades: int = 20,
    min_win_rate: float = 0.45,
    max_drawdown_cap: float = 0.25,
    # Options
    progress: bool = True,
) -> pd.DataFrame:
    """
    Run grid search over parameter combinations.
    
    Process:
        1. Generate all parameter combinations
        2. For each combo, run backtest
        3. Check constraints (min trades, win rate, max DD)
        4. Record results with metrics
    
    Returns:
        DataFrame with columns:
            - All parameter values
            - profit_factor, win_rate, total_trades, max_drawdown
            - total_return, passed_constraints
    
    PORTING: From modular/optimizer.py grid_search_hourly()
    """
    pass


def walk_forward_grid_search(
    windows: list[tuple],  # [(train_start, train_end, test_start, test_end), ...]
    initial_balance: float,
    # Parameter grids (same as grid_search_daily)
    rsi_max_values: list[int],
    volume_surge_values: list[float],
    min_score_values: list[int],
    stop_loss_values: list[float],
    exit_modes: list[str],
    # Fixed params
    top_n: int = 100,
    max_positions: int = 5,
    # Constraints
    min_trades: int = 20,
    min_win_rate: float = 0.45,
    max_drawdown_cap: float = 0.25,
    min_test_trades: int = 10,
    # Options
    progress: bool = True,
    log_to_db: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward analysis with rolling windows.
    
    Process:
        For each window:
            1. Run grid search on TRAIN period
            2. Find best params by profit factor (under constraints)
            3. Run backtest on TEST period with those params
            4. Record OOS (out-of-sample) results
    
    Returns:
        DataFrame with columns:
            - window_id, train_start, train_end, test_start, test_end
            - Best params from train (pick_rsi, pick_vol, pick_score, ...)
            - Train metrics (train_pf, train_wr, train_trades, ...)
            - Test metrics (test_pf, test_wr, test_trades, ...)
    
    PORTING: From modular/optimizer.py walk_forward_grid_search()
    """
    pass


def summarize_oos(
    wf_results: pd.DataFrame,
    min_test_trades: int = 10
) -> dict:
    """
    Aggregate OOS metrics across walks.
    
    Returns:
        {
            'ok': DataFrame of walks with sufficient test trades,
            'oos_summary': {
                'median_pf': float,
                'mean_pf': float,
                'median_wr': float,
                'median_dd': float,
                'total_oos_trades': int,
            }
        }
    
    PORTING: From modular/optimizer.py summarize_oos()
    """
    pass


def top_params_by_stability(
    wf_results: pd.DataFrame,
    min_test_trades: int = 10,
    min_count: int = 2
) -> pd.DataFrame:
    """
    Find parameter combos that were selected across multiple walks.
    
    Stability = robustness indicator (same params work in different periods)
    
    Returns:
        DataFrame with param combos sorted by selection count
    
    PORTING: From modular/optimizer.py top_params_by_stability()
    """
    pass


def generate_windows(
    start_date: str,
    end_date: str,
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 21
) -> list[tuple]:
    """
    Generate rolling train/test windows.
    
    Example:
        generate_windows('2024-01-01', '2024-12-31', 90, 30, 21)
        → [
            ('2024-01-01', '2024-03-31', '2024-04-01', '2024-04-30'),
            ('2024-01-22', '2024-04-21', '2024-04-22', '2024-05-21'),
            ...
          ]
    """
    pass



# =============================================================================
# 7. RUN_BACKTEST.PY — CLI Entry Point
# =============================================================================
"""
PURPOSE:
    Simple CLI to run backtests from command line.
    Also serves as an example of how to use the modules.
"""

# --- run_backtest.py SKELETON ---

import argparse
from datetime import datetime

# from config import *
# from backtesting import run_daily_backtest


def main():
    parser = argparse.ArgumentParser(description='Run JP Stocks Backtest')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=1_000_000, help='Initial balance (JPY)')
    parser.add_argument('--top-n', type=int, default=100, help='Universe size')
    parser.add_argument('--max-positions', type=int, default=5, help='Max concurrent positions')
    parser.add_argument('--exit-mode', type=str, default='default', 
                        choices=['default', 'trailing', 'breakeven', 'breakeven_trailing'])
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"JP Stocks Daily Backtest")
    print(f"{'='*60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Initial Balance: ¥{args.balance:,.0f}")
    print(f"Universe Size: {args.top_n}")
    print(f"Max Positions: {args.max_positions}")
    print(f"Exit Mode: {args.exit_mode}")
    print(f"{'='*60}\n")
    
    # Run backtest
    engine, metrics = run_daily_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        top_n=args.top_n,
        max_positions=args.max_positions,
        exit_mode=args.exit_mode,
        progress=True,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Export trades
    trades_df = pd.DataFrame([t.__dict__ for t in engine.trade_history])
    filename = f"results/backtest_{args.start}_{args.end}.csv"
    trades_df.to_csv(filename, index=False)
    print(f"\nTrades exported to: {filename}")


if __name__ == '__main__':
    main()
"""

===============================================================================
                              END OF CODE SKELETON
===============================================================================

NEXT STEPS:
1. Review this skeleton with the user
2. Create actual module files based on these interfaces
3. Implement each module in phases
4. Test incrementally
"""
