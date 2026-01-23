"""
JP Stocks Modular Trading System — Configuration

Centralized settings for all modules. You can override these in notebook
sessions without editing this file:

    import config
    config.MAX_RSI_ENTRY = 65
    config.UNIVERSE_TOP_N = 50
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).parent
DATABASE_FILE = BASE_DIR / "jp_stocks.db"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Source Settings
# =============================================================================
CACHE_TICKER_LIST_HOURS = 720      # 30 days - ticker list cache
CACHE_JPX_DATA_HOURS = 12          # 12 hours - JPX short data cache
YFINANCE_HISTORY_PERIOD = "5y"     # Default history period for backfill


# =============================================================================
# Universe Filters
# =============================================================================
# Market cap filter (exclude mega-caps for better alpha opportunity)
MAX_MARKET_CAP_JPY = 500_000_000_000     # 500 billion JPY

# Liquidity filters
MIN_AVG_DAILY_VOLUME = 100_000           # Minimum average daily volume
LIQUIDITY_FLOOR_JPY = 100_000_000        # 100 million JPY daily notional

# Universe size
UNIVERSE_TOP_N = 100                     # Top N by volume for backtests
EXCLUDE_NIKKEI_225 = True                # Exclude Nikkei 225 components

# Performance pre-filters (from JP prototype)
PERFORMANCE_LOOKBACK_DAYS = 180
PERFORMANCE_MIN_RISE_PERCENT = 2.0
PERFORMANCE_NEAR_52W_HIGH_THRESHOLD = 0.10


# =============================================================================
# Entry Filters (Scanner Thresholds)
# =============================================================================
ENTRY_FILTERS = {
    'MAX_RSI_ENTRY': 70,          # Maximum RSI for entry
    'MIN_VOLUME_SURGE': 1.5,      # Minimum volume spike factor
    'MIN_SCORE': 50,              # Minimum scanner score to qualify
    'REQUIRE_MACD_BULLISH': False,
}

# Individual variables for direct access
MAX_RSI_ENTRY = ENTRY_FILTERS['MAX_RSI_ENTRY']
MIN_VOLUME_SURGE = ENTRY_FILTERS['MIN_VOLUME_SURGE']
MIN_SCORE = ENTRY_FILTERS['MIN_SCORE']


# =============================================================================
# Scanner-Specific Settings
# =============================================================================
SCANNER_CONFIG = {
    # === Momentum Star ===
    'MOMENTUM_EMA_FAST': 20,
    'MOMENTUM_EMA_SLOW': 50,
    'MOMENTUM_RSI_PERIOD': 14,
    'MOMENTUM_RSI_MIN': 60,
    'MOMENTUM_RSI_MAX': 80,
    'MOMENTUM_VOLUME_SPIKE_FACTOR': 1.5,
    'MOMENTUM_LOW_SHORT_INTEREST': 0.02,    # < 2% is bullish
    'MOMENTUM_HIGH_SHORT_WARNING': 0.08,    # > 8% is warning
    
    # === Reversal Rocket ===
    'REVERSAL_ZSCORE_PERIOD': 30,
    'REVERSAL_ZSCORE_THRESHOLD': -2.0,      # Below -2 std dev
    'REVERSAL_VOLUME_SPIKE_FACTOR': 3.0,    # Capitulation volume
    'REVERSAL_HIGH_SHORT_FUEL': 0.10,       # > 10% shorts = squeeze fuel
    
    # === Consolidation Breakout ===
    'CONSOLIDATION_RANGE_MAX': 0.08,        # Max 8% price range
    'CONSOLIDATION_VOLATILITY_DROP': 0.20,  # 20% ATR contraction
    'CONSOLIDATION_RSI_MIN': 58,
    'CONSOLIDATION_RSI_MAX': 72,
    
    # === Relative Strength ===
    'RS_MOMENTUM_SHORT_MIN': 0.03,          # 3% min 10-day momentum
    'RS_MOMENTUM_SHORT_MAX': 0.15,          # 15% max 10-day momentum
    'RS_MOMENTUM_MID_MIN': 0.05,            # 5% min 30-day momentum
    'RS_MOMENTUM_MID_MAX': 0.25,            # 25% max 30-day momentum
    'RS_R_SQUARED_MIN': 0.5,                # Trend consistency threshold
    
    # === Burst Candidates (Forensic Analysis) ===
    'BURST_RSI_MIN': 30,
    'BURST_RSI_MAX': 70,
    'BURST_VOLUME_RATIO_MIN': 80,           # 80% of prior period
    'BURST_VOLUME_RATIO_MAX': 200,          # 200% of prior period
    'BURST_VOLATILITY_MIN': 20,             # 20% annualized
    'BURST_VOLATILITY_MAX': 60,             # 60% annualized
    'BURST_SMA_RATIO_MIN': 0.85,            # SMA50/SMA200 range
    'BURST_SMA_RATIO_MAX': 1.15,
    
    # === Oversold Bounce ===
    'OVERSOLD_MOMENTUM_10D': -5.0,          # <= -5% in 10 days
    'OVERSOLD_PRICE_BELOW_SMA20': -5.0,     # <= -5% below SMA20
    'OVERSOLD_RSI_MAX': 35,
    'OVERSOLD_VOLUME_SURGE_MIN': 1.5,
    'OVERSOLD_SHORT_INTEREST_MIN': 0.03,    # >= 3% for squeeze
    
    # === Volatility Explosion ===
    'HIGH_VOLATILITY_MIN': 40.0,            # >= 40% annualized
    'VOLATILITY_NEAR_LOW_THRESHOLD': 0.30,  # Bottom 30% of range
    
    # === Risk/Reward ===
    'MIN_RISK_REWARD': 2.0,                 # Minimum R:R ratio
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
    'moderate': {
        'stop_loss_pct': 0.05,
        'target_1_pct': 0.05,
        'target_1_portion': 0.50,
        'target_2_pct': 0.10,
    },
    'aggressive': {
        'stop_loss_pct': 0.07,
        'target_1_pct': 0.07,
        'target_1_portion': 0.33,
        'target_2_pct': 0.15,
    }
}

# Default exit parameters
STOP_LOSS_PCT = 0.05              # 5% stop loss
TARGET_1_PCT = 0.03               # 3% first target
TARGET_1_PORTION = 0.50           # Sell 50% at T1
TARGET_2_PCT = 0.06               # 6% second target (remaining position)

# Exit mode options: 'default', 'trailing', 'breakeven', 'breakeven_trailing'
EXIT_MODE = 'default'
TRAILING_STOP_PCT = 0.03          # 3% trailing stop from peak


# =============================================================================
# Position Sizing
# =============================================================================
MAX_POSITIONS = 5                         # Maximum concurrent positions
POSITION_SIZING_METHOD = 'fixed_fractional'  # 'fixed_fractional' or 'volatility_adjusted'
FIXED_FRACTIONAL_PCT = 0.20               # 20% per position
MAX_POSITION_PCT = 0.25                   # Max 25% in single position
RISK_PER_TRADE_PCT = 0.02                 # 2% risk per trade (for volatility sizing)


# =============================================================================
# Backtest Settings
# =============================================================================
BACKTEST_TIMEFRAME = '1d'                 # Daily bars only for JP stocks
BACKTEST_SLIPPAGE_PCT = 0.001             # 0.1% slippage
BACKTEST_COMMISSION_PCT = 0.001           # 0.1% commission (round-trip estimate)
MIN_HOLD_DAYS = 1                         # Minimum holding period


# =============================================================================
# Optimizer Constraints
# =============================================================================
MAX_DRAWDOWN_CAP = 0.25           # Max 25% drawdown allowed
MIN_WIN_RATE = 0.45               # Min 45% win rate required
MIN_TRADES = 20                   # Minimum trades for statistical validity


# =============================================================================
# Walk-Forward Settings
# =============================================================================
WFA_TRAIN_DAYS = 90               # Training window (3 months)
WFA_TEST_DAYS = 30                # Testing window (1 month)
WFA_STEP_DAYS = 21                # Step size (3 weeks)
WFA_MIN_TEST_TRADES = 10          # Minimum trades in test period


# =============================================================================
# API Settings
# =============================================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
API_RATE_LIMIT_DELAY_SECONDS = 15


# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "jp_stocks.log"


# =============================================================================
# Excluded Symbols (problematic or special cases)
# =============================================================================
EXCLUDED_SYMBOLS = set()  # Add any symbols to exclude, e.g., {'1234.T', '5678.T'}


# =============================================================================
# Helper Functions
# =============================================================================
def get_scanner_config() -> dict:
    """Get a copy of scanner config for use in backtests."""
    return SCANNER_CONFIG.copy()


def get_exit_strategy(name: str = 'moderate') -> dict:
    """Get exit strategy parameters by name."""
    return EXIT_STRATEGIES.get(name, EXIT_STRATEGIES['moderate']).copy()


def print_config_summary():
    """Print a summary of current configuration."""
    print("=" * 60)
    print("JP Stocks Modular - Configuration Summary")
    print("=" * 60)
    print(f"Database: {DATABASE_FILE}")
    print(f"Universe: Top {UNIVERSE_TOP_N} stocks")
    print(f"Max Market Cap: ¥{MAX_MARKET_CAP_JPY:,.0f}")
    print(f"Exclude Nikkei 225: {EXCLUDE_NIKKEI_225}")
    print("-" * 60)
    print("Entry Filters:")
    print(f"  Max RSI: {MAX_RSI_ENTRY}")
    print(f"  Min Volume Surge: {MIN_VOLUME_SURGE}x")
    print(f"  Min Score: {MIN_SCORE}")
    print("-" * 60)
    print("Exit Strategy:")
    print(f"  Stop Loss: {STOP_LOSS_PCT:.1%}")
    print(f"  Target 1: {TARGET_1_PCT:.1%} ({TARGET_1_PORTION:.0%} of position)")
    print(f"  Target 2: {TARGET_2_PCT:.1%}")
    print(f"  Exit Mode: {EXIT_MODE}")
    print("-" * 60)
    print("Position Sizing:")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Method: {POSITION_SIZING_METHOD}")
    print(f"  Size per Trade: {FIXED_FRACTIONAL_PCT:.1%}")
    print("-" * 60)
    print("Optimizer Constraints:")
    print(f"  Max Drawdown: {MAX_DRAWDOWN_CAP:.1%}")
    print(f"  Min Win Rate: {MIN_WIN_RATE:.1%}")
    print(f"  Min Trades: {MIN_TRADES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
