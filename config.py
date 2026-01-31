"""
JP Stocks Modular Trading System — Configuration

config.py is the single source of truth for parameters.
You can override in notebook sessions:

    import config
    config.MIN_SCANNER_SCORE = 35
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
CACHE_TICKER_LIST_HOURS = 720
CACHE_JPX_DATA_HOURS = 12
YFINANCE_HISTORY_PERIOD = "5y"

# =============================================================================
# Auto-Expand DB (incremental daily growth)
# =============================================================================
# If True, precompute.py will auto-download missing tickers before building cache.
AUTO_EXPAND_DB = True
AUTO_EXPAND_DB_MAX_NEW = 1000         # New symbols per run (cap for safety)
AUTO_EXPAND_DB_START_DATE = "2024-01-01"
AUTO_EXPAND_DB_MIN_ROWS = 60
AUTO_EXPAND_DB_SLEEP_SECONDS = 0.2
AUTO_EXPAND_DB_CHUNK_SIZE = 50
AUTO_EXPAND_DB_SHUFFLE = True

# =============================================================================
# Cache Behavior
# =============================================================================
# If True, cache stores all triggered signals (score > 0) and filters later by MIN_SCANNER_SCORE.
CACHE_RAW_SIGNALS = True

# Save LLM research results to results/llm_research_YYYYMMDD_HHMMSS.json
SAVE_LLM_RESULTS = True

# =============================================================================
# Market Cap Ingestion (for universe filter)
# =============================================================================
# If True, precompute will update missing market caps each run (incremental).
AUTO_UPDATE_MARKET_CAP = True
AUTO_UPDATE_MARKET_CAP_MAX = 300     # Symbols per run
AUTO_UPDATE_MARKET_CAP_SLEEP = 0.2


# =============================================================================
# Universe Filters
# =============================================================================
MAX_MARKET_CAP_JPY = 500_000_000_000
# Enforce market-cap filter in universe selection (if market cap data exists)
ENFORCE_MARKET_CAP = True
# Policy when market cap is missing: "include" or "exclude"
MARKET_CAP_MISSING_POLICY = "include"
# NOTE: This is a per-day volume filter (name kept for backward compatibility).
# Lower it to include smaller/micro-cap names that can "burst".
MIN_AVG_DAILY_VOLUME = 20_000
LIQUIDITY_FLOOR_JPY = 100_000_000
# Expanded universe to reduce large-cap bias from "top notional" ranking.
UNIVERSE_TOP_N = 500
EXCLUDE_NIKKEI_225 = True

# Performance pre-filters (from JP prototype)
PERFORMANCE_LOOKBACK_DAYS = 180
PERFORMANCE_MIN_RISE_PERCENT = 2.0
PERFORMANCE_NEAR_52W_HIGH_THRESHOLD = 0.10


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
    
    # === NEW: Smart Money Flow (replaces Power Combinations) ===
    'SMART_MONEY_MIN_RELATIVE_STRENGTH': -2,  # Max 2% down in 10 days
    
    # === NEW: Crash Then Burst (JP penny stock pattern) ===
    'CRASH_MIN_DROP_PCT': -25,              # Minimum 25% crash from high
    'CRASH_VOLUME_CLIMAX': 3.0,             # 3x volume spike on crash
    'CRASH_SHORT_SQUEEZE_MIN': 0.05,        # 5%+ short interest for squeeze
    
    # === NEW: Stealth Accumulation ===
    'STEALTH_MIN_VOLUME_INCREASE': 15,      # 15%+ volume increase
    'STEALTH_MIN_VOLATILITY_COMPRESSION': 20,  # 20%+ volatility squeeze
    'STEALTH_WIN_RATE_MIN': 0.55,           # Subtle bullish bias
    'STEALTH_WIN_RATE_MAX': 0.65,           # Not too obvious
    
    # === NEW: Coiling Pattern ===
    'COILING_MIN_BB_SQUEEZE': 30,           # 30%+ Bollinger Band squeeze
    'COILING_MIN_ATR_COMPRESSION': 25,      # 25%+ ATR compression
}

# =============================================================================
# Early Mode (Pre-Burst) Filters
# =============================================================================
EARLY_MODE_ENABLED = True
EARLY_MODE_SHOW_BOTH = False  # Show legacy mode output alongside early mode
EARLY_MODE_RSI_MAX = 65       # RSI must be <= 65 (not overbought)
EARLY_MODE_10D_RETURN_MAX = 0.15  # 10-day return must be < 15% (close-to-close)
EARLY_MODE_SCANNERS = [
    'oversold_bounce',
    'reversal_rocket',
    'volatility_explosion',
    'coiling_pattern',
    'consolidation_breakout',
]


# =============================================================================
# Exit Strategy Settings
# =============================================================================
STOP_LOSS_PCT = 0.06              # 6% stop loss
TARGET_1_PCT = 0.03               # 3% first target
TARGET_1_PORTION = 0.50           # Sell 50% at T1
TARGET_2_PCT = 0.06               # 6% second target (remaining position)

# Exit mode options: 'default', 'trailing', 'breakeven', 'breakeven_trailing', 'fixed_rr'
EXIT_MODE = 'fixed_rr'
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
# Execution Constraints (Broker Lots)
# =============================================================================
MAX_JPY_PER_TRADE = 100_000               # Max JPY per play (budget cap)
LOT_SIZE = 100                            # JP standard lot size


# =============================================================================
# Backtest Settings
# =============================================================================
BACKTEST_TIMEFRAME = '1d'                 # Daily bars only for JP stocks
BACKTEST_SLIPPAGE_PCT = 0.004             # 0.4% slippage (realistic for small-caps, includes spread)
BACKTEST_COMMISSION_PCT = 0.001           # 0.1% commission
# Total round-trip: ~1% (0.5% entry + 0.5% exit) - conservative for illiquid names
MIN_HOLD_DAYS = 1                         # Minimum holding period


# =============================================================================
# Optimizer Constraints
# =============================================================================
MAX_DRAWDOWN_CAP = 0.25           # Max 25% drawdown allowed
MIN_WIN_RATE = 0.45               # Min 45% win rate required
MIN_TRADES = 20                   # Minimum trades for statistical validity

# =============================================================================
# === FINAL PARAMETERS (Proven: PF 2.46, 59% Win Rate) ===
# =============================================================================
MIN_SCANNER_SCORE = 30            # Sweet spot (tested 15-50)
MIN_HISTORY_DAYS = 60             # Minimum days of history needed

# Exit Strategy
RISK_REWARD_RATIO = 2.0           # 2:1 R:R -> 12% Profit Target

# Scanner Classification
STAR_SCANNERS = ['oversold_bounce', 'burst_candidates', 'momentum_star']
SOLID_SCANNERS = ['relative_strength', 'volatility_explosion', 'coiling_pattern']
DISABLED_SCANNERS = ['crash_then_burst', 'stealth_accumulation']  # PF 0.00

# =============================================================================
# Walk-Forward Settings
# =============================================================================
WFA_TRAIN_DAYS = 180              # Training window (6 months) - longer for stability
WFA_TEST_DAYS = 90                # Testing window (3 months) - longer for validity
WFA_STEP_DAYS = 60                # Step size (2 months)
WFA_MIN_TEST_TRADES = 5           # Minimum trades in test period


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
    print("Signal Filters:")
    print(f"  Min Scanner Score: {MIN_SCANNER_SCORE}")
    print("-" * 60)
    print("Exit Strategy:")
    print(f"  Stop Loss: {STOP_LOSS_PCT:.1%}")
    print(f"  Exit Mode: {EXIT_MODE}")
    print(f"  Risk Reward: {RISK_REWARD_RATIO}")
    print("-" * 60)
    print("Early Mode:")
    print(f"  Enabled: {EARLY_MODE_ENABLED}")
    print(f"  RSI Max: {EARLY_MODE_RSI_MAX}")
    print(f"  10D Return Max: {EARLY_MODE_10D_RETURN_MAX:.0%}")
    print(f"  Scanners: {', '.join(EARLY_MODE_SCANNERS)}")
    print("-" * 60)
    print("Position Sizing:")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Method: {POSITION_SIZING_METHOD}")
    print(f"  Size per Trade: {FIXED_FRACTIONAL_PCT:.1%}")
    print(f"  Max JPY per Trade: {MAX_JPY_PER_TRADE:,.0f} (Lot Size: {LOT_SIZE})")
    print("-" * 60)
    print("Auto-Expand DB:")
    print(f"  Enabled: {AUTO_EXPAND_DB}")
    print(f"  Max New: {AUTO_EXPAND_DB_MAX_NEW}")
    print(f"  Start: {AUTO_EXPAND_DB_START_DATE}")
    print("-" * 60)
    print("Optimizer Constraints:")
    print(f"  Max Drawdown: {MAX_DRAWDOWN_CAP:.1%}")
    print(f"  Min Win Rate: {MIN_WIN_RATE:.1%}")
    print(f"  Min Trades: {MIN_TRADES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
