# JP Stocks Modular Trading System — Architecture Documentation

## 1. Overview

This document describes the architecture for a **modular Japanese stocks trading system** adapted from the existing crypto trading bot. The system focuses on **backtesting and research** (not live trading), using daily bar data with a 1-day delay for signals.

### Core Objectives
- Systematically identify and backtest high-probability, short-term Japanese equity trades
- Port the robust backtesting engine from the crypto bot
- Leverage existing JP stocks prototype scanner logic
- Run walk-forward analysis (WFA) to find stable parameter sets
- Operate on **daily bars** (TSE trading hours: 9:00-15:00 JST)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JP STOCKS MODULAR SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  config.py  │───▶│data_manager │───▶│  scanners   │───▶│ backtesting │  │
│  │             │    │    .py      │    │    .py      │    │    .py      │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  technical  │    │   SQLite    │    │  optimizer  │    │  notebooks/ │  │
│  │ _analysis.py│    │     DB      │    │    .py      │    │  pipeline   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Map

### Core Modules (6 total)

| Module | Purpose | Source |
|--------|---------|--------|
| `config.py` | Centralized settings, API keys, entry filters, backtest knobs | **New** (adapted from crypto) |
| `data_manager.py` | DB access, yfinance/investpy ingestion, JPX short data, universe building | **Adapted** from crypto + JP prototype |
| `technical_analysis.py` | Indicators: RSI, EMA, MACD, ATR, Z-Score, volume metrics | **Ported** from crypto |
| `scanners.py` | Entry signal strategies (Rising Stars strategies) | **Ported** from JP prototype |
| `backtesting.py` | Backtest engine with daily bars, exit modes, metrics | **Ported** from crypto (major adaptation) |
| `optimizer.py` | Grid search, walk-forward analysis, stability ranking | **Ported** from crypto |

### Support Files

| File | Purpose |
|------|---------|
| `run_backtest.py` | CLI entry point for running backtests |
| `notebooks/pipeline_wfa.ipynb` | Research notebook: baseline → grid → WFA |
| `results/` | CSV exports, parameter snapshots (JSON) |

---

## 4. Module Details

### 4.1 `config.py`

**Purpose**: Centralized configuration for all modules.

**Key Settings**:
```python
# Database
DATABASE_FILE = "jp_stocks.db"
CACHE_DIR = "cache/"

# Universe Filters (from JP prototype)
MAX_MARKET_CAP_JPY = 500_000_000_000  # 500B JPY
MIN_AVG_DAILY_VOLUME = 100_000
EXCLUDE_NIKKEI_225 = True

# Entry Filters (tunable)
ENTRY_FILTERS = {
    'MAX_RSI_ENTRY': 70,
    'MIN_VOLUME_SURGE': 1.5,
    'MIN_SCORE': 50,
}

# Exit Strategy
STOP_LOSS_PCT = 0.05
TARGET_1_PCT = 0.03
TARGET_2_PCT = 0.06
EXIT_MODE = 'default'  # 'default', 'trailing', 'breakeven'

# Backtest Settings
BACKTEST_TIMEFRAME = '1d'  # Daily bars only for JP stocks
UNIVERSE_TOP_N = 100
POSITION_SIZING_METHOD = 'fixed_fractional'
MAX_POSITIONS = 5

# Optimizer Constraints
MAX_DRAWDOWN_CAP = 0.25
MIN_WIN_RATE = 0.45
MIN_TRADES = 20

# API Settings
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
```

**Porting from crypto**: Structure identical, just different parameter values.

---

### 4.2 `data_manager.py`

**Purpose**: Data ingestion, caching, universe building, database operations.

**Key Functions**:

| Function | Description | Source |
|----------|-------------|--------|
| `ensure_database()` | Create/verify SQLite schema | New |
| `get_all_tse_tickers()` | Fetch TSE tickers via investpy (with cache) | From JP prototype |
| `get_nikkei_225_components()` | Fetch Nikkei 225 for exclusion | From JP prototype |
| `download_price_history(symbols, start, end)` | Batch download via yfinance | New (adapted) |
| `update_recent_data()` | Incremental daily update | From crypto |
| `fetch_jpx_short_data()` | Scrape JPX short-selling data | From JP prototype |
| `get_daily_bars(symbol, date)` | Retrieve bars for backtest | From crypto |
| `build_liquid_universe(date)` | Filter by volume/market cap for a given date | From crypto |

**Database Schema**:
```sql
-- Daily price data
CREATE TABLE daily_prices (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume INTEGER,
    UNIQUE(symbol, date)
);

-- Symbol metadata
CREATE TABLE symbol_info (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    market_cap REAL,
    is_nikkei_225 BOOLEAN DEFAULT FALSE
);

-- JPX short interest (daily snapshot)
CREATE TABLE jpx_short_interest (
    symbol TEXT,
    date DATE,
    short_volume INTEGER,
    total_volume INTEGER,
    short_ratio REAL,
    PRIMARY KEY(symbol, date)
);

-- Backtest run history
CREATE TABLE backtest_runs (
    id INTEGER PRIMARY KEY,
    run_date DATETIME,
    params_json TEXT,
    profit_factor REAL,
    win_rate REAL,
    max_drawdown REAL,
    total_trades INTEGER
);

-- Indices for performance
CREATE INDEX idx_daily_symbol_date ON daily_prices(symbol, date);
```

**Reusing JP prototype database**: 
- The existing `rising_stars.db` stores scanner reports, not price history
- We'll create a **new** `jp_stocks.db` for modular system with price tables
- Can import cached data from `jpx_data_cache_*.xlsx` files

---

### 4.3 `technical_analysis.py`

**Purpose**: Calculate all technical indicators.

**Key Functions**:

| Function | From | Notes |
|----------|------|-------|
| `calculate_rsi(prices, period)` | Crypto bot | Identical |
| `calculate_ema(prices, span)` | JP prototype | Already exists |
| `calculate_macd(prices, fast, slow, signal)` | Crypto bot | Port directly |
| `calculate_bollinger(prices, period, std_mult)` | Crypto bot | Port directly |
| `calculate_atr(data, period)` | JP prototype | Already exists |
| `calculate_volume_surge(volume, lookback)` | Crypto bot | Port directly |
| `calculate_z_score(prices, period)` | JP prototype | Already exists |
| `detect_higher_lows(lows)` | JP prototype | Already exists |
| `calculate_risk_reward(data)` | JP prototype | Already exists |

**Porting strategy**: Merge best of both — crypto bot's structure with JP prototype's additional indicators.

---

### 4.4 `scanners.py`

**Purpose**: Entry signal detection strategies.

**Key Scanner Functions** (from JP prototype):

| Strategy | Description | Score Threshold |
|----------|-------------|-----------------|
| `scan_momentum_star()` | Uptrend + RSI zone + volume spike | 50+ |
| `scan_reversal_rocket()` | Oversold + Z-score + reversal patterns | 70+ |
| `scan_consolidation_breakout()` | Tight range + volatility contraction | 50+ |
| `scan_relative_strength()` | Price > MA20 > MA50 + consistent trend | 50+ |
| `scan_burst_candidates()` | Forensic signature from winners | 50+ |
| `detect_oversold_bounce()` | Mean reversion on oversold | 50+ |
| `detect_volatility_explosion()` | High volatility + near lows | 50+ |

**Integration with backtest engine**:
```python
def get_entry_signals(symbol, date, data, jpx_data, config) -> list[dict]:
    """
    Returns list of entry signals with scores.
    Called by backtest engine for each symbol on each date.
    """
    signals = []
    
    # Run each scanner
    score, reasons = scan_momentum_star(data, jpx_data, config)
    if score >= config['MIN_SCORE']:
        signals.append({'strategy': 'momentum_star', 'score': score, 'reasons': reasons})
    
    # ... repeat for other scanners
    
    return signals
```

---

### 4.5 `backtesting.py`

**Purpose**: Core backtest engine adapted from crypto.

**Key Differences from Crypto Version**:

| Aspect | Crypto Bot | JP Stocks |
|--------|-----------|-----------|
| Timeframe | Hourly (1h, 4h) | Daily (1d) only |
| Entry timing | Next-hour open | Next-day open |
| Exit checking | Hourly vs high/low | Daily vs high/low |
| Trading hours | 24/7 | 9:00-15:00 JST |
| Universe refresh | Daily notional | Daily volume + market cap |

**Main Backtest Flow**:
```python
def run_daily_backtest(
    start_date: str,
    end_date: str,
    initial_balance: float,
    top_n: int = 100,
    exit_mode: str = 'default',
    stop_loss_pct: float = 0.05,
    # ... other params
) -> tuple[BacktestEngine, dict]:
    """
    Daily backtest engine for JP stocks.
    
    For each trading day:
      1) Build liquid universe (top_n by volume above market cap floor)
      2) Load daily bars for universe
      3) Compute indicators for each symbol
      4) Run scanners to generate entry signals
      5) Rank signals by score, enter top candidates at NEXT day's open
      6) Check exits: stop-loss vs daily low, targets vs daily high
      7) Track positions, PnL, equity curve
    
    Returns engine with trade history and metrics dict.
    """
```

**Exit Modes** (ported from crypto):
- `default`: Fixed targets + stop loss
- `trailing`: Trailing stop from peak
- `breakeven`: Move stop to entry after T1 hit
- `breakeven_trailing`: Combo

**Metrics Computed**:
- `profit_factor`, `win_rate`, `total_trades`
- `max_drawdown`, `total_return`, `final_equity`
- `entries_count`, `avg_entry_value`

---

### 4.6 `optimizer.py`

**Purpose**: Parameter optimization and walk-forward analysis.

**Key Functions** (ported from crypto):

```python
def grid_search_daily(
    start_date, end_date, initial_balance,
    rsi_values: list, volume_values: list, score_values: list,
    stop_losses: list, exit_modes: list,
    # constraints
    min_trades: int, min_win_rate: float, max_drawdown_cap: float,
) -> pd.DataFrame:
    """Run grid search over parameter combinations."""

def walk_forward_grid_search(
    windows: list[tuple],  # (train_start, train_end, test_start, test_end)
    initial_balance: float,
    # param grids...
    min_test_trades: int = 10,
) -> pd.DataFrame:
    """Rolling train/test walk-forward analysis."""

def summarize_oos(wf_results, min_test_trades) -> dict:
    """Aggregate OOS metrics across walks."""

def top_params_by_stability(wf_results, min_count) -> pd.DataFrame:
    """Find parameter combos repeatedly selected across walks."""
```

**Porting**: Nearly identical to crypto — just reference different scanner parameters.

---

## 5. Data Flow

```
                    ┌──────────────────┐
                    │   investpy/      │
                    │   yfinance       │
                    └────────┬─────────┘
                             │
                             ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ JPX Website  │───▶│  data_manager.py │◀──▶│  jp_stocks.db    │
│ (Short Data) │    │                  │    │  (SQLite)        │
└──────────────┘    └────────┬─────────┘    └──────────────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
        ┌────────────┐ ┌──────────┐ ┌────────────┐
        │ scanners.py│ │technical │ │config.py   │
        │            │ │_analysis │ │            │
        └──────┬─────┘ └────┬─────┘ └────────────┘
               │            │
               └─────┬──────┘
                     ▼
             ┌──────────────┐
             │backtesting.py│
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ optimizer.py │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │  notebooks/  │
             │  results/    │
             └──────────────┘
```

---

## 6. What Can Be Reused

### From Crypto Bot (`modular/`)

| Component | Reuse Level | Notes |
|-----------|-------------|-------|
| `config.py` structure | 90% | Different params, same pattern |
| `technical_analysis.py` | 80% | Add JP-specific indicators |
| `backtesting.py` engine logic | 70% | Adapt hourly → daily, same exit logic |
| `optimizer.py` | 95% | Nearly identical, different param names |
| Notebook structure | 90% | Same research workflow |
| DB schema pattern | 60% | Add JP-specific tables |

### From JP Prototype (`JP_stocks_rising/`)

| Component | Reuse Level | Notes |
|-----------|-------------|-------|
| `get_all_tse_tickers()` | 100% | Direct port |
| `get_nikkei_225_components()` | 100% | Direct port |
| `fetch_and_process_jpx_data()` | 100% | Direct port |
| `RisingStarsScanner` strategies | 90% | Refactor into `scanners.py` |
| `calculate_rsi/ema/atr/z_score` | 100% | Already identical |
| Gemini integration | 80% | Optional enhancement |
| LSTM forecasting | 50% | Optional, not critical |

### Existing Databases

| Database | Reuse? | Notes |
|----------|--------|-------|
| `rising_stars.db` | No | Scanner reports only, not price data |
| `*.xlsx` cache files | Yes | Can import JPX data |
| `tse_tickers_cache.csv` | Yes | Ticker list cache |
| `checkpoint_*.csv` | Partial | Review data format |

---

## 7. Implementation Phases

### Phase 1: Foundation (Day 1)
- [ ] Create `config.py` with all settings
- [ ] Create `data_manager.py` with DB schema + investpy/yfinance ingestion
- [ ] Port `technical_analysis.py` indicators

### Phase 2: Scanners (Day 1-2)
- [ ] Create `scanners.py` from `RisingStarsScanner` class
- [ ] Refactor to functional style matching backtest interface
- [ ] Add scoring thresholds to config

### Phase 3: Backtest Engine (Day 2-3)
- [ ] Port `backtesting.py` from crypto
- [ ] Adapt hourly → daily bar handling
- [ ] Adapt entry timing (next-day open)
- [ ] Keep exit modes intact
- [ ] Test with single strategy

### Phase 4: Optimizer & WFA (Day 3-4)
- [ ] Port `optimizer.py` grid search
- [ ] Port walk-forward functions
- [ ] Create research notebook

### Phase 5: Validation (Day 4-5)
- [ ] Run baseline backtest
- [ ] Run grid search on small parameter space
- [ ] Run WFA with rolling windows
- [ ] Document results

---

## 8. File Structure

```
JP_stocks_modular/
├── config.py              # Centralized settings
├── data_manager.py        # DB + data ingestion
├── technical_analysis.py  # Indicators
├── scanners.py            # Entry signal strategies
├── backtesting.py         # Backtest engine
├── optimizer.py           # Grid search + WFA
├── run_backtest.py        # CLI entry point
├── .env                   # API keys (gitignored)
├── jp_stocks.db           # SQLite database
├── cache/                 # Cached data files
│   ├── tse_tickers.csv
│   └── jpx_short_*.xlsx
├── results/               # Backtest outputs
│   ├── trades_*.csv
│   └── params_*.json
├── notebooks/
│   └── pipeline_wfa.ipynb
└── documentation.md       # This file
```

---

## 9. Key Design Decisions

1. **Daily bars only**: TSE has limited trading hours; hourly data adds complexity without proportional benefit for this strategy type.

2. **Next-day entry**: Signal on day T → enter at open of day T+1 (1-day delay, no lookahead).

3. **No live trading**: Focus on research and backtesting; simpler architecture.

4. **Single database**: New `jp_stocks.db` with clean schema; import cached data as needed.

5. **Scanner-first architecture**: Scanners produce scored candidates; backtest engine consumes them.

6. **Modular strategies**: Each scanner is independent; easy to add/remove/tune.

---

## 10. Next Steps

1. **Review this plan** — confirm module responsibilities and data flow
2. **Create code skeleton** — see `code_skeleton.py` for detailed interfaces
3. **Implement Phase 1** — foundation modules
4. **Iterate** — test each phase before proceeding

