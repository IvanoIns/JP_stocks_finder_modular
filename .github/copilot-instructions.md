# Copilot Instructions: JP Stocks Modular Trading System

## Project Overview

This is a **hybrid Quant + AI trading system** for Japanese small-cap stocks. The system identifies burst candidates using 9 technical scanners with confluence scoring, then validates findings using Perplexity Sonar Pro AI research.

**Core thesis**: Find stocks with 3+ technical signals agreeing (confluence ≥ 3) AND positive news/catalysts → high-probability entries.

---

## Architecture Essentials

### Three-Layer Design

1. **Data Layer** (`data_manager.py`, `config.py`)
   - Fetches JP small-cap universe (~100 stocks filtered by market cap, volume, liquidity)
   - Pulls OHLCV from yfinance; handles cache expiry (12hr for price data, 720hr for tickers)
   - Reads JPX short-selling regulations; filters by universe settings in `config.py`

2. **Signal Layer** (`scanners.py`, `technical_analysis.py`)
   - 9 independent scanners (each returns score 0-100 + list of signal reasons)
   - **Key principle**: Confluence scoring = Signals agreeing + Trend confirmation
   - Pure functions; no DB access; all config via `SCANNER_CONFIG` in `config.py`

3. **AI Research Layer** (`llm_research.py`)
   - Perplexity Sonar Pro integration (grounded web search)
   - Returns structured result: news summary, catalysts (list), sentiment, risks (list)
   - Scoring adjustment: +10 for positive news, +5 per catalyst, -3 per risk

### Production Flow

`generate_signals_with_research.py` chains these steps:
1. Load precomputed cache (`results/precomputed_cache.pkl` from walk-forward backtest)
2. Get latest date signals; filter by `MIN_SCANNER_SCORE` (default 30)
3. **For top N picks**: Call `llm_research.research_jp_stock()` to add news/catalysts
4. Output CSV with: symbol, technical score, adjusted score (tech + AI), news summary, risks, confluence count

---

## Critical Patterns & Conventions

### Configuration Hierarchy

**config.py** → **Runtime overrides**

```python
# In generate_signals_with_research.py:
min_score = config.MIN_SCANNER_SCORE  # From config.py
config.MIN_SCANNER_SCORE = 40  # Override for this session
```

**Edit `config.py`** for persistent changes (backtest params, scanner thresholds).

### Scanner Return Type: `ScanResult`

```python
ScanResult = Tuple[int, List[str]]  # (score, reasons)
# Example: (65, ["Uptrend (EMA Fast>Slow)", "Volume Spike (2.1x)"])
# Return (0, []) if no signal
```

All 9 scanners follow this pattern. Check `scanners.py` lines 1-80 for the template.

### LLM Research Error Handling

Perplexity API failures are **silent** (logged but not raised). If `LLM_AVAILABLE = False` or request fails, system degrades gracefully:
- Research columns become empty
- Technical score is used as final ranking score

See `llm_research.py` lines 160-200 for retry logic and timeout handling.

### Database Schema

JP Stocks SQLite DB (created via `data_manager.py`):
- `stocks` table: symbol, name, market_cap_jpy, avg_daily_volume, short_interest_pct
- Indexed on symbol for fast lookups
- Use `data_manager.get_connection()` for all DB access (uses WAL pragma for performance)

---

## Key Files & Their Responsibilities

| File | Purpose | Maintainer Notes |
|------|---------|------------------|
| `generate_signals_with_research.py` | **Production entry point** - run daily for next-day picks | Change `top_n` for research coverage; edit `min_score` in config, not here |
| `scanners.py` | 9 signal algorithms | Add new scanner by copying template (score 0-100 + reasons list) |
| `llm_research.py` | Perplexity integration | API key in `.env` as `PERPLEXITY_API_KEY`; lookback = 14 days |
| `technical_analysis.py` | Pure indicator calculations | All SMA, EMA, RSI, MACD, Bollinger Bands; zero side effects |
| `config.py` | Parameter source of truth | Edit here for persistent changes |
| `data_manager.py` | Data fetching & universe building | Handles caching; min volume 100k; excludes Nikkei 225 by default |

---

## Common Development Workflows

### Adding a New Scanner

1. In `scanners.py`, create function `scan_new_signal(data, jpx_data, scanner_config)`:
   ```python
   def scan_new_signal(...) -> ScanResult:
       """Signal description."""
       score = 0
       signals = []
       # ... logic, accumulate score ...
       return score, signals
   ```
2. In `scanners.py` line ~1100, add to `ACTIVE_SCANNERS` dict
3. Update `SCANNER_CONFIG` in `config.py` with any new thresholds
4. Backtest via `python run_walk_forward.py` to check Profit Factor

### Debugging Scanner Performance

Use `scanner_diagnostics.py`:
```bash
python scanner_diagnostics.py --scanner oversold_bounce --days 30
```
Shows per-trade breakdowns, false signals, confluence patterns.

### Tweaking Parameters Without Code

Edit `config.py`:
- `MIN_SCANNER_SCORE`: Lower = more signals (more noise)
- `SCANNER_CONFIG['MOMENTUM_RSI_MAX']`: Raise to catch overbought signals
- `UNIVERSE_TOP_N`: Universe size (increase for more opportunities, slower backtests)

Then run: `python run_walk_forward.py` (validation only; doesn't overwrite production params)

---

## Known Edge Cases & Gotchas

1. **JP Small Caps Have Low News Frequency**
   - Lookback window set to 30 days for research (not 7 days like US stocks)
   - Many tickers return "No recent news" → filter these manually if needed

2. **Market Cap in JPY, Not USD**
   - Settings use JPY thresholds (`max_market_cap_jpy: 500_000_000_000`)
   - Conversion not needed; data_manager fetches JPY natively

3. **Confluence Plateau at 3**
   - Having 4+ scanners agree ≠ proportionally better signal
   - Recommendation: position size by confluence (3+ = full size, 2 = 70%, 1 = 50%)

4. **Precomputed Cache is Time-Locked**
   - `results/precomputed_cache.pkl` contains fixed historical dates
   - For new calendar dates, must regenerate via `run_walk_forward.py`

5. **LLM Sentiment is Qualitative**
   - `news_sentiment` is categorical: "Positive" | "Negative" | "Neutral" | "Mixed"
   - Use only as veto filter; don't rely for scoring

---

## Testing & Validation

- **Unit**: `test_jpx_parser.py` - validates data fetch
- **Integration**: `run_backtest.py` - single-date backtest
- **Optimization**: `run_walk_forward.py` - multi-date, preserves best params in `results/best_params.json`
- **Paper Trade**: Run `generate_signals_with_research.py` daily for 2 weeks, compare to actual market moves

---

## Performance Locked Settings (Do Not Change Lightly)

These parameters produced PF 2.46 in backtests. Changes require re-validation:

```python
MIN_SCANNER_SCORE = 30        # ← Core gatekeeper
STOP_LOSS_PCT = 0.06          # ← Tight for small-cap volatility
RISK_REWARD_RATIO = 2.0       # ← Fixed target
EXIT_MODE = 'fixed_rr'        # ← NOT trailing (shakeout risk)
```

---

## When Adding Features

- **Scanner logic**: Must be pure function, return ScanResult, configurable via `config.py`
- **Data fetching**: Add to `data_manager.py`, cache with expiry hours in `config.py`
- **New indicators**: Add to `technical_analysis.py` (no side effects)
- **UI/reporting**: Can mutate `generate_signals_with_research.py` or notebook; avoid changing core modules
