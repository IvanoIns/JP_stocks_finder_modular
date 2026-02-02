# JP Stocks Modular — Documentation

## System Overview

Japanese liquidity-selected stock scanner targeting "burst" candidates (10-30% moves). Combines **Quantitative Scanners** with **Qualitative AI Research**.

**Proven Performance**: PF 2.46, 59% Win Rate (Score 30, R:R 2.0)

**Source of truth**: `config.py` (legacy YAML config removed).  
**Execution mode**: Manual / paper trading (no broker API connected yet).

---

## 🧠 AI Research Module (New)

Located in `llm_research.py`, this module adds a "human-like" analyst review to the automated signals.

### Features
- **Engine**: Perplexity Sonar Pro (Grounded Search)
- **Scope**: Last 30 days of news + Social Media (X/Twitter)
- **Language**: English & Japanese (`{Asset} ニュース`)
- **Output**: JSON structured data (Catalysts, Risks, Sentiment)

### Scoring Adjustment
The AI modifies the raw Scanner Score to create a final **Adjusted Score**:
- **Sentiment**: +10 (Positive) to -10 (Negative)
- **Catalysts**: +5 per valid catalyst (Max +20)
- **Risks**: -3 per key risk (Max -15)

Negative sentiment is a **soft penalty**, not a veto (a very strong technical score can still rank highly).

---

## Current Parameters

```python
MIN_SCANNER_SCORE = 30      # Filter threshold
STOP_LOSS_PCT = 0.06        # 6% stop
RISK_REWARD_RATIO = 2.0     # 12% target
EXIT_MODE = 'fixed_rr'      # Exit at target or stop
```

---

## Universe Selection (Aligned)

Universe is built daily using `data_manager.build_liquid_universe`:  
top `UNIVERSE_TOP_N` by notional (with `MIN_AVG_DAILY_VOLUME`), excluding Nikkei 225 if enabled. Defaults: `UNIVERSE_TOP_N=1500`, `MIN_AVG_DAILY_VOLUME=20_000`.

This is shared by **backtests** and **precompute cache** for consistency.  
If you change universe settings in `config.py`, rebuild the cache.  
Cache now stores all triggered signals (score > 0) and filters by `MIN_SCANNER_SCORE` at runtime.  
Market-cap filtering is enforced; `symbol_info.market_cap` is auto‑populated incrementally.

---

## Known Issues / Risks

- JPX short-interest is live-only context for LLM research (not used in backtests/scanner scoring; missing data is neutral).
- Fast cache must be rebuilt after changing universe or scanner settings.
- Delisted tickers are cached in `cache/bad_yfinance_tickers.txt` and skipped on future runs.

---

## Cache Workflow (Daily)

Signal generation is cache-based for speed.

1. Update DB: `python -c "import data_manager as dm; dm.update_recent_data(days=5)"`
   - Skips symbols already up-to-date and avoids pre‑close (before 16:00 JST) requests.
2. Rebuild cache: `python precompute.py` (auto-expands DB + updates market caps each run)
3. Generate signals: `python generate_signals.py` (scanner only) or `python generate_signals_with_research.py` (scanner + LLM, auto-saves to `results/llm_research_*.json` and `.csv`)
4. Plot charts (optional): `python plot_signals_charts.py --top 20 --days 180`
5. Terminal dashboard (optional): `python signals_dashboard.py --top 20 --days 180`
6. Streamlit dashboard (recommended): `streamlit run streamlit_dashboard.py`  
   - If empty, select a **date with signals** (checkbox on by default) or lower **Min Score**.
7. Run All: `python run_all.py` (defaults editable in file)

Stops/targets are % based (6% stop, 2R target) and should be computed from your actual entry fill at the open.  
Output is split into lot-affordable vs over-budget picks using `MAX_JPY_PER_TRADE` and `LOT_SIZE` from `config.py`.  
Early Mode is default: 10‑day return < 15%, RSI ≤ 65, and early‑scanner subset. Toggle legacy output via `EARLY_MODE_SHOW_BOTH`.  
Signal entry prices use the **next trading day open** when available; if not, they fall back to the last close and are marked with `*`.

---

## Expanding DB Coverage (One-Time / Occasional)

If a ticker is not present in `jp_stocks.db` (table `daily_prices`), it cannot be scanned and will never appear in signals.

Use incremental downloads (resumable):
- Add ~1,000 new tickers: `python expand_db.py --max-new 1000 --start 2024-01-01`
- Repeat as needed, then rebuild the cache: `python precompute.py`

Auto-expand is enabled by default in `config.py` and runs inside `precompute.py`.

---

## Script Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_signals_with_research.py` | **Main Tool**: Scans + AI Research | Daily |
| `llm_research.py` | AI Module (Perplexity API) | Import only |
| `generate_signals.py` | Legacy: Scans only (faster) | Quick check |
| `run_all.py` | Full pipeline (expand + caps + cache + signals) | Daily |
| `run_walk_forward.py` | Optimization | Research |

---

## Scanners (9 Active)

### Star Performers
- **`oversold_bounce`** — PF 11.13, RSI oversold + bounce
- **`burst_candidates`** — PF 4.30, Historical pattern match
- **`momentum_star`** — PF 4.77, EMA cross + volume

### Solid Contributors  
- **`relative_strength`** — PF 1.98, MA alignment
- **`volatility_explosion`** — PF 2.20, High vol near lows
- **`consolidation_breakout`** — Tight range breakout

---

## Confluence Bonus

When multiple scanners fire on same symbol:
```python
bonus = (num_scanners - 1) * 10
# 2 scanners = +10, 3 scanners = +20, etc.
```

---

## Transaction Costs

- Slippage: 0.4%
- Commission: 0.1%  
- **Total: 1.0% round-trip**

---

*Last updated: 2026-02-02*
