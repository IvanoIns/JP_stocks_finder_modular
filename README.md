# JP Stocks Modular Trading System

> **A Japanese liquidity-selected stock scanner that identifies "burst" candidates using 9 technical scanners + AI-powered research.**

## Quick Start

```bash
cd JP_stocks_modular

# 1. Setup API Key (for AI research)
cp .env.example .env
# Edit .env and add your PERPLEXITY_API_KEY

# 2. Build / refresh scanner cache (required)
python precompute.py

# 3. Get AI-Analyzed Picks for Tomorrow
python generate_signals_with_research.py
```

## Core Features

1. **Foundational Scanners**: 9 technical strategies (RSI, Volume, Pattern Matching)
2. **Confluence Scoring**: Signals get +10 points for each additional scanner that agrees.
3. **AI Analyst (New)**: Perplexity Sonar Pro researches top 20 picks for:
   - 📰 **News**: Earnings, partnerships, M&A (Multilingual EN/JA)
   - 🗣️ **Social**: Sentiment from X (Twitter) & retail forums
   - 📅 **Catalysts**: Upcoming events vs. priced-in news
   - ⚖️ **Verdict**: Adjusts scanner score (soft penalty/bonus, ~-25 to +30) based on findings

**Trading mode**: Manual / paper trading (no broker API connected yet).
**Universe**: Built daily by liquidity (top N by notional / volume floor), Nikkei 225 excluded. Defaults: `UNIVERSE_TOP_N=500`, `MIN_AVG_DAILY_VOLUME=20_000`. Market-cap filter is enforced and `symbol_info.market_cap` is auto‑populated incrementally.
**Source of truth**: `config.py` (legacy YAML config removed).
**Early Mode (default)**: Pre‑burst focus with filters (10‑day return < 15%, RSI ≤ 65) and early‑scanner subset.

## Current Configuration (Proven: PF 2.46)

| Parameter | Value |
|-----------|-------|
| Min Score | 30 |
| Stop Loss | 6% |
| R:R Ratio | 2:1 |
| Exit Mode | fixed_rr |

## Early Mode (Default)

- Filters: 10‑day return < 15%, RSI ≤ 65
- Scanners: `oversold_bounce`, `reversal_rocket`, `volatility_explosion`, `coiling_pattern`, `consolidation_breakout`
- Toggle: set `EARLY_MODE_SHOW_BOTH=True` to also print legacy (momentum‑allowed) output

## File Structure

```
JP_stocks_modular/
├── llm_research.py        # 🧠 AI Research Engine (Perplexity)
├── generate_signals_with_research.py # 🚀 Main production script
├── generate_signals.py    # Legacy (scanner only)
├── run_walk_forward.py    # Optimization
├── config.py              # Parameters
├── scanners.py            # 9 Strategies
├── results/               
│   ├── precomputed_cache.pkl
│   └── best_params.json
└── .env                   # API Keys
```

## Daily Workflow

1. **Update DB**: `python -c "import data_manager as dm; dm.update_recent_data(days=5)"`
2. **Rebuild Cache**: `python precompute.py` (auto-expands DB + updates market caps each run)
3. **Run Generator**: `python generate_signals_with_research.py` (auto-saves LLM results to `results/llm_research_*.json` and `.csv`)
4. **Plot Charts (optional)**: `python plot_signals_charts.py --top 20 --days 180`
5. **Terminal Dashboard (optional)**: `python signals_dashboard.py --top 20 --days 180`
6. **Streamlit Dashboard (recommended)**: `streamlit run streamlit_dashboard.py`  
   - If the dashboard is empty, pick a **date with signals** (checkbox is on by default) or lower **Min Score**.
7. **Run All (no LLM by default)**: `python run_all.py`  
   - Includes optional DB expand + market-cap fill (can disable): `python run_all.py --no-expand-db --no-fill-market-caps`  
   - Add LLM: `python run_all.py --with-llm`
8. **Review Output**:
   - Check **Adjusted Score** (Scanner + AI Bonus)
   - Read **News Summary** & **Risks**
   - Verify **Confluence** count
   - Review **budget split** (lot-affordable vs over-budget) using `MAX_JPY_PER_TRADE` and `LOT_SIZE`
   - Entry uses **next open** when available; if not, last close is marked with `*`
7. **Execute Trades**:
   - Entry: Market Open (use your actual fill to set stop/target)
   - Stop: -6%
   - Target: +12%

## Active Scanners (9)

| Scanner | Performance | Role |
|---------|-------------|------|
| `oversold_bounce` | ⭐ PF 11.13 | Sniper (rare, accurate) |
| `burst_candidates` | ⭐ PF 4.30 | Core signal generator |
| `momentum_star` | ⭐ PF 4.77 | Trend follower |
| `relative_strength` | ✅ PF 1.98 | Volume generator |
| `volatility_explosion` | ✅ PF 2.20 | Mean reversion |
| `consolidation_breakout` | ✅ Rare | Breakout detector |
| `reversal_rocket` | ✅ Mixed | Oversold bounce |
| `smart_money_flow` | ⚠️ PF 1.26 | Institutional flow |
| `coiling_pattern` | ⚠️ Mixed | BB squeeze |

**Disabled**: `crash_then_burst`, `stealth_accumulation` (PF 0.00)

## Known Issues / Risks

- JPX short-interest is live-only context for LLM research (not used in backtests/scanner scoring; missing data is neutral).
- Fast cache must be rebuilt after changing universe or scanner settings.
- DB coverage matters: a symbol not present in `jp_stocks.db` can never be signaled. Use `python expand_db.py --max-new 1000` to grow coverage incrementally.

## License

Research/educational use only. Not financial advice.
