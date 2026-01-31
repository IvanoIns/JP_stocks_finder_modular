# JP Stocks Modular — Complete Project Handoff

> **Purpose**: Enable another LLM or developer to continue this project.  
> **Date**: 2026-01-31

---

## 1. PROJECT OVERVIEW

### What It Does
A hybrid **Quant + AI** trading system for Japanese stocks (liquidity-selected; **market-cap filter enforced**).
**Early Mode (default)** focuses on pre-burst setups: 10-day return < 15% and RSI <= 65 using early scanners.
1. **Quant**: 9 technical scanners identify "burst" patterns (confluence scoring).
2. **AI**: Perplexity Sonar Pro researches top picks to validate news/catalysts.

### Core Strategy
```
IF scanner_score >= 30
THEN candidate.

AI research adjusts the score (soft penalty/bonus).
Negative sentiment is a penalty, not a veto.

Execution plan (manual / paper trading):
  - Entry: next session open (signals use next-day open when available; otherwise last close is marked with `*`)
  - Stop Loss: 6%
  - Target: 12% (2:1 R:R)
```

---

## 2. FILE STRUCTURE & ARCHITECTURE

```
JP_stocks_modular/
|-- generate_signals_with_research.py  # Production script (scanner + LLM)
|-- generate_signals.py                # Scanner-only
|-- llm_research.py                    # AI Module (Perplexity Integration)
|-- streamlit_dashboard.py             # Streamlit UI dashboard
|-- plot_signals_charts.py             # PNG charts for top signals
|-- config.py                          # Parameters (Score 30, Stop 6%)
|-- scanners.py                        # 9 active strategies
|-- .env                               # API Keys (Perplexity)
|-- results/
|   |-- precomputed_cache.pkl          # Cached scanner signals
|   |-- llm_research_*.json/.csv        # Saved AI results
```

---

## 3. AI RESEARCH MODULE (`llm_research.py`)

**Engine**: Perplexity Sonar Pro  
**Integration**: `generate_signals_with_research.py` calls this for the top picks.

**Key Features**:
- **Multilingual**: Searches Japanese and English
- **Social**: Scans X (Twitter) and forums for retail buzz
- **Scoring**:
  - `Positive` News: +10 points
  - `Catalyst` Found: +5 points each
  - `Risk` Found: -3 points each

---

## 4. SCANNERS (9 Active)

| Scanner | Type | PF | Status |
|---------|------|-----|--------|
| `oversold_bounce` | Mean-reversion | 11.13 | Star |
| `burst_candidates` | Pattern match | 4.30 | Star |
| `momentum_star` | Trend | 4.77 | Star |
| `relative_strength` | Trend | 1.98 | Solid |
| `volatility_explosion` | Mean-reversion | 2.20 | Solid |
| `consolidation_breakout` | Breakout | Rare | Solid |
| `reversal_rocket` | Oversold | Mixed | Active |
| `smart_money_flow` | Institutional | 1.26 | Monitor |
| `coiling_pattern` | BB Squeeze | Mixed | Monitor |

**Disabled**: `crash_then_burst`, `stealth_accumulation` (PF 0.00)

---

## 5. HOW TO RUN

### Step 1: Setup API
Copy `.env.example` to `.env` and add `PERPLEXITY_API_KEY`.

### Step 2: Build/Refresh Cache (Required)
```bash
python precompute.py
```

### Step 3: Run Analysis
```bash
python generate_signals_with_research.py
```

### Optional: Dashboards
```bash
streamlit run streamlit_dashboard.py
python plot_signals_charts.py --top 20 --days 180
```

---

## 6. KNOWN ISSUES / RISKS

1. **Market-cap data coverage**: auto-population is enabled; monitor completeness for your universe.
2. **Short interest is live-only**: JPX short-interest is used only as optional LLM context (not used in backtests/scanner scoring; missing data is neutral).
3. **Cache sensitivity**: changes to universe or scanner settings require a cache rebuild for correctness.
4. **Signal price vs fill**: stops/targets are % based and should be computed from your actual entry fill at the open.

---

## 7. NEXT STEPS

1. **Live Paper Trading**: Run daily for 2 weeks to verify AI verdict accuracy.
2. **Sentiment Weighting**: Tune how much the AI score affects the final rank (currently conservative).
3. **Social-Only Mode**: Create a mode that only scans for social buzz spikes.

---

*End of Handoff*
