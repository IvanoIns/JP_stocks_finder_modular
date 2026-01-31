# JP Stocks Modular — Complete Project Handoff

> **Purpose**: Enable another LLM or developer to continue this project.  
> **Date**: 2026-01-29

---

## 1. PROJECT OVERVIEW

### What It Does
A hybrid **Quant + AI** trading system for Japanese stocks (liquidity-selected; market-cap filter pending).
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
├── generate_signals_with_research.py # 🚀 PRODUCTION SCRIPT
├── llm_research.py         # AI Module (Perplexity Integration)
├── config.py               # Parameters (Score 30, Stop 6%)
├── scanners.py             # 9 active strategies
├── .env                    # API Keys (Perplexity)
└── results/
    ├── precomputed_cache.pkl # Cached scanner signals
```

---

## 3. AI RESEARCH MODULE (`llm_research.py`)

**Engine**: Perplexity Sonar Pro
**Integration**: `generate_signals_with_research.py` calls this for the top 20 technical picks.

**Key Features**:
- **Multilingual**: Searches Japanese ("銘柄名 ニュース") and English.
- **Social**: Scans X (Twitter) and forums for retail buzz.
- **Scoring**:
  - `Positive` News: +10 points
  - `Catalyst` Found: +5 points each
  - `Risk` Found: -3 points each

---

## 4. SCANNERS (9 Active)

| Scanner | Type | PF | Status |
|---------|------|-----|--------|
| `oversold_bounce` | Mean-reversion | 11.13 | ⭐ Star |
| `burst_candidates` | Pattern match | 4.30 | ⭐ Star |
| `momentum_star` | Trend | 4.77 | ⭐ Star |
| `relative_strength` | Trend | 1.98 | ✅ Solid |
| `volatility_explosion` | Mean-reversion | 2.20 | ✅ Solid |
| `consolidation_breakout` | Breakout | Rare | ✅ Solid |
| `reversal_rocket` | Oversold | Mixed | ✅ Active |
| `smart_money_flow` | Institutional | 1.26 | ⚠️ Monitor |
| `coiling_pattern` | BB Squeeze | Mixed | ⚠️ Monitor |

**Disabled**: `crash_then_burst`, `stealth_accumulation` (PF 0.00)

---

## 5. HOW TO RUN

### step 1: Setup API
Copy `.env.example` to `.env` and add `PERPLEXITY_API_KEY`.

### Step 2: Build/Refresh Cache (Required)
```bash
python precompute.py
```

### Step 3: Run Analysis
```bash
python generate_signals_with_research.py
```
This produces a ranked list of ~20 stocks with technical scores AND news summaries.

---

## 6. KEY LEARNINGS

1. **Confluence is King**: 3 scanners agreeing is much stronger than 1 scanner with high score.
2. **AI Filters Noise**: The LLM successfully catches "good chart, bad news" scenarios.
3. **Small Caps need Looser Filters**: We increased news lookback to 30 days because JP small caps don't have daily news.
4. **Fixed R:R wins**: Trailing stops tend to shake out early in volatile JP markets.

---

## 7. KNOWN ISSUES / RISKS

1. **Market-cap data coverage**: auto-population is enabled; monitor completeness for your universe.
2. **Short interest is live-only**: JPX short-interest is used only as optional LLM context (not used in backtests/scanner scoring; missing data is neutral).
3. **Cache sensitivity**: changes to universe or scanner settings require a cache rebuild for correctness.
4. **Signal price vs fill**: stops/targets are % based and should be computed from your actual entry fill at the open.

---

## 8. NEXT STEPS

1. **Live Paper Trading**: Run daily for 2 weeks to verify AI verdict accuracy.
2. **Sentiment Weighting**: Tune how much the AI score affects the final rank (currently conservative).
3. **Social-Only Mode**: Create a mode that *only* scans for Twitter buzz spikes.

---

*End of Handoff*
