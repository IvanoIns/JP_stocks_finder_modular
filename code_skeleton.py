"""
JP Stocks Modular — Code Skeleton

Quick reference for system architecture and key functions.
Last updated: 2026-01-29
"""

# =============================================================================
# FILE STRUCTURE
# =============================================================================
"""
JP_stocks_modular/
├── generate_signals_with_research.py # 🚀 MAIN SCRIPT (Scanner + AI)
├── llm_research.py         # 🧠 Perplexity API Wrapper
├── config.py               # Parameters
├── scanners.py             # 9 scanner strategies
├── backtesting.py          # Backtest engine
├── precompute.py           # Cache engine
├── data_manager.py         # DB access
└── .env                    # API Keys
"""

# =============================================================================
# KEY MODULES
# =============================================================================

# --- llm_research.py ---
"""
class ResearchResult:
    recent_news_summary: str
    upcoming_catalysts: List[str]
    news_sentiment: "Positive" | "Negative" | "Neutral"
    key_risks: List[str]

research_asset(name, ticker, kind="jp_stock") -> ResearchResult
    # Uses Perplexity Sonar Pro
    # Searches EN + JA news + Social (X/Twitter)
    # Returns structured JSON
"""

# --- generate_signals_with_research.py ---
"""
generate_signals_with_research(top_n=20)
    1. Loads scanner cache
    2. Filters signals >= MIN_SCANNER_SCORE (30) at runtime (cache stores raw signals)
    3. Calls research_asset() for top N picks
    4. Calculates Adjusted Score:
       Base + Sentiment(+10) + Catalysts(+20) - Risks(-15)
    5. Prints detailed report
"""

# --- config.py ---
"""
MIN_SCANNER_SCORE = 30
STOP_LOSS_PCT = 0.06
RISK_REWARD_RATIO = 2.0
"""

# =============================================================================
# DATA FLOW
# =============================================================================
"""
1. jp_stocks.db (raw prices)
      ↓
2. precompute.py → precomputed_cache.pkl (signals cached)
      ↓
3. generate_signals_with_research.py (loads cache)
      ↓
4. Scanners Filter → Top 20 Candidates
      ↓
5. llm_research.py ← Queries Perplexity API (News/Social)
      ↓
6. Final Output: Ranked list with Tech Score + AI Insights
"""

# =============================================================================
# SETUP
# =============================================================================
"""
1. cp .env.example .env
2. Add PERPLEXITY_API_KEY
3. python precompute.py (auto-expands DB ~1000 tickers/run)
4. python generate_signals_with_research.py
"""
