# JP Stocks Modular - TODO & Status

## Status: PRODUCTION READY

---

## Completed

### Phase 1: Foundation
- [x] config.py - Parameters centralized
- [x] data_manager.py - SQLite access
- [x] technical_analysis.py - All indicators

### Phase 2: Scanners
- [x] 9 active scanners (Confluence bonus enabled)
- [x] Disabled weak scanners (PF 0.00)

### Phase 3: AI Research Integration
- [x] `llm_research.py` - Perplexity Sonar Pro integration
- [x] Multilingual search (EN + JA)
- [x] Social sentiment (X/Twitter)
- [x] `generate_signals_with_research.py` - Hybrid scoring

### Phase 4: Optimization
- [x] Walk-forward analysis complete
- [x] Winning Params: Score 30, R:R 2.0 (PF 2.46)

---

## Active Scanners (9)

- momentum_star
- consolidation_breakout
- relative_strength
- burst_candidates
- reversal_rocket
- oversold_bounce
- volatility_explosion
- smart_money_flow
- coiling_pattern

---

## Next Actions

### Priority Fixes (Backtest / Data Correctness)

1. [x] **Slow backtest bars coverage**: In `run_daily_backtest`, load bars for `universe ∪ open_positions ∪ pending_entries` (scan only the universe).
2. [x] **JPX short-interest (live-only)**: Remove from backtests/cache, treat missing as neutral, and pass live short ratio into LLM research prompts.
3. [x] **Early Mode default**: Pre-burst filters (10D return < 15%, RSI <= 65) + early scanner subset, with optional legacy output.
4. [x] **Universe definition enforcement**: Enforce `MAX_MARKET_CAP_JPY` when market-cap data exists; missing caps pass by default.
5. [x] **Cache supports `min_score` optimization**: Cache stores raw signals (score > 0) and filters by `MIN_SCANNER_SCORE` at runtime.
6. [ ] **Signal output vs real entry**: Make stops/targets explicitly based on actual next-open fill (or output % only + optional CSV for paper trading).

### Ongoing (Paper Trading)

7. [ ] **Daily run**: Update DB -> rebuild cache -> run `generate_signals_with_research.py`
8. [ ] **Monitor**: Check if AI sentiment correlates with actual price moves
9. [ ] **Refine**: Adjust AI score weights if needed (keep sentiment as penalty, not veto)
10. [ ] **Expand DB coverage**: Use `python expand_db.py --max-new 1000 --start 2024-01-01` (repeat) so microcaps can be scanned.

---

*Last updated: 2026-01-29*
