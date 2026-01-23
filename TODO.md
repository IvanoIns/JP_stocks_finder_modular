# JP Stocks Finder Modular — TODO

## Phase 1: Foundation ✅
- [x] `config.py` — All settings
- [x] `data_manager.py` — DB, ingestion, universe building
- [x] `technical_analysis.py` — Indicators

## Phase 2: Scanners ✅
- [x] `scanners.py`
  - [x] `scan_momentum_star()`
  - [x] `scan_reversal_rocket()`
  - [x] `scan_consolidation_breakout()`
  - [x] `scan_relative_strength()`
  - [x] `scan_burst_candidates()`
  - [x] `scan_oversold_bounce()`
  - [x] `scan_volatility_explosion()`
  - [x] `scan_power_combinations()`
  - [x] `get_all_signals()` aggregator

## Phase 3: Backtest Engine ✅
- [x] `backtesting.py`
  - [x] `Position` and `Trade` dataclasses
  - [x] `BacktestEngine` class
  - [x] Position sizing logic
  - [x] Entry queueing (next-day open)
  - [x] Exit checking (stops, targets)
  - [x] Exit modes (default, trailing, breakeven)
  - [x] `run_daily_backtest()` main function

## Phase 4: Optimizer ✅
- [x] `optimizer.py`
  - [x] `grid_search_daily()`
  - [x] `walk_forward_grid_search()`
  - [x] `summarize_oos()`
  - [x] `top_params_by_stability()`
- [x] `run_backtest.py` CLI

## Phase 5: Validation
- [ ] `notebooks/pipeline_wfa.ipynb`
- [ ] Run baseline backtest
- [ ] Verify trade log
- [ ] Run mini grid search
- [ ] Run WFA
- [ ] Document results

---
Last updated: 2026-01-23
