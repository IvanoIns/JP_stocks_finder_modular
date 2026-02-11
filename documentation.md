# JP Stocks Modular - Documentation

## Overview

Japanese stock scanner targeting **pre-burst** setups (manual / paper trading). It combines:
- **Quant scanners** (technical signals + confluence scoring)
- Optional **LLM research** (news/catalysts/risks sentiment as a *soft* score adjustment)

**Source of truth:** `config.py` (ignore `settings.yaml` if present).

---

## Daily Pipeline (Cache-Based)

The system is designed so `generate_signals.py` is fast (seconds) by using a precomputed cache.

1) **Update DB (incremental):**
   - `python -c "import data_manager as dm; dm.update_recent_data(days=5)"`
   - Updates only "active" symbols (volume >= `MIN_AVG_DAILY_VOLUME` on the latest DB date).
   - Uses batched multi-ticker downloads for short ranges (faster, fewer rate limits).
   - Avoids wasting time on a date Yahoo hasn't published yet:
     - `YFINANCE_DAILY_READY_HOUR` (JST) controls when "today" is allowed.
     - `YFINANCE_USE_LATEST_DATE_GUARD=True` clamps requests to Yahoo's latest available daily bar.
   - Bad/invalid tickers may be listed in `cache/bad_yfinance_tickers.txt`.
     - Default policy: only skip symbols that have **no rows in the DB** (`YFINANCE_BAD_TICKERS_POLICY="skip_if_no_db_data"`).
     - If the file gets polluted, prune it with: `python clean_bad_yfinance_tickers.py`

2) **Update cache (fast scanning, incremental):**
    - `python precompute.py`
    - Produces/updates `results/precomputed_cache.pkl`
    - Default behavior:
      - computes only missing trading dates
      - refreshes indicator data for the current universe symbols
      - prunes old dates outside the requested window
    - Rebuild triggers:
      - fingerprint mismatch on code/config/runtime knobs
      - explicit `python precompute.py --rebuild`
    - Fingerprint covers hashes of:
      - `scanners.py`
      - `technical_analysis.py`
      - `data_manager.py`
      - `config.py`
      - `precompute.py`
    - Also optionally:
      - expands DB coverage (`AUTO_EXPAND_DB`)
      - fills missing market caps incrementally (`AUTO_UPDATE_MARKET_CAP`)

3) **Generate signals (scanner-only):**
   - `python generate_signals.py`
   - Saves picks to:
     - `results/daily_picks/daily_picks_YYYY-MM-DD.csv`
     - `results/daily_picks/daily_picks_YYYY-MM-DD.json`

4) **Burst audit (real market recall vs picks):**
   - `python burst_audit.py daily --pending --start 2026-01-31`
   - Auto-collects DB bursts (`close(D) >= close(prev_day) * 1.10`) into:
     - `results/burst_audit/bursts_log.csv`
   - Audits each burst date against previous-day candidates and saves:
     - `results/burst_audit/audit_YYYY-MM-DD.csv`
     - `results/burst_audit/audit_summary_YYYY-MM-DD.json`
     - `results/burst_audit/audit_master.csv`
   - If A/B is enabled, it also compares baseline vs shadow universe (top-N):
     - Shadow picks: `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.csv|json`
     - A/B daily audit: `results/burst_audit/ab_audit_YYYY-MM-DD.csv`
     - A/B daily summary: `results/burst_audit/ab_audit_summary_YYYY-MM-DD.json`
     - A/B master: `results/burst_audit/ab_audit_master.csv`
   - If you miss days, `--pending` catches up all missing dates.
   - You can manually append missing external bursts to `bursts_log.csv`; audit re-runs only when payload changes.

5) **Generate signals + LLM research (optional):**
   - `python generate_signals_with_research.py --top 20`
   - Saves:
     - `results/llm_research_YYYYMMDD_HHMMSS.json`
     - `results/llm_research_YYYYMMDD_HHMMSS.csv`

6) **Dashboards (optional):**
   - Streamlit (recommended): `streamlit run streamlit_dashboard.py`
   - Charts (PNG): `python plot_signals_charts.py --top 20 --days 180`

7) **Run everything (no LLM by default):**
   - `python run_all.py`
   - `run_all.py` now includes burst audit catch-up by default (`--pending --start BURST_AUDIT_START_DATE`).
   - A/B is run by default from config. Override from CLI:
     - `--no-burst-ab`
     - `--burst-ab-top-n 10`
     - `--burst-ab-shadow-min-volume 5000`
   - Edit defaults in `config.py` and `run_all.py` (or use CLI flags).
   - Expansion paths:
     - explicit: `run_all.py --expand-db --fill-market-caps`
     - implicit: `precompute.py` runs `AUTO_EXPAND_DB` / `AUTO_UPDATE_MARKET_CAP` if enabled in `config.py`

---

## Universe Selection (Aligned With Backtests)

Universe is built daily by `data_manager.build_liquid_universe`:
- Volume floor: `MIN_AVG_DAILY_VOLUME`
- Optional market-cap ceiling: `ENFORCE_MARKET_CAP=True` + `MAX_MARKET_CAP_JPY`
  - Missing market caps follow `MARKET_CAP_MISSING_POLICY` (default: include)
- Nikkei 225 excluded when enabled: `EXCLUDE_NIKKEI_225=True`
- **No top-N cap by default:** `UNIVERSE_TOP_N=None`

If you change universe/scanner parameters in `config.py`, the next cache run detects the fingerprint change and rebuilds automatically.

---

## Early Mode (Default)

The "Early Mode" output is meant to bias toward **soon-to-burst** candidates instead of already-pumped momentum names.

Default filters:
- RSI <= `EARLY_MODE_RSI_MAX` (currently 65)
- 10D return < `EARLY_MODE_10D_RETURN_MAX` (currently 15%)

Default scanner subset:
- `oversold_bounce`, `reversal_rocket`, `volatility_explosion`, `coiling_pattern`, `consolidation_breakout`

Optional legacy (momentum-allowed) output:
- `EARLY_MODE_SHOW_BOTH=True`

---

## Signal Evaluation (Did Picks Actually Work?)

Daily bars can't reveal intraday order, so evaluation uses a conservative rule:
- **Stop is checked before target** on the same day (matches `backtesting.BacktestEngine` behavior).

### Single day
- `python evaluate_signals_forward.py --date 2026-02-02 --top 20 --horizon 10`
- Saves: `results/signal_eval_YYYY-MM-DD_topN_hH.csv`

### Date range mapping (Jan 27 -> today, etc.)
- `python evaluate_signals_range.py --start 2026-01-27 --end 2026-02-02 --top 20 --horizon 5`
- Saves:
  - `results/signal_eval_range_<start>_to_<end>_topN_hH.csv` (daily summary)
  - `results/signal_eval_range_<start>_to_<end>_topN_hH_trades.csv` (trade details)

Evaluation note:
- By default, evaluation requires a full horizon worth of future bars in the cache. If the cache ends too soon, trades are labeled `horizon_truncated` and excluded from the win-rate/avg-return stats.
- Use `--allow-truncated-horizon` if you intentionally want partial (not fully matured) results.

### A/B scorecard (Universe Phase 2 decision gate)
- `python ab_scorecard.py --ab-master results/burst_audit/ab_audit_master.csv --top-n 10 --window-days 20 --min-days 20`
- Saves:
  - `results/burst_audit/ab_scorecard_daily.csv`
  - `results/burst_audit/ab_scorecard_latest.json`
- Decision logic:
  - Requires at least `min_days` observed burst dates in the rolling window
  - Pass capture gate if relative capture-rate improvement (B vs A) >= `min_rel_capture_improvement`
  - Pass precision gate if relative precision drop (B vs A) <= `max_rel_precision_drop`

---

## Common Outputs (Where Data Is Stored)

- Price DB: `jp_stocks.db`
  - `daily_prices` (OHLCV)
  - `symbol_info` (market cap, name, sector)
  - `jpx_short_interest` (live-only context; optional)
- Cache: `results/precomputed_cache.pkl` (incrementally updated by default)
- Daily picks: `results/daily_picks/daily_picks_YYYY-MM-DD.csv|json`
- Burst log: `results/burst_audit/bursts_log.csv`
- Burst audit: `results/burst_audit/audit_YYYY-MM-DD.csv|json` and `results/burst_audit/audit_master.csv`
- Burst audit A/B: `results/burst_audit/ab_audit_YYYY-MM-DD.csv|json` and `results/burst_audit/ab_audit_master.csv`
- A/B scorecard: `results/burst_audit/ab_scorecard_daily.csv` and `results/burst_audit/ab_scorecard_latest.json`
- Shadow picks cache: `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.csv|json`
- LLM research: `results/llm_research_*.json|csv`
- Evaluation: `results/signal_eval_*.csv` and `results/signal_eval_range_*.csv`

---

## Notes / Limitations

- Manual execution only (no broker API yet).
- Daily OHLC evaluation is approximate (no intraday sequencing).
- If you want to reduce the cost of market-cap updates, set `AUTO_UPDATE_MARKET_CAP=False` once market caps are mostly filled.

---

## Related Docs

- Daily priorities and retune gates: `NEXT_STEPS.md`
- Full architecture and handoff details: `PROJECT_HANDOFF.md`
- Release-level change history: `CHANGELOG.md`

## Module Ownership

- Library modules:
  - `config.py`
  - `data_manager.py`
  - `technical_analysis.py`
  - `scanners.py`
  - `backtesting.py`
  - `optimizer.py`
  - `llm_research.py`
- Core operational CLI scripts:
  - `run_all.py`
  - `precompute.py`
  - `generate_signals.py`
  - `generate_signals_with_research.py`
  - `burst_audit.py`
- Analysis/reporting CLI scripts:
  - `evaluate_signals_forward.py`
  - `evaluate_signals_range.py`
  - `ab_scorecard.py`
  - `diagnose_missed_signals.py`
  - `plot_signals_charts.py`
- Maintenance CLI scripts:
  - `expand_db.py`
  - `fill_market_caps.py`
  - `clean_bad_yfinance_tickers.py`
  - `yfinance_field_mapper.py`

## A/B Testing Path

Current implemented phase:

1) Universe A/B (Phase 2):
   - A (baseline): current universe filters
   - B (shadow): same filters, lower min volume (`BURST_AB_SHADOW_MIN_VOLUME`)
   - Comparison metric: burst capture at `top N` (`BURST_AB_TOP_N`)

2) Single-signal A/B (Phase 3):
   - A (baseline): current ranking
   - B (shadow): same universe, enforce minimum single-scanner presence in top-N
   - Controls:
     - `BURST_AB_SINGLE_MIN_COUNT`
     - `BURST_AB_SINGLE_MIN_SCORE`
     - `BURST_AB_TOP_N`
   - Select variant with:
     - `BURST_AB_VARIANT = "single_signal_mix"`
     - or CLI `--ab-variant single_signal_mix`

Planned next phases:

1) Scanner-subset A/B based on observed miss reasons

Rules:
- Change one variable at a time
- Evaluate with `top 10` as primary cut
- Require a minimum 20-trading-day window per phase before promoting changes

---

Last updated: 2026-02-11
