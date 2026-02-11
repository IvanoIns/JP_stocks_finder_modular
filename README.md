# JP Stocks Modular Trading System

A Japan stock scanner focused on **pre-burst** setups using:
- Technical scanners (+ confluence bonus)
- Optional LLM research (news/catalysts/risks sentiment as a *soft* score adjustment)

Manual / paper-trading only (no broker integration yet).

## Quick Start

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) (Optional) LLM research key
# macOS/Linux
cp .env.example .env
# Windows PowerShell
Copy-Item .env.example .env
# edit .env and add PERPLEXITY_API_KEY (or the configured provider key)

# 3) Run the full daily pipeline
python run_all.py
```

## Daily Workflow (What To Run)

Scanner-only:
```bash
python -c "import data_manager as dm; dm.update_recent_data(days=5)"
python precompute.py
python generate_signals.py
python burst_audit.py daily --pending --start 2026-01-31
```

`precompute.py` is incremental by default:
- Reuses the existing cache and computes only missing trading dates
- Auto-rebuilds when fingerprinted settings/code changed
- Force full rebuild with: `python precompute.py --rebuild`

DB expansion / market-cap update have two paths:
- Explicit path (manual toggle): `python run_all.py --expand-db --fill-market-caps`
- Implicit path (config-driven inside cache update): `precompute.py` runs `AUTO_EXPAND_DB` / `AUTO_UPDATE_MARKET_CAP` when enabled in `config.py`

Scanner + LLM research:
```bash
python run_all.py --with-llm
```

Dashboards:
```bash
streamlit run streamlit_dashboard.py
```

## Universe (No Top-N Cap By Default)

Universe is built daily using:
- `MIN_AVG_DAILY_VOLUME` (volume floor)
- optional market-cap ceiling (`ENFORCE_MARKET_CAP` + `MAX_MARKET_CAP_JPY`)
- Nikkei 225 exclusion (`EXCLUDE_NIKKEI_225`)
- `UNIVERSE_TOP_N=None` by default (no rank cap)

All parameters live in `config.py`.

## Early Mode (Default Output)

Early Mode is the main report:
- RSI <= 65
- 10D return < 15%
- scanner subset: oversold/reversal/volatility/coiling/consolidation

Optional legacy (momentum-allowed) output:
- `EARLY_MODE_SHOW_BOTH=True`

## Where Outputs Are Saved

- Cache: `results/precomputed_cache.pkl`
- Daily picks (scanner-only): `results/daily_picks/daily_picks_YYYY-MM-DD.csv|json`
- Burst log (auto + manual additions): `results/burst_audit/bursts_log.csv`
- Burst audit outputs: `results/burst_audit/audit_YYYY-MM-DD.csv|json`, `results/burst_audit/audit_master.csv`
- Burst audit A/B outputs: `results/burst_audit/ab_audit_YYYY-MM-DD.csv|json`, `results/burst_audit/ab_audit_master.csv`
- Shadow picks (A/B): `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.csv|json`
- LLM research: `results/llm_research_*.json|csv`

## Burst Audit Workflow

- Burst rule: close-to-close +10% (`close(D) >= close(prev_trading_day(D)) * 1.10`)
- Comparison: burst date `D` is audited against candidates from `daily_picks_{prev_trading_day(D)}.csv`
- Catch-up mode:
```bash
python burst_audit.py daily --pending --start 2026-01-31
```
- `run_all.py` includes this catch-up workflow by default (toggle with `--no-burst-audit`).
- A/B universe comparison is also available in the same flow:
```bash
python run_all.py --burst-ab-top-n 10 --burst-ab-shadow-min-volume 5000
```
- Phase 3 single-signal shadow variant:
```bash
python run_all.py --burst-ab-variant single_signal_mix --burst-ab-top-n 10 --burst-ab-single-min-count 3 --burst-ab-single-min-score 70
```

## Evaluate "Did These Picks Work?"

Single day:
```bash
python evaluate_signals_forward.py --date 2026-02-02 --top 20 --horizon 10
```

Date range mapping:
```bash
python evaluate_signals_range.py --start 2026-01-27 --end 2026-02-02 --top 20 --horizon 5
```

Notes:
- By default, evaluation requires a full horizon worth of future bars in the cache. If the cache ends too soon, those trades are labeled `horizon_truncated` and excluded from win-rate/avg-return stats.
- Use `--allow-truncated-horizon` if you intentionally want partial (not fully matured) results.

## A/B Decision Scorecard (Universe Phase 2)

Build the rolling decision report from burst-audit A/B outputs:

```bash
python ab_scorecard.py --ab-master results/burst_audit/ab_audit_master.csv --top-n 10 --window-days 20 --min-days 20
```

Outputs:
- `results/burst_audit/ab_scorecard_daily.csv`
- `results/burst_audit/ab_scorecard_latest.json`

## Diagnose "Why Was This Not In The List?"

```bash
python diagnose_missed_signals.py --date 2026-02-03 --symbols 7901,7771,7810,8920,2962,7922,4960,6495,6433
```

## Notes

- Yahoo daily bars can lag. `data_manager.update_recent_data()` clamps requests to Yahoo's latest published daily bar to prevent misleading "possibly delisted / no price data found" spam.
- Recent updates use batched multi-ticker downloads for short date ranges (faster, fewer rate limits).
- If a ticker isn't in `jp_stocks.db`, it can't be signaled. DB expansion is incremental and resumable (`expand_db.py`), and is optionally run inside `precompute.py`.
- Cache validation uses code + config fingerprints (`scanners.py`, `technical_analysis.py`, `data_manager.py`, `config.py`, `precompute.py`). If any hash changes, cache rebuild is triggered.

## Project Docs

- Workflow and operations: `documentation.md`
- Current priorities: `NEXT_STEPS.md`
- Full project handoff: `PROJECT_HANDOFF.md`
- Change history: `CHANGELOG.md`

## Module Ownership

- Library modules (imported by others): `config.py`, `data_manager.py`, `technical_analysis.py`, `scanners.py`, `backtesting.py`, `optimizer.py`, `llm_research.py`
- Main operational CLIs: `run_all.py`, `precompute.py`, `generate_signals.py`, `generate_signals_with_research.py`, `burst_audit.py`
- Analysis/reporting CLIs: `evaluate_signals_forward.py`, `evaluate_signals_range.py`, `ab_scorecard.py`, `diagnose_missed_signals.py`, `plot_signals_charts.py`
- Maintenance CLIs: `expand_db.py`, `fill_market_caps.py`, `clean_bad_yfinance_tickers.py`, `yfinance_field_mapper.py`

## Disclaimer

Research/educational use only. Not financial advice.
