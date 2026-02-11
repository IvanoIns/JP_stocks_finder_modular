# JP Stocks Modular - Project Handoff

Date: 2026-02-11

---

## 1) Project Objective

A JP stocks scanning system for manual/paper trading.

Core layers:
1. Quant scanners produce candidate picks from cached daily data.
2. Optional LLM research adjusts ranking (soft penalties/bonuses).
3. Burst-audit validates whether real +10% close-to-close movers were captured.

This is not auto-execution. No broker API is integrated.

---

## 2) Current Production Workflow

Primary command:

```bash
python run_all.py
```

What it runs:
1. Refresh recent DB bars (`data_manager.update_recent_data(days=5)`).
2. Update cache (`precompute.py`, incremental by default; full rebuild only on fingerprint mismatch or `--rebuild`).
3. Generate daily scanner picks (`generate_signals.py`).
4. Run burst audit catch-up (`burst_audit.py daily --pending --start BURST_AUDIT_START_DATE`).
   - Includes optional A/B shadow comparison (top-N capture).
   - Supported variants: `universe_min_volume` (Phase 2), `single_signal_mix` (Phase 3).
5. Optional: LLM research if `--with-llm` is used.

---

## 3) Core Modules

- `config.py`: source of truth for parameters.
- `data_manager.py`: DB I/O, universe construction, recent updates.
- `precompute.py`: incrementally updates `results/precomputed_cache.pkl` (or rebuilds when required).
- `generate_signals.py`: scanner output and daily picks export.
- `generate_signals_with_research.py`: scanner + LLM ranking.
- `burst_audit.py`: burst collection and miss/capture diagnostics.
- `run_all.py`: orchestration wrapper.

---

## 4) Current Strategy Settings

From `config.py`:

- `MIN_SCANNER_SCORE = 30`
- `STOP_LOSS_PCT = 0.06`
- `RISK_REWARD_RATIO = 2.0`
- `EARLY_MODE_ENABLED = True`
- `EARLY_MODE_RSI_MAX = 65`
- `EARLY_MODE_10D_RETURN_MAX = 0.15`
- `EARLY_MODE_SCANNERS = [oversold_bounce, reversal_rocket, volatility_explosion, coiling_pattern, consolidation_breakout]`

Universe defaults:
- `UNIVERSE_TOP_N = None` (no rank cap)
- liquidity floor active (`MIN_AVG_DAILY_VOLUME`)
- market-cap enforcement optional via config

---

## 5) Burst-Audit Logic

Burst definition:
- `close(D) >= close(prev_trading_day(D)) * 1.10`

Audit alignment:
- Bursts on day `D` are checked against candidate picks from day `D-1`:
  - `results/daily_picks/daily_picks_{D-1}.csv`

Status and idempotency:
- `results/burst_audit/collect_status.csv`
- `results/burst_audit/audit_status.csv`

Main outputs:
- `results/burst_audit/bursts_log.csv`
- `results/burst_audit/audit_YYYY-MM-DD.csv`
- `results/burst_audit/audit_summary_YYYY-MM-DD.json`
- `results/burst_audit/audit_master.csv`
- `results/burst_audit/ab_audit_YYYY-MM-DD.csv`
- `results/burst_audit/ab_audit_summary_YYYY-MM-DD.json`
- `results/burst_audit/ab_audit_master.csv`
- `results/burst_audit/ab_scorecard_daily.csv`
- `results/burst_audit/ab_scorecard_latest.json`
- `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.csv|json`

---

## 6) Daily Outputs To Review

- Scanner picks:
  - `results/daily_picks/daily_picks_YYYY-MM-DD.csv`
  - `results/daily_picks/daily_picks_YYYY-MM-DD.json`
- Burst diagnostics:
  - `results/burst_audit/audit_summary_YYYY-MM-DD.json`
  - `results/burst_audit/audit_YYYY-MM-DD.csv`
  - `results/burst_audit/ab_audit_summary_YYYY-MM-DD.json`
  - `results/burst_audit/ab_audit_YYYY-MM-DD.csv`

Recommended operational check:
1. Run `python run_all.py`.
2. If needed, append external missed bursts to `bursts_log.csv`.
3. Re-run `python burst_audit.py audit --pending --start 2026-01-31`.
4. Refresh decision gate: `python ab_scorecard.py --ab-master results/burst_audit/ab_audit_master.csv --top-n 10 --window-days 20 --min-days 20`.

---

## 7) Known Risks / Current Findings

From recent audited window:
- Main miss reason is `early_filter_fail` (RSI/10D-return gates).
- Many captured bursts appear at low ranking positions (outside top-20).

Implication:
- The implementation is functioning, but current gating/ranking is misaligned with the burst-capture objective.

---

## 8) Recommended Next Work

1. Continue collecting burst-audit data for 15-20 trading days before retuning.
2. Track stability of miss-reason distribution over time.
3. Only then test controlled parameter changes (A/B style), starting from:
   - Phase 2 already implemented: universe shadow (`MIN_AVG_DAILY_VOLUME` only), compare `capture@topN`
   - early gates (`RSI`, `10D return`)
   - scanner subset
   - ranking logic

---

End of handoff.
