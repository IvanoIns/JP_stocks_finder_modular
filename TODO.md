# JP Stocks Modular - TODO

## Completed (Key Fixes)

1) Backtest correctness: daily bars loaded for `universe + open_positions + pending_entries` (scan only universe).
2) Early Mode default: RSI <= 65 and 10D return < 15% + early scanner subset; optional legacy output toggle.
3) Universe enforcement: market-cap filter enforced; missing caps handled by policy; no top-N cap by default.
4) Cache correctness: cache stores raw signals (score > 0) and applies `MIN_SCANNER_SCORE` at runtime.
5) Entry realism: stop/target computed from next-day open when available; otherwise close is marked as estimate (`*`).
6) Persist outputs:
   - Scanner-only picks saved by `generate_signals.py` to `results/daily_picks/`.
   - LLM results saved by `generate_signals_with_research.py` to `results/llm_research_*.json|csv`.
7) Evaluation tools:
   - `evaluate_signals_forward.py` (single day)
   - `evaluate_signals_range.py` (date range mapping)
8) Yahoo lag guard: clamp updates to Yahoo's latest published daily bar to avoid mass "possibly delisted" spam.
9) Faster DB refresh: short-range updates use batched multi-ticker yfinance downloads; bad-ticker cache only skips symbols with no DB rows.
10) Evaluation correctness: trades with insufficient future bars are labeled `horizon_truncated` and excluded by default.
11) Diagnostics: `diagnose_missed_signals.py --symbols ...` explains universe/early-mode exclusion on a per-ticker basis.
12) Burst audit workflow:
   - `burst_audit.py collect|audit|daily`
   - Backfill/pending catch-up from a start date
   - Outputs in `results/burst_audit/` and integrates with `run_all.py`.

## Next Actions (Paper Trading / Research)

1) Daily run: update DB -> update cache incrementally -> generate signals (+ optional LLM research).
2) Keep `results/burst_audit/bursts_log.csv` complete (manual external bursts not in DB), then rerun pending audit.
3) Expand DB coverage (occasional): `python expand_db.py --max-new 1000 --start 2024-01-01` then run `python precompute.py` (incremental; full rebuild only if fingerprint changed or `--rebuild` is used).
4) YFinance field mapping (research): sample per market-cap decile and label usable `fast_info`/`info` fields.

## A/B Backlog (After Current RSI Test)

1) Universe A/B (implemented):
   - A: baseline picks from `results/daily_picks/`
   - B: shadow picks from `results/daily_picks_ab/` with lower min volume
   - Metric: capture at `BURST_AB_TOP_N` in `results/burst_audit/ab_audit_*.csv`
   - Decision report: `python ab_scorecard.py ...` -> `results/burst_audit/ab_scorecard_latest.json`
2) Single-signal A/B (implemented):
   - A: current ranking behavior
   - B: `single_signal_mix` variant that enforces a minimum number of strong single-scanner names in top-N
3) Scanner-subset A/B:
   - A: current early scanner subset
   - B: controlled subset expansion based on miss diagnostics
4) Keep one-variable-at-a-time rule for every phase.

Last updated: 2026-02-11
