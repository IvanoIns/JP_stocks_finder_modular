# Changelog

## 2026-02-11

### Cache / Performance
- `precompute.py` now updates cache incrementally by default instead of rebuilding all dates every run.
- Added cache fingerprint validation with file hashes for:
  - `scanners.py`
  - `technical_analysis.py`
  - `data_manager.py`
  - `config.py`
  - `precompute.py`
- Cache auto-rebuilds when fingerprinted logic/config changes; force rebuild with `python precompute.py --rebuild`.
- Added fingerprint diff logging to show why rebuild is required.
- Added atomic cache save (`.tmp` then replace) to reduce risk of partial/corrupt cache files.
- Optimized per-date signal computation loop in `precompute.py` to avoid repeated string-based date masks.

### Data Integrity
- `data_manager.get_daily_bars()` now normalizes datetime index, sorts, and deduplicates dates.
- `data_manager.get_daily_bars_batch()` now applies the same normalization/dedup per symbol.

### Docs
- Updated `README.md`, `documentation.md`, and `TODO.md` to describe incremental cache behavior and rebuild triggers.

## 2026-02-08

### Burst Audit A/B (Universe Phase 2)
- Added shadow-universe A/B comparison to `burst_audit.py` (`--ab`).
- Shadow universe keeps all baseline rules and lowers only volume floor via `--ab-shadow-min-volume` (default from config).
- Added top-N capture comparison via `--ab-top-n` (default from config).
- New outputs:
  - `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.csv|json`
  - `results/burst_audit/ab_audit_YYYY-MM-DD.csv`
  - `results/burst_audit/ab_audit_summary_YYYY-MM-DD.json`
  - `results/burst_audit/ab_audit_master.csv`
- Added Phase-2 config defaults in `config.py`:
  - `BURST_AB_ENABLED`
  - `BURST_AB_TOP_N`
  - `BURST_AB_SHADOW_MIN_VOLUME`
- Added `ab_scorecard.py` for rolling A/B decision reporting with configurable gates and outputs:
  - `results/burst_audit/ab_scorecard_daily.csv`
  - `results/burst_audit/ab_scorecard_latest.json`

### Burst Audit A/B (Phase 3: Single-Signal Mix)
- Added shadow variant selector:
  - `universe_min_volume` (Phase 2)
  - `single_signal_mix` (Phase 3)
- Added Phase 3 controls:
  - `BURST_AB_VARIANT`
  - `BURST_AB_SINGLE_MIN_COUNT`
  - `BURST_AB_SINGLE_MIN_SCORE`
- Added CLI pass-through in `run_all.py` and `burst_audit.py`:
  - `--burst-ab-variant / --ab-variant`
  - `--burst-ab-single-min-count / --ab-single-min-count`
  - `--burst-ab-single-min-score / --ab-single-min-score`
- Shadow pick metadata now records variant and parameters in `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.json`.

### Burst Audit
- Added `burst_audit.py` with `collect`, `audit`, and `daily` commands.
- Added backfill/pending behavior to catch up missed days across a date range.
- Burst collection now scans all DB symbols for close-to-close >= +10% and stores results in `results/burst_audit/bursts_log.csv`.
- Audit compares burst day `D` vs candidate picks from `daily_picks_{prev_trading_day(D)}` and stores:
  - `results/burst_audit/audit_YYYY-MM-DD.csv`
  - `results/burst_audit/audit_summary_YYYY-MM-DD.json`
  - `results/burst_audit/audit_master.csv`
- Added idempotent status tracking files:
  - `results/burst_audit/collect_status.csv`
  - `results/burst_audit/audit_status.csv`

### Workflow
- `run_all.py` now supports burst-audit integration by default (`--burst-audit`, `--burst-start`, `--burst-threshold`, `--burst-force`).
- `run_all.py` now supports A/B pass-through flags:
  - `--burst-ab` / `--no-burst-ab`
  - `--burst-ab-top-n`
  - `--burst-ab-shadow-min-volume`
- Added burst-audit defaults in `config.py`:
  - `BURST_AUDIT_ENABLED`
  - `BURST_AUDIT_START_DATE`
  - `BURST_AUDIT_THRESHOLD_CLOSE_PCT`

## 2026-02-04

### Data / Pipeline
- Added a Yahoo "latest daily bar" guard to avoid requesting dates Yahoo hasn't published yet (prevents mass "possibly delisted / no price data found" spam).
- Recent DB refresh (`update_recent_data`) now uses batched multi-ticker yfinance downloads for short date ranges (much faster, fewer rate limits).
- Improved bad-ticker handling:
  - yfinance logging is silenced (to avoid misleading spam).
  - `cache/bad_yfinance_tickers.txt` is treated as "skip only if the symbol has no DB rows" by default.

### Signals / Outputs
- `generate_signals.py` now persists scanner-only picks to `results/daily_picks/daily_picks_YYYY-MM-DD.csv` and `.json`.

### Evaluation
- Added `evaluate_signals_range.py` to map signal performance over a date range and export summary + trade-detail CSVs.
- Evaluation now labels trades as `horizon_truncated` (and excludes them by default) when the cache doesn't have enough future bars to fully evaluate the requested horizon.

### Docs
- Refreshed `documentation.md` and `README.md` to reflect the current pipeline and outputs.
