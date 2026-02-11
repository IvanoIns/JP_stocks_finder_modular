# Next Steps

## Current Baseline

- Pipeline is stable and automated with `run_all.py`.
- Daily picks are stored in `results/daily_picks/daily_picks_YYYY-MM-DD.csv|json`.
- Burst audit is active and persistent in `results/burst_audit/`.
- Current live objective is capture of next-day close-to-close bursts (`>= +10%`).

---

## Immediate Priorities (No Strategy Retune Yet)

1. Keep collecting daily burst-audit data for at least 15-20 trading days.
2. Keep `results/burst_audit/bursts_log.csv` complete (add external bursts not found in DB).
3. Monitor these metrics daily:
   - capture rate
   - `capture@top20`
   - miss-reason mix (`early_filter_fail`, `excluded_by_early_scanner_subset`, `no_signal`, `not_in_db`, `not_in_universe`)
4. Do not change production thresholds before sample size is adequate.

---

## Daily Operating Procedure

1. Run full pipeline:
   - `python run_all.py`
2. Review:
   - `results/burst_audit/audit_summary_YYYY-MM-DD.json`
   - `results/burst_audit/audit_YYYY-MM-DD.csv`
   - `results/burst_audit/ab_audit_summary_YYYY-MM-DD.json`
   - `results/burst_audit/ab_scorecard_latest.json`
3. Append missing external bursts to:
   - `results/burst_audit/bursts_log.csv`
4. Re-run pending audit:
   - `python burst_audit.py audit --pending --start 2026-01-31`
5. Refresh A/B decision report:
   - `python ab_scorecard.py --ab-master results/burst_audit/ab_audit_master.csv --top-n 10 --window-days 20 --min-days 20`

---

## Decision Gate For Any Retune

Retune only when all are true:

1. At least 15-20 audited trading days.
2. Miss reasons are stable (not random noise by day).
3. Candidate forward returns do not degrade after retune.

---

## Candidate Retunes (After Data Window)

1. Early-mode gates:
   - `EARLY_MODE_RSI_MAX`
   - `EARLY_MODE_10D_RETURN_MAX`
2. Early scanner subset:
   - expand/restrict `EARLY_MODE_SCANNERS`
3. Ranking quality:
   - prioritize by expected burst capture, not raw scanner score alone
4. Execution cut:
   - evaluate impact of top-N constraint on capture and PnL

---

## A/B Roadmap (After RSI 65 vs 70)

Run one change at a time, keep all other settings fixed, and evaluate with `top 10` as primary cut.

1. Phase 1 (current): `EARLY_MODE_RSI_MAX=65` vs `70`
2. Phase 2 (implemented): baseline universe vs shadow universe with lower min volume (`BURST_AB_SHADOW_MIN_VOLUME`) to reduce `not_in_universe`
3. Phase 3 (implemented): baseline ranking vs shadow variant that enforces minimum single-scanner names in top-N
4. Phase 4 (scanner subset): baseline early scanner list vs controlled expansion using scanners that dominate missed bursts

Decision gate per phase:
- Minimum 20 audited trading days
- Improvement in `capture@10`
- No material degradation in precision / forward returns

Phase 2 outputs now available daily:
- `results/daily_picks_ab/daily_picks_shadow_YYYY-MM-DD.csv|json`
- `results/burst_audit/ab_audit_YYYY-MM-DD.csv`
- `results/burst_audit/ab_audit_summary_YYYY-MM-DD.json`
- `results/burst_audit/ab_audit_master.csv`

---

## Operational Risks To Keep Monitoring

1. `not_in_db` misses: DB expansion lag vs market reality.
2. `not_in_universe` misses: universe filters excluding future bursts.
3. High captured ranks (e.g. captures mostly outside top-20): ranking mismatch.
4. Cache drift: run `python precompute.py` after config/universe changes (it will incrementally update or auto-rebuild if fingerprint changed).

---

Last updated: 2026-02-11
