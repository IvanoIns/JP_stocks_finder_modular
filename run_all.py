"""
Run the full daily pipeline (including optional DB expansion and market-cap fill).

Edit the DEFAULT_* settings below to change behavior without CLI flags.

Usage:
  python run_all.py
  python run_all.py --with-llm
  python run_all.py --no-burst-ab
  python run_all.py --burst-ab-top-n 10 --burst-ab-shadow-min-volume 5000
  python run_all.py --burst-ab-variant single_signal_mix --burst-ab-single-min-count 3 --burst-ab-single-min-score 70
  python run_all.py --update-days 5 --precompute-args "--top 1500"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

import config


def _run(cmd: list[str]) -> int:
    print(f"\n>> {' '.join(cmd)}")
    return subprocess.call(cmd)


# =========================
# Editable defaults (toggles)
# =========================
RUN_EXPAND_DB = False
RUN_FILL_MARKET_CAPS = False
UPDATE_DAYS = 5
EXPAND_MAX_NEW = 1000
EXPAND_START = "2024-01-01"
MC_BATCH = 300
MC_SLEEP = 0.2
MC_MAX_BATCHES = 5
RUN_BURST_AUDIT = getattr(config, "BURST_AUDIT_ENABLED", True)
BURST_AUDIT_START = getattr(config, "BURST_AUDIT_START_DATE", "2026-01-31")
BURST_AUDIT_THRESHOLD = getattr(config, "BURST_AUDIT_THRESHOLD_CLOSE_PCT", 0.10)
RUN_BURST_AB = getattr(config, "BURST_AB_ENABLED", True)
BURST_AB_TOP_N = getattr(config, "BURST_AB_TOP_N", 10)
BURST_AB_SHADOW_MIN_VOLUME = getattr(config, "BURST_AB_SHADOW_MIN_VOLUME", 5_000)
BURST_AB_VARIANT = getattr(config, "BURST_AB_VARIANT", "single_signal_mix")
BURST_AB_SINGLE_MIN_COUNT = getattr(config, "BURST_AB_SINGLE_MIN_COUNT", 3)
BURST_AB_SINGLE_MIN_SCORE = getattr(config, "BURST_AB_SINGLE_MIN_SCORE", 70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-days", type=int, default=UPDATE_DAYS, help="Days to refresh in DB")
    parser.add_argument("--with-llm", action="store_true", help="Run LLM research script at the end")
    parser.add_argument("--precompute-args", type=str, default="", help="Extra args for precompute.py")
    parser.add_argument("--signals-args", type=str, default="", help="Extra args for generate_signals.py")
    parser.add_argument("--llm-args", type=str, default="", help="Extra args for generate_signals_with_research.py")
    parser.add_argument("--expand-db", action="store_true", default=RUN_EXPAND_DB, help="Run expand_db.py")
    parser.add_argument("--no-expand-db", action="store_false", dest="expand_db")
    parser.add_argument("--fill-market-caps", action="store_true", default=RUN_FILL_MARKET_CAPS, help="Run fill_market_caps.py")
    parser.add_argument("--no-fill-market-caps", action="store_false", dest="fill_market_caps")
    parser.add_argument("--expand-max-new", type=int, default=EXPAND_MAX_NEW, help="Max new symbols for expand_db.py")
    parser.add_argument("--expand-start", type=str, default=EXPAND_START, help="Start date for expand_db.py")
    parser.add_argument("--mc-batch", type=int, default=MC_BATCH, help="Batch size for fill_market_caps.py")
    parser.add_argument("--mc-sleep", type=float, default=MC_SLEEP, help="Sleep seconds for fill_market_caps.py")
    parser.add_argument("--mc-max-batches", type=int, default=MC_MAX_BATCHES, help="Max batches for fill_market_caps.py")
    parser.add_argument("--burst-audit", action="store_true", default=RUN_BURST_AUDIT, help="Run burst_audit.py daily workflow")
    parser.add_argument("--no-burst-audit", action="store_false", dest="burst_audit")
    parser.add_argument("--burst-start", type=str, default=BURST_AUDIT_START, help="Burst audit backfill start date (YYYY-MM-DD)")
    parser.add_argument("--burst-threshold", type=float, default=BURST_AUDIT_THRESHOLD, help="Burst threshold for collect (default 0.10 = +10%% close)")
    parser.add_argument("--burst-force", action="store_true", help="Force burst audit reprocessing even if already done")
    parser.add_argument("--burst-ab", action="store_true", default=RUN_BURST_AB, help="Run A/B universe comparison during burst audit")
    parser.add_argument("--no-burst-ab", action="store_false", dest="burst_ab")
    parser.add_argument("--burst-ab-top-n", type=int, default=BURST_AB_TOP_N, help="Top-N picks used for burst A/B capture comparison")
    parser.add_argument("--burst-ab-shadow-min-volume", type=int, default=BURST_AB_SHADOW_MIN_VOLUME, help="Shadow universe min volume for A/B")
    parser.add_argument("--burst-ab-variant", type=str, default=BURST_AB_VARIANT, choices=["universe_min_volume", "single_signal_mix"], help="A/B shadow variant type")
    parser.add_argument("--burst-ab-single-min-count", type=int, default=BURST_AB_SINGLE_MIN_COUNT, help="For single_signal_mix: minimum single-scanner names in top-N")
    parser.add_argument("--burst-ab-single-min-score", type=float, default=BURST_AB_SINGLE_MIN_SCORE, help="For single_signal_mix: minimum score for promoted single-scanner names")
    args = parser.parse_args()

    # 0) Optional: expand DB coverage
    if args.expand_db:
        rc = _run([
            sys.executable,
            "expand_db.py",
            "--max-new",
            str(args.expand_max_new),
            "--start",
            args.expand_start,
        ])
        if rc != 0:
            return rc

    # 0b) Optional: fill market caps
    if args.fill_market_caps:
        rc = _run([
            sys.executable,
            "fill_market_caps.py",
            "--batch",
            str(args.mc_batch),
            "--sleep",
            str(args.mc_sleep),
            "--max-batches",
            str(args.mc_max_batches),
        ])
        if rc != 0:
            return rc

    # 1) Update DB (recent data)
    rc = _run([sys.executable, "-c", f"import data_manager as dm; dm.update_recent_data(days={args.update_days})"])
    if rc != 0:
        return rc

    # 2) Precompute cache
    pre_args = shlex.split(args.precompute_args) if args.precompute_args else []
    rc = _run([sys.executable, "precompute.py", *pre_args])
    if rc != 0:
        return rc

    # 3) Generate signals (scanner only)
    sig_args = shlex.split(args.signals_args) if args.signals_args else []
    rc = _run([sys.executable, "generate_signals.py", *sig_args])
    if rc != 0:
        return rc

    # 4) Burst audit workflow (collect + audit, pending catch-up)
    if args.burst_audit:
        burst_cmd = [
            sys.executable,
            "burst_audit.py",
            "daily",
            "--pending",
            "--start",
            args.burst_start,
            "--threshold",
            str(args.burst_threshold),
        ]
        if args.burst_ab:
            burst_cmd.extend(
                [
                    "--ab",
                    "--ab-top-n",
                    str(args.burst_ab_top_n),
                    "--ab-shadow-min-volume",
                    str(args.burst_ab_shadow_min_volume),
                    "--ab-variant",
                    str(args.burst_ab_variant),
                    "--ab-single-min-count",
                    str(args.burst_ab_single_min_count),
                    "--ab-single-min-score",
                    str(args.burst_ab_single_min_score),
                ]
            )
        if args.burst_force:
            burst_cmd.append("--force")
        rc = _run(burst_cmd)
        if rc != 0:
            return rc

    # 5) Optional LLM research
    if args.with_llm:
        llm_args = shlex.split(args.llm_args) if args.llm_args else []
        rc = _run([sys.executable, "generate_signals_with_research.py", *llm_args])
        if rc != 0:
            return rc

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
