"""
Run the full daily pipeline (including optional DB expansion and market-cap fill).

Edit the DEFAULT_* settings below to change behavior without CLI flags.

Usage:
  python run_all.py
  python run_all.py --with-llm
  python run_all.py --update-days 5 --precompute-args "--top 1500"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


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

    # 4) Optional LLM research
    if args.with_llm:
        llm_args = shlex.split(args.llm_args) if args.llm_args else []
        rc = _run([sys.executable, "generate_signals_with_research.py", *llm_args])
        if rc != 0:
            return rc

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
