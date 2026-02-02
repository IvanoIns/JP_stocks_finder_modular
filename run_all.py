"""
Run the full daily pipeline (excluding LLM research unless opted-in).

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-days", type=int, default=5, help="Days to refresh in DB")
    parser.add_argument("--with-llm", action="store_true", help="Run LLM research script at the end")
    parser.add_argument("--precompute-args", type=str, default="", help="Extra args for precompute.py")
    parser.add_argument("--signals-args", type=str, default="", help="Extra args for generate_signals.py")
    parser.add_argument("--llm-args", type=str, default="", help="Extra args for generate_signals_with_research.py")
    args = parser.parse_args()

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
