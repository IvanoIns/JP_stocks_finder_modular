"""
Batch updater for market caps using yfinance.

Usage:
  python fill_market_caps.py --batch 300 --sleep 0.2 --max-batches 10
"""

from __future__ import annotations

import argparse

import data_manager as dm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=300, help="Max symbols per batch")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests")
    parser.add_argument("--max-batches", type=int, default=10, help="Stop after N batches (0 = unlimited)")
    args = parser.parse_args()

    batch = 0
    total_updated = 0
    while True:
        batch += 1
        updated = dm.update_market_caps_incremental(
            max_symbols=args.batch,
            sleep_seconds=args.sleep,
            progress=True,
        )
        total_updated += updated
        print(f"[Batch {batch}] updated={updated} total_updated={total_updated}")

        if updated == 0:
            print("No more missing market caps.")
            break
        if args.max_batches and batch >= args.max_batches:
            print("Reached max batches.")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
