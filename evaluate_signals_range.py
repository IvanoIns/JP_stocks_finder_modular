"""
Evaluate signals over a date range with a short forward "micro backtest".

This is the batch version of `evaluate_signals_forward.py`.

Typical use:
  python evaluate_signals_range.py --start 2026-01-27 --end 2026-02-03 --top 20 --horizon 10

Outputs:
  - results/signal_eval_range_<start>_to_<end>_topN_hH.csv          (daily summary)
  - results/signal_eval_range_<start>_to_<end>_topN_hH_trades.csv   (per-trade details)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
import precompute
from evaluate_signals_forward import _next_trading_date, _simulate_fixed_rr


def _select_signal_dates(pc, start: str, end: str) -> list[str]:
    dates = [d for d in pc.trading_dates if start <= d <= end]
    # Need an entry day (next trading day) to evaluate
    return [d for d in dates if _next_trading_date(pc, d) is not None]


def _lot_cost(entry: float | None, lot_size: int) -> float | None:
    if entry is None:
        return None
    return float(entry) * float(lot_size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate signals over a date range.")
    parser.add_argument("--start", type=str, required=True, help="Start signal date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End signal date (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=20, help="Top N signals per day (sorted by score)")
    parser.add_argument("--horizon", type=int, default=10, help="Max holding period in trading days (from entry day)")
    parser.add_argument("--allow-truncated-horizon", action="store_true", help="Allow evaluating trades even if future bars are missing in cache")
    parser.add_argument("--min-score", type=int, default=None, help="Min score filter (default: config.MIN_SCANNER_SCORE)")
    parser.add_argument("--budget-jpy", type=float, default=None, help="Budget per trade for lot affordability (default: config.MAX_JPY_PER_TRADE)")
    parser.add_argument("--lot-size", type=int, default=None, help="Lot size (default: config.LOT_SIZE)")
    parser.add_argument("--only-affordable", action="store_true", help="Evaluate only lot-affordable picks (lot_cost <= budget)")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output prefix (default: results/signal_eval_range_...)")
    args = parser.parse_args()

    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        raise SystemExit("Cache not found. Build it with: python precompute.py")

    pc = precompute.load_precomputed(cache_path)

    signal_dates = _select_signal_dates(pc, args.start, args.end)
    if not signal_dates:
        raise SystemExit(
            f"No evaluatable trading dates in range {args.start}..{args.end} "
            f"(cache: {pc.start_date}..{pc.end_date})."
        )

    min_score = args.min_score if args.min_score is not None else config.MIN_SCANNER_SCORE
    stop_loss_pct = config.STOP_LOSS_PCT
    rr_ratio = config.RISK_REWARD_RATIO
    slippage_pct = config.BACKTEST_SLIPPAGE_PCT
    commission_pct = config.BACKTEST_COMMISSION_PCT

    budget_jpy = args.budget_jpy if args.budget_jpy is not None else float(getattr(config, "MAX_JPY_PER_TRADE", 0))
    lot_size = args.lot_size if args.lot_size is not None else int(getattr(config, "LOT_SIZE", 100))

    trades: list[dict] = []
    for d in signal_dates:
        day_signals = list(pc.signals_by_date.get(d, []))
        if not day_signals:
            continue

        day_signals = [s for s in day_signals if float(s.get("score", 0) or 0) >= min_score]
        day_signals.sort(key=lambda s: float(s.get("score", 0) or 0), reverse=True)

        if args.top and args.top > 0:
            day_signals = day_signals[: args.top]

        for s in day_signals:
            sym = s.get("symbol")
            if not sym:
                continue

            r = _simulate_fixed_rr(
                pc=pc,
                symbol=sym,
                signal_date=d,
                horizon_trading_days=args.horizon,
                stop_loss_pct=stop_loss_pct,
                rr_ratio=rr_ratio,
                slippage_pct=slippage_pct,
                commission_pct=commission_pct,
                require_full_horizon=not args.allow_truncated_horizon,
            )

            # Budget bucket uses raw entry_open (no slippage) for lot sizing.
            lot_cost = _lot_cost(r.entry_open, lot_size)
            budget_ok = (lot_cost is not None) and (lot_cost <= budget_jpy)
            if args.only_affordable and not budget_ok:
                continue

            trades.append(
                {
                    **r.__dict__,
                    "strategy": s.get("strategy"),
                    "score": s.get("score"),
                    "lot_size": lot_size,
                    "budget_jpy": budget_jpy,
                    "lot_cost": lot_cost,
                    "budget_ok": budget_ok,
                }
            )

    if not trades:
        raise SystemExit("No trades selected after filtering.")

    trades_df = pd.DataFrame(trades)
    ok = trades_df["return_pct_net"].notna()

    # Daily summary
    def _summarize(df: pd.DataFrame) -> pd.Series:
        ok_df = df[df["return_pct_net"].notna()].copy()
        out = {
            "trades_selected": len(df),
            "trades_evaluated": len(ok_df),
            "win_rate_net": None,
            "hit_target_rate": None,
            "hit_stop_rate": None,
            "avg_return_net": None,
            "median_return_net": None,
        }
        if len(ok_df) > 0:
            out["win_rate_net"] = float((ok_df["return_pct_net"] > 0).mean())
            out["hit_target_rate"] = float((ok_df["exit_reason"] == "profit_target").mean())
            out["hit_stop_rate"] = float((ok_df["exit_reason"] == "stop_loss").mean())
            out["avg_return_net"] = float(ok_df["return_pct_net"].mean())
            out["median_return_net"] = float(ok_df["return_pct_net"].median())
        return pd.Series(out)

    daily_all = trades_df.groupby("signal_date", as_index=True).apply(_summarize, include_groups=False)
    daily_aff = (
        trades_df[trades_df["budget_ok"]]
        .groupby("signal_date", as_index=True)
        .apply(_summarize, include_groups=False)
    )

    daily = daily_all.add_prefix("all_").join(daily_aff.add_prefix("affordable_"), how="left").reset_index()

    # Print overall summary
    print("=" * 70)
    print("SIGNAL RANGE FORWARD EVALUATION")
    print("=" * 70)
    print(f"Signal dates: {signal_dates[0]} .. {signal_dates[-1]} ({len(signal_dates)} days)")
    print(f"Top/day: {args.top} | Horizon: {args.horizon} trading days | Min score: {min_score}")
    print(f"Costs: slippage {slippage_pct:.2%} | commission {commission_pct:.2%} (both sides in net)")
    print(f"Budget: JPY {budget_jpy:,.0f} | Lot size: {lot_size} | Only affordable: {args.only_affordable}")

    if ok.any():
        eval_df = trades_df[ok].copy()
        print(f"\nTrades evaluated: {len(eval_df)}/{len(trades_df)}")
        print(f"Win rate (net): {(eval_df['return_pct_net'] > 0).mean():.1%}")
        print(f"Avg return (net): {eval_df['return_pct_net'].mean():.2%} | Median: {eval_df['return_pct_net'].median():.2%}")
    else:
        print("\nNo trades could be evaluated (missing entry/exit bars).")

    reason_counts = trades_df["exit_reason"].fillna("none").value_counts()
    print("\nExit reasons (all selected trades):")
    for k, v in reason_counts.items():
        print(f"  {k}: {int(v)}")
    if "horizon_truncated" in reason_counts:
        print(f"\nNOTE: horizon_truncated means the cache does not have enough future bars yet (cache end: {pc.end_date}).")

    # Save outputs
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else Path("results") / f"signal_eval_range_{args.start}_to_{args.end}_top{args.top}_h{args.horizon}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_suffix(".csv")
    trades_path = prefix.parent / f"{prefix.name}_trades.csv"

    daily.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print(f"\nSaved daily summary: {summary_path}")
    print(f"Saved trade details: {trades_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
