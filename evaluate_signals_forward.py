"""
Evaluate one day's signals with a short forward "micro backtest".

Goal:
  Given a signal date (T), simulate entry at next trading day open (T+1)
  and exit using stop/target rules (fixed R:R) or a time-based horizon.

This answers: "If I traded the picks from date X, did they actually work?"

Notes:
  - Uses `results/precomputed_cache.pkl` for the signal list and trading calendar.
  - Uses daily bars (OHLC) and checks exits against daily high/low.
  - Daily bars can't tell intra-day order; we use a conservative rule:
      Stop is checked before Target (same as backtesting.BacktestEngine).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
import precompute


@dataclass(frozen=True)
class EvalResult:
    symbol: str
    strategy: str | None
    score: float | None
    signal_date: str
    entry_date: str | None
    entry_open: float | None
    entry_exec: float | None
    stop_price: float | None
    target_price: float | None
    exit_date: str | None
    exit_reason: str | None
    exit_price: float | None
    exit_exec: float | None
    return_pct: float | None
    return_pct_net: float | None
    notes: str | None = None


def _find_default_signal_date(pc) -> str:
    # Pick latest date with signals that is not the last trading date
    for d in reversed(pc.trading_dates[:-1]):
        if pc.signals_by_date.get(d):
            return d
    raise SystemExit("No signals found in cache.")


def _next_trading_date(pc, date: str) -> str | None:
    try:
        i = pc.trading_dates.index(date)
    except ValueError:
        return None
    if i + 1 >= len(pc.trading_dates):
        return None
    return pc.trading_dates[i + 1]


def _trading_dates_slice(pc, start_date: str, trading_days: int) -> list[str]:
    if trading_days <= 0:
        return []
    try:
        i0 = pc.trading_dates.index(start_date)
    except ValueError:
        return []
    i1 = min(i0 + trading_days, len(pc.trading_dates))
    return pc.trading_dates[i0:i1]


def _simulate_fixed_rr(
    pc,
    symbol: str,
    signal_date: str,
    horizon_trading_days: int,
    stop_loss_pct: float,
    rr_ratio: float,
    slippage_pct: float,
    commission_pct: float,
    require_full_horizon: bool = True,
) -> EvalResult:
    entry_date = _next_trading_date(pc, signal_date)
    if entry_date is None:
        return EvalResult(
            symbol=symbol,
            strategy=None,
            score=None,
            signal_date=signal_date,
            entry_date=None,
            entry_open=None,
            entry_exec=None,
            stop_price=None,
            target_price=None,
            exit_date=None,
            exit_reason=None,
            exit_price=None,
            exit_exec=None,
            return_pct=None,
            return_pct_net=None,
            notes="no_next_trading_day_in_cache",
        )

    entry_bar = pc.bars_by_date.get(entry_date, {}).get(symbol)
    if not entry_bar or not entry_bar.get("open"):
        return EvalResult(
            symbol=symbol,
            strategy=None,
            score=None,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_open=None,
            entry_exec=None,
            stop_price=None,
            target_price=None,
            exit_date=None,
            exit_reason=None,
            exit_price=None,
            exit_exec=None,
            return_pct=None,
            return_pct_net=None,
            notes="missing_entry_open",
        )

    entry_open = float(entry_bar["open"])
    entry_exec = entry_open * (1.0 + slippage_pct)

    stop_price = entry_exec * (1.0 - stop_loss_pct)
    target_price = entry_exec * (1.0 + stop_loss_pct * rr_ratio)

    # Iterate horizon days (starting at entry_date)
    dates = _trading_dates_slice(pc, entry_date, horizon_trading_days)
    if not dates:
        return EvalResult(
            symbol=symbol,
            strategy=None,
            score=None,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_open=entry_open,
            entry_exec=entry_exec,
            stop_price=stop_price,
            target_price=target_price,
            exit_date=None,
            exit_reason=None,
            exit_price=None,
            exit_exec=None,
            return_pct=None,
            return_pct_net=None,
            notes="no_horizon_dates",
        )

    if require_full_horizon and len(dates) < horizon_trading_days:
        return EvalResult(
            symbol=symbol,
            strategy=None,
            score=None,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_open=entry_open,
            entry_exec=entry_exec,
            stop_price=stop_price,
            target_price=target_price,
            exit_date=dates[-1],
            exit_reason="horizon_truncated",
            exit_price=None,
            exit_exec=None,
            return_pct=None,
            return_pct_net=None,
            notes=f"need_{horizon_trading_days}_have_{len(dates)}",
        )

    exit_date = None
    exit_reason = None
    exit_price = None

    for d in dates:
        bar = pc.bars_by_date.get(d, {}).get(symbol)
        if not bar:
            # If trading is suspended / missing, we can't reliably simulate intraday exits
            exit_date = d
            exit_reason = "missing_bar"
            exit_price = None
            break

        low = float(bar["low"])
        high = float(bar["high"])

        # Conservative: stop first, then target (same as BacktestEngine)
        if low <= stop_price:
            exit_date = d
            exit_reason = "stop_loss"
            exit_price = stop_price
            break
        if high >= target_price:
            exit_date = d
            exit_reason = "profit_target"
            exit_price = target_price
            break

    if exit_reason is None:
        # No stop/target hit. Exit at close on the last horizon day.
        last_d = dates[-1]
        last_bar = pc.bars_by_date.get(last_d, {}).get(symbol)
        if not last_bar or not last_bar.get("close"):
            return EvalResult(
                symbol=symbol,
                strategy=None,
                score=None,
                signal_date=signal_date,
                entry_date=entry_date,
                entry_open=entry_open,
                entry_exec=entry_exec,
                stop_price=stop_price,
                target_price=target_price,
                exit_date=last_d,
                exit_reason="missing_close",
                exit_price=None,
                exit_exec=None,
                return_pct=None,
                return_pct_net=None,
                notes="missing_horizon_close",
            )
        exit_date = last_d
        exit_reason = "horizon_close"
        exit_price = float(last_bar["close"])

    if exit_price is None or entry_exec <= 0:
        return EvalResult(
            symbol=symbol,
            strategy=None,
            score=None,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_open=entry_open,
            entry_exec=entry_exec,
            stop_price=stop_price,
            target_price=target_price,
            exit_date=exit_date,
            exit_reason=exit_reason,
            exit_price=exit_price,
            exit_exec=None,
            return_pct=None,
            return_pct_net=None,
            notes="no_exit_price",
        )

    exit_exec = float(exit_price) * (1.0 - slippage_pct)
    return_pct = (exit_exec / entry_exec) - 1.0

    # Approximate net return including both-side commissions.
    # (Engine metrics currently undercount entry commission; we prefer clarity here.)
    entry_cost = entry_exec * (1.0 + commission_pct)
    exit_net = exit_exec * (1.0 - commission_pct)
    return_pct_net = (exit_net / entry_cost) - 1.0

    return EvalResult(
        symbol=symbol,
        strategy=None,
        score=None,
        signal_date=signal_date,
        entry_date=entry_date,
        entry_open=entry_open,
        entry_exec=entry_exec,
        stop_price=stop_price,
        target_price=target_price,
        exit_date=exit_date,
        exit_reason=exit_reason,
        exit_price=float(exit_price),
        exit_exec=exit_exec,
        return_pct=return_pct,
        return_pct_net=return_pct_net,
        notes=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one day of signals with a short forward test.")
    parser.add_argument("--date", type=str, default=None, help="Signal date (YYYY-MM-DD). Default: latest evaluatable date.")
    parser.add_argument("--top", type=int, default=20, help="Evaluate top N signals by score")
    parser.add_argument("--horizon", type=int, default=10, help="Max holding period in trading days (starting from entry day)")
    parser.add_argument("--allow-truncated-horizon", action="store_true", help="Allow evaluating trades even if future bars are missing in cache")
    parser.add_argument("--min-score", type=int, default=None, help="Min score filter (default: config.MIN_SCANNER_SCORE)")
    parser.add_argument("--output", type=str, default=None, help="CSV output path (default: results/signal_eval_...)")
    args = parser.parse_args()

    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        raise SystemExit("Cache not found. Build it with: python precompute.py")

    pc = precompute.load_precomputed(cache_path)

    signal_date = args.date or _find_default_signal_date(pc)
    if signal_date not in pc.trading_dates:
        raise SystemExit(f"Date {signal_date} not found in cache trading dates ({pc.start_date}..{pc.end_date}).")

    # Need at least one future trading day to simulate entry
    if _next_trading_date(pc, signal_date) is None:
        raise SystemExit(f"Date {signal_date} is the last trading date in cache. Pick an earlier date.")

    signals = list(pc.signals_by_date.get(signal_date, []))
    if not signals:
        raise SystemExit(f"No signals in cache for {signal_date}.")

    min_score = args.min_score if args.min_score is not None else config.MIN_SCANNER_SCORE
    signals = [s for s in signals if float(s.get("score", 0) or 0) >= min_score]
    signals.sort(key=lambda s: float(s.get("score", 0) or 0), reverse=True)
    signals = signals[: max(args.top, 0)]

    stop_loss_pct = config.STOP_LOSS_PCT
    rr_ratio = config.RISK_REWARD_RATIO
    slippage_pct = config.BACKTEST_SLIPPAGE_PCT
    commission_pct = config.BACKTEST_COMMISSION_PCT

    results: list[EvalResult] = []
    for s in signals:
        sym = s.get("symbol")
        if not sym:
            continue
        r = _simulate_fixed_rr(
            pc=pc,
            symbol=sym,
            signal_date=signal_date,
            horizon_trading_days=args.horizon,
            stop_loss_pct=stop_loss_pct,
            rr_ratio=rr_ratio,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            require_full_horizon=not args.allow_truncated_horizon,
        )
        results.append(EvalResult(**{**r.__dict__, "strategy": s.get("strategy"), "score": s.get("score")}))

    df = pd.DataFrame([r.__dict__ for r in results])
    ok = df["return_pct_net"].notna()

    print("=" * 70)
    print("SIGNAL FORWARD EVALUATION")
    print("=" * 70)
    print(f"Signal date: {signal_date} | Top: {len(df)} | Horizon: {args.horizon} trading days")
    print(f"Min score: {min_score} | Stop: {stop_loss_pct:.1%} | R:R: {rr_ratio:.2f}")
    print(f"Costs: slippage {slippage_pct:.2%} | commission {commission_pct:.2%} (both sides in net)")

    if ok.any():
        eval_df = df[ok].copy()
        win = eval_df["return_pct_net"] > 0
        hit_target = eval_df["exit_reason"] == "profit_target"
        hit_stop = eval_df["exit_reason"] == "stop_loss"
        print(f"\nEvaluated trades: {len(eval_df)}/{len(df)} (others missing data)")
        print(f"Win rate (net): {win.mean():.1%}")
        print(f"Hit target: {hit_target.mean():.1%} | Hit stop: {hit_stop.mean():.1%}")
        print(f"Avg return (net): {eval_df['return_pct_net'].mean():.2%} | Median: {eval_df['return_pct_net'].median():.2%}")
    else:
        print("\nNo trades could be evaluated (missing entry/exit bars).")

    reason_counts = df["exit_reason"].fillna("none").value_counts()
    print("\nExit reasons (all selected trades):")
    for k, v in reason_counts.items():
        print(f"  {k}: {int(v)}")
    if "horizon_truncated" in reason_counts:
        print(f"\nNOTE: horizon_truncated means the cache does not have enough future bars yet (cache end: {pc.end_date}).")

    # Save CSV
    out_path = Path(args.output) if args.output else Path("results") / f"signal_eval_{signal_date}_top{len(df)}_h{args.horizon}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
