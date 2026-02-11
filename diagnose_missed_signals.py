"""
Diagnose "missed" signals for a given date.

This answers questions like:
- Were there legacy (non-early) signals that got excluded by Early Mode?
- Were there signals below MIN_SCANNER_SCORE?
- Did the universe filter exclude most candidates?

Outputs a CSV in results/ with per-symbol details.

Example:
  python diagnose_missed_signals.py --date 2026-02-02 --top 1500 --lookback-days 400
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import config
import data_manager as dm
import scanners as sc
import technical_analysis as ta


@dataclass
class Row:
    symbol: str
    in_universe: bool
    rsi: float | None
    ret_10d: float | None
    early_filter_pass: bool
    early_strategy: str | None
    early_score: float | None
    legacy_strategy: str | None
    legacy_score: float | None
    excluded_reason: str | None


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _compute_early_filter(df: pd.DataFrame) -> tuple[float | None, float | None, bool]:
    if df is None or df.empty:
        return None, None, False
    if "rsi" not in df.columns:
        return None, None, False
    if len(df) < 11:
        rsi = float(df["rsi"].iloc[-1])
        return rsi, None, False
    rsi = float(df["rsi"].iloc[-1])
    ret_10d = float((df["close"].iloc[-1] / df["close"].iloc[-11]) - 1)
    passes = (rsi <= config.EARLY_MODE_RSI_MAX) and (ret_10d < config.EARLY_MODE_10D_RETURN_MAX)
    return rsi, ret_10d, bool(passes)


def _top_signal(signals: list[dict]) -> dict | None:
    if not signals:
        return None
    return max(signals, key=lambda s: s.get("score", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose missed signals for a date")
    parser.add_argument("--date", required=True, help="Trading date to analyze (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=config.UNIVERSE_TOP_N, help="Universe top N by notional")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma/space-separated tickers to analyze (optional, e.g. 7901,7771,4960 or 7901.T 4960.T)",
    )
    parser.add_argument("--lookback-days", type=int, default=400, help="Calendar days of lookback to load")
    parser.add_argument("--chunk-size", type=int, default=800, help="DB batch chunk size (<= ~900)")
    parser.add_argument("--output", default="", help="Output CSV path (default: results/missed_signals_<date>.csv)")
    args = parser.parse_args()

    def _normalize_symbol(s: str) -> str:
        s = (s or "").strip().upper()
        if not s:
            return ""
        if "." in s:
            return s
        return f"{s}.T"

    user_symbols: list[str] = []
    if args.symbols:
        raw = args.symbols.replace(",", " ").split()
        user_symbols = [_normalize_symbol(s) for s in raw if _normalize_symbol(s)]

    date = args.date
    lookback_start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    universe = dm.build_liquid_universe(date, top_n=args.top)
    universe_set = set(universe)

    print(f"Date: {date}")
    print(f"Universe: {len(universe)} symbols (top {args.top})")
    print(f"Loading bars: {lookback_start} -> {date}")

    scanner_config = config.get_scanner_config()
    early_scanners = set(config.EARLY_MODE_SCANNERS)
    min_score = config.MIN_SCANNER_SCORE

    rows: list[Row] = []

    symbols_to_analyze = user_symbols if user_symbols else universe
    if user_symbols:
        print(f"Symbols requested: {len(user_symbols)}")

    # Fetch data in chunks to avoid SQLite variable limits
    for chunk in _chunked(symbols_to_analyze, args.chunk_size):
        batch = dm.get_daily_bars_batch(chunk, start_date=lookback_start, end_date=date)
        for symbol in chunk:
            df = batch.get(symbol)
            if df is None or df.empty or len(df) < 60:
                rows.append(
                    Row(
                        symbol=symbol,
                        in_universe=symbol in universe_set,
                        rsi=None,
                        ret_10d=None,
                        early_filter_pass=False,
                        early_strategy=None,
                        early_score=None,
                        legacy_strategy=None,
                        legacy_score=None,
                        excluded_reason="no_data",
                    )
                )
                continue

            in_universe = symbol in universe_set
            df_ind = ta.calculate_all_indicators(df, scanner_config)
            rsi, ret_10d, early_pass = _compute_early_filter(df_ind)

            legacy_signals = sc.get_all_signals(
                symbol,
                df_ind,
                jpx_data={},
                scanner_config=scanner_config,
                min_score=0,
                early_mode=False,
            )
            early_signals = sc.get_all_signals(
                symbol,
                df_ind,
                jpx_data={},
                scanner_config=scanner_config,
                min_score=0,
                early_mode=True,
            )

            legacy_top = _top_signal(legacy_signals)
            early_top = _top_signal(early_signals)

            legacy_score = float(legacy_top["score"]) if legacy_top else None
            legacy_strategy = legacy_top.get("strategy") if legacy_top else None
            early_score = float(early_top["score"]) if early_top else None
            early_strategy = early_top.get("strategy") if early_top else None

            excluded_reason = None
            if not in_universe:
                excluded_reason = "not_in_universe"
            else:
                early_ok = (early_score is not None) and (early_score >= min_score)
                legacy_ok = (legacy_score is not None) and (legacy_score >= min_score)

                if early_ok:
                    excluded_reason = None
                elif legacy_ok:
                    # Legacy qualifies, but early might be filtering it out
                    if not early_top:
                        if not early_pass:
                            excluded_reason = "early_filter_fail"
                        else:
                            # Passed early pre-filter, but no early scanner signal
                            if legacy_strategy and legacy_strategy not in early_scanners:
                                excluded_reason = "excluded_by_early_scanner_subset"
                            else:
                                excluded_reason = "no_early_signal"
                    else:
                        excluded_reason = "early_score_below_min"
                else:
                    # Nothing qualifies (useful when --symbols is used)
                    if early_top and (early_score is not None) and early_score < min_score:
                        excluded_reason = "early_score_below_min"
                    elif legacy_top and (legacy_score is not None) and legacy_score < min_score:
                        excluded_reason = "legacy_score_below_min"
                    elif not early_top and not legacy_top:
                        excluded_reason = "no_signal"

            rows.append(
                Row(
                    symbol=symbol,
                    in_universe=in_universe,
                    rsi=rsi,
                    ret_10d=ret_10d,
                    early_filter_pass=early_pass,
                    early_strategy=early_strategy,
                    early_score=early_score,
                    legacy_strategy=legacy_strategy,
                    legacy_score=legacy_score,
                    excluded_reason=excluded_reason,
                )
            )

    df_out = pd.DataFrame([asdict(r) for r in rows])

    # Summary prints
    legacy_ok = df_out[(df_out["legacy_score"].notna()) & (df_out["legacy_score"] >= min_score)]
    early_ok = df_out[(df_out["early_score"].notna()) & (df_out["early_score"] >= min_score)]
    excluded = legacy_ok[legacy_ok["excluded_reason"].notna()]

    print("\nSummary:")
    print(f"  Legacy signals >= {min_score}: {len(legacy_ok)}")
    print(f"  Early signals  >= {min_score}: {len(early_ok)}")
    print(f"  Excluded legacy signals:       {len(excluded)}")
    if len(excluded):
        print("\nTop excluded (by legacy score):")
        top_ex = excluded.sort_values("legacy_score", ascending=False).head(15)
        for _, r in top_ex.iterrows():
            rsi_s = f"{r['rsi']:.1f}" if pd.notna(r["rsi"]) else "NA"
            ret_s = f"{r['ret_10d']*100:.1f}%" if pd.notna(r["ret_10d"]) else "NA"
            print(
                f"  {r['symbol']}: legacy {r['legacy_strategy']} {r['legacy_score']:.0f} | "
                f"early {r.get('early_strategy')} {r.get('early_score')} | "
                f"RSI {rsi_s} 10D {ret_s} | {r['excluded_reason']}"
            )

    # Save CSV
    out_path = Path(args.output) if args.output else Path("results") / f"missed_signals_{date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
