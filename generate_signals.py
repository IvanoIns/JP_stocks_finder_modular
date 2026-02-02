"""
Trade Signal Generator - Find Tomorrow's Burst Candidates

Scans the most recent data and outputs trade signals with entry/stop/target prices.
Use config.py parameters (Score 30, Stop 6%, R:R 2.0).
"""

from pathlib import Path
import sqlite3
import config
from precompute import load_precomputed
import scanners as sc


def _get_db_latest_date(db_file: Path) -> str | None:
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_prices")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _warn_if_cache_mismatch(precomputed) -> None:
    cache_min_score = getattr(precomputed, "min_score", None)
    cache_raw = getattr(precomputed, "raw_signals", False)
    if cache_raw:
        return
    if cache_min_score is not None and cache_min_score != config.MIN_SCANNER_SCORE:
        print(
            "WARNING: Cache was built with MIN_SCANNER_SCORE="
            f"{cache_min_score}, but config.MIN_SCANNER_SCORE={config.MIN_SCANNER_SCORE}. "
            f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
        )

    cache_scanner_config = getattr(precomputed, "scanner_config", None)
    if cache_scanner_config and cache_scanner_config != config.get_scanner_config():
        print(
            "WARNING: Cache scanner_config differs from config.py. "
            f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
        )

    cache_universe_top_n = getattr(precomputed, "universe_top_n", None)
    cache_universe_min_volume = getattr(precomputed, "universe_min_volume", None)
    cache_universe_exclude_nikkei = getattr(precomputed, "universe_exclude_nikkei", None)
    if cache_universe_top_n is None and cache_universe_min_volume is None:
        print("WARNING: Cache missing universe metadata. Rebuild recommended.")
    else:
        if cache_universe_top_n is not None and cache_universe_top_n != config.UNIVERSE_TOP_N:
            print(
                "WARNING: Cache was built with UNIVERSE_TOP_N="
                f"{cache_universe_top_n}, but config.UNIVERSE_TOP_N={config.UNIVERSE_TOP_N}. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )
        if cache_universe_min_volume is not None and cache_universe_min_volume != config.MIN_AVG_DAILY_VOLUME:
            print(
                "WARNING: Cache was built with MIN_AVG_DAILY_VOLUME="
                f"{cache_universe_min_volume}, but config.MIN_AVG_DAILY_VOLUME={config.MIN_AVG_DAILY_VOLUME}. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )
        if cache_universe_exclude_nikkei is not None and cache_universe_exclude_nikkei != config.EXCLUDE_NIKKEI_225:
            print(
                "WARNING: Cache was built with EXCLUDE_NIKKEI_225="
                f"{cache_universe_exclude_nikkei}, but config.EXCLUDE_NIKKEI_225={config.EXCLUDE_NIKKEI_225}. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )

    cache_early_mode = getattr(precomputed, "early_mode_enabled", None)
    cache_early_rsi = getattr(precomputed, "early_mode_rsi_max", None)
    cache_early_return = getattr(precomputed, "early_mode_return_max", None)
    cache_early_scanners = getattr(precomputed, "early_mode_scanners", None)
    if cache_early_mode is None:
        print("WARNING: Cache missing early-mode metadata. Rebuild recommended.")
    else:
        if cache_early_mode != config.EARLY_MODE_ENABLED:
            print(
                "WARNING: Cache was built with EARLY_MODE_ENABLED="
                f"{cache_early_mode}, but config.EARLY_MODE_ENABLED={config.EARLY_MODE_ENABLED}. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )
        if cache_early_rsi is not None and cache_early_rsi != config.EARLY_MODE_RSI_MAX:
            print(
                "WARNING: Cache was built with EARLY_MODE_RSI_MAX="
                f"{cache_early_rsi}, but config.EARLY_MODE_RSI_MAX={config.EARLY_MODE_RSI_MAX}. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )
        if cache_early_return is not None and cache_early_return != config.EARLY_MODE_10D_RETURN_MAX:
            print(
                "WARNING: Cache was built with EARLY_MODE_10D_RETURN_MAX="
                f"{cache_early_return}, but config.EARLY_MODE_10D_RETURN_MAX={config.EARLY_MODE_10D_RETURN_MAX}. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )
        if cache_early_scanners and cache_early_scanners != config.EARLY_MODE_SCANNERS:
            print(
                "WARNING: Cache EARLY_MODE_SCANNERS differs from config.py. "
                f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
            )


def _get_latest_date_with_signals(precomputed) -> str | None:
    """
    Return the most recent trading date that has at least one cached signal.
    """
    for date in reversed(precomputed.trading_dates):
        if precomputed.signals_by_date.get(date):
            return date
    return None


def _get_budget_params() -> tuple[float, int]:
    max_jpy = getattr(config, "MAX_JPY_PER_TRADE", 100_000)
    lot_size = getattr(config, "LOT_SIZE", 100)
    if not isinstance(lot_size, int) or lot_size <= 0:
        lot_size = 100
    return float(max_jpy), lot_size


def _lot_cost(entry: float, lot_size: int) -> float:
    if not entry:
        return 0.0
    return float(entry) * lot_size


def _max_shares(entry: float, max_jpy: float) -> int:
    if not entry or entry <= 0:
        return 0
    return int(max_jpy // float(entry))


def _split_by_budget(signals: list[dict], max_jpy: float, lot_size: int) -> tuple[list[dict], list[dict]]:
    affordable = []
    over_budget = []
    for sig in signals:
        entry = sig.get("entry_price", sig.get("price", 0)) or 0
        if entry > 0 and _lot_cost(entry, lot_size) <= max_jpy:
            affordable.append(sig)
        else:
            over_budget.append(sig)
    return affordable, over_budget


def _get_next_trading_date(precomputed, date: str) -> str | None:
    try:
        idx = precomputed.trading_dates.index(date)
    except ValueError:
        return None
    if idx + 1 >= len(precomputed.trading_dates):
        return None
    return precomputed.trading_dates[idx + 1]


def _attach_entry_prices(precomputed, date: str, signals: list[dict]) -> bool:
    """
    Attach entry_price and entry_source to each signal.
    Returns True if any entries are estimated from close.
    """
    estimated = False
    next_date = _get_next_trading_date(precomputed, date)
    for sig in signals:
        symbol = sig.get("symbol")
        fallback = sig.get("price", 0) or 0
        entry_price = None
        entry_source = "close_est"
        if next_date:
            bar = precomputed.bars_by_date.get(next_date, {}).get(symbol)
            if bar and bar.get("open"):
                entry_price = bar["open"]
                entry_source = "next_open"
        if entry_price is None:
            entry_price = fallback
            entry_source = "close_est"
            estimated = True
        sig["entry_price"] = entry_price
        sig["entry_source"] = entry_source
    return estimated


def _compute_signals_for_date(precomputed, date: str, min_score: int, scanner_config: dict, early_mode: bool) -> list[dict]:
    signals = []
    universe_symbols = set(getattr(precomputed, "universe_by_date", {}).get(date, precomputed.universe_symbols))
    for symbol, df in precomputed.symbol_data.items():
        if symbol not in universe_symbols:
            continue
        data_up_to = df[df.index.strftime('%Y-%m-%d') <= date]
        if len(data_up_to) < 60:
            continue
        sigs = sc.get_all_signals(symbol, data_up_to, None, scanner_config, min_score, early_mode=early_mode)
        for sig in sigs:
            sig['date'] = date
            signals.append(sig)
    signals.sort(key=lambda x: x['score'], reverse=True)
    return signals


def _get_10d_return(precomputed, symbol: str, date: str) -> float | None:
    df = precomputed.symbol_data.get(symbol)
    if df is None:
        return None
    df = df[df.index.strftime('%Y-%m-%d') <= date]
    if len(df) < 11:
        return None
    return (df['close'].iloc[-1] / df['close'].iloc[-11]) - 1


def generate_signals():
    """Generate trade signals for the next trading day."""
    
    print("=" * 70)
    print("TRADE SIGNAL GENERATOR")
    print("=" * 70)
    
    # Load precomputed data
    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        print("ERROR: No cache found.")
        print(f"Build it with: python precompute.py --top {config.UNIVERSE_TOP_N}")
        return
    
    print("Loading data...")
    precomputed = load_precomputed(cache_path)
    
    print(f"   Symbols: {precomputed.num_symbols}")
    print(f"   Date range: {precomputed.start_date} to {precomputed.end_date}")
    
    # Get the most recent trading date that actually has signals
    latest_db_date = precomputed.trading_dates[-1]
    latest_signal_date = _get_latest_date_with_signals(precomputed)
    latest_date = latest_signal_date or latest_db_date
    if latest_signal_date and latest_signal_date != latest_db_date:
        print(f"\nLatest data: {latest_db_date}")
        print(f"   No signals for {latest_db_date}. Falling back to {latest_signal_date}.")
    else:
        print(f"\nLatest data: {latest_date}")
    print(f"   Signals generated for NEXT trading day")

    _warn_if_cache_mismatch(precomputed)
    db_latest_date = _get_db_latest_date(config.DATABASE_FILE)
    if db_latest_date and db_latest_date > latest_date:
        print(
            f"WARNING: Database has newer data ({db_latest_date}) than cache ({latest_date}). "
            f"Rebuild cache: python precompute.py --top {config.UNIVERSE_TOP_N}"
        )
    
    # Get signals for latest date
    signals = precomputed.signals_by_date.get(latest_date, [])
    
    if not signals:
        print(f"\nWARNING: No signals found for {latest_date}")
        return
    
    # Filter by min score
    min_score = config.MIN_SCANNER_SCORE
    stop_loss_pct = config.STOP_LOSS_PCT
    rr_ratio = config.RISK_REWARD_RATIO
    
    print(f"\nParameters:")
    print(f"   Min Score: {min_score}")
    print(f"   Stop Loss: {stop_loss_pct*100:.1f}%")
    print(f"   R:R Ratio: {rr_ratio}")
    if config.EARLY_MODE_ENABLED:
        print(f"   Early Mode: ON (RSI <= {config.EARLY_MODE_RSI_MAX}, 10D Return < {config.EARLY_MODE_10D_RETURN_MAX:.0%})")
        print(f"   Early Scanners: {', '.join(config.EARLY_MODE_SCANNERS)}")
    else:
        print("   Early Mode: OFF")
    
    filtered = [s for s in signals if s['score'] >= min_score]
    
    # Sort by score descending
    filtered.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nFound {len(filtered)} signals (from {len(signals)} raw)")
    
    if not filtered:
        print("   No signals passed the min_score filter.")
        return

    used_estimated_entry = _attach_entry_prices(precomputed, latest_date, filtered)

    max_jpy, lot_size = _get_budget_params()
    max_entry_price = max_jpy / lot_size if lot_size else 0
    affordable, over_budget = _split_by_budget(filtered, max_jpy, lot_size)

    print(f"Budget: Max JPY {max_jpy:,.0f} | Lot size {lot_size} (<= JPY {max_entry_price:,.0f} per 100-share lot)")
    print(f"Affordable lots: {len(affordable)} | Over-budget (odd-lot): {len(over_budget)}")

    def print_candidates(title: str, candidates: list[dict]) -> None:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        if not candidates:
            print("   No candidates.")
            return
        print(f"\n{'Rank':<5} {'Symbol':<12} {'Score':<7} {'Entry':<10} {'LotCost':<12} {'Stop':<10} {'Target':<10} {'Strategy':<20}")
        print("-" * 100)
        for i, sig in enumerate(candidates[:10], 1):
            symbol = sig['symbol']
            score = sig['score']
            entry = sig.get('entry_price', sig['price'])
            lot_cost = _lot_cost(entry, lot_size)
            stop = entry * (1 - stop_loss_pct)
            target = entry * (1 + stop_loss_pct * rr_ratio)
            strategy = sig['strategy']
            confluence = sig.get('confluence_count', 1)
            if confluence > 1:
                strategy = f"{strategy} (+{confluence-1})"
            entry_label = "*" if sig.get("entry_source") == "close_est" else ""
            print(
                f"{i:<5} {symbol:<12} {score:<7.0f} JPY {entry:<9,.0f}{entry_label} "
                f"JPY {lot_cost:<10,.0f} JPY {stop:<9,.0f} JPY {target:<9,.0f} {strategy:<20}"
            )

    def print_detailed(title: str, candidates: list[dict]) -> None:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        if not candidates:
            print("   No candidates.")
            return
        for i, sig in enumerate(candidates[:5], 1):
            symbol = sig['symbol']
            entry = sig.get('entry_price', sig['price'])
            stop = entry * (1 - stop_loss_pct)
            target = entry * (1 + stop_loss_pct * rr_ratio)
            risk = entry - stop
            reward = target - entry
            lot_cost = _lot_cost(entry, lot_size)
            max_shares = _max_shares(entry, max_jpy)
            print(f"\n#{i} {symbol}")
            print(f"   Score: {sig['score']:.0f} | Strategy: {sig['strategy']}")
            if sig.get("entry_source") == "close_est":
                print(f"   Entry (est close): JPY {entry:,.0f}")
            else:
                print(f"   Entry (next open): JPY {entry:,.0f}")
            print(f"   Lot cost (x{lot_size}): JPY {lot_cost:,.0f}")
            if lot_cost > max_jpy:
                print(f"   Max shares within budget: {max_shares} (est JPY {max_shares * entry:,.0f})")
            print(f"   Stop:  JPY {stop:,.0f} (-{stop_loss_pct*100:.1f}%)")
            print(f"   Target: JPY {target:,.0f} (+{stop_loss_pct*rr_ratio*100:.1f}%)")
            print(f"   Risk: JPY {risk:,.0f} -> Reward: JPY {reward:,.0f}")
            if sig.get('confluence_scanners'):
                scanners = sig['confluence_scanners']
                print(f"   Confluence ({len(scanners)} scanners): {', '.join(scanners)}")

    print_candidates("TOP TRADE CANDIDATES (EARLY MODE · LOT-AFFORDABLE)", affordable)
    print_candidates("TOP TRADE CANDIDATES (EARLY MODE · OVER-BUDGET / ODD-LOT)", over_budget)
    print_detailed("DETAILED VIEW (EARLY MODE · LOT-AFFORDABLE)", affordable)
    print_detailed("DETAILED VIEW (EARLY MODE · OVER-BUDGET / ODD-LOT)", over_budget)

    if used_estimated_entry:
        print("\n* Entry uses last close when next-day open is not yet available.")
        print("  Recalculate stop/target from your actual fill at the open.")

    if config.EARLY_MODE_SHOW_BOTH:
        print("\n" + "=" * 70)
        print("LEGACY MODE (Momentum Allowed)")
        print("=" * 70)
        legacy_signals = _compute_signals_for_date(
            precomputed=precomputed,
            date=latest_date,
            min_score=min_score,
            scanner_config=config.get_scanner_config(),
            early_mode=False,
        )
        _attach_entry_prices(precomputed, latest_date, legacy_signals)
        if not legacy_signals:
            print("   No legacy signals.")
        else:
            return_max = config.EARLY_MODE_10D_RETURN_MAX
            still_no_momo = []
            already_momo = []
            for sig in legacy_signals:
                ret_10d = _get_10d_return(precomputed, sig['symbol'], latest_date)
                if ret_10d is None or ret_10d < return_max:
                    still_no_momo.append(sig)
                else:
                    already_momo.append(sig)
            print(f"Legacy still-no-momentum: {len(still_no_momo)} | already-in-momentum: {len(already_momo)}")
            print_candidates("LEGACY · STILL NO MOMENTUM (<15% / 10D)", still_no_momo)
            print_candidates("LEGACY · ALREADY IN MOMENTUM (>=15% / 10D)", already_momo)
    
    # === TOP PICK BY SCANNER TYPE ===
    print("\n" + "=" * 70)
    print("TOP PICK BY SCANNER (diverse view)")
    print("=" * 70)
    
    # Group by primary scanner
    scanner_best = {}
    for sig in filtered:
        scanner = sig['strategy']
        if scanner not in scanner_best:
            scanner_best[scanner] = sig
    
    print(f"\n{'Scanner':<25} {'Symbol':<10} {'Score':<7} {'Entry':<10} {'LotCost':<12} {'Confluence'}")
    print("-" * 85)
    
    # Sort by historical PF (star scanners first)
    star_order = ['oversold_bounce', 'burst_candidates', 'momentum_star', 
                  'volatility_explosion', 'relative_strength', 'consolidation_breakout',
                  'coiling_pattern', 'smart_money_flow', 'reversal_rocket']
    
    for scanner in star_order:
        if scanner in scanner_best:
            sig = scanner_best[scanner]
            conf = sig.get('confluence_count', 1)
            conf_str = f"+{conf-1}" if conf > 1 else "-"
            entry = sig.get("entry_price", sig["price"])
            entry_label = "*" if sig.get("entry_source") == "close_est" else ""
            lot_cost = _lot_cost(entry, lot_size)
            print(
                f"* {scanner:<22} {sig['symbol']:<10} {sig['score']:<7.0f} "
                f"JPY {entry:<9,.0f}{entry_label} JPY {lot_cost:<10,.0f} {conf_str}"
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("TRADING NOTES")
    print("=" * 70)
    print(f"- Total candidates: {len(filtered)}")
    print(f"- Budget per play: JPY {max_jpy:,.0f} (lot size {lot_size}, max entry ~ JPY {max_entry_price:,.0f})")
    print(f"- Entry timing: Open of next trading day")
    print(f"- Position size: Max {config.MAX_POSITIONS} concurrent positions")
    print(f"- Risk per trade: {stop_loss_pct*100:.1f}% stop loss")
    print(f"- Reward target: {stop_loss_pct*rr_ratio*100:.1f}% profit target")
    print("- Exit: First to hit (stop OR target)")
    print("\nTIP: High confluence (3+) = multiple scanners agree = stronger signal")
    print("=" * 70)


if __name__ == "__main__":
    generate_signals()
