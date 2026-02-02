"""
Trade Signal Generator with LLM Research

Enhanced version that:
1. Generates scanner-based signals (like generate_signals.py)
2. Adds LLM research via Perplexity Sonar Pro for top picks
3. Outputs combined analysis with news, catalysts, and risks

Usage:
    python generate_signals_with_research.py
    python generate_signals_with_research.py --top 10  # Research top 10 picks
"""

import argparse
import json
import csv
from datetime import datetime
import sys
import sqlite3
from pathlib import Path
import config
from precompute import load_precomputed
import scanners as sc


def _configure_utf8_output() -> None:
    """
    Prevent UnicodeEncodeError on Windows when printing non-ASCII text (e.g., Japanese).

    If the console can't render some characters, they'll be replaced instead of crashing.
    """
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_configure_utf8_output()


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


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed

# Import the LLM research module
try:
    from llm_research import research_jp_stock, ResearchResult
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("WARNING: llm_research module not available. Install: pip install requests")


def get_company_name(symbol: str) -> str:
    """
    Get company name for a JP stock symbol.
    Uses the local SQLite database (symbol_info) if available, otherwise returns symbol.
    """
    try:
        conn = sqlite3.connect(config.DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM symbol_info WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    
    # Fallback: clean up symbol
    return symbol.replace(".T", "")


def generate_signals_with_research(top_n: int = 20):
    """
    Generate trade signals and add LLM research for top picks.
    
    Args:
        top_n: Number of top picks to research with LLM
    """
    
    print("=" * 70)
    print("TRADE SIGNAL GENERATOR + LLM RESEARCH")
    print("=" * 70)
    
    # Load precomputed data
    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        print("ERROR: No cache found.")
        print(f"Build it with: python precompute.py --top {config.UNIVERSE_TOP_N}")
        return
    
    print("Loading scanner data...")
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
    filtered.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nFound {len(filtered)} signals")
    
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
        print(
            f"\n{'Rank':<5} {'Symbol':<12} {'Score':<7} {'Entry':<10} "
            f"{'LotCost':<12} {'Confluence':<12} {'Strategy'}"
        )
        print("-" * 95)
        for i, sig in enumerate(candidates[:10], 1):
            symbol = sig['symbol']
            score = sig['score']
            entry = sig.get('entry_price', sig['price'])
            lot_cost = _lot_cost(entry, lot_size)
            confluence = sig.get('confluence_count', 1)
            strategy = sig['strategy']
            entry_label = "*" if sig.get("entry_source") == "close_est" else ""
            print(
                f"{i:<5} {symbol:<12} {score:<7.0f} JPY {entry:<9,.0f}{entry_label} "
                f"JPY {lot_cost:<10,.0f} +{confluence-1:<11} {strategy}"
            )

    print_candidates("TOP TRADE CANDIDATES (EARLY MODE · LOT-AFFORDABLE)", affordable)
    print_candidates("TOP TRADE CANDIDATES (EARLY MODE · OVER-BUDGET / ODD-LOT)", over_budget)
    if used_estimated_entry:
        print("\n* Entry uses last close when next-day open is not yet available.")
        print("  Recalculate stop/target from your actual fill at the open.")

    top_picks_affordable = affordable[:top_n]
    top_picks_over = over_budget[:top_n]
    top_picks = top_picks_affordable + top_picks_over

    legacy_signals = []
    legacy_still_no_momo = []
    legacy_already_momo = []
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
            for sig in legacy_signals:
                ret_10d = _get_10d_return(precomputed, sig['symbol'], latest_date)
                if ret_10d is None or ret_10d < return_max:
                    legacy_still_no_momo.append(sig)
                else:
                    legacy_already_momo.append(sig)
            print(f"Legacy still-no-momentum: {len(legacy_still_no_momo)} | already-in-momentum: {len(legacy_already_momo)}")
            print_candidates("LEGACY · STILL NO MOMENTUM (<15% / 10D)", legacy_still_no_momo)
            print_candidates("LEGACY · ALREADY IN MOMENTUM (>=15% / 10D)", legacy_already_momo)
    
    # === LLM RESEARCH ===
    if not LLM_AVAILABLE:
        print("\nWARNING: LLM research unavailable. Skipping...")
        return
    
    import os
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("\nWARNING: PERPLEXITY_API_KEY not set. Skipping LLM research.")
        print("   Set with: set PERPLEXITY_API_KEY=your_key_here")
        return

    # Optional: add live-only short-interest context for the LLM.
    # This is NOT used in backtests or scanner scoring.
    jpx_snapshot = {}
    try:
        import data_manager as dm

        print("\nFetching live JPX short-interest snapshot...")
        jpx_snapshot = dm.fetch_jpx_short_data()
        if jpx_snapshot:
            print(f"   Loaded short data for {len(jpx_snapshot)} symbols")
        else:
            print("   No short-interest data available")
    except Exception:
        print("   No short-interest data available")
    
    include_momentum_bucket = False
    if config.EARLY_MODE_SHOW_BOTH and legacy_already_momo:
        choice = input("\nInclude 'already in momentum' bucket in LLM research? (y/N): ").strip().lower()
        include_momentum_bucket = choice in ("y", "yes")
        if include_momentum_bucket:
            top_picks += legacy_already_momo[:top_n]
            # De-duplicate by symbol
            seen = set()
            unique = []
            for sig in top_picks:
                sym = sig.get("symbol")
                if sym in seen:
                    continue
                seen.add(sym)
                unique.append(sig)
            top_picks = unique

    print("\n" + "=" * 70)
    print(
        f"LLM RESEARCH (Top {len(top_picks_affordable)} affordable + "
        f"{len(top_picks_over)} over-budget"
        f"{' + momentum bucket' if include_momentum_bucket else ''} | {len(top_picks)} total)"
    )
    print("=" * 70)
    
    research_results = []
    
    for i, sig in enumerate(top_picks, 1):
        symbol = sig['symbol']
        company_name = get_company_name(symbol)
        
        print(f"\n[{i}/{len(top_picks)}] Researching {symbol} ({company_name})...")

        short_info = jpx_snapshot.get(symbol) if isinstance(jpx_snapshot, dict) else None
        short_ratio = None
        if isinstance(short_info, dict) and short_info.get("short_ratio") is not None:
            try:
                short_ratio = float(short_info["short_ratio"])
            except (TypeError, ValueError):
                short_ratio = None

        extra_context = None
        if short_ratio is not None:
            extra_lines = [f"JPX reported short ratio: {short_ratio:.2%}."]
            short_vol = short_info.get("short_volume") if isinstance(short_info, dict) else None
            if short_vol:
                try:
                    extra_lines.append(f"Reported short shares (sum across filers): {int(short_vol):,}.")
                except Exception:
                    pass
            extra_lines.append("Note: high shorts can increase squeeze potential, but can also reflect bearish positioning.")
            extra_context = "\n".join(extra_lines)

        if short_ratio is not None:
            print(f"   Live short ratio: {short_ratio:.2%}")

        result = research_jp_stock(symbol, company_name, extra_context=extra_context)
        research_results.append({
            "signal": sig,
            "research": result,
        })
        
        # Display summary
        if result.success:
            print(f"   NEWS: {result.recent_news_summary[:80]}...")
            print(f"   Sentiment: {result.news_sentiment}")
            print(f"   Catalysts: {len(result.upcoming_catalysts)}")
            if result.upcoming_catalysts:
                print(f"      - {result.upcoming_catalysts[0][:60]}...")
        else:
            print(f"   ERROR: Research failed: {result.error_message}")
        
        # Rate limiting (2 seconds between calls)
        if i < len(top_picks):
            import time
            time.sleep(2)
    
    # === COMBINED ANALYSIS ===
    print("\n" + "=" * 70)
    print("COMBINED ANALYSIS (Scanner + LLM)")
    print("=" * 70)
    
    for i, item in enumerate(research_results, 1):
        sig = item["signal"]
        res = item["research"]
        
        symbol = sig['symbol']
        entry = sig.get('entry_price', sig['price'])
        stop = entry * (1 - stop_loss_pct)
        target = entry * (1 + stop_loss_pct * rr_ratio)
        
        # Score adjustments based on research
        base_score = sig['score']
        sentiment_bonus = res._sentiment_to_score() * 10 if res.success else 0
        catalyst_bonus = min(len(res.upcoming_catalysts) * 5, 20) if res.success else 0
        risk_penalty = min(len(res.key_risks) * 3, 15) if res.success else 0
        
        adjusted_score = base_score + sentiment_bonus + catalyst_bonus - risk_penalty
        
        print("\n" + "=" * 60)
        print(f"#{i} {symbol}")
        print("=" * 60)
        
        print(f"\nTRADE SETUP")
        if sig.get("entry_source") == "close_est":
            print(f"   Entry (est close):  JPY {entry:,.0f}")
        else:
            print(f"   Entry (next open):  JPY {entry:,.0f}")
        print(f"   Stop:   JPY {stop:,.0f} (-{stop_loss_pct*100:.1f}%)")
        print(f"   Target: JPY {target:,.0f} (+{stop_loss_pct*rr_ratio*100:.1f}%)")
        lot_cost = _lot_cost(entry, lot_size)
        max_shares = _max_shares(entry, max_jpy)
        print(f"   Lot cost (x{lot_size}): JPY {lot_cost:,.0f}")
        if lot_cost > max_jpy:
            print(f"   Max shares within budget: {max_shares} (est JPY {max_shares * entry:,.0f})")
        
        print(f"\nSCORING")
        print(f"   Scanner Score:    {base_score:.0f}")
        print(f"   Sentiment Bonus:  {sentiment_bonus:+.1f}")
        print(f"   Catalyst Bonus:   {catalyst_bonus:+.1f}")
        print(f"   Risk Penalty:     {-risk_penalty:.1f}")
        print("   " + "-" * 22)
        print(f"   ADJUSTED SCORE:   {adjusted_score:.0f}")
        
        if res.success:
            print(f"\nNEWS SUMMARY")
            print(f"   {res.recent_news_summary}")
            
            if res.upcoming_catalysts:
                print(f"\nUPCOMING CATALYSTS")
                for cat in res.upcoming_catalysts[:3]:
                    print(f"   - {cat}")
            
            if res.key_risks:
                print(f"\nKEY RISKS")
                for risk in res.key_risks[:2]:
                    print(f"   - {risk}")
            
            print(f"\nSENTIMENT: {res.news_sentiment}")

    # === OPTIONAL SAVE ===
    if getattr(config, "SAVE_LLM_RESULTS", True):
        results_dir = Path(getattr(config, "RESULTS_DIR", Path("results")))
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"llm_research_{ts}.json"

        def _serialize_result(item: dict) -> dict:
            sig = item["signal"]
            res = item["research"]
            return {
                "symbol": sig.get("symbol"),
                "strategy": sig.get("strategy"),
                "score": sig.get("score"),
                "entry_price": sig.get("entry_price", sig.get("price")),
                "entry_source": sig.get("entry_source", "close_est"),
                "confluence_scanners": sig.get("confluence_scanners", []),
                "news_sentiment": res.news_sentiment if res.success else "N/A",
                "recent_news_summary": res.recent_news_summary if res.success else "",
                "upcoming_catalysts": res.upcoming_catalysts if res.success else [],
                "key_risks": res.key_risks if res.success else [],
                "error": res.error_message if not res.success else "",
            }

        payload = {
            "run_timestamp": ts,
            "latest_date": latest_date,
            "min_score": min_score,
            "stop_loss_pct": stop_loss_pct,
            "risk_reward_ratio": rr_ratio,
            "early_mode_enabled": config.EARLY_MODE_ENABLED,
            "results": [_serialize_result(x) for x in research_results],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved LLM results to: {out_path}")

        # Also save CSV
        csv_path = results_dir / f"llm_research_{ts}.csv"
        rows = payload["results"]
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved LLM results to: {csv_path}")
    
    # === FINAL RANKING ===
    print("\n" + "=" * 70)
    print("FINAL RANKING (Adjusted Scores)")
    print("=" * 70)
    
    ranked = []
    for item in research_results:
        sig = item["signal"]
        res = item["research"]
        
        base_score = sig['score']
        sentiment_bonus = res._sentiment_to_score() * 10 if res.success else 0
        catalyst_bonus = min(len(res.upcoming_catalysts) * 5, 20) if res.success else 0
        risk_penalty = min(len(res.key_risks) * 3, 15) if res.success else 0
        
        adjusted_score = base_score + sentiment_bonus + catalyst_bonus - risk_penalty
        entry = sig.get("entry_price", sig["price"])
        lot_cost = _lot_cost(entry, lot_size)
        budget_ok = entry > 0 and lot_cost <= max_jpy
        max_shares = _max_shares(entry, max_jpy)
        
        ranked.append({
            "symbol": sig["symbol"],
            "original_score": base_score,
            "adjusted_score": adjusted_score,
            "sentiment": res.news_sentiment if res.success else "N/A",
            "catalysts": len(res.upcoming_catalysts) if res.success else 0,
            "entry": entry,
            "entry_source": sig.get("entry_source", "close_est"),
            "lot_cost": lot_cost,
            "max_shares": max_shares,
            "budget_ok": budget_ok,
        })
    
    ranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
    ranked_affordable = [r for r in ranked if r["budget_ok"]]
    ranked_over = [r for r in ranked if not r["budget_ok"]]

    def print_ranked(title: str, rows: list[dict], show_max_shares: bool = False) -> None:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        if not rows:
            print("   No candidates.")
            return
        if show_max_shares:
            print(f"\n{'Rank':<5} {'Symbol':<10} {'Orig':<6} {'Adj':<6} {'Sent':<10} {'Cat':<5} {'Entry':<10} {'MaxSh':<6} {'EstCost'}")
            print("-" * 90)
            for i, r in enumerate(rows, 1):
                est_cost = r["max_shares"] * r["entry"]
                entry_label = "*" if r.get("entry_source") == "close_est" else ""
                print(
                    f"{i:<5} {r['symbol']:<10} {r['original_score']:<6.0f} {r['adjusted_score']:<6.0f} "
                    f"{r['sentiment']:<10} {r['catalysts']:<5} JPY {r['entry']:<9,.0f}{entry_label} "
                    f"{r['max_shares']:<6} JPY {est_cost:,.0f}"
                )
        else:
            print(f"\n{'Rank':<5} {'Symbol':<10} {'Orig':<6} {'Adj':<6} {'Sent':<10} {'Cat':<5} {'Entry':<10} {'LotCost'}")
            print("-" * 80)
            for i, r in enumerate(rows, 1):
                entry_label = "*" if r.get("entry_source") == "close_est" else ""
                print(
                    f"{i:<5} {r['symbol']:<10} {r['original_score']:<6.0f} {r['adjusted_score']:<6.0f} "
                    f"{r['sentiment']:<10} {r['catalysts']:<5} JPY {r['entry']:<9,.0f}{entry_label} "
                    f"JPY {r['lot_cost']:,.0f}"
                )

    print_ranked("FINAL RANKING (LOT-AFFORDABLE)", ranked_affordable, show_max_shares=False)
    print_ranked("FINAL RANKING (OVER-BUDGET / ODD-LOT)", ranked_over, show_max_shares=True)
    
    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate signals with LLM research")
    parser.add_argument("--top", type=int, default=20, help="Number of top picks to research")
    parser.add_argument("--no-save", action="store_true", help="Disable saving LLM results to results/")
    args = parser.parse_args()
    if args.no_save:
        setattr(config, "SAVE_LLM_RESULTS", False)

    generate_signals_with_research(top_n=args.top)
