"""
Comprehensive Scanner Diagnostics

Analyzes:
1. Raw signal counts by scanner (before filtering)
2. Per-scanner performance after trades
3. Scanner co-occurrence (which scanners fire together)
4. Why some scanners aren't generating trades
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backtesting import run_fast_backtest, get_trade_history_df
from precompute import precompute_all_data, load_precomputed
import config


def analyze_raw_signals(precomputed, min_score: int = 30):
    """
    Analyze raw signal generation by each scanner.
    Shows how many signals each scanner produces before any filtering.
    """
    print("\n" + "=" * 70)
    print("RAW SIGNAL COUNTS BY SCANNER")
    print(f"(min_score threshold: {min_score})")
    print("=" * 70)
    
    scanner_counts = defaultdict(int)
    scanner_signals = defaultdict(list)  # Store all scores
    by_date = defaultdict(lambda: defaultdict(list))  # date -> scanner -> symbols
    
    for date, signals in precomputed.signals_by_date.items():
        for signal in signals:
            scanner = signal.get('strategy', 'unknown')
            score = signal.get('score', 0)
            symbol = signal.get('symbol', '')
            
            # Count ALL signals (regardless of score)
            scanner_signals[scanner].append(score)
            
            # Count signals above threshold
            if score >= min_score:
                scanner_counts[scanner] += 1
                by_date[date][scanner].append(symbol)
    
    # Print summary
    print(f"\n{'Scanner':<25} {'Raw Signals':>12} {'Above {0}'.format(min_score):>12} {'Avg Score':>10} {'Max Score':>10}")
    print("-" * 70)
    
    for scanner in sorted(scanner_signals.keys()):
        scores = scanner_signals[scanner]
        raw = len(scores)
        above = scanner_counts[scanner]
        avg = np.mean(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        status = "✅" if above > 50 else "⚠️" if above > 10 else "❌"
        print(f"{status} {scanner:<23} {raw:>12,} {above:>12,} {avg:>10.1f} {max_score:>10}")
    
    print("-" * 70)
    total_raw = sum(len(s) for s in scanner_signals.values())
    total_above = sum(scanner_counts.values())
    print(f"{'TOTAL':<25} {total_raw:>12,} {total_above:>12,}")
    
    return scanner_counts, scanner_signals


def analyze_scanner_cooccurrence(precomputed, min_score: int = 30):
    """
    Analyze which scanners fire together on the same symbol/date.
    """
    print("\n" + "=" * 70)
    print("SCANNER CO-OCCURRENCE ANALYSIS")
    print(f"(How often do scanners fire together on same symbol/date)")
    print("=" * 70)
    
    # Build: (date, symbol) -> list of scanners that fired
    cooccurrence = defaultdict(list)
    
    for date, signals in precomputed.signals_by_date.items():
        for signal in signals:
            if signal.get('score', 0) >= min_score:
                key = (date, signal['symbol'])
                cooccurrence[key].append(signal['strategy'])
    
    # Count pairs
    pair_counts = defaultdict(int)
    single_counts = defaultdict(int)
    
    for key, scanners in cooccurrence.items():
        scanners = list(set(scanners))  # Unique scanners
        
        if len(scanners) == 1:
            single_counts[scanners[0]] += 1
        else:
            # Count all pairs
            for i, s1 in enumerate(scanners):
                for s2 in scanners[i+1:]:
                    pair = tuple(sorted([s1, s2]))
                    pair_counts[pair] += 1
    
    # Print single-scanner signals
    print(f"\n📊 Single-Scanner Signals (only one scanner fired):")
    for scanner, count in sorted(single_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {scanner:<25}: {count:>6}")
    
    # Print common pairs
    print(f"\n🔗 Scanner Pairs (both fired on same symbol/date):")
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:15]
    
    if not sorted_pairs:
        print("   No scanner pairs found - scanners are firing independently")
    else:
        for (s1, s2), count in sorted_pairs:
            print(f"   {s1:<20} + {s2:<20}: {count:>5}")
    
    return pair_counts, single_counts


def analyze_why_missing(precomputed, scanner_signals):
    """
    Analyze why certain scanners aren't generating enough signals.
    """
    print("\n" + "=" * 70)
    print("WHY ARE SOME SCANNERS NOT FIRING?")
    print("=" * 70)
    
    for scanner, scores in sorted(scanner_signals.items()):
        if len(scores) < 50:  # Low signal count
            percentiles = np.percentile(scores, [25, 50, 75, 90, 95]) if scores else [0,0,0,0,0]
            print(f"\n❌ {scanner}:")
            print(f"   Total signals: {len(scores)}")
            print(f"   Score distribution: P25={percentiles[0]:.0f}, P50={percentiles[1]:.0f}, P75={percentiles[2]:.0f}, P90={percentiles[3]:.0f}, P95={percentiles[4]:.0f}")
            
            if max(scores) < 50 if scores else True:
                print(f"   ⚠️ Issue: Max score is {max(scores) if scores else 0}, never reaches min_score=50")
                print(f"   → Consider adjusting scanner thresholds in config.py")


def run_backtest_all_scores(precomputed, min_score: int = 30):
    """
    Run backtest with lower min_score to get more trades.
    """
    print("\n" + "=" * 70)
    print(f"BACKTEST WITH MIN_SCORE={min_score}")
    print("=" * 70)
    
    engine, metrics = run_fast_backtest(
        precomputed=precomputed,
        initial_balance=1_000_000,
        stop_loss_pct=0.08,
        exit_mode='fixed_rr',
        risk_reward_ratio=2.0,
        min_score=min_score,
        progress=False,
    )
    
    print(f"\n   Total trades: {metrics['total_trades']}")
    print(f"   Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"   Profit factor: {metrics['profit_factor']:.2f}")
    print(f"   Total return: {metrics['total_return']*100:.1f}%")
    
    return engine, metrics


def print_scanner_performance(trades_df: pd.DataFrame):
    """
    Detailed per-scanner performance.
    """
    print("\n" + "=" * 70)
    print("DETAILED SCANNER PERFORMANCE")
    print("=" * 70)
    
    if trades_df.empty or 'scanner_name' not in trades_df.columns:
        print("No scanner data available.")
        return
    
    print(f"\n{'Scanner':<25} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'PF':>8} {'Avg Win':>10} {'Avg Loss':>10} {'Net P/L':>12}")
    print("-" * 95)
    
    results = []
    for scanner, group in trades_df.groupby('scanner_name'):
        wins = group[group['pnl'] > 0]
        losses = group[group['pnl'] <= 0]
        
        total = len(group)
        win_count = len(wins)
        wr = win_count / total if total > 0 else 0
        
        gp = wins['pnl'].sum() if len(wins) > 0 else 0
        gl = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        pf = gp / gl if gl > 0 else float('inf')
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        net = gp - gl
        
        status = "✅" if pf > 1.5 else "⚠️" if pf > 1.0 else "❌"
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        
        print(f"{status} {scanner:<23} {total:>6} {win_count:>6} {wr*100:>6.1f}% {pf_str:>8} {avg_win:>10,.0f} {avg_loss:>10,.0f} {net:>12,.0f}")
        
        results.append({
            'scanner': scanner, 'trades': total, 'wins': win_count,
            'win_rate': wr, 'profit_factor': pf, 'net_pnl': net
        })
    
    print("-" * 95)
    
    # Save results
    pd.DataFrame(results).to_csv("results/scanner_detailed_performance.csv", index=False)
    print(f"\n💾 Saved to results/scanner_detailed_performance.csv")


def main():
    print("🔬 COMPREHENSIVE SCANNER DIAGNOSTICS")
    print("=" * 70)
    
    # Load precomputed data
    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        print("⚠️ No cache found. Run run_walk_forward.py first.")
        return
    
    precomputed = load_precomputed(cache_path)
    print(f"📦 Loaded: {precomputed.num_symbols} symbols, {precomputed.num_days} days")
    
    # 1. Raw signal counts
    scanner_counts, scanner_signals = analyze_raw_signals(precomputed, min_score=30)
    
    # 2. Why are some missing?
    analyze_why_missing(precomputed, scanner_signals)
    
    # 3. Scanner co-occurrence
    analyze_scanner_cooccurrence(precomputed, min_score=30)
    
    # 4. Backtest with lower threshold
    engine, metrics = run_backtest_all_scores(precomputed, min_score=30)
    trades_df = get_trade_history_df(engine)
    trades_df.to_csv("results/all_trades_min30.csv", index=False)
    
    # 5. Detailed scanner performance
    print_scanner_performance(trades_df)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
