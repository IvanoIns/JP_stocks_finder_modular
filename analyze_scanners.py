"""
Scanner Performance Analysis

Analyzes per-scanner performance from walk-forward results.
Identifies which of the 11 scanners generate profitable vs losing trades.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backtesting import run_fast_backtest, get_trade_history_df
from precompute import precompute_all_data, load_precomputed


def analyze_scanner_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by scanner.
    
    Args:
        trades_df: DataFrame with trade history including 'scanner_name' column
        
    Returns:
        DataFrame with per-scanner metrics
    """
    if trades_df.empty or 'scanner_name' not in trades_df.columns:
        print("⚠️ No scanner_name in trade history. Run backtest first.")
        return pd.DataFrame()
    
    # Group by scanner
    scanner_stats = []
    for scanner, group in trades_df.groupby('scanner_name'):
        wins = group[group['pnl'] > 0]
        losses = group[group['pnl'] <= 0]
        
        total_trades = len(group)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        scanner_stats.append({
            'scanner': scanner,
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'net_pnl': gross_profit - gross_loss,
        })
    
    results_df = pd.DataFrame(scanner_stats)
    results_df = results_df.sort_values('profit_factor', ascending=False)
    return results_df


def print_scanner_analysis(df: pd.DataFrame):
    """Pretty print scanner analysis."""
    print("\n" + "=" * 70)
    print("SCANNER PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    if df.empty:
        print("No data available.")
        return
    
    print(f"\n{'Scanner':<25} {'Trades':>7} {'Win%':>7} {'PF':>8} {'Net P/L':>12}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        pf = row['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        
        # Color-code by profitability
        status = "✅" if row['profit_factor'] > 1.2 else "⚠️" if row['profit_factor'] > 1.0 else "❌"
        
        print(f"{status} {row['scanner']:<23} {row['total_trades']:>6} "
              f"{row['win_rate']*100:>6.1f}% {pf_str:>8} {row['net_pnl']:>12,.0f}")
    
    print("-" * 70)
    
    # Summary recommendations
    weak_scanners = df[df['profit_factor'] < 1.0]['scanner'].tolist()
    strong_scanners = df[df['profit_factor'] > 1.5]['scanner'].tolist()
    
    if weak_scanners:
        print(f"\n❌ WEAK SCANNERS (PF < 1.0): {', '.join(weak_scanners)}")
        print("   Consider removing or improving these.")
    
    if strong_scanners:
        print(f"\n✅ STRONG SCANNERS (PF > 1.5): {', '.join(strong_scanners)}")
        print("   Focus on these patterns.")


def run_scanner_analysis():
    """Run a full backtest and analyze scanner performance."""
    print("🔍 Running Scanner Performance Analysis...")
    
    # Load precomputed data
    cache_path = Path("results/precomputed_cache.pkl")
    if cache_path.exists():
        print("📦 Loading cached data...")
        precomputed = load_precomputed(cache_path)
        print(f"   ✓ {precomputed.num_symbols} symbols, {precomputed.num_days} days")
    else:
        print("⚠️ No cache found. Run run_walk_forward.py first.")
        return
    
    # Load best params from walk-forward results
    import json
    params_path = Path("results/best_params.json")
    if params_path.exists():
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print(f"📋 Loaded best_params.json:")
        print(f"   Min Score: {best_params.get('min_score', 'N/A')}")
        print(f"   Stop Loss: {best_params.get('stop_loss', 'N/A')}")
        print(f"   R:R Ratio: {best_params.get('risk_reward_ratio', 'N/A')}")
    else:
        print("⚠️ No best_params.json found. Using defaults.")
        best_params = {
            'min_score': 30,
            'stop_loss': 0.06,
            'exit_mode': 'fixed_rr',
            'risk_reward_ratio': 2.0,
        }
    
    # Run backtest with LOADED params (not hardcoded!)
    print("📊 Running backtest with best params...")
    engine, metrics = run_fast_backtest(
        precomputed=precomputed,
        initial_balance=1_000_000,
        stop_loss_pct=best_params.get('stop_loss', 0.06),
        exit_mode=best_params.get('exit_mode', 'fixed_rr'),
        risk_reward_ratio=best_params.get('risk_reward_ratio', 2.0),
        min_score=best_params.get('min_score', 30),
        progress=True,
    )
    
    print(f"\n   Total trades: {metrics['total_trades']}")
    print(f"   Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"   Profit factor: {metrics['profit_factor']:.2f}")
    
    # Get trade history and analyze
    trades_df = get_trade_history_df(engine)
    
    if trades_df.empty:
        print("⚠️ No trades to analyze.")
        return
    
    # Save for reference
    trades_df.to_csv("results/trade_history_with_scanners.csv", index=False)
    print(f"   💾 Trade history saved to results/trade_history_with_scanners.csv")
    
    # Analyze scanner performance
    scanner_df = analyze_scanner_performance(trades_df)
    print_scanner_analysis(scanner_df)
    
    # Save analysis
    scanner_df.to_csv("results/scanner_performance.csv", index=False)
    print(f"\n💾 Scanner analysis saved to results/scanner_performance.csv")


if __name__ == "__main__":
    run_scanner_analysis()
