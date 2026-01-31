"""
Full Optimization Script — 2024-2025 Data

This script runs:
1. Extended Grid Search with many parameter combinations
2. Walk-Forward Analysis for robustness
3. Stability ranking to find best parameters

Run this file to optimize your strategy.
"""

import sys
import pandas as pd
from datetime import datetime

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================

# Date range (use latest available data)
START_DATE = "2024-01-01"
END_DATE = "2026-01-23"  # Today or latest data

# Initial balance
INITIAL_BALANCE = 1_000_000

# Grid search parameter ranges (comprehensive)
PARAM_GRID = {
    'rsi_max_values': [60, 65, 70, 75, 80],
    'volume_surge_values': [1.2, 1.5, 2.0, 2.5],
    'min_score_values': [40, 50, 60, 70],
    'stop_loss_values': [0.03, 0.04, 0.05, 0.06, 0.07],
    'exit_modes': ['default', 'trailing', 'breakeven'],
}

# Walk-forward settings
WFA_SETTINGS = {
    'train_days': 90,   # 3 months training
    'test_days': 30,    # 1 month testing
    'step_days': 21,    # 3 weeks step
}

# Constraints
CONSTRAINTS = {
    'min_trades': 15,       # Lowered for shorter periods
    'min_win_rate': 0.40,
    'max_drawdown_cap': 0.30,
    'min_test_trades': 5,
}

# =============================================================================
# RUN OPTIMIZATION
# =============================================================================

def main():
    print("=" * 70)
    print("JP STOCKS FINDER — FULL PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Data Period: {START_DATE} to {END_DATE}")
    print(f"Initial Balance: JPY {INITIAL_BALANCE:,}")
    print("=" * 70)
    
    # Count combinations
    total_combos = 1
    for key, values in PARAM_GRID.items():
        total_combos *= len(values)
    print(f"\nGrid Search: {total_combos} parameter combinations")
    print(f"Parameters:")
    for key, values in PARAM_GRID.items():
        print(f"  {key}: {values}")
    print()
    
    from optimizer import grid_search_daily, walk_forward_grid_search, generate_windows
    from optimizer import summarize_oos, top_params_by_stability
    
    # =========================================================================
    # STEP 1: GRID SEARCH
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: GRID SEARCH")
    print("=" * 70)
    print("Finding best parameters across full period...")
    print("This may take a while...\n")
    
    grid_results = grid_search_daily(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_balance=INITIAL_BALANCE,
        rsi_max_values=PARAM_GRID['rsi_max_values'],
        volume_surge_values=PARAM_GRID['volume_surge_values'],
        min_score_values=PARAM_GRID['min_score_values'],
        stop_loss_values=PARAM_GRID['stop_loss_values'],
        exit_modes=PARAM_GRID['exit_modes'],
        min_trades=CONSTRAINTS['min_trades'],
        min_win_rate=CONSTRAINTS['min_win_rate'],
        max_drawdown_cap=CONSTRAINTS['max_drawdown_cap'],
        progress=True,
    )
    
    # Show top results
    print("\n" + "-" * 70)
    print("TOP 10 PARAMETER COMBINATIONS (by Profit Factor)")
    print("-" * 70)
    
    if len(grid_results) > 0:
        passed = grid_results[grid_results['passed_constraints'] == True]
        top10 = passed.head(10) if len(passed) >= 10 else passed
        
        print(top10.to_string(index=False))
        
        # Save full results
        grid_results.to_csv("results/grid_search_results.csv", index=False)
        print(f"\nFull results saved to: results/grid_search_results.csv")
    else:
        print("No results found!")
        return
    
    # =========================================================================
    # STEP 2: WALK-FORWARD ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: WALK-FORWARD ANALYSIS")
    print("=" * 70)
    
    # Generate windows
    windows = generate_windows(
        START_DATE, END_DATE,
        train_days=WFA_SETTINGS['train_days'],
        test_days=WFA_SETTINGS['test_days'],
        step_days=WFA_SETTINGS['step_days'],
    )
    
    print(f"Rolling windows: {len(windows)}")
    print(f"  Train: {WFA_SETTINGS['train_days']} days")
    print(f"  Test: {WFA_SETTINGS['test_days']} days")
    print(f"  Step: {WFA_SETTINGS['step_days']} days\n")
    
    wfa_results = walk_forward_grid_search(
        windows=windows,
        initial_balance=INITIAL_BALANCE,
        rsi_max_values=PARAM_GRID['rsi_max_values'],
        volume_surge_values=PARAM_GRID['volume_surge_values'],
        min_score_values=PARAM_GRID['min_score_values'],
        stop_loss_values=PARAM_GRID['stop_loss_values'],
        exit_modes=PARAM_GRID['exit_modes'],
        min_trades=CONSTRAINTS['min_trades'],
        min_win_rate=CONSTRAINTS['min_win_rate'],
        max_drawdown_cap=CONSTRAINTS['max_drawdown_cap'],
        min_test_trades=CONSTRAINTS['min_test_trades'],
        progress=True,
    )
    
    # =========================================================================
    # STEP 3: SUMMARIZE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: OUT-OF-SAMPLE SUMMARY")
    print("=" * 70)
    
    if len(wfa_results) > 0:
        oos = summarize_oos(wfa_results, min_test_trades=CONSTRAINTS['min_test_trades'])
        
        print("\nOOS Performance:")
        for key, value in oos['oos_summary'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Save WFA results
        wfa_results.to_csv("results/wfa_results.csv", index=False)
        print(f"\nWFA results saved to: results/wfa_results.csv")
        
        # Stability ranking
        print("\n" + "-" * 70)
        print("STABLE PARAMETER COMBINATIONS")
        print("(Selected across multiple walk-forward windows)")
        print("-" * 70)
        
        stable = top_params_by_stability(wfa_results, min_count=2)
        if len(stable) > 0:
            print(stable.to_string(index=False))
            stable.to_csv("results/stable_params.csv", index=False)
            print(f"\nStable params saved to: results/stable_params.csv")
        else:
            print("No stable parameters found (none selected multiple times)")
    else:
        print("No WFA results!")
    
    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)
    
    if len(grid_results) > 0:
        best = grid_results[grid_results['passed_constraints'] == True].iloc[0] if len(passed) > 0 else grid_results.iloc[0]
        
        print("\nBest parameters from grid search:")
        print(f"  RSI Max: {int(best['rsi_max'])}")
        print(f"  Volume Surge: {best['volume_surge']}")
        print(f"  Min Score: {int(best['min_score'])}")
        print(f"  Stop Loss: {best['stop_loss']:.1%}")
        print(f"  Exit Mode: {best['exit_mode']}")
        print()
        print(f"  Expected PF: {best['profit_factor']:.2f}")
        print(f"  Expected WR: {best['win_rate']:.1%}")
        print(f"  Expected DD: {best['max_drawdown']:.1%}")
        
        # Update config.py recommendation
        print("\n" + "-" * 70)
        print("To use these settings, update config.py:")
        print("-" * 70)
        print(f"MIN_SCANNER_SCORE = {int(best['min_score'])}")
        print(f"STOP_LOSS_PCT = {best['stop_loss']}")
        print(f"EXIT_MODE = '{best['exit_mode']}'")
        print(f"SCANNER_CONFIG['MOMENTUM_RSI_MAX'] = {int(best['rsi_max'])}")
        print(f"SCANNER_CONFIG['MOMENTUM_VOLUME_SPIKE_FACTOR'] = {best['volume_surge']}")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  - results/grid_search_results.csv")
    print("  - results/wfa_results.csv")
    print("  - results/stable_params.csv")
    
    print("\nPress Enter to exit...")
    input()


if __name__ == "__main__":
    main()
