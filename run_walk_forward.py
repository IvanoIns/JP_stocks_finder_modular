"""
Walk-Forward Optimization
=========================
Proper validation using the existing optimizer.py walk-forward analysis.

This replaces run_chunked_optimization.py which was incorrectly doing
in-sample optimization only (training and testing on the same data).

Walk-forward process:
1. Divide data into rolling train/test windows
2. For each window: optimize on TRAIN, validate on TEST (unseen)
3. Find params that work consistently across multiple windows
4. Those stable params are most likely to work in live trading

Usage:
    python run_walk_forward.py           # Run walk-forward analysis
    python run_walk_forward.py --quick   # Quick test (fewer windows)
"""

import sys
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config
from optimizer import (
    generate_windows,
    walk_forward_grid_search,
    summarize_oos,
    top_params_by_stability,
    save_best_params,
)
from precompute import precompute_all_data, load_precomputed

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range for walk-forward analysis
START_DATE = "2024-01-01"
END_DATE = "2026-01-24"

# Walk-forward window settings
TRAIN_DAYS = 180  # 6 months training
TEST_DAYS = 90    # 3 months testing (out-of-sample)
STEP_DAYS = 60    # 2 months step (overlapping windows for more data points)

# Initial capital
INITIAL_BALANCE = 1_000_000  # JPY

# === PHASE 3 REFINEMENT GRID ===
# Targeting the "Middle Ground" between Safe (Phase 1) and Risky (Phase 2)
PARAM_GRID = {
    'rsi_max_values': [70],          
    'volume_surge_values': [1.3],    
    
    # Testing the sweet spot between noise (15) and strict (30)
    'min_score_values': [20, 25, 30], 
    
    # Testing if higher R:R pays off with wider stops
    'risk_reward_ratios': [2.0, 2.5, 3.0],
    
    # Testing wider stops (up to 12%) to let winners run
    'stop_loss_values': [0.06, 0.08, 0.10, 0.12],
    
    'exit_modes': ['fixed_rr']       
}

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
MIN_WIN_RATE = 0.25  # Lowered to 25% because at 4:1 R:R, you only need 20% to breakeven
MIN_TRADES = 10        # Minimum trades per window
MAX_DRAWDOWN = 0.30    # Maximum 30% drawdown

# Stability threshold
MIN_SELECTION_COUNT = 2  # Params must be selected in at least 2 windows

PRECOMPUTE_CACHE = Path("results/precomputed_cache.pkl")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_precomputed_data():
    """Load or generate precomputed data for 100x speedup."""
    if PRECOMPUTE_CACHE.exists():
        print("\n📦 Loading cached pre-computed data...")
        try:
            precomputed = load_precomputed(PRECOMPUTE_CACHE)
            # Basic validation
            if precomputed.start_date == START_DATE and precomputed.end_date == END_DATE:
                print(f"   ✓ Loaded {precomputed.num_symbols} symbols, {precomputed.num_days} days")
                return precomputed
        except Exception as e:
            print(f"   ⚠️ Cache load failed: {e}")
            print("   Re-computing...")
    
    print("\n🚀 Pre-computing signals (one-time setup)...")
    precomputed = precompute_all_data(
        start_date=START_DATE,
        end_date=END_DATE,
        top_n=config.UNIVERSE_TOP_N if hasattr(config, 'UNIVERSE_TOP_N') else 200,
        progress=True
    )
    
    # Save cache
    PRECOMPUTE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRECOMPUTE_CACHE, 'wb') as f:
        pickle.dump(precomputed, f)
    print(f"   💾 Saved cache to {PRECOMPUTE_CACHE}")
    
    return precomputed

# ============================================================================
# MAIN
# ============================================================================

def run_walk_forward(quick_mode: bool = False):
    """Run walk-forward analysis with proper OOS validation."""
    
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION (PROPER OOS VALIDATION)")
    print("="*70)
    
    # Load data first (FAST ENGINE)
    precomputed = get_precomputed_data()
    
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Train: {TRAIN_DAYS} days, Test: {TEST_DAYS} days, Step: {STEP_DAYS} days")
    print(f"Transaction costs: {config.BACKTEST_SLIPPAGE_PCT*100:.1f}% slippage + {config.BACKTEST_COMMISSION_PCT*100:.1f}% commission")
    print("="*70)
    
    # Reduce windows for quick mode
    train_days = 90 if quick_mode else TRAIN_DAYS
    test_days = 45 if quick_mode else TEST_DAYS
    step_days = 45 if quick_mode else STEP_DAYS
    
    # Generate rolling windows
    print("\n[1/4] Generating rolling train/test windows...")
    windows = generate_windows(
        start_date=START_DATE,
        end_date=END_DATE,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    
    print(f"    Generated {len(windows)} windows:")
    for i, (ts, te, vs, ve) in enumerate(windows[:3]):
        print(f"    Window {i+1}: Train {ts} to {te} | Test {vs} to {ve}")
    if len(windows) > 3:
        print(f"    ... and {len(windows) - 3} more")
    
    # Calculate total combinations
    total_combos = 1
    for vals in PARAM_GRID.values():
        total_combos *= len(vals)
    print(f"\n    Parameter combinations: {total_combos}")
    print(f"    Total backtests: {total_combos * len(windows)} (train) + {len(windows)} (test)")
    
    # Run walk-forward analysis
    print("\n[2/4] Running walk-forward grid search...")
    print("    This utilizes the ULTRA-FAST pre-computed engine.")
    print()
    
    start_time = datetime.now()
    
    wf_results = walk_forward_grid_search(
        windows=windows,
        initial_balance=INITIAL_BALANCE,
        rsi_max_values=PARAM_GRID['rsi_max_values'],
        volume_surge_values=PARAM_GRID['volume_surge_values'],
        min_score_values=PARAM_GRID['min_score_values'],
        stop_loss_values=PARAM_GRID['stop_loss_values'],
        exit_modes=PARAM_GRID['exit_modes'],
        risk_reward_ratios=PARAM_GRID['risk_reward_ratios'],
        top_n=config.UNIVERSE_TOP_N,
        max_positions=config.MAX_POSITIONS,
        min_trades=MIN_TRADES,
        min_win_rate=MIN_WIN_RATE,
        max_drawdown_cap=MAX_DRAWDOWN,
        min_test_trades=5,  # Need at least 5 test trades
        progress=True,
        precomputed=precomputed,  # PASS FAST DATA ENGINE
    )
    
    elapsed = datetime.now() - start_time
    print(f"\n    Completed in {elapsed}")
    
    # Summarize OOS results
    print("\n[3/4] Analyzing out-of-sample results...")
    
    # Convert to DataFrame if list (handle older pandas versions or list return)
    if isinstance(wf_results, list):
         wf_results = pd.DataFrame(wf_results)

    if len(wf_results) == 0:
        print("    ❌ No results! Check logs for errors.")
        return
    
    summary = summarize_oos(wf_results, min_test_trades=5)
    oos = summary['oos_summary']
    
    if not oos:
        print("    ❌ No valid OOS results with sufficient trades.")
        return
    
    print(f"""
    OUT-OF-SAMPLE PERFORMANCE (what matters!):
    ─────────────────────────────────────────
    Median Profit Factor: {oos['median_pf']:.2f}
    Mean Profit Factor:   {oos['mean_pf']:.2f}
    Median Win Rate:      {oos['median_wr']*100:.1f}%
    Median Max Drawdown:  {oos['median_dd']*100:.1f}%
    Total OOS Trades:     {oos['total_oos_trades']}
    Valid Windows:        {oos['windows_ok']}/{oos['windows_total']}
    """)
    
    # Find stable parameters
    print("[4/4] Finding stable parameters (selected in multiple windows)...")
    
    stable = top_params_by_stability(
        wf_results,
        min_test_trades=5,
        min_count=MIN_SELECTION_COUNT,
    )
    
    if len(stable) == 0:
        print("    ⚠️ No stable parameters found!")
        print("    This suggests the strategy may not be robust.")
        print("    Consider: simpler parameters, longer train periods, or different scanners.")
    else:
        print(f"\n    STABLE PARAMETERS (selected in {MIN_SELECTION_COUNT}+ windows):")
        print("    " + "─"*65)
        
        for i, row in stable.head(5).iterrows():
            print(f"""
    Rank {i+1}:
      Params: RSI Max={row['pick_rsi_max']}, Vol Surge={row['pick_vol_surge']}, 
              Min Score={row['pick_min_score']}, Stop Loss={row['pick_stop_loss']*100:.0f}%, 
              Exit={row['pick_exit_mode']}
      Selected in: {int(row['selection_count'])} windows
      Avg OOS PF:  {row['avg_test_pf']:.2f}
      Avg OOS WR:  {row['avg_test_wr']*100:.1f}%
            """)
        
        # Save best stable params
        if len(stable) > 0:
            best = stable.iloc[0]
            best_params = {
                'rsi_max': int(best['pick_rsi_max']),
                'volume_surge': float(best['pick_vol_surge']),
                'min_score': int(best['pick_min_score']),
                'stop_loss': float(best['pick_stop_loss']),
                'exit_mode': best['pick_exit_mode'],
                'risk_reward_ratio': float(best.get('pick_risk_reward', 2.0)),
                'selection_count': int(best['selection_count']),
                'avg_oos_pf': float(best['avg_test_pf']),
            }
            save_best_params(best_params)
            print(f"    ✓ Best stable params saved to results/best_params.json")
    
    # Export full results
    results_file = Path("results/walk_forward_results.csv")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    wf_results.to_csv(results_file, index=False)
    print(f"\n    ✓ Full results saved to {results_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("WALK-FORWARD COMPLETE")
    print("="*70)
    
    if len(stable) > 0 and stable.iloc[0]['avg_test_pf'] > 1.2:
        print("✅ Found stable parameters with OOS edge!")
        print("   These params worked across multiple market periods.")
        print("   Consider paper trading before live deployment.")
    elif len(stable) > 0:
        print("⚠️ Found stable parameters but weak OOS edge.")
        print("   Profit factor < 1.2 on unseen data is concerning.")
        print("   Consider refining scanners or entry criteria.")
    else:
        print("❌ No robust strategy found.")
        print("   The strategy may be overfit to historical quirks.")
        print("   Go back to strategy design before optimization.")
    
    print("="*70)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        print("Running in QUICK mode (shorter windows for testing)")
    run_walk_forward(quick_mode=quick)
