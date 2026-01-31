"""
Chunked Optimization with Checkpoint/Resume
============================================
Runs optimization in small chunks, saves after each chunk, and can resume
from where it left off. Perfect for long-running optimizations.

Usage:
    python run_chunked_optimization.py           # Run next chunk
    python run_chunked_optimization.py --status  # Check progress
    python run_chunked_optimization.py --reset   # Start fresh
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from itertools import product

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

import config
from data_manager import get_daily_bars_batch, get_nikkei_225_components, fetch_jpx_short_data, get_available_symbols
from backtesting import run_daily_backtest

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================

CHUNK_SIZE = 20  # Number of strategies per chunk (increased for speed)
RESULTS_FILE = Path("results/optimization_checkpoint.json")
SUMMARY_FILE = Path("results/optimization_summary.csv")

# Time period for backtesting - CHANGE THESE AS NEEDED
# Full 2-year period: 2024-01-01 to today
START_DATE = "2024-01-01"  # Start from beginning of 2024
END_DATE = "2026-01-24"    # Today

# Initial capital for backtesting
INITIAL_BALANCE = 1_000_000  # JPY 1,000,000

# Parameter grid - Comprehensive coverage
# Total combinations: 2*2*3*3*3*2 = 216 combinations
PARAM_GRID = {
    'min_signals': [2, 3],           # Entry filter strictness
    'max_positions': [3, 5],          # Portfolio concentration
    'position_pct': [0.10, 0.15, 0.25],  # Position sizing (10%, 15%, 25%)
    'profit_target': [0.08, 0.12, 0.15], # Target 1 (8%, 12%, 15%)
    'stop_loss': [0.04, 0.06, 0.08],     # Stop loss (4%, 6%, 8%)
    'trailing_stop_pct': [0.03, 0.05],   # Trailing stop (3%, 5%)
}

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def get_param_hash(params: dict) -> str:
    """Create unique hash for parameter combination."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]

def load_checkpoint() -> dict:
    """Load checkpoint data."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": {}, "start_time": None, "total_combinations": 0}

def save_checkpoint(data: dict):
    """Save checkpoint data."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def export_summary(checkpoint: dict):
    """Export results summary to CSV."""
    if not checkpoint["completed"]:
        return
    
    rows = []
    for hash_id, result in checkpoint["completed"].items():
        row = {**result["params"], **result["metrics"]}
        row["hash"] = hash_id
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Only sort by profit_factor if column exists (not just errors)
    if "profit_factor" in df.columns:
        df = df.sort_values("profit_factor", ascending=False)
    
    df.to_csv(SUMMARY_FILE, index=False)
    print(f"\n📊 Summary exported to: {SUMMARY_FILE}")

def generate_all_combinations() -> list:
    """Generate all parameter combinations."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    
    combinations = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)
    
    return combinations

# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def run_single_backtest_fast(params: dict, precomputed) -> dict:
    """Run backtest using pre-computed data (5-10x faster)."""
    from backtesting import run_fast_backtest
    
    # Run fast backtest with parameters
    result = run_fast_backtest(
        precomputed=precomputed,
        initial_balance=INITIAL_BALANCE,
        max_positions=params['max_positions'],
        position_size_pct=params['position_pct'],
        stop_loss_pct=params['stop_loss'],
        target_1_pct=params['profit_target'],
        trailing_stop_pct=params['trailing_stop_pct'],
        progress=False,
    )
    
    if result is None:
        return {"error": "Backtest failed"}
    
    engine, metrics = result
    
    return {
        "total_return": metrics.get("total_return", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "win_rate": metrics.get("win_rate", 0),
        "max_drawdown": metrics.get("max_drawdown", 0),
        "total_trades": metrics.get("total_trades", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
    }

PRECOMPUTE_CACHE = Path("results/precomputed_cache.pkl")

def preload_data_fast():
    """Load pre-computed data from cache, or compute if not cached."""
    import pickle
    from precompute import precompute_all_data, load_precomputed
    
    # Check if cache exists and is valid
    if PRECOMPUTE_CACHE.exists():
        print("\n" + "="*60)
        print("📦 LOADING PRE-COMPUTED DATA FROM CACHE")
        print("="*60)
        
        try:
            precomputed = load_precomputed(PRECOMPUTE_CACHE)
            
            # Verify cache is for same date range
            if precomputed.start_date == START_DATE and precomputed.end_date == END_DATE:
                print(f"✓ Loaded cache: {precomputed.num_symbols} symbols × {precomputed.num_days} days")
                total_signals = sum(len(s) for s in precomputed.signals_by_date.values())
                print(f"✓ {total_signals:,} pre-computed signals")
                print("="*60)
                return precomputed
            else:
                print("⚠️ Cache date range mismatch, re-computing...")
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}, re-computing...")
    
    # Compute if no cache
    print("\n" + "="*60)
    print("🚀 PRE-COMPUTING ALL SIGNALS (one-time)")
    print("="*60)
    print("This takes ~15 min but will be CACHED for future runs!")
    print()
    
    precomputed = precompute_all_data(
        start_date=START_DATE,
        end_date=END_DATE,
        top_n=200,
        progress=True
    )
    
    # Save to cache
    print("\n💾 Saving to cache...")
    try:
        with open(PRECOMPUTE_CACHE, 'wb') as f:
            pickle.dump(precomputed, f)
        print(f"✓ Saved to {PRECOMPUTE_CACHE}")
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")
    
    print("="*60)
    
    return precomputed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def show_status():
    """Show current optimization status."""
    checkpoint = load_checkpoint()
    all_combos = generate_all_combinations()
    
    completed = len(checkpoint["completed"])
    total = len(all_combos)
    remaining = total - completed
    
    print("\n" + "="*60)
    print("OPTIMIZATION STATUS")
    print("="*60)
    print(f"Completed: {completed}/{total} ({100*completed/total:.1f}%)")
    print(f"Remaining: {remaining}")
    
    if checkpoint["start_time"]:
        print(f"Started: {checkpoint['start_time']}")
    
    if completed > 0:
        # Show top 5 results
        print("\nTop 5 Results So Far:")
        results = [(h, r) for h, r in checkpoint["completed"].items()]
        results.sort(key=lambda x: x[1]["metrics"].get("profit_factor", 0), reverse=True)
        
        for hash_id, result in results[:5]:
            pf = result["metrics"].get("profit_factor", 0)
            wr = result["metrics"].get("win_rate", 0) * 100
            trades = result["metrics"].get("total_trades", 0)
            print(f"  PF={pf:.2f}, WR={wr:.1f}%, Trades={trades}")
    
    print("="*60)

def reset_optimization():
    """Reset and start fresh."""
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()
    print("✓ Optimization reset. Run again to start fresh.")

def run_next_chunk():
    """Run the next chunk of optimizations."""
    checkpoint = load_checkpoint()
    all_combos = generate_all_combinations()
    
    # Find remaining combinations
    remaining = []
    for params in all_combos:
        hash_id = get_param_hash(params)
        if hash_id not in checkpoint["completed"]:
            remaining.append((hash_id, params))
    
    if not remaining:
        print("\n✅ ALL COMBINATIONS COMPLETED!")
        export_summary(checkpoint)
        show_status()
        return
    
    # Initialize start time if first run
    if not checkpoint["start_time"]:
        checkpoint["start_time"] = datetime.now().isoformat()
        checkpoint["total_combinations"] = len(all_combos)
    
    # Take next chunk
    chunk = remaining[:CHUNK_SIZE]
    completed_count = len(checkpoint["completed"])
    total = len(all_combos)
    
    print("\n" + "="*60)
    print("CHUNKED OPTIMIZATION (FAST MODE)")
    print("="*60)
    print(f"Progress: {completed_count}/{total} completed")
    print(f"Running chunk: {len(chunk)} combinations")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("="*60)
    
    # Pre-compute all indicators ONCE (major speedup!)
    precomputed = preload_data_fast()
    
    # Calculate average time per combo from previous runs
    elapsed_times = []
    for result in checkpoint["completed"].values():
        if "elapsed_seconds" in result and "error" not in result.get("metrics", {}):
            elapsed_times.append(result["elapsed_seconds"])
    
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else None
    
    if avg_time:
        chunk_eta = avg_time * len(chunk)
        total_eta = avg_time * (total - completed_count)
        print(f"\n⏱️  Estimated times:")
        print(f"   This chunk ({len(chunk)} combos): ~{chunk_eta/60:.0f} min")
        print(f"   Total remaining ({total - completed_count} combos): ~{total_eta/60:.0f} min ({total_eta/3600:.1f} hours)")
    
    print(f"\n🚀 Running chunk ({len(chunk)} combinations)...")
    
    chunk_start = time.time()
    
    for i, (hash_id, params) in enumerate(chunk):
        progress = completed_count + i + 1
        
        # Progress bar
        pct = (progress / total) * 100
        bar_len = 30
        filled = int(bar_len * progress / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\n[{bar}] {pct:.1f}% ({progress}/{total})")
        print(f"Testing: {params}")
        
        start = time.time()
        try:
            metrics = run_single_backtest_fast(params, precomputed)
            elapsed = time.time() - start
            
            # Save result
            checkpoint["completed"][hash_id] = {
                "params": params,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed
            }
            
            # Save checkpoint after each combination
            save_checkpoint(checkpoint)
            
            pf = metrics.get("profit_factor", 0)
            wr = metrics.get("win_rate", 0) * 100
            trades = metrics.get("total_trades", 0)
            print(f"    ✓ PF={pf:.2f}, WR={wr:.1f}%, Trades={trades} ({elapsed/60:.1f} min)")
            
            # Update ETA estimate
            remaining_in_chunk = len(chunk) - (i + 1)
            if remaining_in_chunk > 0:
                eta_chunk = remaining_in_chunk * elapsed
                print(f"    ⏱️  ~{eta_chunk/60:.0f} min remaining in this chunk")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            elapsed = time.time() - start
            # Save error
            checkpoint["completed"][hash_id] = {
                "params": params,
                "metrics": {"error": str(e)},
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed
            }
            save_checkpoint(checkpoint)
    
    # Export summary after chunk
    export_summary(checkpoint)
    
    # Show status
    new_completed = len(checkpoint["completed"])
    new_remaining = total - new_completed
    
    print("\n" + "="*60)
    print(f"✅ CHUNK COMPLETE!")
    print(f"Progress: {new_completed}/{total} ({100*new_completed/total:.1f}%)")
    print(f"Remaining: {new_remaining}")
    
    if new_remaining > 0:
        print(f"\n💡 Run again to continue: python run_chunked_optimization.py")
    else:
        print(f"\n🎉 ALL DONE! Check {SUMMARY_FILE} for results.")
    print("="*60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--reset" in sys.argv:
        reset_optimization()
    else:
        run_next_chunk()
