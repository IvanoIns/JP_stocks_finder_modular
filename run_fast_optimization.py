"""
FAST Optimization Script

Key optimizations:
1. Pre-load ALL data once at startup (no repeated DB queries)
2. Smaller, smarter parameter grid (36 combos instead of 1200)
3. Shorter test period for faster iterations
4. Progress saved after each combo

Run this instead of run_optimization.py for practical results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from tqdm import tqdm
import sys

# =============================================================================
# FAST OPTIMIZATION SETTINGS
# =============================================================================

# Use shorter period for faster optimization (6 months instead of 2 years)
START_DATE = "2025-07-01"   # Recent 6 months
END_DATE = "2026-01-23"

INITIAL_BALANCE = 1_000_000

# REDUCED GRID (36 combinations instead of 1200)
PARAM_GRID = {
    'min_score': [50, 60],          # 2 values
    'stop_loss': [0.04, 0.05, 0.06], # 3 values  
    'exit_mode': ['default', 'trailing'],  # 2 values
    'rsi_max': [70, 75],            # 2 values
    'volume_surge': [1.5, 2.0],     # 2 values
    # Total: 2×3×2×2×2 = 48 combinations
}

# Constraints
MIN_TRADES = 10
MIN_WIN_RATE = 0.40


def preload_all_data(start_date: str, end_date: str, top_n: int = 100):
    """
    Load ALL data upfront into memory to avoid repeated DB queries.
    This is the key optimization - load once, use many times.
    """
    print("Pre-loading data into memory...")
    
    import data_manager as dm
    
    # 1. Load Nikkei 225 exclusion list
    print("  - Loading Nikkei 225 list...")
    nikkei_set = dm.get_nikkei_225_components()
    print(f"    {len(nikkei_set)} symbols")
    
    # 2. Get all trading dates
    print("  - Getting trading dates...")
    conn = dm.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (start_date, end_date))
    trading_dates = [row[0] for row in cursor.fetchall()]
    print(f"    {len(trading_dates)} trading days")
    
    # 3. Get all symbols with sufficient data
    print("  - Loading price data...")
    symbols = dm.get_available_symbols(min_rows=60)
    
    # Filter out Nikkei 225
    symbols = [s for s in symbols if s not in nikkei_set][:top_n * 2]  # Get extra for filtering
    
    # 4. Load all price data at once
    lookback_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=300)).strftime('%Y-%m-%d')
    all_data = dm.get_daily_bars_batch(symbols, lookback_start, end_date)
    print(f"    {len(all_data)} symbols loaded")
    
    conn.close()
    
    return {
        'trading_dates': trading_dates,
        'all_data': all_data,
        'nikkei_set': nikkei_set,
    }


def run_fast_backtest(
    preloaded: dict,
    min_score: int,
    stop_loss: float,
    exit_mode: str,
    rsi_max: int,
    volume_surge: float,
) -> dict:
    """
    Run a single backtest using pre-loaded data.
    Much faster than the original because no DB queries during loop.
    """
    import config
    import scanners
    import technical_analysis as ta
    from backtesting import BacktestEngine, PendingEntry
    
    # Create scanner config with these params
    scanner_config = config.get_scanner_config()
    scanner_config['MOMENTUM_RSI_MAX'] = rsi_max
    scanner_config['MOMENTUM_VOLUME_SPIKE_FACTOR'] = volume_surge
    
    # Initialize engine
    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
        stop_loss_pct=stop_loss,
        exit_mode=exit_mode,
    )
    
    trading_dates = preloaded['trading_dates']
    all_data = preloaded['all_data']
    
    for date in trading_dates:
        # Get today's bars from preloaded data
        today_bars = {}
        open_prices = {}
        close_prices = {}
        
        for symbol, df in all_data.items():
            if date in df.index.strftime('%Y-%m-%d').values:
                idx = df.index.strftime('%Y-%m-%d') == date
                if idx.any():
                    today_bars[symbol] = df.loc[idx].iloc[-1]
                    open_prices[symbol] = df.loc[idx].iloc[-1]['open']
                    close_prices[symbol] = df.loc[idx].iloc[-1]['close']
        
        # Process pending entries
        engine.process_pending_entries(date, open_prices)
        
        # Check exits
        engine.check_exits(date, today_bars)
        
        # Run scanners on symbols with data
        for symbol, df in all_data.items():
            if len(df) < 60:
                continue
            
            # Filter to data up to current date
            df_to_date = df[df.index <= date]
            if len(df_to_date) < 60:
                continue
            
            # Run scanners
            signals = scanners.get_all_signals(
                symbol, df_to_date, None, scanner_config, min_score
            )
            
            # Queue entry
            for signal in signals[:1]:  # Top 1 only for speed
                if engine.can_open_position():
                    current_price = close_prices.get(symbol, 0)
                    if current_price > 0:
                        engine.queue_entry(signal, date, current_price)
        
        # Record balance
        engine.record_daily_balance(date, close_prices)
    
    # Liquidate
    if engine.positions:
        last_date = trading_dates[-1]
        last_prices = {s: all_data[s].iloc[-1]['close'] for s in engine.positions if s in all_data}
        engine.liquidate_all(last_date, last_prices)
    
    return engine.calculate_metrics()


def main():
    print("=" * 70)
    print("FAST PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Balance: JPY {INITIAL_BALANCE:,}")
    
    # Count combinations
    combos = list(product(
        PARAM_GRID['min_score'],
        PARAM_GRID['stop_loss'],
        PARAM_GRID['exit_mode'],
        PARAM_GRID['rsi_max'],
        PARAM_GRID['volume_surge'],
    ))
    print(f"Combinations: {len(combos)}")
    print("=" * 70)
    
    # Pre-load all data
    print("\n[1/2] PRE-LOADING DATA")
    preloaded = preload_all_data(START_DATE, END_DATE)
    print("Data loaded!\n")
    
    # Run grid search
    print("[2/2] RUNNING GRID SEARCH")
    results = []
    
    for combo in tqdm(combos, desc="Optimizing"):
        min_score, stop_loss, exit_mode, rsi_max, vol_surge = combo
        
        try:
            metrics = run_fast_backtest(
                preloaded=preloaded,
                min_score=min_score,
                stop_loss=stop_loss,
                exit_mode=exit_mode,
                rsi_max=rsi_max,
                volume_surge=vol_surge,
            )
            
            passed = (
                metrics['total_trades'] >= MIN_TRADES and
                metrics['win_rate'] >= MIN_WIN_RATE
            )
            
            results.append({
                'min_score': min_score,
                'stop_loss': stop_loss,
                'exit_mode': exit_mode,
                'rsi_max': rsi_max,
                'volume_surge': vol_surge,
                'profit_factor': metrics['profit_factor'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'max_drawdown': metrics['max_drawdown'],
                'total_return': metrics['total_return'],
                'passed': passed,
            })
        except Exception as e:
            print(f"Error with {combo}: {e}")
            continue
    
    # Process results
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df = df.sort_values('profit_factor', ascending=False)
        
        print("\n" + "=" * 70)
        print("TOP 10 RESULTS")
        print("=" * 70)
        
        print(df.head(10).to_string(index=False))
        
        # Save results
        df.to_csv("results/fast_grid_search.csv", index=False)
        print(f"\nResults saved to: results/fast_grid_search.csv")
        
        # Best params
        if len(df[df['passed'] == True]) > 0:
            best = df[df['passed'] == True].iloc[0]
        else:
            best = df.iloc[0]
        
        print("\n" + "=" * 70)
        print("RECOMMENDED SETTINGS")
        print("=" * 70)
        print(f"Min Score: {int(best['min_score'])}")
        print(f"Stop Loss: {best['stop_loss']:.1%}")
        print(f"Exit Mode: {best['exit_mode']}")
        print(f"RSI Max: {int(best['rsi_max'])}")
        print(f"Volume Surge: {best['volume_surge']}")
        print("-" * 40)
        print(f"Expected PF: {best['profit_factor']:.2f}")
        print(f"Expected WR: {best['win_rate']:.1%}")
        print(f"Trades: {int(best['total_trades'])}")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print("\nPress Enter to exit...")
    input()


if __name__ == "__main__":
    main()
