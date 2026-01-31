"""
JP Stocks Modular Trading System — Optimizer

Grid search, walk-forward analysis, and stability ranking.
"""

import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import json
import logging

import config
import data_manager as dm
from backtesting import run_daily_backtest, run_fast_backtest, BacktestEngine

logger = logging.getLogger(__name__)


# =============================================================================
# Grid Search
# =============================================================================

def _is_precomputed_compatible(
    precomputed,
    rsi_max_values: List[int],
    volume_surge_values: List[float],
    min_score_values: List[int],
) -> bool:
    if precomputed is None:
        return True

    precomputed_scanner_config = getattr(precomputed, "scanner_config", None)
    precomputed_min_score = getattr(precomputed, "min_score", None)

    if not precomputed_scanner_config or precomputed_min_score is None:
        logger.warning("Precomputed cache missing metadata; disabling fast mode for correctness.")
        return False

    unique_rsi = set(rsi_max_values) if rsi_max_values else set()
    unique_vol = set(volume_surge_values) if volume_surge_values else set()
    unique_min_score = set(min_score_values) if min_score_values else set()

    if len(unique_rsi) != 1 or len(unique_vol) != 1 or len(unique_min_score) != 1:
        logger.warning("Parameter grid varies scanner/min_score; disabling fast mode for correctness.")
        return False

    rsi_max = unique_rsi.pop()
    vol_surge = unique_vol.pop()
    min_score = unique_min_score.pop()

    if precomputed_scanner_config.get('MOMENTUM_RSI_MAX') != rsi_max:
        logger.warning("Precomputed cache uses different MOMENTUM_RSI_MAX; disabling fast mode.")
        return False
    if precomputed_scanner_config.get('MOMENTUM_VOLUME_SPIKE_FACTOR') != vol_surge:
        logger.warning("Precomputed cache uses different MOMENTUM_VOLUME_SPIKE_FACTOR; disabling fast mode.")
        return False
    if precomputed_min_score != min_score:
        logger.warning("Precomputed cache uses different min_score; disabling fast mode.")
        return False

    return True


def grid_search_daily(
    start_date: str,
    end_date: str,
    initial_balance: float,
    # Parameter grids
    rsi_max_values: List[int] = None,
    volume_surge_values: List[float] = None,
    min_score_values: List[int] = None,
    stop_loss_values: List[float] = None,
    exit_modes: List[str] = None,
    risk_reward_ratios: List[float] = None,  # NEW: R:R ratios
    # Fixed params
    top_n: int = None,
    max_positions: int = None,
    # Constraints
    min_trades: int = None,
    min_win_rate: float = None,
    max_drawdown_cap: float = None,
    # Options
    progress: bool = True,
    precomputed = None,  # PrecomputedData object (optional)
) -> pd.DataFrame:
    """
    Run grid search over parameter combinations.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_balance: Starting capital
        rsi_max_values: RSI max threshold values to test
        volume_surge_values: Volume surge threshold values
        min_score_values: Minimum scanner score values
        stop_loss_values: Stop loss percentages
        exit_modes: Exit mode strategies
        top_n: Universe size
        max_positions: Max concurrent positions
        min_trades: Minimum trades constraint
        min_win_rate: Minimum win rate constraint
        max_drawdown_cap: Maximum drawdown constraint
        progress: Show progress bar
        precomputed: Optional PrecomputedData for 10-100x speedup
    
    Returns:
        DataFrame with all combinations and their metrics
    """
    # Set defaults
    if rsi_max_values is None:
        rsi_max_values = [65, 70, 75]
    if volume_surge_values is None:
        volume_surge_values = [1.3, 1.5, 2.0]
    if min_score_values is None:
        min_score_values = [50, 60, 70]
    if stop_loss_values is None:
        stop_loss_values = [0.04, 0.05, 0.06]
    if exit_modes is None:
        exit_modes = ['default', 'trailing']
    if risk_reward_ratios is None:
        risk_reward_ratios = [2.0]  # Default 2:1 R:R
    if top_n is None:
        top_n = config.UNIVERSE_TOP_N
    if max_positions is None:
        max_positions = config.MAX_POSITIONS
    if min_trades is None:
        min_trades = config.MIN_TRADES
    if min_win_rate is None:
        min_win_rate = config.MIN_WIN_RATE
    if max_drawdown_cap is None:
        max_drawdown_cap = config.MAX_DRAWDOWN_CAP

    if precomputed and not _is_precomputed_compatible(
        precomputed, rsi_max_values, volume_surge_values, min_score_values
    ):
        precomputed = None
    
    # Generate all combinations
    combinations = list(product(
        rsi_max_values,
        volume_surge_values,
        min_score_values,
        stop_loss_values,
        exit_modes,
        risk_reward_ratios,  # NEW
    ))
    
    logger.info(f"Grid search: {len(combinations)} combinations")
    
    results = []
    iterator = combinations
    if progress:
        iterator = tqdm(combinations, desc="Grid search")
    
    for rsi_max, vol_surge, min_score, stop_loss, exit_mode, rr_ratio in iterator:
        # Create scanner config for this combo
        scanner_config = config.get_scanner_config()
        scanner_config['MOMENTUM_RSI_MAX'] = rsi_max
        scanner_config['MOMENTUM_VOLUME_SPIKE_FACTOR'] = vol_surge
        
        try:
            # Run backtest (FAST or SLOW)
            if precomputed:
                # Fast mode (RAM-based)
                engine, metrics = run_fast_backtest(
                    precomputed=precomputed,
                    initial_balance=initial_balance,
                    max_positions=max_positions,
                    stop_loss_pct=stop_loss,
                    exit_mode=exit_mode,
                    min_score=min_score,
                    scanner_config=scanner_config,
                    progress=False,
                    start_date=start_date,
                    end_date=end_date,
                    risk_reward_ratio=rr_ratio,  # NEW
                )
            else:
                # Slow mode (DB-based)
                engine, metrics = run_daily_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    initial_balance=initial_balance,
                    top_n=top_n,
                    max_positions=max_positions,
                    stop_loss_pct=stop_loss,
                    exit_mode=exit_mode,
                    min_score=min_score,
                    scanner_config=scanner_config,
                    progress=False,
                    risk_reward_ratio=rr_ratio,  # NEW
                )

            
            # Check constraints
            passed = (
                metrics['total_trades'] >= min_trades and
                metrics['win_rate'] >= min_win_rate and
                metrics['max_drawdown'] <= max_drawdown_cap
            )
            
            results.append({
                'rsi_max': rsi_max,
                'volume_surge': vol_surge,
                'min_score': min_score,
                'stop_loss': stop_loss,
                'exit_mode': exit_mode,
                'risk_reward_ratio': rr_ratio,  # NEW
                'profit_factor': metrics['profit_factor'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'max_drawdown': metrics['max_drawdown'],
                'total_return': metrics['total_return'],
                'passed_constraints': passed,
            })
            
        except Exception as e:
            logger.warning(f"Error in grid search combo: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Sort by profit factor (descending), filtered by constraints
        df = df.sort_values('profit_factor', ascending=False)
        passed_count = df['passed_constraints'].sum()
        logger.info(f"Grid search complete: {len(df)} results, {passed_count} passed constraints")
    
    return df


# =============================================================================
# Walk-Forward Analysis
# =============================================================================

def generate_windows(
    start_date: str,
    end_date: str,
    train_days: int = None,
    test_days: int = None,
    step_days: int = None
) -> List[Tuple[str, str, str, str]]:
    """
    Generate rolling train/test windows.
    
    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    if train_days is None:
        train_days = config.WFA_TRAIN_DAYS
    if test_days is None:
        test_days = config.WFA_TEST_DAYS
    if step_days is None:
        step_days = config.WFA_STEP_DAYS
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    windows = []
    current = start
    
    while current + timedelta(days=train_days + test_days) <= end:
        train_start = current.strftime('%Y-%m-%d')
        train_end = (current + timedelta(days=train_days - 1)).strftime('%Y-%m-%d')
        test_start = (current + timedelta(days=train_days)).strftime('%Y-%m-%d')
        test_end = (current + timedelta(days=train_days + test_days - 1)).strftime('%Y-%m-%d')
        
        windows.append((train_start, train_end, test_start, test_end))
        current += timedelta(days=step_days)
    
    return windows


def walk_forward_grid_search(
    windows: List[Tuple[str, str, str, str]],
    initial_balance: float,
    # Parameter grids
    rsi_max_values: List[int] = None,
    volume_surge_values: List[float] = None,
    min_score_values: List[int] = None,
    stop_loss_values: List[float] = None,
    exit_modes: List[str] = None,
    risk_reward_ratios: List[float] = None,
    # Fixed params
    top_n: int = None,
    max_positions: int = None,
    # Constraints
    min_trades: int = None,
    min_win_rate: float = None,
    max_drawdown_cap: float = None,
    min_test_trades: int = None,
    # Options
    progress: bool = True,
    log_to_db: bool = True,
    precomputed = None, # PrecomputedData object (optional)
) -> pd.DataFrame:
    """
    Walk-forward analysis with rolling windows.
    
    For each window:
        1. Run grid search on TRAIN period
        2. Find best params by profit factor (under constraints)
        3. Run backtest on TEST period with those params
        4. Record OOS (out-of-sample) results
    
    Returns:
        DataFrame with train/test results for each window
    """
    if min_test_trades is None:
        min_test_trades = config.WFA_MIN_TEST_TRADES
    
    # Set defaults
    if rsi_max_values is None:
        rsi_max_values = [65, 70, 75]
    if volume_surge_values is None:
        volume_surge_values = [1.3, 1.5, 2.0]
    if min_score_values is None:
        min_score_values = [50, 60, 70]
    if stop_loss_values is None:
        stop_loss_values = [0.04, 0.05, 0.06]
    if exit_modes is None:
        exit_modes = ['default', 'trailing']
    if risk_reward_ratios is None:
        risk_reward_ratios = [1.5, 2.0, 2.5]

    if precomputed and not _is_precomputed_compatible(
        precomputed, rsi_max_values, volume_surge_values, min_score_values
    ):
        precomputed = None
    
    logger.info(f"Walk-forward analysis: {len(windows)} windows")
    
    results = []
    iterator = enumerate(windows)
    if progress:
        iterator = tqdm(list(iterator), desc="Walk-forward")
    
    for i, (train_start, train_end, test_start, test_end) in iterator:
        # 1. Grid search on training period
        train_results = grid_search_daily(
            start_date=train_start,
            end_date=train_end,
            initial_balance=initial_balance,
            rsi_max_values=rsi_max_values,
            volume_surge_values=volume_surge_values,
            min_score_values=min_score_values,
            stop_loss_values=stop_loss_values,
            exit_modes=exit_modes,
            risk_reward_ratios=risk_reward_ratios,
            top_n=top_n,
            max_positions=max_positions,
            min_trades=min_trades,
            min_win_rate=min_win_rate,
            max_drawdown_cap=max_drawdown_cap,
            progress=False,
            precomputed=precomputed,
        )
        
        if len(train_results) == 0:
            logger.warning(f"Window {i}: No training results")
            continue
        
        # 2. Find best params (passed constraints, highest profit factor)
        passed = train_results[train_results['passed_constraints'] == True]
        if len(passed) == 0:
            # Fallback: best overall
            best = train_results.iloc[0]
        else:
            best = passed.iloc[0]
        
        # 3. Run backtest on test period with best params
        scanner_config = config.get_scanner_config()
        scanner_config['MOMENTUM_RSI_MAX'] = best['rsi_max']
        scanner_config['MOMENTUM_VOLUME_SPIKE_FACTOR'] = best['volume_surge']
        
        try:
            if precomputed:
                 test_engine, test_metrics = run_fast_backtest(
                    precomputed=precomputed,
                    initial_balance=initial_balance,
                    max_positions=max_positions,
                    stop_loss_pct=best['stop_loss'],
                    exit_mode=best['exit_mode'],
                    min_score=int(best['min_score']),
                    scanner_config=scanner_config,
                    progress=False,
                    start_date=test_start,
                    end_date=test_end,
                )
            else:
                test_engine, test_metrics = run_daily_backtest(
                    start_date=test_start,
                    end_date=test_end,
                    initial_balance=initial_balance,
                    top_n=top_n,
                    max_positions=max_positions,
                    stop_loss_pct=best['stop_loss'],
                    exit_mode=best['exit_mode'],
                    min_score=int(best['min_score']),
                    scanner_config=scanner_config,
                    progress=False,
                    risk_reward_ratio=best.get('risk_reward_ratio', 2.0),
                )

            
            results.append({
                'window_id': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                # Best params from train
                'pick_rsi_max': best['rsi_max'],
                'pick_vol_surge': best['volume_surge'],
                'pick_min_score': best['min_score'],
                'pick_stop_loss': best['stop_loss'],
                'pick_exit_mode': best['exit_mode'],
                'pick_risk_reward': best.get('risk_reward_ratio', 2.0),
                # Train metrics
                'train_pf': best['profit_factor'],
                'train_wr': best['win_rate'],
                'train_trades': best['total_trades'],
                'train_dd': best['max_drawdown'],
                # Test metrics (OOS)
                'test_pf': test_metrics['profit_factor'],
                'test_wr': test_metrics['win_rate'],
                'test_trades': test_metrics['total_trades'],
                'test_dd': test_metrics['max_drawdown'],
                'test_return': test_metrics['total_return'],
            })
            
            # Log to database
            if log_to_db:
                params = {
                    'rsi_max': int(best['rsi_max']),
                    'vol_surge': float(best['volume_surge']),
                    'min_score': int(best['min_score']),
                    'stop_loss': float(best['stop_loss']),
                    'exit_mode': best['exit_mode'],
                }
                dm.log_backtest_run(
                    test_start, test_end, params, test_metrics,
                    notes=f"WFA window {i}"
                )
                
        except Exception as e:
            logger.warning(f"Window {i}: Error in test period: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        logger.info(f"Walk-forward complete: {len(df)} windows")
    
    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def summarize_oos(
    wf_results: pd.DataFrame,
    min_test_trades: int = None
) -> dict:
    """
    Aggregate OOS metrics across walks.
    
    Returns:
        Dict with 'ok' DataFrame and 'oos_summary' stats
    """
    if min_test_trades is None:
        min_test_trades = config.WFA_MIN_TEST_TRADES
    
    if len(wf_results) == 0:
        return {'ok': pd.DataFrame(), 'oos_summary': {}}
    
    # Filter walks with sufficient test trades
    ok = wf_results[wf_results['test_trades'] >= min_test_trades].copy()
    
    if len(ok) == 0:
        return {'ok': pd.DataFrame(), 'oos_summary': {}}
    
    summary = {
        'median_pf': ok['test_pf'].median(),
        'mean_pf': ok['test_pf'].mean(),
        'median_wr': ok['test_wr'].median(),
        'median_dd': ok['test_dd'].median(),
        'total_oos_trades': ok['test_trades'].sum(),
        'windows_ok': len(ok),
        'windows_total': len(wf_results),
    }
    
    return {'ok': ok, 'oos_summary': summary}


def top_params_by_stability(
    wf_results: pd.DataFrame,
    min_test_trades: int = None,
    min_count: int = 2
) -> pd.DataFrame:
    """
    Find parameter combos that were selected across multiple walks.
    
    Stability = robustness indicator (same params work in different periods)
    
    Returns:
        DataFrame with param combos sorted by selection count
    """
    if min_test_trades is None:
        min_test_trades = config.WFA_MIN_TEST_TRADES
    
    if len(wf_results) == 0:
        return pd.DataFrame()
    
    # Filter valid walks
    ok = wf_results[wf_results['test_trades'] >= min_test_trades].copy()
    
    if len(ok) == 0:
        return pd.DataFrame()
    
    # Group by parameter combination
    param_cols = ['pick_rsi_max', 'pick_vol_surge', 'pick_min_score', 
                  'pick_stop_loss', 'pick_exit_mode', 'pick_risk_reward']
    
    grouped = ok.groupby(param_cols).agg({
        'test_pf': ['mean', 'median', 'count'],
        'test_wr': 'mean',
        'test_trades': 'sum',
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Rename for clarity
    grouped = grouped.rename(columns={
        'test_pf_count': 'selection_count',
        'test_pf_mean': 'avg_test_pf',
        'test_pf_median': 'median_test_pf',
        'test_wr_mean': 'avg_test_wr',
        'test_trades_sum': 'total_test_trades',
    })
    
    # Filter by minimum count
    grouped = grouped[grouped['selection_count'] >= min_count]
    
    # Sort by count, then by average PF
    grouped = grouped.sort_values(
        ['selection_count', 'avg_test_pf'], 
        ascending=[False, False]
    )
    
    return grouped


def save_best_params(params: dict, filename: str = None):
    """Save best parameters to JSON file."""
    if filename is None:
        filename = config.RESULTS_DIR / "best_params.json"
    
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Saved params to {filename}")


def load_best_params(filename: str = None) -> dict:
    """Load parameters from JSON file."""
    if filename is None:
        filename = config.RESULTS_DIR / "best_params.json"
    
    with open(filename, 'r') as f:
        return json.load(f)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing optimizer module...")
    
    print("[+] generate_windows()")
    windows = generate_windows('2024-01-01', '2024-12-31', 90, 30, 30)
    print(f"    Generated {len(windows)} rolling windows")
    if windows:
        print(f"    First: train {windows[0][0]} to {windows[0][1]}, test {windows[0][2]} to {windows[0][3]}")
    
    print("[+] Grid search and WFA functions defined")
    print("[+] summarize_oos() and top_params_by_stability() defined")
    
    print("\nTo run optimization:")
    print("  from optimizer import grid_search_daily, walk_forward_grid_search")
    print("  results = grid_search_daily('2024-01-01', '2024-06-30', 1_000_000)")
    print("  wf = walk_forward_grid_search(windows, 1_000_000)")
    
    print("\noptimizer module tests passed!")
