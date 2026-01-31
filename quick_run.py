"""
Quick Run Script — Double-click or run this file to execute backtests.
Modify the settings below and run.
"""

# =============================================================================
# SETTINGS — Modify these as needed
# =============================================================================

import config

BACKTEST_SETTINGS = {
    'start_date': '2024-01-01',      # Start date (YYYY-MM-DD)
    'end_date': '2024-06-30',        # End date (YYYY-MM-DD)
    'initial_balance': 1_000_000,    # Starting capital in JPY
    'top_n': config.UNIVERSE_TOP_N,          # Universe size
    'max_positions': config.MAX_POSITIONS,   # Max concurrent positions
    'exit_mode': config.EXIT_MODE,           # 'default', 'trailing', 'breakeven', 'fixed_rr'
    'stop_loss_pct': config.STOP_LOSS_PCT,   # Stop loss
    'min_score': config.MIN_SCANNER_SCORE,   # Minimum scanner score
}

# =============================================================================
# RUN THE BACKTEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("JP Stocks Finder — Running Backtest")
    print("=" * 60)
    
    from backtesting import run_daily_backtest, get_trade_history_df
    
    print(f"Period: {BACKTEST_SETTINGS['start_date']} to {BACKTEST_SETTINGS['end_date']}")
    print(f"Balance: JPY {BACKTEST_SETTINGS['initial_balance']:,}")
    print(f"Exit Mode: {BACKTEST_SETTINGS['exit_mode']}")
    print("=" * 60)
    print()
    
    # Run backtest
    engine, metrics = run_daily_backtest(
        start_date=BACKTEST_SETTINGS['start_date'],
        end_date=BACKTEST_SETTINGS['end_date'],
        initial_balance=BACKTEST_SETTINGS['initial_balance'],
        top_n=BACKTEST_SETTINGS['top_n'],
        max_positions=BACKTEST_SETTINGS['max_positions'],
        exit_mode=BACKTEST_SETTINGS['exit_mode'],
        stop_loss_pct=BACKTEST_SETTINGS['stop_loss_pct'],
        min_score=BACKTEST_SETTINGS['min_score'],
        progress=True,
    )
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Profit Factor:   {metrics['profit_factor']:.4f}")
    print(f"Win Rate:        {metrics['win_rate']:.2%}")
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Max Drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"Total Return:    {metrics['total_return']:.2%}")
    print(f"Final Equity:    JPY {metrics['final_equity']:,.0f}")
    print("-" * 60)
    print(f"Winners:         {metrics['winners']}")
    print(f"Losers:          {metrics['losers']}")
    print(f"Avg Win:         JPY {metrics['avg_win']:,.0f}")
    print(f"Avg Loss:        JPY {metrics['avg_loss']:,.0f}")
    print("=" * 60)
    
    # Export trades to CSV
    trades_df = get_trade_history_df(engine)
    if len(trades_df) > 0:
        export_path = "results/trades_latest.csv"
        trades_df.to_csv(export_path, index=False)
        print(f"\nTrades exported to: {export_path}")
    
    print("\nDone! Press Enter to exit...")
    input()
