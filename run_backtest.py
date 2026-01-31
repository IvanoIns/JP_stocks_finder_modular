#!/usr/bin/env python
"""
JP Stocks Modular Trading System — CLI Entry Point

Usage examples:
    python run_backtest.py --start 2024-01-01 --end 2024-06-30 --balance 1000000
    python run_backtest.py --start 2024-01-01 --end 2024-12-31 --exit-mode trailing
    python run_backtest.py --download  # Download data first
"""

import argparse
import sys
from datetime import datetime

import config


def main():
    parser = argparse.ArgumentParser(
        description='JP Stocks Modular Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --download                           # Download price data
  python run_backtest.py --start 2024-01-01 --end 2024-06-30  # Run backtest
  python run_backtest.py --config                             # Show configuration
        """
    )
    
    # Action flags
    parser.add_argument('--download', action='store_true', 
                        help='Download price data from yfinance')
    parser.add_argument('--config', action='store_true',
                        help='Show current configuration')
    
    # Backtest parameters
    parser.add_argument('--start', type=str, 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=1_000_000, 
                        help='Initial balance (JPY, default: 1,000,000)')
    parser.add_argument('--top-n', type=int, default=config.UNIVERSE_TOP_N,
                        help=f'Universe size (default: {config.UNIVERSE_TOP_N})')
    parser.add_argument('--max-positions', type=int, default=config.MAX_POSITIONS,
                        help=f'Max concurrent positions (default: {config.MAX_POSITIONS})')
    parser.add_argument('--exit-mode', type=str, default=config.EXIT_MODE,
                        choices=['default', 'trailing', 'breakeven', 'breakeven_trailing', 'fixed_rr'],
                        help=f'Exit mode (default: {config.EXIT_MODE})')
    parser.add_argument('--stop-loss', type=float, default=config.STOP_LOSS_PCT,
                        help=f'Stop loss percentage (default: {config.STOP_LOSS_PCT})')
    parser.add_argument('--min-score', type=int, default=config.MIN_SCANNER_SCORE,
                        help=f'Minimum scanner score (default: {config.MIN_SCANNER_SCORE})')
    
    # Output options
    parser.add_argument('--export', type=str, 
                        help='Export trades to CSV file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Show config
    if args.config:
        config.print_config_summary()
        return 0
    
    # Download data
    if args.download:
        print("\n" + "="*60)
        print("Downloading Price Data")
        print("="*60 + "\n")
        
        import data_manager as dm
        
        # Setup database
        dm.setup_database()
        
        # Get tickers
        print("Fetching TSE ticker list...")
        tickers = dm.get_all_tse_tickers()
        print(f"Found {len(tickers)} tickers")
        
        # Download
        print(f"\nDownloading price history (this may take a while)...")
        count = dm.download_price_history(tickers[:500], progress=not args.quiet)  # Limit for speed
        print(f"\nDownloaded data for {count} symbols")
        
        # Fetch JPX data
        print("\nFetching JPX short-selling data...")
        jpx = dm.fetch_jpx_short_data()
        print(f"Processed {len(jpx)} symbols")
        
        print("\nData download complete!")
        return 0
    
    # Run backtest
    if args.start and args.end:
        print("\n" + "="*60)
        print("JP Stocks Daily Backtest")
        print("="*60)
        print(f"Period: {args.start} to {args.end}")
        print(f"Initial Balance: JPY {args.balance:,.0f}")
        print(f"Universe Size: {args.top_n}")
        print(f"Max Positions: {args.max_positions}")
        print(f"Exit Mode: {args.exit_mode}")
        print(f"Stop Loss: {args.stop_loss:.1%}")
        print(f"Min Score: {args.min_score}")
        print("="*60 + "\n")
        
        from backtesting import run_daily_backtest, get_trade_history_df
        
        # Run backtest
        engine, metrics = run_daily_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_balance=args.balance,
            top_n=args.top_n,
            max_positions=args.max_positions,
            exit_mode=args.exit_mode,
            stop_loss_pct=args.stop_loss,
            min_score=args.min_score,
            progress=not args.quiet,
        )
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Profit Factor:   {metrics['profit_factor']:.4f}")
        print(f"Win Rate:        {metrics['win_rate']:.2%}")
        print(f"Total Trades:    {metrics['total_trades']}")
        print(f"Max Drawdown:    {metrics['max_drawdown']:.2%}")
        print(f"Total Return:    {metrics['total_return']:.2%}")
        print(f"Final Equity:    JPY {metrics['final_equity']:,.0f}")
        print("-"*60)
        print(f"Winners:         {metrics['winners']}")
        print(f"Losers:          {metrics['losers']}")
        print(f"Avg Win:         JPY {metrics['avg_win']:,.0f}")
        print(f"Avg Loss:        JPY {metrics['avg_loss']:,.0f}")
        print("="*60)
        
        # Export trades
        if args.export:
            trades_df = get_trade_history_df(engine)
            if len(trades_df) > 0:
                trades_df.to_csv(args.export, index=False)
                print(f"\nTrades exported to: {args.export}")
        
        return 0
    
    # No valid action
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
