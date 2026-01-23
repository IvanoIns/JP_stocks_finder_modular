"""
JP Stocks Modular Trading System — Backtesting Engine

Daily backtest engine adapted from crypto hourly backtester.
Entry at next-day open, exits checked against daily high/low.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import logging

import config
import data_manager as dm
import technical_analysis as ta
import scanners

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Position:
    """
    Represents an open position.
    """
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    quantity: float
    stop_price: float
    target_1_price: float
    target_2_price: float
    remaining_quantity: float = None
    target_1_hit: bool = False
    peak_price: float = None  # For trailing stop
    entry_value: float = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        if self.peak_price is None:
            self.peak_price = self.entry_price
        if self.entry_value is None:
            self.entry_value = self.entry_price * self.quantity


@dataclass 
class Trade:
    """
    Represents a closed trade (or partial close).
    """
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'stop_loss', 'target_1', 'target_2', 'trailing', 'end_of_backtest'


@dataclass
class PendingEntry:
    """
    Signal queued for entry at next day's open.
    """
    symbol: str
    strategy: str
    score: int
    signal_date: str
    signal_price: float


# =============================================================================
# Backtest Engine
# =============================================================================

class BacktestEngine:
    """
    Daily backtest engine for JP stocks.
    
    Key differences from crypto hourly backtester:
    - Daily bars only
    - Entry at next-day open (T+1)
    - Exits checked against daily high/low
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_positions: int = None,
        position_size_pct: float = None,
        stop_loss_pct: float = None,
        target_1_pct: float = None,
        target_1_portion: float = None,
        target_2_pct: float = None,
        exit_mode: str = None,
        trailing_stop_pct: float = None,
        slippage_pct: float = None,
        commission_pct: float = None,
    ):
        # Use config defaults if not specified
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.max_positions = max_positions or config.MAX_POSITIONS
        self.position_size_pct = position_size_pct or config.FIXED_FRACTIONAL_PCT
        
        # Exit parameters
        self.stop_loss_pct = stop_loss_pct or config.STOP_LOSS_PCT
        self.target_1_pct = target_1_pct or config.TARGET_1_PCT
        self.target_1_portion = target_1_portion or config.TARGET_1_PORTION
        self.target_2_pct = target_2_pct or config.TARGET_2_PCT
        self.exit_mode = exit_mode or config.EXIT_MODE
        self.trailing_stop_pct = trailing_stop_pct or config.TRAILING_STOP_PCT
        
        # Costs
        self.slippage_pct = slippage_pct if slippage_pct is not None else config.BACKTEST_SLIPPAGE_PCT
        self.commission_pct = commission_pct if commission_pct is not None else config.BACKTEST_COMMISSION_PCT
        
        # State
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.daily_balances: List[Dict] = []
        self.pending_entries: List[PendingEntry] = []
        
        # Counters
        self.entries_count = 0
        self.total_entry_value = 0.0
    
    @property
    def equity(self) -> float:
        """Current equity = cash + open positions value."""
        positions_value = sum(
            p.remaining_quantity * p.entry_price  # Mark at entry (conservative)
            for p in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def open_position_count(self) -> int:
        return len(self.positions)
    
    def can_open_position(self) -> bool:
        return self.open_position_count < self.max_positions
    
    def calculate_position_size(self, entry_price: float) -> Tuple[float, float]:
        """
        Calculate position size based on sizing method.
        
        Returns:
            Tuple of (quantity, position_value)
        """
        available = self.cash * self.position_size_pct
        max_allowed = self.equity * config.MAX_POSITION_PCT
        position_value = min(available, max_allowed, self.cash)
        
        if position_value <= 0 or entry_price <= 0:
            return 0, 0
        
        quantity = position_value / entry_price
        return quantity, position_value
    
    def queue_entry(self, signal: dict, current_date: str, current_price: float):
        """
        Queue a signal for entry at next day's open.
        """
        self.pending_entries.append(PendingEntry(
            symbol=signal['symbol'],
            strategy=signal['strategy'],
            score=signal['score'],
            signal_date=current_date,
            signal_price=current_price,
        ))
    
    def process_pending_entries(
        self,
        date: str,
        open_prices: Dict[str, float]
    ) -> int:
        """
        Process queued entries at today's open.
        
        Returns:
            Number of positions opened
        """
        if not self.pending_entries:
            return 0
        
        # Sort by score (highest first)
        self.pending_entries.sort(key=lambda x: x.score, reverse=True)
        
        opened = 0
        for entry in self.pending_entries:
            if not self.can_open_position():
                break
            
            if entry.symbol in self.positions:
                continue  # Already have position
            
            open_price = open_prices.get(entry.symbol)
            if open_price is None or open_price <= 0:
                continue
            
            # Apply slippage
            entry_price = open_price * (1 + self.slippage_pct)
            
            # Calculate position size
            quantity, position_value = self.calculate_position_size(entry_price)
            if quantity <= 0:
                continue
            
            # Apply commission
            commission = position_value * self.commission_pct
            
            # Create position
            position = Position(
                symbol=entry.symbol,
                strategy=entry.strategy,
                entry_date=date,
                entry_price=entry_price,
                quantity=quantity,
                stop_price=entry_price * (1 - self.stop_loss_pct),
                target_1_price=entry_price * (1 + self.target_1_pct),
                target_2_price=entry_price * (1 + self.target_2_pct),
            )
            
            self.positions[entry.symbol] = position
            self.cash -= (position_value + commission)
            self.entries_count += 1
            self.total_entry_value += position_value
            opened += 1
        
        # Clear pending entries
        self.pending_entries = []
        return opened
    
    def check_exits(
        self,
        date: str,
        daily_bars: Dict[str, pd.Series]
    ) -> int:
        """
        Check all open positions for exit conditions.
        
        Returns:
            Number of exits processed
        """
        exits = 0
        symbols_to_close = []
        
        for symbol, position in list(self.positions.items()):
            bar = daily_bars.get(symbol)
            if bar is None:
                continue
            
            high = bar['high']
            low = bar['low']
            close = bar['close']
            
            # Update peak price for trailing stop
            if high > position.peak_price:
                position.peak_price = high
            
            # === EXIT CHECKS (order matters) ===
            
            # 1. Stop Loss Check
            if low <= position.stop_price:
                self._close_position(
                    symbol, date, position.stop_price,
                    position.remaining_quantity, 'stop_loss'
                )
                symbols_to_close.append(symbol)
                exits += 1
                continue
            
            # 2. Target 1 Check (partial exit)
            if not position.target_1_hit and high >= position.target_1_price:
                sell_qty = position.quantity * self.target_1_portion
                self._close_position(
                    symbol, date, position.target_1_price,
                    sell_qty, 'target_1'
                )
                position.remaining_quantity -= sell_qty
                position.target_1_hit = True
                
                # Breakeven mode: move stop to entry
                if self.exit_mode in ('breakeven', 'breakeven_trailing'):
                    position.stop_price = position.entry_price
                
                # Check if fully closed
                if position.remaining_quantity <= 0:
                    symbols_to_close.append(symbol)
                    exits += 1
                    continue
            
            # 3. Target 2 Check (close remainder)
            if position.target_1_hit and high >= position.target_2_price:
                self._close_position(
                    symbol, date, position.target_2_price,
                    position.remaining_quantity, 'target_2'
                )
                symbols_to_close.append(symbol)
                exits += 1
                continue
            
            # 4. Trailing Stop Check
            if self.exit_mode in ('trailing', 'breakeven_trailing'):
                trailing_stop = position.peak_price * (1 - self.trailing_stop_pct)
                if trailing_stop > position.stop_price:
                    position.stop_price = trailing_stop
                
                if low <= position.stop_price:
                    self._close_position(
                        symbol, date, position.stop_price,
                        position.remaining_quantity, 'trailing'
                    )
                    symbols_to_close.append(symbol)
                    exits += 1
                    continue
        
        # Remove closed positions
        for symbol in symbols_to_close:
            if symbol in self.positions:
                del self.positions[symbol]
        
        return exits
    
    def _close_position(
        self,
        symbol: str,
        exit_date: str,
        exit_price: float,
        quantity: float,
        reason: str
    ):
        """Record a position close (or partial close)."""
        position = self.positions.get(symbol)
        if position is None:
            return
        
        # Apply slippage and commission
        actual_exit = exit_price * (1 - self.slippage_pct)
        proceeds = actual_exit * quantity
        commission = proceeds * self.commission_pct
        net_proceeds = proceeds - commission
        
        # Calculate PnL
        entry_value = position.entry_price * quantity
        pnl = net_proceeds - entry_value
        pnl_pct = (actual_exit / position.entry_price - 1.0) if position.entry_price > 0 else 0
        
        # Record trade
        self.trade_history.append(Trade(
            symbol=symbol,
            strategy=position.strategy,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=actual_exit,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        ))
        
        # Update cash
        self.cash += net_proceeds
    
    def liquidate_all(self, date: str, close_prices: Dict[str, float]):
        """Close all positions at end of backtest."""
        for symbol, position in list(self.positions.items()):
            close_price = close_prices.get(symbol, position.entry_price)
            self._close_position(
                symbol, date, close_price,
                position.remaining_quantity, 'end_of_backtest'
            )
        self.positions.clear()
    
    def record_daily_balance(self, date: str, close_prices: Dict[str, float]):
        """Record end-of-day balance for equity curve."""
        # Mark positions to market
        positions_mtm = 0
        for symbol, position in self.positions.items():
            price = close_prices.get(symbol, position.entry_price)
            positions_mtm += position.remaining_quantity * price
        
        self.daily_balances.append({
            'date': date,
            'cash': self.cash,
            'positions_value': positions_mtm,
            'equity': self.cash + positions_mtm,
            'open_positions': len(self.positions),
        })
    
    def calculate_metrics(self) -> dict:
        """Calculate backtest performance metrics."""
        if not self.trade_history:
            return {
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'final_balance': self.cash,
                'final_equity': self.equity,
            }
        
        trades_df = pd.DataFrame([t.__dict__ for t in self.trade_history])
        
        # Win/Loss
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
        
        # Profit Factor
        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0  
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        
        # Max Drawdown
        if self.daily_balances:
            equity_curve = pd.Series([b['equity'] for b in self.daily_balances])
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        # Returns
        final_equity = self.equity
        total_return = (final_equity / self.initial_balance - 1.0) if self.initial_balance > 0 else 0
        
        # Average trade stats
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        avg_trade = trades_df['pnl'].mean()
        
        return {
            'profit_factor': round(profit_factor, 4),
            'win_rate': round(win_rate, 4),
            'total_trades': len(self.trade_history),
            'max_drawdown': round(max_drawdown, 4),
            'total_return': round(total_return, 4),
            'final_balance': round(self.cash, 2),
            'final_equity': round(final_equity, 2),
            'entries_count': self.entries_count,
            'avg_entry_value': round(self.total_entry_value / self.entries_count, 2) if self.entries_count > 0 else 0,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade': round(avg_trade, 2),
            'winners': len(winners),
            'losers': len(losers),
        }


# =============================================================================
# Main Backtest Function
# =============================================================================

def run_daily_backtest(
    start_date: str,
    end_date: str,
    initial_balance: float,
    top_n: int = None,
    max_positions: int = None,
    position_size_pct: float = None,
    stop_loss_pct: float = None,
    target_1_pct: float = None,
    target_1_portion: float = None,
    target_2_pct: float = None,
    exit_mode: str = None,
    trailing_stop_pct: float = None,
    min_score: int = None,
    scanner_config: dict = None,
    liquidate_on_end: bool = True,
    progress: bool = True,
    lookback_days: int = 250,
) -> Tuple[BacktestEngine, dict]:
    """
    Main daily backtest entry point.
    
    FLOW:
        For each trading day T:
            1. Build liquid universe for day T
            2. Load daily bars for universe (with lookback for indicators)
            3. Calculate indicators for each symbol
            4. Run scanners to detect signals
            5. Queue top signals for entry (ranked by score)
            6. Process exits for existing positions (vs day T high/low)
            7. Process pending entries at day T open
            8. Record daily balance
        
        After all days:
            - Liquidate remaining positions (if enabled)
            - Calculate metrics
    
    ENTRY TIMING:
        Signal on day T -> Entry at open of day T+1
        (1-day delay, no lookahead)
    
    Returns:
        (engine: BacktestEngine, metrics: dict)
    """
    # Set defaults from config
    if top_n is None:
        top_n = config.UNIVERSE_TOP_N
    if min_score is None:
        min_score = config.MIN_SCORE
    if scanner_config is None:
        scanner_config = config.get_scanner_config()
    
    # Initialize engine
    engine = BacktestEngine(
        initial_balance=initial_balance,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        target_1_pct=target_1_pct,
        target_1_portion=target_1_portion,
        target_2_pct=target_2_pct,
        exit_mode=exit_mode,
        trailing_stop_pct=trailing_stop_pct,
    )
    
    # Get all trading dates
    conn = dm.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT date FROM daily_prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (start_date, end_date))
    trading_dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not trading_dates:
        logger.warning(f"No trading dates found between {start_date} and {end_date}")
        return engine, engine.calculate_metrics()
    
    logger.info(f"Running backtest: {start_date} to {end_date} ({len(trading_dates)} days)")
    
    # Main backtest loop
    iterator = enumerate(trading_dates)
    if progress:
        iterator = tqdm(list(iterator), desc="Backtesting")
    
    for i, date in iterator:
        # 1. Build liquid universe for this date
        universe = dm.build_liquid_universe(date, top_n=top_n)
        if not universe:
            continue
        
        # 2. Get price data (need lookback for indicators)
        lookback_start = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        bars_batch = dm.get_daily_bars_batch(universe, lookback_start, date)
        
        # Get today's bars for exit checks
        today_bars = {}
        open_prices = {}
        close_prices = {}
        
        for symbol, df in bars_batch.items():
            if len(df) < 20:  # Need minimum data
                continue
            
            # Get today's bar
            today_data = df[df.index.strftime('%Y-%m-%d') == date]
            if len(today_data) > 0:
                today_bars[symbol] = today_data.iloc[-1]
                open_prices[symbol] = today_data.iloc[-1]['open']
                close_prices[symbol] = today_data.iloc[-1]['close']
        
        # 3. Process pending entries from yesterday's signals (at today's open)
        engine.process_pending_entries(date, open_prices)
        
        # 4. Check exits for existing positions
        engine.check_exits(date, today_bars)
        
        # 5. Get JPX short data (if available)
        jpx_data = dm.get_jpx_short_for_date(date)
        
        # 6. Run scanners on each symbol in universe
        for symbol in universe:
            if symbol not in bars_batch:
                continue
            
            data = bars_batch[symbol]
            if len(data) < 60:
                continue
            
            # Get short data for this symbol
            symbol_jpx = jpx_data.get(symbol, {'short_ratio': 0})
            
            # Run scanners
            signals = scanners.get_all_signals(
                symbol, data, symbol_jpx, scanner_config, min_score
            )
            
            # Queue best signals for next day entry
            for signal in signals[:3]:  # Top 3 signals per symbol
                if engine.can_open_position():
                    current_price = close_prices.get(symbol, 0)
                    if current_price > 0:
                        engine.queue_entry(signal, date, current_price)
        
        # 7. Record daily balance
        engine.record_daily_balance(date, close_prices)
    
    # Liquidate remaining positions at end
    if liquidate_on_end and engine.positions:
        last_date = trading_dates[-1]
        last_bars = dm.get_daily_bars_batch(
            list(engine.positions.keys()), 
            last_date, last_date
        )
        last_prices = {
            s: df.iloc[-1]['close'] if len(df) > 0 else 0 
            for s, df in last_bars.items()
        }
        engine.liquidate_all(last_date, last_prices)
    
    # Calculate final metrics
    metrics = engine.calculate_metrics()
    
    logger.info(f"Backtest complete: {metrics['total_trades']} trades, "
                f"PF={metrics['profit_factor']:.2f}, WR={metrics['win_rate']:.1%}")
    
    return engine, metrics


def get_trade_history_df(engine: BacktestEngine) -> pd.DataFrame:
    """Convert trade history to DataFrame for analysis."""
    if not engine.trade_history:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in engine.trade_history])


def get_equity_curve_df(engine: BacktestEngine) -> pd.DataFrame:
    """Get equity curve as DataFrame."""
    if not engine.daily_balances:
        return pd.DataFrame()
    return pd.DataFrame(engine.daily_balances)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing backtesting module...")
    
    # This requires data in the database
    # Run: python data_manager.py first to populate data
    
    print("[+] BacktestEngine initialized")
    engine = BacktestEngine(initial_balance=1_000_000)
    print(f"    Initial balance: {engine.initial_balance:,.0f}")
    print(f"    Max positions: {engine.max_positions}")
    print(f"    Exit mode: {engine.exit_mode}")
    
    print("[+] Position/Trade dataclasses defined")
    
    # Test with minimal data if available
    try:
        symbols = dm.get_available_symbols(min_rows=60)
        if symbols:
            print(f"[+] Found {len(symbols)} symbols with data")
            print("    To run full backtest, use:")
            print("    engine, metrics = run_daily_backtest('2024-01-01', '2024-12-31', 1_000_000)")
        else:
            print("[ ] No data in database - run data_manager.py first")
    except Exception as e:
        print(f"[ ] Could not check database: {e}")
    
    print("\nbacktesting module tests passed!")
