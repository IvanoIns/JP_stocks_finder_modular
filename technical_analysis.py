"""
JP Stocks Modular Trading System — Technical Analysis

Indicator calculations for scanners and backtesting.
All functions are pure (no side effects, no database access).
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple

import config


# =============================================================================
# Moving Averages
# =============================================================================

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Price series (usually close prices)
        period: Lookback period
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period, min_periods=1).mean()


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Price series
        span: EMA span (equivalent to period)
    
    Returns:
        Series of EMA values
    """
    return prices.ewm(span=span, adjust=False).mean()


# =============================================================================
# Momentum Indicators
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Price series
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.0001)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal line, and Histogram.
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def is_macd_bullish(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    """
    Check if MACD is bullish (MACD line above signal line).
    
    Returns:
        True if bullish
    """
    macd, sig, hist = calculate_macd(prices, fast, slow, signal)
    if len(hist) < 1:
        return False
    return hist.iloc[-1] > 0


# =============================================================================
# Volatility Indicators
# =============================================================================

def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        period: Moving average period (default 20)
        std_mult: Standard deviation multiplier (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(prices, period)
    std = prices.rolling(window=period, min_periods=1).std()
    
    upper = middle + (std * std_mult)
    lower = middle - (std * std_mult)
    
    return upper, middle, lower


def is_bb_overbought(prices: pd.Series, period: int = 20, std_mult: float = 2.0) -> bool:
    """
    Check if price is above upper Bollinger Band (overbought).
    """
    upper, middle, lower = calculate_bollinger_bands(prices, period, std_mult)
    if len(prices) < 1:
        return False
    return prices.iloc[-1] > upper.iloc[-1]


def is_bb_oversold(prices: pd.Series, period: int = 20, std_mult: float = 2.0) -> bool:
    """
    Check if price is below lower Bollinger Band (oversold).
    """
    upper, middle, lower = calculate_bollinger_bands(prices, period, std_mult)
    if len(prices) < 1:
        return False
    return prices.iloc[-1] < lower.iloc[-1]


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)
    
    Returns:
        Series of ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the rolling mean of TR
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr


def calculate_volatility(prices: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate historical volatility (standard deviation of returns).
    
    Args:
        prices: Price series
        period: Lookback period
        annualize: If True, annualize the volatility (assumes 252 trading days)
    
    Returns:
        Series of volatility values (as percentages if annualized)
    """
    returns = prices.pct_change()
    vol = returns.rolling(window=period, min_periods=1).std()
    
    if annualize:
        vol = vol * np.sqrt(252) * 100  # Convert to annual percentage
    
    return vol


def calculate_z_score(prices: pd.Series, period: int = 30) -> pd.Series:
    """
    Calculate Z-Score (standard deviations from rolling mean).
    
    Args:
        prices: Price series
        period: Lookback period for mean and std
    
    Returns:
        Series of Z-Score values
    """
    rolling_mean = prices.rolling(window=period, min_periods=1).mean()
    rolling_std = prices.rolling(window=period, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, 0.0001)
    
    z_score = (prices - rolling_mean) / rolling_std
    
    return z_score


# =============================================================================
# Volume Indicators
# =============================================================================

def calculate_volume_surge(volume: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Calculate volume relative to moving average.
    
    Args:
        volume: Volume series
        lookback: Period for average volume calculation
    
    Returns:
        Series of volume surge ratios (1.0 = average, 2.0 = 2x average)
    """
    avg_volume = volume.rolling(window=lookback, min_periods=1).mean()
    avg_volume = avg_volume.replace(0, 1)  # Avoid division by zero
    
    return volume / avg_volume


def calculate_volume_trend(volume: pd.Series, short_period: int = 10, long_period: int = 30) -> float:
    """
    Calculate volume trend (recent vs prior volume).
    
    Returns:
        Ratio of recent volume to prior volume (>1 = increasing, <1 = decreasing)
    """
    if len(volume) < long_period:
        return 1.0
    
    recent_vol = volume.iloc[-short_period:].mean()
    prior_vol = volume.iloc[-long_period:-short_period].mean()
    
    if prior_vol == 0:
        return 1.0
    
    return recent_vol / prior_vol


# =============================================================================
# Trend Indicators
# =============================================================================

def calculate_trend_strength(prices: pd.Series, period: int = 20) -> Tuple[float, float]:
    """
    Calculate linear regression slope and R-squared.
    
    Args:
        prices: Price series
        period: Lookback period
    
    Returns:
        Tuple of (slope, r_squared)
    """
    if len(prices) < period:
        return 0.0, 0.0
    
    recent_prices = prices.iloc[-period:].values
    x = np.arange(len(recent_prices))
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
        r_squared = r_value ** 2
        return slope, r_squared
    except Exception:
        return 0.0, 0.0


def detect_higher_lows(lows: pd.Series, min_periods: int = 3, min_improvement: float = 0.001) -> bool:
    """
    Detect if we have a higher lows pattern (accumulation signature).
    
    Args:
        lows: Series of low prices
        min_periods: Minimum number of local minima to check
        min_improvement: Minimum improvement ratio between lows
    
    Returns:
        True if higher lows pattern detected
    """
    if len(lows) < 5:
        return False
    
    lows_arr = lows.values
    
    # Find local minima
    local_mins = []
    for i in range(1, len(lows_arr) - 1):
        if lows_arr[i] < lows_arr[i-1] and lows_arr[i] < lows_arr[i+1]:
            local_mins.append((i, lows_arr[i]))
    
    if len(local_mins) < min_periods:
        return False
    
    # Check if lows are ascending
    for i in range(1, len(local_mins)):
        if local_mins[i][1] <= local_mins[i-1][1] * (1 + min_improvement):
            return False
    
    return True


def detect_higher_highs(highs: pd.Series, min_periods: int = 3) -> bool:
    """
    Detect if we have a higher highs pattern (uptrend confirmation).
    """
    if len(highs) < 5:
        return False
    
    highs_arr = highs.values
    
    # Find local maxima
    local_maxs = []
    for i in range(1, len(highs_arr) - 1):
        if highs_arr[i] > highs_arr[i-1] and highs_arr[i] > highs_arr[i+1]:
            local_maxs.append((i, highs_arr[i]))
    
    if len(local_maxs) < min_periods:
        return False
    
    # Check if highs are ascending
    for i in range(1, len(local_maxs)):
        if local_maxs[i][1] <= local_maxs[i-1][1]:
            return False
    
    return True


# =============================================================================
# Risk/Reward
# =============================================================================

def calculate_risk_reward(data: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate risk/reward ratio for current setup.
    
    Risk = distance to support (recent low)
    Reward = distance to resistance (recent high)
    
    Args:
        data: DataFrame with 'high', 'low', 'close' columns
        lookback: Period for high/low calculation
    
    Returns:
        Risk/Reward ratio (higher is better)
    """
    if len(data) < lookback:
        return 0.0
    
    recent = data.iloc[-lookback:]
    high_20d = recent['high'].max()
    low_20d = recent['low'].min()
    current_price = data['close'].iloc[-1]
    
    # Distance to resistance (potential reward)
    resistance_dist = (high_20d - current_price) / current_price
    
    # Distance to support (potential risk)
    support_dist = (current_price - low_20d) / current_price
    
    if support_dist <= 0:
        return 0.0
    
    return resistance_dist / support_dist


def calculate_position_in_range(data: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate where current price sits in recent range.
    
    Returns:
        0.0 = at low, 1.0 = at high, 0.5 = middle
    """
    if len(data) < lookback:
        return 0.5
    
    recent = data.iloc[-lookback:]
    high = recent['high'].max()
    low = recent['low'].min()
    current = data['close'].iloc[-1]
    
    if high == low:
        return 0.5
    
    return (current - low) / (high - low)


# =============================================================================
# Candlestick Patterns
# =============================================================================

def check_bullish_reversal_pattern(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Detect bullish reversal candlestick patterns.
    
    Patterns detected:
        - Hammer
        - Bullish Engulfing
        - Morning Star
    
    Args:
        data: DataFrame with OHLC columns
    
    Returns:
        Tuple of (pattern_detected, pattern_name)
    """
    if len(data) < 3:
        return False, "N/A"
    
    day1 = data.iloc[-3]  # Two days ago
    day2 = data.iloc[-2]  # Yesterday
    day3 = data.iloc[-1]  # Today
    
    # --- Hammer ---
    body = abs(day3['close'] - day3['open'])
    total_range = day3['high'] - day3['low']
    lower_wick = min(day3['open'], day3['close']) - day3['low']
    
    if total_range > 0.01 and body > 0.01:
        body_ratio = body / total_range
        wick_ratio = lower_wick / body if body > 0 else 0
        
        if body_ratio < 0.3 and wick_ratio > 2:
            return True, "Hammer"
    
    # --- Bullish Engulfing ---
    if (day2['open'] > day2['close'] and  # Day 2 is bearish
        day3['close'] > day3['open'] and  # Day 3 is bullish
        day3['close'] > day2['open'] and  # Day 3 close > Day 2 open
        day3['open'] < day2['close']):    # Day 3 open < Day 2 close
        return True, "Bullish Engulfing"
    
    # --- Morning Star ---
    day1_body = day1['open'] - day1['close']
    day3_body = day3['close'] - day3['open']
    
    if (day1['open'] > day1['close'] and        # Day 1 bearish
        abs(day2['open'] - day2['close']) < day1_body * 0.3 and  # Day 2 small body
        day3['close'] > day3['open'] and        # Day 3 bullish
        day3['close'] > (day1['open'] + day1['close']) / 2):     # Day 3 closes above Day 1 midpoint
        return True, "Morning Star"
    
    return False, "N/A"


def check_bearish_reversal_pattern(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Detect bearish reversal candlestick patterns.
    
    Patterns detected:
        - Shooting Star
        - Bearish Engulfing
        - Evening Star
    """
    if len(data) < 3:
        return False, "N/A"
    
    day1 = data.iloc[-3]
    day2 = data.iloc[-2]
    day3 = data.iloc[-1]
    
    # --- Shooting Star ---
    body = abs(day3['close'] - day3['open'])
    total_range = day3['high'] - day3['low']
    upper_wick = day3['high'] - max(day3['open'], day3['close'])
    
    if total_range > 0.01 and body > 0.01:
        body_ratio = body / total_range
        wick_ratio = upper_wick / body if body > 0 else 0
        
        if body_ratio < 0.3 and wick_ratio > 2:
            return True, "Shooting Star"
    
    # --- Bearish Engulfing ---
    if (day2['close'] > day2['open'] and  # Day 2 is bullish
        day3['open'] > day3['close'] and  # Day 3 is bearish
        day3['open'] > day2['close'] and  # Day 3 open > Day 2 close
        day3['close'] < day2['open']):    # Day 3 close < Day 2 open
        return True, "Bearish Engulfing"
    
    return False, "N/A"


# =============================================================================
# Consolidated Indicator Function
# =============================================================================

def calculate_all_indicators(
    data: pd.DataFrame,
    scanner_config: dict = None
) -> pd.DataFrame:
    """
    Calculate all indicators for a price DataFrame.
    Adds columns for each indicator.
    
    OPTIMIZATION: If indicators are already calculated (e.g., from pre-compute),
    this function returns the data as-is, avoiding expensive recalculation.
    
    Args:
        data: DataFrame with columns: open, high, low, close, volume
        scanner_config: Configuration dict (uses config.SCANNER_CONFIG if None)
    
    Returns:
        DataFrame with original + indicator columns
    """
    # FAST PATH: Skip if indicators already calculated
    # This is the key optimization for pre-computed data!
    if 'rsi' in data.columns and 'ema_fast' in data.columns:
        return data  # Already has indicators, no need to recalculate!
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    df = data.copy()
    
    # Moving averages
    df['sma_20'] = calculate_sma(df['close'], 20)
    df['sma_50'] = calculate_sma(df['close'], 50)
    df['sma_200'] = calculate_sma(df['close'], 200)
    df['ema_fast'] = calculate_ema(df['close'], scanner_config.get('MOMENTUM_EMA_FAST', 20))
    df['ema_slow'] = calculate_ema(df['close'], scanner_config.get('MOMENTUM_EMA_SLOW', 50))
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], scanner_config.get('MOMENTUM_RSI_PERIOD', 14))
    
    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    # Volatility
    df['atr'] = calculate_atr(df)
    df['volatility'] = calculate_volatility(df['close'])
    df['z_score'] = calculate_z_score(df['close'], scanner_config.get('REVERSAL_ZSCORE_PERIOD', 30))
    
    # Volume
    df['volume_surge'] = calculate_volume_surge(df['volume'])
    
    return df


def get_indicator_snapshot(data: pd.DataFrame, scanner_config: dict = None) -> dict:
    """
    Get a snapshot of current indicator values (latest row).
    
    Returns:
        Dict with indicator names and values
    """
    df = calculate_all_indicators(data, scanner_config)
    
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    return {
        'close': latest['close'],
        'rsi': latest['rsi'],
        'macd': latest['macd'],
        'macd_signal': latest['macd_signal'],
        'macd_hist': latest['macd_hist'],
        'macd_bullish': latest['macd'] > latest['macd_signal'],
        'ema_fast': latest['ema_fast'],
        'ema_slow': latest['ema_slow'],
        'ema_bullish': latest['ema_fast'] > latest['ema_slow'],
        'sma_20': latest['sma_20'],
        'sma_50': latest['sma_50'],
        'bb_upper': latest['bb_upper'],
        'bb_lower': latest['bb_lower'],
        'bb_overbought': latest['close'] > latest['bb_upper'],
        'bb_oversold': latest['close'] < latest['bb_lower'],
        'atr': latest['atr'],
        'volatility': latest['volatility'],
        'z_score': latest['z_score'],
        'volume_surge': latest['volume_surge'],
    }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing technical_analysis module...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(100) * 0.01),
        'high': close * (1 + abs(np.random.randn(100) * 0.02)),
        'low': close * (1 - abs(np.random.randn(100) * 0.02)),
        'close': close,
        'volume': np.random.randint(100000, 1000000, 100),
    }, index=dates)
    
    # Test indicators
    print(f"✓ RSI: {calculate_rsi(data['close']).iloc[-1]:.2f}")
    
    macd, sig, hist = calculate_macd(data['close'])
    print(f"✓ MACD: {macd.iloc[-1]:.4f}, Signal: {sig.iloc[-1]:.4f}")
    
    print(f"✓ ATR: {calculate_atr(data).iloc[-1]:.4f}")
    print(f"✓ Z-Score: {calculate_z_score(data['close']).iloc[-1]:.2f}")
    print(f"✓ Volume Surge: {calculate_volume_surge(data['volume']).iloc[-1]:.2f}x")
    
    slope, r2 = calculate_trend_strength(data['close'])
    print(f"✓ Trend: slope={slope:.4f}, R²={r2:.4f}")
    
    rr = calculate_risk_reward(data)
    print(f"✓ Risk/Reward: {rr:.2f}")
    
    # Test consolidated function
    df_with_indicators = calculate_all_indicators(data)
    print(f"✓ All indicators: {len(df_with_indicators.columns)} columns")
    
    # Test snapshot
    snapshot = get_indicator_snapshot(data)
    print(f"✓ Snapshot: RSI={snapshot['rsi']:.1f}, MACD bullish={snapshot['macd_bullish']}")
    
    print("\ntechnical_analysis module tests passed!")
