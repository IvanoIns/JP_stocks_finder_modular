"""
JP Stocks Modular Trading System — Scanners

Entry signal detection strategies ported from RisingStarsScanner.
Each scanner returns a (score, reasons) tuple or (0, []) if no signal.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import config
import technical_analysis as ta


# =============================================================================
# Scanner Type Alias
# =============================================================================
ScanResult = Tuple[int, List[str]]  # (score, list of reasons)


# =============================================================================
# 1. MOMENTUM STAR
# =============================================================================

def scan_momentum_star(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find stocks in strong uptrend with controlled RSI and volume.
    
    Signals:
        +30: EMA Fast > EMA Slow (uptrend confirmed)
        +20: RSI in sweet spot (60-80)
        +20: Volume spike > 1.5x average
        +20: Low short interest (< 2%)
        -30: High short interest warning (> 8%)
    
    Threshold: 50 points minimum
    
    Source: JP prototype scan_for_momentum_star()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    
    # 1. Uptrend check: EMA Fast > EMA Slow
    if latest['ema_fast'] > latest['ema_slow']:
        score += 30
        signals.append("Uptrend (EMA Fast>Slow)")
    else:
        # Must be in uptrend for momentum star
        return 0, []
    
    # 2. RSI in sweet spot (strong but not overbought)
    rsi_min = scanner_config.get('MOMENTUM_RSI_MIN', 60)
    rsi_max = scanner_config.get('MOMENTUM_RSI_MAX', 80)
    if rsi_min < latest['rsi'] < rsi_max:
        score += 20
        signals.append(f"Strong RSI ({latest['rsi']:.1f})")
    
    # 3. Volume spike
    vol_threshold = scanner_config.get('MOMENTUM_VOLUME_SPIKE_FACTOR', 1.5)
    if latest['volume_surge'] > vol_threshold:
        score += 20
        signals.append(f"Volume Spike ({latest['volume_surge']:.1f}x)")
    
    # 4. Short interest check
    short_ratio = jpx_data.get('short_ratio', 0) if isinstance(jpx_data, dict) else 0
    low_short = scanner_config.get('MOMENTUM_LOW_SHORT_INTEREST', 0.02)
    high_short = scanner_config.get('MOMENTUM_HIGH_SHORT_WARNING', 0.08)
    
    if short_ratio < low_short:
        score += 20
        signals.append("Low Short Interest")
    elif short_ratio > high_short:
        score -= 30
        signals.append("Warning: High Shorts")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 2. REVERSAL ROCKET
# =============================================================================

def scan_reversal_rocket(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find oversold stocks with reversal potential.
    
    Signals:
        +20: Price < EMA Slow (downtrend prerequisite)
        +30: Z-Score < -2.0 (oversold)
        +20: Capitulation volume (3x+ average)
        +20: Bullish reversal pattern detected
        +30: High short interest (squeeze fuel > 10%)
    
    Threshold: 70 points minimum
    
    Source: JP prototype scan_for_reversal_rocket()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    
    # 1. Downtrend check (prerequisite for reversal)
    if latest['close'] < latest['ema_slow']:
        score += 20
        signals.append("Downtrend (Price<EMA Slow)")
    else:
        # Must be in downtrend for reversal
        return 0, []
    
    # 2. Z-Score oversold
    z_threshold = scanner_config.get('REVERSAL_ZSCORE_THRESHOLD', -2.0)
    if latest['z_score'] < z_threshold:
        score += 30
        signals.append(f"Oversold (Z-Score {latest['z_score']:.2f})")
    
    # 3. Capitulation volume
    vol_threshold = scanner_config.get('REVERSAL_VOLUME_SPIKE_FACTOR', 3.0)
    if latest['volume_surge'] > vol_threshold:
        score += 20
        signals.append("Capitulation Volume")
    
    # 4. Bullish reversal pattern
    has_pattern, pattern_name = ta.check_bullish_reversal_pattern(data)
    if has_pattern:
        score += 20
        signals.append(f"Reversal Pattern ({pattern_name})")
    
    # 5. High short interest (squeeze fuel)
    short_ratio = jpx_data.get('short_ratio', 0) if isinstance(jpx_data, dict) else 0
    fuel_threshold = scanner_config.get('REVERSAL_HIGH_SHORT_FUEL', 0.10)
    
    if short_ratio >= fuel_threshold:
        score += 30
        signals.append("High Short Interest (Squeeze Fuel)")
    
    return (score, signals) if score >= 70 else (0, [])


# =============================================================================
# 3. CONSOLIDATION BREAKOUT
# =============================================================================

def scan_consolidation_breakout(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find stocks consolidating before potential breakout.
    
    Signals:
        +30: Tight price range (< 8% over 20 days)
        +10: Positioned near highs of range (>60%)
        +20: ATR contraction (volatility decreasing)
        +15: Volume drying up
        +20: RSI in ready zone (58-72)
        +15: Near 52-week high (within striking distance)
    
    Threshold: 50 points minimum
    
    Source: JP prototype scan_consolidation_breakout()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    recent = data.iloc[-20:]
    
    # 1. Price consolidation (tight range)
    high_20d = recent['high'].max()
    low_20d = recent['low'].min()
    current_price = latest['close']
    
    if high_20d > 0:
        consolidation_range = (high_20d - low_20d) / high_20d
        range_max = scanner_config.get('CONSOLIDATION_RANGE_MAX', 0.08)
        
        if consolidation_range < range_max:
            score += 30
            signals.append(f"Tight Range ({consolidation_range:.1%})")
            
            # Position in range (prefer near highs)
            if high_20d > low_20d:
                position_in_range = (current_price - low_20d) / (high_20d - low_20d)
                if position_in_range > 0.6:
                    score += 10
                    signals.append("Near Range Highs")
    
    # 2. Volatility contraction
    if len(data) >= 40:
        atr_10 = ta.calculate_atr(data.iloc[-10:]).iloc[-1]
        atr_30 = ta.calculate_atr(data.iloc[-30:]).iloc[-1]
        
        if atr_30 > 0:
            contraction = (atr_30 - atr_10) / atr_30
            vol_drop = scanner_config.get('CONSOLIDATION_VOLATILITY_DROP', 0.20)
            if contraction > vol_drop:
                score += 20
                signals.append(f"Volatility Contracting ({contraction:.0%})")
    
    # 3. Volume drying up
    vol_recent = data['volume'].iloc[-5:].mean()
    vol_prior = data['volume'].iloc[-20:-5].mean()
    
    if vol_prior > 0:
        vol_change = (vol_recent - vol_prior) / vol_prior
        if -0.3 < vol_change < -0.1:
            score += 15
            signals.append("Volume Drying Up")
    
    # 4. RSI in ready zone
    rsi_min = scanner_config.get('CONSOLIDATION_RSI_MIN', 58)
    rsi_max = scanner_config.get('CONSOLIDATION_RSI_MAX', 72)
    if rsi_min <= latest['rsi'] <= rsi_max:
        score += 20
        signals.append(f"RSI Ready ({latest['rsi']:.0f})")
    
    # 5. Near 52-week high
    if len(data) >= 252:
        high_52w = data['high'].iloc[-252:].max()
        distance = (high_52w - current_price) / current_price
        
        if 0.02 < distance < 0.08:
            score += 15
            signals.append("Near 52w High")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 4. RELATIVE STRENGTH
# =============================================================================

def scan_relative_strength(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find stocks showing relative strength.
    
    Signals:
        +30: Bullish MA alignment (Price > SMA20 > SMA50)
        +25: Short-term momentum (3-15% in 10 days)
        +20: Medium-term momentum (5-25% in 30 days)
        +25: High R-squared on trend (consistent)
    
    Threshold: 50 points minimum
    
    Source: JP prototype scan_relative_strength()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    close = data['close']
    current_price = latest['close']
    
    # 1. Bullish MA alignment: Price > SMA20 > SMA50
    if current_price > latest['sma_20'] > latest['sma_50']:
        score += 30
        signals.append("Price>MA20>MA50")
    
    # 2. Short-term momentum (10-day)
    if len(close) >= 10:
        momentum_10d = (current_price / close.iloc[-10] - 1)
        mom_min = scanner_config.get('RS_MOMENTUM_SHORT_MIN', 0.03)
        mom_max = scanner_config.get('RS_MOMENTUM_SHORT_MAX', 0.15)
        
        if mom_min < momentum_10d < mom_max:
            score += 25
            signals.append(f"Momentum +{momentum_10d:.1%}/10d")
    
    # 3. Medium-term momentum (30-day)
    if len(close) >= 30:
        momentum_30d = (current_price / close.iloc[-30] - 1)
        mom_min = scanner_config.get('RS_MOMENTUM_MID_MIN', 0.05)
        mom_max = scanner_config.get('RS_MOMENTUM_MID_MAX', 0.25)
        
        if mom_min < momentum_30d < mom_max:
            score += 20
            signals.append(f"+{momentum_30d:.1%}/30d")
    
    # 4. Trend consistency (R-squared)
    slope, r_squared = ta.calculate_trend_strength(close, period=20)
    r2_min = scanner_config.get('RS_R_SQUARED_MIN', 0.5)
    
    if r_squared > r2_min and slope > 0:
        score += 25
        signals.append("Strong Trend")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 5. BURST CANDIDATES
# =============================================================================

def scan_burst_candidates(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find stocks matching "forensic signature" of past winners.
    
    Signals:
        +25: RSI in range (30-70)
        +25: Volume ratio in range (80-200%)
        +25: Volatility in range (20-60%)
        +25: SMA ratio in range (0.85-1.15)
    
    Pre-filter: Risk/reward >= 2.0
    Threshold: 50 points minimum
    
    Source: JP prototype scan_for_burst_candidates()
    """
    if len(data) < 200:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    # Risk/Reward pre-filter
    risk_reward = ta.calculate_risk_reward(data)
    min_rr = scanner_config.get('MIN_RISK_REWARD', 2.0)
    if risk_reward < min_rr:
        return 0, [f"Poor R/R: {risk_reward:.2f}"]
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    close = data['close'].values
    volume = data['volume'].values
    
    # 1. RSI in range
    rsi_min = scanner_config.get('BURST_RSI_MIN', 30)
    rsi_max = scanner_config.get('BURST_RSI_MAX', 70)
    if rsi_min <= latest['rsi'] <= rsi_max:
        score += 25
        signals.append(f"RSI:{latest['rsi']:.0f}")
    
    # 2. Volume ratio
    vol_30d = volume[-30:].mean() if len(volume) >= 30 else 0
    vol_90d_prior = volume[-120:-30].mean() if len(volume) >= 120 else vol_30d
    volume_ratio = (vol_30d / vol_90d_prior * 100) if vol_90d_prior > 0 else 100
    
    vol_min = scanner_config.get('BURST_VOLUME_RATIO_MIN', 80)
    vol_max = scanner_config.get('BURST_VOLUME_RATIO_MAX', 200)
    if vol_min <= volume_ratio <= vol_max:
        score += 25
        signals.append(f"Vol:{volume_ratio:.0f}%")
    
    # 3. Volatility in range
    volatility = latest['volatility']
    vol_low = scanner_config.get('BURST_VOLATILITY_MIN', 20)
    vol_high = scanner_config.get('BURST_VOLATILITY_MAX', 60)
    if vol_low <= volatility <= vol_high:
        score += 25
        signals.append(f"σ:{volatility:.0f}%")
    
    # 4. SMA ratio (SMA50/SMA200)
    sma_50 = close[-50:].mean() if len(close) >= 50 else close[-1]
    sma_200 = close[-200:].mean() if len(close) >= 200 else sma_50
    sma_ratio = sma_50 / sma_200 if sma_200 > 0 else 1.0
    
    sma_min = scanner_config.get('BURST_SMA_RATIO_MIN', 0.85)
    sma_max = scanner_config.get('BURST_SMA_RATIO_MAX', 1.15)
    if sma_min <= sma_ratio <= sma_max:
        score += 25
        signals.append(f"SMA:{sma_ratio:.2f}")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 6. OVERSOLD BOUNCE
# =============================================================================

def scan_oversold_bounce(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Mean reversion on oversold stocks.
    
    Signals:
        +30: 10-day momentum <= -5%
        +25: Price <= -5% below SMA20
        +25: RSI <= 35
        +15: Volume surge >= 1.5x
        +15: Short interest >= 3%
        +10: Combo: RSI <= 30 AND shorts > 5% (squeeze)
        +5: Monday (reversal day)
    
    Threshold: 50 points minimum
    
    Source: JP prototype detect_oversold_bounce()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    close = data['close']
    
    # 1. 10-day momentum check
    if len(close) >= 10:
        momentum_10d = (latest['close'] / close.iloc[-10] - 1) * 100
        mom_threshold = scanner_config.get('OVERSOLD_MOMENTUM_10D', -5.0)
        if momentum_10d <= mom_threshold:
            score += 30
            signals.append(f"Mom10d {momentum_10d:.1f}%")
    
    # 2. Price vs SMA20
    sma_20 = latest['sma_20']
    price_vs_sma = (latest['close'] / sma_20 - 1) * 100 if sma_20 > 0 else 0
    sma_threshold = scanner_config.get('OVERSOLD_PRICE_BELOW_SMA20', -5.0)
    if price_vs_sma <= sma_threshold:
        score += 25
        signals.append(f"SMA20 {price_vs_sma:.1f}%")
    
    # 3. RSI check
    rsi_threshold = scanner_config.get('OVERSOLD_RSI_MAX', 35)
    if latest['rsi'] <= rsi_threshold:
        score += 25
        signals.append(f"RSI {latest['rsi']:.0f}")
    
    # 4. Volume surge
    vol_threshold = scanner_config.get('OVERSOLD_VOLUME_SURGE_MIN', 1.5)
    if latest['volume_surge'] >= vol_threshold:
        score += 15
        signals.append(f"Vol {latest['volume_surge']:.1f}x")
    
    # 5. Short interest
    short_ratio = jpx_data.get('short_ratio', 0) if isinstance(jpx_data, dict) else 0
    short_threshold = scanner_config.get('OVERSOLD_SHORT_INTEREST_MIN', 0.03)
    if short_ratio >= short_threshold:
        score += 15
        signals.append(f"Short {short_ratio*100:.1f}%")
        
        # Combo: Oversold + High Shorts = Squeeze potential
        if latest['rsi'] <= 30 and short_ratio > 0.05:
            score += 10
            signals.append("SQUEEZE!")
    
    # 6. Day of week (Monday reversal)
    day_of_week = datetime.now().weekday()
    if day_of_week == 0:  # Monday
        score += 5
        signals.append("Monday Reversal")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 7. VOLATILITY EXPLOSION
# =============================================================================

def scan_volatility_explosion(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    High volatility stocks near lows.
    
    Signals:
        +40: High volatility (>= 40% annualized)
        +30: Near lows of range (bottom 30%)
        +20: Volume surge >= 1.5x
        +10: Short interest >= 3% (fuel)
    
    Threshold: 50 points minimum
    
    Source: JP prototype detect_volatility_explosion()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    
    # 1. High volatility check
    vol_threshold = scanner_config.get('HIGH_VOLATILITY_MIN', 40.0)
    if latest['volatility'] >= vol_threshold:
        score += 40
        signals.append(f"HighVol {latest['volatility']:.0f}%")
        
        # 2. Position in recent range (prefer near lows)
        recent = data.iloc[-20:]
        high_20 = recent['high'].max()
        low_20 = recent['low'].min()
        current = latest['close']
        
        if high_20 > low_20:
            position = (current - low_20) / (high_20 - low_20)
            near_low_threshold = scanner_config.get('VOLATILITY_NEAR_LOW_THRESHOLD', 0.30)
            
            if position <= near_low_threshold:
                score += 30
                signals.append("Near Lows")
        
        # 3. Volume confirmation
        if latest['volume_surge'] >= 1.5:
            score += 20
            signals.append(f"Vol {latest['volume_surge']:.1f}x")
        
        # 4. Short interest adds fuel
        short_ratio = jpx_data.get('short_ratio', 0) if isinstance(jpx_data, dict) else 0
        if short_ratio >= 0.03:
            score += 10
            signals.append("Shorts Trapped")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 8. POWER COMBINATIONS
# =============================================================================

def scan_power_combinations(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Multiple signals together = higher probability.
    
    Combos:
        +50: TRIPLE (RSI <= 30 + Volume >= 1.5x + Shorts >= 3%)
        +40: EXTREME (Momentum <= -8% + Volatility >= 40%)
        +30: Friday Oversold (Friday + RSI <= 35)
        +20: Month-end Oversold (day >= 25 + Momentum <= -5%)
    
    Threshold: 50 points minimum
    
    Source: JP prototype detect_power_combinations()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    # Calculate indicators
    df = ta.calculate_all_indicators(data, scanner_config)
    latest = df.iloc[-1]
    close = data['close']
    
    # Pre-calculate values
    rsi = latest['rsi']
    vol_surge = latest['volume_surge']
    volatility = latest['volatility']
    short_ratio = jpx_data.get('short_ratio', 0) if isinstance(jpx_data, dict) else 0
    
    momentum_10d = (latest['close'] / close.iloc[-10] - 1) * 100 if len(close) >= 10 else 0
    
    # COMBO 1: Triple threat
    if rsi <= 30 and vol_surge >= 1.5 and short_ratio >= 0.03:
        score += 50
        signals.append("TRIPLE: RSI+Vol+Short")
    
    # COMBO 2: Extreme oversold + high volatility
    if momentum_10d <= -8 and volatility >= 40:
        score += 40
        signals.append("EXTREME: Oversold+Volatile")
    
    # COMBO 3: Friday oversold
    if datetime.now().weekday() == 4 and rsi <= 35:  # Friday
        score += 30
        signals.append("Friday Oversold")
    
    # COMBO 4: Month-end oversold
    if datetime.now().day >= 25 and momentum_10d <= -5:
        score += 20
        signals.append("Month-end Oversold")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# AGGREGATOR
# =============================================================================

def get_all_signals(
    symbol: str,
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None,
    min_score: int = None
) -> List[Dict]:
    """
    Run all scanners and return list of qualifying signals.
    
    A stock can have multiple signals (different strategies).
    
    Args:
        symbol: Stock symbol
        data: DataFrame with OHLCV data
        jpx_data: Short interest data dict for this symbol
        scanner_config: Scanner configuration
        min_score: Minimum score to include (default from config)
    
    Returns:
        List of dicts: [
            {'symbol': '7203.T', 'strategy': 'momentum_star', 'score': 70, 'reasons': [...]},
            ...
        ]
    """
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    if min_score is None:
        min_score = config.MIN_SCORE
    
    # Ensure jpx_data is a dict
    if not isinstance(jpx_data, dict):
        jpx_data = {'short_ratio': 0}
    
    signals = []
    
    scanners = [
        ('momentum_star', scan_momentum_star),
        ('reversal_rocket', scan_reversal_rocket),
        ('consolidation_breakout', scan_consolidation_breakout),
        ('relative_strength', scan_relative_strength),
        ('burst_candidates', scan_burst_candidates),
        ('oversold_bounce', scan_oversold_bounce),
        ('volatility_explosion', scan_volatility_explosion),
        ('power_combinations', scan_power_combinations),
    ]
    
    for name, scanner_func in scanners:
        try:
            score, reasons = scanner_func(data, jpx_data, scanner_config)
            if score >= min_score:
                signals.append({
                    'symbol': symbol,
                    'strategy': name,
                    'score': score,
                    'reasons': reasons,
                    'price': data['close'].iloc[-1] if len(data) > 0 else 0,
                })
        except Exception as e:
            # Log but don't fail the whole scan
            continue
    
    # Sort by score descending
    signals.sort(key=lambda x: x['score'], reverse=True)
    
    return signals


def scan_universe(
    symbols_data: Dict[str, pd.DataFrame],
    jpx_data: Dict[str, dict],
    scanner_config: dict = None,
    min_score: int = None,
    progress: bool = False
) -> List[Dict]:
    """
    Scan multiple symbols and aggregate all signals.
    
    Args:
        symbols_data: Dict of {symbol: DataFrame}
        jpx_data: Dict of {symbol: {'short_ratio': float}}
        scanner_config: Scanner configuration
        min_score: Minimum score
        progress: Show progress bar
    
    Returns:
        List of all qualifying signals, sorted by score
    """
    all_signals = []
    
    iterator = symbols_data.items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Scanning symbols")
    
    for symbol, data in iterator:
        symbol_jpx = jpx_data.get(symbol, {'short_ratio': 0})
        signals = get_all_signals(symbol, data, symbol_jpx, scanner_config, min_score)
        all_signals.extend(signals)
    
    # Sort all signals by score
    all_signals.sort(key=lambda x: x['score'], reverse=True)
    
    return all_signals


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing scanners module...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    close = 100 + np.cumsum(np.random.randn(250) * 0.5)
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(250) * 0.01),
        'high': close * (1 + abs(np.random.randn(250) * 0.02)),
        'low': close * (1 - abs(np.random.randn(250) * 0.02)),
        'close': close,
        'volume': np.random.randint(100000, 1000000, 250),
    }, index=dates)
    
    jpx_data = {'short_ratio': 0.02}
    
    # Test each scanner
    scanners = [
        ('momentum_star', scan_momentum_star),
        ('reversal_rocket', scan_reversal_rocket),
        ('consolidation_breakout', scan_consolidation_breakout),
        ('relative_strength', scan_relative_strength),
        ('burst_candidates', scan_burst_candidates),
        ('oversold_bounce', scan_oversold_bounce),
        ('volatility_explosion', scan_volatility_explosion),
        ('power_combinations', scan_power_combinations),
    ]
    
    for name, func in scanners:
        score, reasons = func(data, jpx_data)
        status = "[+]" if score > 0 else "[ ]"
        print(f"{status} {name}: score={score}, reasons={reasons[:2] if reasons else []}")
    
    # Test aggregator
    signals = get_all_signals('TEST.T', data, jpx_data)
    print(f"\n[+] get_all_signals: {len(signals)} signals found")
    if signals:
        print(f"  Best: {signals[0]['strategy']} (score={signals[0]['score']})")
    
    print("\nscanners module tests passed!")
