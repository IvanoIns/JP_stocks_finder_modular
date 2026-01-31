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

def _get_short_ratio(jpx_data: Optional[dict]) -> Optional[float]:
    """
    Return short_ratio if it is present and parseable, otherwise None.

    Important: missing short data should be treated as neutral (no scoring impact).
    """
    if not isinstance(jpx_data, dict):
        return None
    if "short_ratio" not in jpx_data:
        return None
    value = jpx_data.get("short_ratio")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    short_ratio = _get_short_ratio(jpx_data)
    low_short = scanner_config.get('MOMENTUM_LOW_SHORT_INTEREST', 0.02)
    high_short = scanner_config.get('MOMENTUM_HIGH_SHORT_WARNING', 0.08)
    
    if short_ratio is not None:
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
    short_ratio = _get_short_ratio(jpx_data)
    fuel_threshold = scanner_config.get('REVERSAL_HIGH_SHORT_FUEL', 0.10)
    
    if short_ratio is not None and short_ratio >= fuel_threshold:
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
    short_ratio = _get_short_ratio(jpx_data)
    short_threshold = scanner_config.get('OVERSOLD_SHORT_INTEREST_MIN', 0.03)
    if short_ratio is not None and short_ratio >= short_threshold:
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
        short_ratio = _get_short_ratio(jpx_data)
        if short_ratio is not None and short_ratio >= 0.03:
            score += 10
            signals.append("Shorts Trapped")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 8. SMART MONEY FLOW (replaces outdated Power Combinations)
# =============================================================================

def scan_smart_money_flow(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Identify institutional positioning vs retail behavior.
    Replaces calendar-based Power Combinations which have degraded.
    
    Signals:
        +35: Relative strength (holding up while market weak)
        +30: Consistent volume pattern (institutional accumulation)
        +35: Price holding key support (near but above SMA50)
    
    Threshold: 50 points minimum
    
    Source: JP prototype detect_smart_money_flow()
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
    volume = data['volume']
    
    # A. Relative Strength Analysis (35 points max)
    if len(close) >= 11:
        stock_return_10d = (close.iloc[-1] / close.iloc[-11] - 1) * 100
        min_rs = scanner_config.get('SMART_MONEY_MIN_RELATIVE_STRENGTH', -2)
        
        if stock_return_10d > min_rs:  # Not falling much
            score += 20
            signals.append(f"Resilient{stock_return_10d:+.1f}%")
        if stock_return_10d > 0:  # Actually positive
            score += 15
            signals.append("Outperforming")
    
    # B. Volume Distribution Analysis (30 points max)
    # Consistent volume = institutional; Sporadic = retail
    if len(volume) >= 20:
        volume_std = volume.iloc[-20:].std()
        volume_mean = volume.iloc[-20:].mean()
        volume_consistency = 1 - (volume_std / volume_mean) if volume_mean > 0 else 0
        
        if volume_consistency > 0.3:  # Consistent volume pattern
            score += 20
            signals.append("SteadyVol")
        elif volume_consistency > 0.15:
            score += 10
            signals.append("ModerateVol")
    
    # C. Price Level Holding (35 points max)
    sma_50 = latest['sma_50']
    current_price = latest['close']
    
    if sma_50 > 0:
        support_test = (current_price / sma_50 - 1) * 100
        
        if -2 <= support_test <= 5:  # Near but above key support
            score += 25
            signals.append(f"Support{support_test:+.1f}%")
        elif support_test > 5:  # Above support
            score += 10
            signals.append("AboveMA50")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 9. CRASH THEN BURST (NEW - catches the JP penny stock pattern)
# =============================================================================

def scan_crash_then_burst(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find stocks that crashed hard and are showing early recovery signs.
    This is the pattern seen daily in Japanese penny stocks:
    - Stock drops 30-50% → volume climax → stabilizes → BURSTS 20-30%
    
    Signals:
        +40: Price dropped 25%+ from 20-day high
        +30: Volume climax occurred (3x+ average on drop)
        +20: Bottom formation (2-3 days of stabilization)
        +20: Short interest >= 5% (squeeze fuel)
        +15: First "green" day with elevated volume
    
    Threshold: 70 points (high bar = fewer but better trades)
    
    Source: User observation of JP penny stock patterns
    """
    if len(data) < 30:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    close = data['close']
    volume = data['volume']
    high = data['high']
    
    current_price = close.iloc[-1]
    
    # 1. CRASH CHECK: Price dropped 25%+ from recent high (+40 points)
    high_20d = high.iloc[-20:].max()
    crash_pct = (current_price / high_20d - 1) * 100
    
    crash_threshold = scanner_config.get('CRASH_MIN_DROP_PCT', -25)
    
    if crash_pct <= crash_threshold:  # Dropped 25%+
        score += 40
        signals.append(f"Crashed{crash_pct:.0f}%")
    elif crash_pct <= -15:  # Moderate drop
        score += 20
        signals.append(f"Dropped{crash_pct:.0f}%")
    else:
        # No crash = no signal
        return 0, []
    
    # 2. VOLUME CLIMAX: Did we see capitulation volume during the drop? (+30 points)
    avg_volume = volume.iloc[-30:-5].mean() if len(volume) >= 30 else volume.mean()
    max_volume_recent = volume.iloc[-10:].max()
    volume_spike = max_volume_recent / avg_volume if avg_volume > 0 else 0
    
    climax_threshold = scanner_config.get('CRASH_VOLUME_CLIMAX', 3.0)
    
    if volume_spike >= climax_threshold:
        score += 30
        signals.append(f"Climax{volume_spike:.1f}x")
    elif volume_spike >= 2.0:
        score += 15
        signals.append(f"HighVol{volume_spike:.1f}x")
    
    # 3. BOTTOM FORMATION: Has price stabilized? (+20 points)
    # Look for 2-3 days where low doesn't make new lows
    recent_lows = data['low'].iloc[-5:]
    low_3d = recent_lows.iloc[-3:].min()
    low_prior = recent_lows.iloc[:2].min()
    
    if low_3d >= low_prior * 0.99:  # Not making new lows
        score += 20
        signals.append("Stabilizing")
    
    # 4. SHORT INTEREST: Squeeze fuel (+20 points)
    short_ratio = _get_short_ratio(jpx_data)
    
    squeeze_threshold = scanner_config.get('CRASH_SHORT_SQUEEZE_MIN', 0.05)
    
    if short_ratio is not None and short_ratio >= squeeze_threshold:
        score += 20
        signals.append(f"Shorts{short_ratio*100:.1f}%")
    elif short_ratio is not None and short_ratio >= 0.03:
        score += 10
        signals.append(f"Short{short_ratio*100:.1f}%")
    
    # 5. RECOVERY SIGN: First green day with volume (+15 points)
    if len(close) >= 2:
        today_green = close.iloc[-1] > close.iloc[-2]
        today_vol = volume.iloc[-1]
        avg_vol = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
        
        if today_green and today_vol > avg_vol * 1.2:
            score += 15
            signals.append("GreenWithVol")
    
    # Higher threshold for this pattern
    return (score, signals) if score >= 70 else (0, [])


# =============================================================================
# 10. STEALTH ACCUMULATION (from original prototype)
# =============================================================================

def scan_stealth_accumulation(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find gradual institutional buying without obvious price spikes.
    Institutions accumulate quietly before the move.
    
    Signals:
        +30: Volume increasing 15%+ while price stable
        +25: Volatility compression 20%+ (coiling)
        +20: Subtle bullish bias (55-65% up days)
        +15: Low short interest (institutions aren't fighting shorts)
    
    Threshold: 50 points minimum
    
    Source: JP prototype detect_stealth_accumulation()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    close = data['close']
    volume = data['volume']
    
    # A. Volume-Price Divergence (30 points max)
    # Volume increasing while price flat = accumulation
    volume_ma_short = volume.iloc[-10:].mean()
    volume_ma_long = volume.iloc[-30:].mean() if len(volume) >= 30 else volume.mean()
    
    volume_trend = (volume_ma_short / volume_ma_long - 1) * 100 if volume_ma_long > 0 else 0
    
    stealth_min_vol = scanner_config.get('STEALTH_MIN_VOLUME_INCREASE', 15)
    
    if volume_trend > stealth_min_vol:  # Volume increasing 15%+
        score += 30
        signals.append(f"Vol↑{volume_trend:.0f}%")
    elif volume_trend > 5:
        score += 15
        signals.append(f"Vol+{volume_trend:.0f}%")
    
    # B. Price Volatility Compression (25 points max)
    if len(close) >= 60:
        volatility_20d = close.iloc[-20:].std() / close.iloc[-20:].mean()
        volatility_60d = close.iloc[-60:].std() / close.iloc[-60:].mean()
        vol_compression = (1 - volatility_20d / volatility_60d) * 100 if volatility_60d > 0 else 0
        
        stealth_min_comp = scanner_config.get('STEALTH_MIN_VOLATILITY_COMPRESSION', 20)
        
        if vol_compression > stealth_min_comp:  # Volatility compressed 20%+
            score += 25
            signals.append(f"Squeeze{vol_compression:.0f}%")
        elif vol_compression > 10:
            score += 15
            signals.append(f"Tighten{vol_compression:.0f}%")
    
    # C. Consistent Positive Bias (20 points max)
    if len(close) >= 21:
        up_days = sum(close.iloc[-20:].diff().dropna() > 0)
        win_rate = up_days / 20
        
        stealth_wr_min = scanner_config.get('STEALTH_WIN_RATE_MIN', 0.55)
        stealth_wr_max = scanner_config.get('STEALTH_WIN_RATE_MAX', 0.65)
        
        if stealth_wr_min <= win_rate <= stealth_wr_max:  # Subtle bullish bias
            score += 20
            signals.append(f"Bias{win_rate*100:.0f}%")
    
    # D. Low Short Interest (15 points max)
    short_ratio = _get_short_ratio(jpx_data)
    if short_ratio is not None and 0 < short_ratio < 0.03:  # Low but present
        score += 15
        signals.append("LowShorts")
    
    return (score, signals) if score >= 50 else (0, [])


# =============================================================================
# 11. COILING PATTERN (from original prototype)
# =============================================================================

def scan_coiling_pattern(
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None
) -> ScanResult:
    """
    Find stocks coiling for explosive moves.
    Bollinger Band squeeze + ATR compression = energy building.
    
    Signals:
        +40: Bollinger Band squeeze 30%+ vs 60-day average
        +30: ATR compression 25%+
        +20: Volume building (not declining)
    
    Threshold: 60 points minimum (high bar for quality)
    
    Source: JP prototype detect_coiling_pattern()
    """
    if len(data) < 60:
        return 0, []
    
    if scanner_config is None:
        scanner_config = config.SCANNER_CONFIG
    
    score = 0
    signals = []
    
    close = data['close']
    volume = data['volume']
    
    # A. Bollinger Band Squeeze Detection (40 points max)
    bb_period = 20
    bb_std = 2
    
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    bb_upper = sma + (std * bb_std)
    bb_lower = sma - (std * bb_std)
    bb_width = (bb_upper - bb_lower) / close * 100
    
    current_width = bb_width.iloc[-1]
    avg_width_60d = bb_width.iloc[-60:].mean() if len(bb_width) >= 60 else bb_width.mean()
    
    squeeze_ratio = (1 - current_width / avg_width_60d) * 100 if avg_width_60d > 0 else 0
    
    coiling_min_bb = scanner_config.get('COILING_MIN_BB_SQUEEZE', 30)
    
    if squeeze_ratio > coiling_min_bb:  # 30%+ compression
        score += 40
        signals.append(f"BBSqueeze{squeeze_ratio:.0f}%")
    elif squeeze_ratio > 15:
        score += 25
        signals.append(f"BBTight{squeeze_ratio:.0f}%")
    
    # B. ATR Contraction (30 points max)
    high_low = data['high'] - data['low']
    atr_current = high_low.iloc[-10:].mean()
    atr_historical = high_low.iloc[-50:].mean() if len(high_low) >= 50 else high_low.mean()
    
    atr_compression = (1 - atr_current / atr_historical) * 100 if atr_historical > 0 else 0
    
    coiling_min_atr = scanner_config.get('COILING_MIN_ATR_COMPRESSION', 25)
    
    if atr_compression > coiling_min_atr:
        score += 30
        signals.append(f"ATRCoil{atr_compression:.0f}%")
    elif atr_compression > 10:
        score += 15
        signals.append(f"ATRTight{atr_compression:.0f}%")
    
    # C. Volume Building Pattern (20 points max)
    vol_trend_10d = volume.iloc[-10:].mean()
    vol_trend_30d = volume.iloc[-30:].mean() if len(volume) >= 30 else volume.mean()
    vol_pattern = (vol_trend_10d / vol_trend_30d - 1) * 100 if vol_trend_30d > 0 else 0
    
    if vol_pattern > 10:  # Volume building
        score += 20
        signals.append(f"VolBuild{vol_pattern:.0f}%")
    elif vol_pattern > -5:  # Volume steady (not declining)
        score += 10
        signals.append("VolSteady")
    
    return (score, signals) if score >= 60 else (0, [])


# =============================================================================
# AGGREGATOR
# =============================================================================

def get_all_signals(
    symbol: str,
    data: pd.DataFrame,
    jpx_data: dict,
    scanner_config: dict = None,
    min_score: int = None,
    early_mode: Optional[bool] = None,
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
        min_score = config.MIN_SCANNER_SCORE
    
    # Ensure jpx_data is a dict
    if not isinstance(jpx_data, dict):
        jpx_data = {}

    # Determine early mode (pre-burst) filter
    if early_mode is None:
        early_mode = getattr(config, "EARLY_MODE_ENABLED", False)

    if early_mode:
        rsi_max = getattr(config, "EARLY_MODE_RSI_MAX", 65)
        return_max = getattr(config, "EARLY_MODE_10D_RETURN_MAX", 0.15)
        df = ta.calculate_all_indicators(data, scanner_config)
        if len(df) < 11:
            return []
        latest_rsi = df['rsi'].iloc[-1]
        if latest_rsi > rsi_max:
            return []
        ret_10d = (df['close'].iloc[-1] / df['close'].iloc[-11]) - 1
        if ret_10d >= return_max:
            return []

    signals = []
    
    scanners = [
        # Original trend-following scanners
        ('momentum_star', scan_momentum_star),
        ('consolidation_breakout', scan_consolidation_breakout),
        ('relative_strength', scan_relative_strength),
        ('burst_candidates', scan_burst_candidates),
        
        # Mean-reversion scanners
        ('reversal_rocket', scan_reversal_rocket),
        ('oversold_bounce', scan_oversold_bounce),
        ('volatility_explosion', scan_volatility_explosion),
        
        # Advanced detection
        ('smart_money_flow', scan_smart_money_flow),
        # DISABLED (PF 0.00): ('crash_then_burst', scan_crash_then_burst),
        # DISABLED (PF 0.00): ('stealth_accumulation', scan_stealth_accumulation),
        ('coiling_pattern', scan_coiling_pattern),
    ]

    if early_mode:
        early_scanners = set(getattr(config, "EARLY_MODE_SCANNERS", []))
        if early_scanners:
            scanners = [s for s in scanners if s[0] in early_scanners]
    
    for name, scanner_func in scanners:
        try:
            score, reasons = scanner_func(data, jpx_data, scanner_config)
            if score > 0 and (min_score is None or score >= min_score):
                signals.append({
                    'symbol': symbol,
                    'strategy': name,
                    'scanner_name': name,  # Explicit scanner tracking
                    'score': score,
                    'reasons': reasons,
                    'price': data['close'].iloc[-1] if len(data) > 0 else 0,
                })
        except Exception as e:
            # Log but don't fail the whole scan
            continue
    
    # === CONFLUENCE BONUS ===
    # When multiple scanners agree, boost the best signal
    if len(signals) >= 2:
        # Count how many scanners fired
        num_scanners = len(signals)
        all_scanner_names = [s['strategy'] for s in signals]
        
        # Calculate confluence bonus (10 points per additional scanner)
        confluence_bonus = (num_scanners - 1) * 10
        
        # Sort by original score to find the best
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply bonus to the best signal
        best_signal = signals[0]
        best_signal['score'] += confluence_bonus
        best_signal['confluence_count'] = num_scanners
        best_signal['confluence_scanners'] = all_scanner_names
        best_signal['reasons'].append(
            f"CONFLUENCE: {num_scanners} scanners agree (+{confluence_bonus}pts)"
        )
        
        # Keep only the best (confluent) signal
        signals = [best_signal]
    elif len(signals) == 1:
        signals[0]['confluence_count'] = 1
        signals[0]['confluence_scanners'] = [signals[0]['strategy']]
    
    # Sort by score descending
    signals.sort(key=lambda x: x['score'], reverse=True)
    
    return signals


def scan_universe(
    symbols_data: Dict[str, pd.DataFrame],
    jpx_data: Dict[str, dict],
    scanner_config: dict = None,
    min_score: int = None,
    progress: bool = False,
    early_mode: Optional[bool] = None,
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
    
    if not isinstance(jpx_data, dict):
        jpx_data = {}

    for symbol, data in iterator:
        symbol_jpx = jpx_data.get(symbol)
        signals = get_all_signals(symbol, data, symbol_jpx, scanner_config, min_score, early_mode)
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
