# Future Improvements & Testing Strategies

## Current State ✅

**Locked Parameters** (PF 2.46):
- Min Score: 30
- Stop Loss: 6%
- R:R Ratio: 2.0
- Active Scanners: 9 (2 disabled)

---

## Critical Fixes (Before Live)

1. **Validate market-cap coverage**: Auto-population is enabled; monitor completeness for your universe.
2. **Cache supports `min_score` optimization**: Fast mode should allow sweeping thresholds without rebuilding cache.
3. **Signal output vs real entry**: Stops/targets should be computed from actual next-open fill (or output % only).

---

## Strategy 1: Paper Trading ⭐ RECOMMENDED

**Goal**: Validate in real-time before live money

**How**:
1. Update DB: `python -c "import data_manager as dm; dm.update_recent_data(days=5)"`
2. Rebuild cache: `python precompute.py` (auto-expands DB, ~1000 new tickers/run)
3. Run `generate_signals.py` (scanner) or `generate_signals_with_research.py` (scanner + LLM)
4. Log signals to spreadsheet
5. Track theoretical P/L vs actual market moves
6. Run for 2-4 weeks

**Success Criteria**: 
- PF > 1.5 on paper trades
- Win rate > 45%

---

## Strategy 2: Scanner-Specific Tuning

**Goal**: Optimize each star scanner individually

**Candidates**:
| Scanner | Current PF | Test |
|---------|-----------|------|
| oversold_bounce | 11.13 | Tighter score filter? |
| burst_candidates | 4.30 | Different stop sizes? |
| momentum_star | 4.77 | Higher R:R ratio? |

---

## Strategy 3: Position Sizing by Confluence

**Goal**: Bet bigger on stronger signals

```python
if confluence >= 3:
    position_size = 30%
elif confluence == 2:
    position_size = 20%
else:
    position_size = 15%
```

---

## Strategy 4: Time-Based Filters

**Goal**: Find optimal entry timing

**Test**:
- Day of week effects (Mon vs Fri)
- Monthly seasonality
- Earnings season avoidance

---

## Strategy 5: Multi-Timeframe

**Goal**: Weekly trend confirmation

**Concept**:
- Only take daily signals when weekly RSI > 40
- Avoid counter-trend entries

---

## What NOT To Do

❌ Keep running walk-forward hoping for better "stable" params
❌ Lower min_score below 25 (noise)
❌ Use tight stops (< 5%) on small caps
❌ Chase every optimization metric

---

## Recommended Order

1. **Paper Trading** (immediate)
2. **Confluence Position Sizing** (quick win)
3. **Scanner-Specific Tuning** (medium effort)
4. **Time-Based Filters** (research project)

---

*Last updated: 2026-01-29*
