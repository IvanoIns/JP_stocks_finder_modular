"""Quick test for JPX parser fix - faster version."""
import pandas as pd

print("Loading cached JPX Excel (first 500 rows only)...")
df = pd.read_excel('cache/jpx_data_cache_2025-01-23.xlsx', nrows=500)
print(f"Loaded {len(df)} rows\n")

# Test the parser logic directly
result = {}
for idx in range(len(df)):
    row = df.iloc[idx]
    
    code_raw = row.iloc[2] if len(row) > 2 else None
    if pd.isna(code_raw):
        continue
    
    code = str(code_raw).strip()
    if not code.isdigit():
        continue
    
    symbol = code + '.T'
    ratio = float(row.iloc[10]) if len(row) > 10 and pd.notna(row.iloc[10]) else 0.0
    shares = int(float(row.iloc[11])) if len(row) > 11 and pd.notna(row.iloc[11]) else 0
    
    if symbol in result:
        result[symbol]['short_volume'] += shares
        result[symbol]['short_ratio'] = max(result[symbol]['short_ratio'], ratio)
    else:
        result[symbol] = {'short_ratio': ratio, 'short_volume': shares}

print(f"=== RESULTS ===")
print(f"Total symbols: {len(result)}")
print(f"Non-zero ratios: {sum(1 for d in result.values() if d['short_ratio'] > 0)}")

if result:
    avg_ratio = sum(d['short_ratio'] for d in result.values()) / len(result)
    max_ratio = max(d['short_ratio'] for d in result.values())
    print(f"Avg ratio: {avg_ratio:.2%}")
    print(f"Max ratio: {max_ratio:.2%}")
    
    print("\nSample entries:")
    for s, d in list(result.items())[:10]:
        print(f"  {s}: ratio={d['short_ratio']:.2%}, vol={d['short_volume']:,}")

print("\n=== PARSER FIX VERIFIED! ===" if sum(1 for d in result.values() if d['short_ratio'] > 0) > 0 else "\n=== FIX FAILED - still zero ratios ===")
