"""
Build a field mapping from yfinance for JP tickers across market-cap deciles.

Example:
  python yfinance_field_mapper.py --samples-per-layer 3 --layers 10 --sleep 0.5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

import config


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value[:5]]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in list(value.items())[:10]}
    return str(value)


def _value_type(value: Any) -> str:
    if value is None:
        return "none"
    return type(value).__name__


def _short_value(value: Any, max_len: int = 160) -> str:
    v = _json_safe(value)
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _load_symbols_with_market_cap() -> pd.DataFrame:
    conn = sqlite3.connect(config.DATABASE_FILE)
    df = pd.read_sql(
        """
        SELECT symbol, market_cap
        FROM symbol_info
        WHERE market_cap IS NOT NULL AND market_cap > 0
        """,
        conn,
    )
    conn.close()
    return df


def _assign_layers(df: pd.DataFrame, layers: int) -> pd.DataFrame:
    if df.empty:
        return df.assign(layer=pd.Series(dtype=int))
    try:
        df = df.copy()
        df["layer"] = pd.qcut(df["market_cap"], layers, labels=False, duplicates="drop") + 1
        return df
    except ValueError:
        df = df.copy()
        df["layer"] = 1
        return df


def _sample_tickers(df: pd.DataFrame, samples_per_layer: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    sampled = []
    for layer, group in df.groupby("layer"):
        n = min(samples_per_layer, len(group))
        sampled.append(group.sample(n=n, random_state=seed))
    return pd.concat(sampled, ignore_index=True)


def _fetch_ticker_fields(symbol: str) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    fields: dict[str, Any] = {}
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info:
        try:
            for k, v in fast_info.items():
                fields[f"fast_info.{k}"] = v
        except Exception:
            pass
    try:
        info = ticker.info or {}
        for k, v in info.items():
            fields[f"info.{k}"] = v
    except Exception:
        pass
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=10, help="Market-cap layers (deciles)")
    parser.add_argument("--samples-per-layer", type=int, default=3, help="Sample tickers per layer")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = _load_symbols_with_market_cap()
    if df.empty:
        print("No symbols with market cap found in symbol_info.")
        print("Run: python -c \"import data_manager as dm; dm.update_market_caps_incremental(max_symbols=1000)\"")
        return 1

    df = _assign_layers(df, args.layers)
    sampled = _sample_tickers(df, args.samples_per_layer, args.seed)
    if sampled.empty:
        print("No symbols sampled.")
        return 1

    field_stats = defaultdict(lambda: {"type": None, "samples": [], "count": 0})
    sample_meta = []

    for _, row in sampled.iterrows():
        symbol = row["symbol"]
        layer = int(row["layer"])
        market_cap = float(row["market_cap"])
        fields = _fetch_ticker_fields(symbol)

        sample_meta.append(
            {
                "symbol": symbol,
                "layer": layer,
                "market_cap": market_cap,
                "field_count": len(fields),
            }
        )

        for field, value in fields.items():
            stat = field_stats[field]
            stat["count"] += 1
            if stat["type"] is None and value is not None:
                stat["type"] = _value_type(value)
            if len(stat["samples"]) < 3:
                stat["samples"].append(
                    {
                        "symbol": symbol,
                        "layer": layer,
                        "value": _json_safe(value),
                    }
                )

        time.sleep(args.sleep)

    total_samples = len(sample_meta)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    field_rows = []
    for field, stat in field_stats.items():
        sample_values = [_short_value(s["value"]) for s in stat["samples"]]
        sample_symbols = [s["symbol"] for s in stat["samples"]]
        field_rows.append(
            {
                "field": field,
                "source": field.split(".", 1)[0],
                "value_type": stat["type"] or "unknown",
                "non_null_count": stat["count"],
                "total_samples": total_samples,
                "sample_values": " | ".join(sample_values),
                "sample_tickers": " | ".join(sample_symbols),
            }
        )

    field_rows = sorted(field_rows, key=lambda r: (r["source"], r["field"]))
    df_fields = pd.DataFrame(field_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"yfinance_field_samples_{timestamp}.json"
    csv_path = results_dir / f"yfinance_field_samples_{timestamp}.csv"
    template_path = results_dir / f"yfinance_field_mapping_template_{timestamp}.csv"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "layers": args.layers,
        "samples_per_layer": args.samples_per_layer,
        "sampled_tickers": sample_meta,
        "field_stats": field_stats,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    df_fields.to_csv(csv_path, index=False, encoding="utf-8")

    df_template = df_fields[["field", "source"]].copy()
    df_template["description"] = ""
    df_template["use_in_analysis"] = ""
    df_template["notes"] = ""
    df_template.to_csv(template_path, index=False, encoding="utf-8")

    print(f"Samples saved: {json_path}")
    print(f"Field CSV saved: {csv_path}")
    print(f"Mapping template saved: {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
