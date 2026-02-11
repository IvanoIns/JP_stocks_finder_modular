"""
Burst audit workflow:

1) Collect real "close +10%" bursts from DB for one/many dates.
2) Keep an appendable master log (`results/burst_audit/bursts_log.csv`).
3) Audit burst dates against previous-day candidates in `results/daily_picks/`.
4) Support backfill/pending catch-up so missing days can be reprocessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd

import config
import data_manager as dm
import scanners as sc
import technical_analysis as ta
from generate_signals import _attach_entry_prices, _get_budget_params, _save_daily_picks
from precompute import load_precomputed


RESULTS_DIR = Path(getattr(config, "RESULTS_DIR", Path("results")))
AUDIT_DIR = RESULTS_DIR / "burst_audit"
BURSTS_LOG_PATH = AUDIT_DIR / "bursts_log.csv"
COLLECT_STATUS_PATH = AUDIT_DIR / "collect_status.csv"
AUDIT_STATUS_PATH = AUDIT_DIR / "audit_status.csv"
AUDIT_MASTER_PATH = AUDIT_DIR / "audit_master.csv"
AB_AUDIT_MASTER_PATH = AUDIT_DIR / "ab_audit_master.csv"
CANDIDATES_AB_DIR = RESULTS_DIR / "daily_picks_ab"

CACHE_PATH = RESULTS_DIR / "precomputed_cache.pkl"

BURSTS_LOG_COLUMNS = [
    "burst_date",
    "symbol",
    "burst_pct_close",
    "source",
    "notes",
    "close_prev",
    "close_burst",
    "signal_date",
    "last_updated_at",
]

COLLECT_STATUS_COLUMNS = ["burst_date", "auto_count", "processed_at"]
AUDIT_STATUS_COLUMNS = ["burst_date", "row_count", "payload_hash", "audited_at"]

DAILY_PICKS_COLUMNS = [
    "run_timestamp",
    "signal_date",
    "entry_date",
    "symbol",
    "strategy",
    "score",
    "close_price",
    "entry_price",
    "entry_source",
    "stop_price",
    "target_price",
    "lot_size",
    "max_jpy_per_trade",
    "lot_cost",
    "budget_ok",
    "max_shares_within_budget",
    "confluence_count",
    "confluence_scanners",
    "bucket",
]

AB_AUDIT_COLUMNS = [
    "burst_date",
    "signal_date",
    "symbol",
    "captured_a_topn",
    "rank_a",
    "score_a",
    "strategy_a",
    "captured_b_topn",
    "rank_b",
    "score_b",
    "strategy_b",
    "delta_capture_b_minus_a",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dirs() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_AB_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if value in {"", "NAN", "NONE", "NULL"}:
        return ""
    if "." in value:
        return value
    return f"{value}.T"


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    payload = df.copy()
    for col in columns:
        if col not in payload.columns:
            payload[col] = None
    payload = payload[columns]
    payload.to_csv(path, index=False, encoding="utf-8")


def _load_cache():
    if not CACHE_PATH.exists():
        raise SystemExit("Cache not found. Build it first: python precompute.py")
    return load_precomputed(CACHE_PATH)


def _get_db_max_date() -> str | None:
    try:
        conn = sqlite3.connect(config.DATABASE_FILE)
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM daily_prices")
        row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _latest_available_trading_date(pc) -> str:
    db_max = _get_db_max_date()
    if not db_max:
        return pc.trading_dates[-1]
    candidates = [d for d in pc.trading_dates if d <= db_max]
    return candidates[-1] if candidates else pc.trading_dates[-1]


def _resolve_dates(pc, *, date: str | None, start: str | None, end: str | None, pending: bool) -> list[str]:
    latest = _latest_available_trading_date(pc)

    if date:
        if date not in pc.trading_dates:
            raise SystemExit(f"Date {date} is not a trading date in cache.")
        return [date]

    if start is None:
        if pending:
            start = getattr(config, "BURST_AUDIT_START_DATE", pc.trading_dates[0])
        else:
            start = latest
    if end is None:
        end = latest

    dates = [d for d in pc.trading_dates if start <= d <= end and d <= latest]
    if not dates:
        raise SystemExit(f"No trading dates in range {start}..{end}.")
    return dates


def _prev_trading_date(pc, date: str) -> str | None:
    try:
        idx = pc.trading_dates.index(date)
    except ValueError:
        return None
    if idx <= 0:
        return None
    return pc.trading_dates[idx - 1]


def _signal_date_for_burst_date(pc, burst_date: str) -> str | None:
    return _prev_trading_date(pc, burst_date)


def _get_next_trading_date(pc, date: str) -> str | None:
    try:
        idx = pc.trading_dates.index(date)
    except ValueError:
        return None
    if idx + 1 >= len(pc.trading_dates):
        return None
    return pc.trading_dates[idx + 1]


def _load_bursts_log() -> pd.DataFrame:
    df = _read_csv(BURSTS_LOG_PATH, BURSTS_LOG_COLUMNS)
    if df.empty:
        return df

    df["burst_date"] = df["burst_date"].astype(str).str.strip()
    df["burst_date"] = df["burst_date"].replace({"nan": "", "NaN": "", "None": "", "NONE": "", "NULL": ""})
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df["source"] = (
        df["source"].astype(str).str.strip().str.lower().replace({"": "manual", "nan": "manual"})
    )
    df["notes"] = df["notes"].fillna("").astype(str)
    df["burst_pct_close"] = pd.to_numeric(df["burst_pct_close"], errors="coerce")
    df["close_prev"] = pd.to_numeric(df["close_prev"], errors="coerce")
    df["close_burst"] = pd.to_numeric(df["close_burst"], errors="coerce")
    df["signal_date"] = df["signal_date"].fillna("").astype(str)
    df["last_updated_at"] = df["last_updated_at"].fillna("").astype(str)

    now_iso = _now_iso()
    df.loc[df["last_updated_at"] == "", "last_updated_at"] = now_iso
    df = df[(df["burst_date"] != "") & (df["symbol"] != "")]

    source_rank = {"manual": 0, "auto_db": 1}
    df["_source_rank"] = df["source"].map(source_rank).fillna(9)
    df = df.sort_values(
        ["burst_date", "symbol", "_source_rank", "last_updated_at"],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(subset=["burst_date", "symbol"], keep="first")
    df = df.drop(columns=["_source_rank"])
    return df


def _save_bursts_log(df: pd.DataFrame) -> None:
    payload = df.copy()
    if not payload.empty:
        payload = payload.sort_values(["burst_date", "symbol"]).reset_index(drop=True)
    _write_csv(BURSTS_LOG_PATH, payload, BURSTS_LOG_COLUMNS)


def _load_collect_status() -> pd.DataFrame:
    return _read_csv(COLLECT_STATUS_PATH, COLLECT_STATUS_COLUMNS)


def _save_collect_status(df: pd.DataFrame) -> None:
    payload = df.copy()
    if not payload.empty:
        payload = payload.sort_values("burst_date").drop_duplicates("burst_date", keep="last")
    _write_csv(COLLECT_STATUS_PATH, payload, COLLECT_STATUS_COLUMNS)


def _upsert_collect_status(status_df: pd.DataFrame, burst_date: str, auto_count: int) -> pd.DataFrame:
    payload = status_df[status_df["burst_date"] != burst_date].copy() if not status_df.empty else status_df.copy()
    row = pd.DataFrame(
        [{"burst_date": burst_date, "auto_count": int(auto_count), "processed_at": _now_iso()}]
    )
    return pd.concat([payload, row], ignore_index=True)


def _load_audit_status() -> pd.DataFrame:
    return _read_csv(AUDIT_STATUS_PATH, AUDIT_STATUS_COLUMNS)


def _save_audit_status(df: pd.DataFrame) -> None:
    payload = df.copy()
    if not payload.empty:
        payload = payload.sort_values("burst_date").drop_duplicates("burst_date", keep="last")
    _write_csv(AUDIT_STATUS_PATH, payload, AUDIT_STATUS_COLUMNS)


def _upsert_audit_status(status_df: pd.DataFrame, burst_date: str, row_count: int, payload_hash: str) -> pd.DataFrame:
    payload = status_df[status_df["burst_date"] != burst_date].copy() if not status_df.empty else status_df.copy()
    row = pd.DataFrame(
        [
            {
                "burst_date": burst_date,
                "row_count": int(row_count),
                "payload_hash": payload_hash,
                "audited_at": _now_iso(),
            }
        ]
    )
    return pd.concat([payload, row], ignore_index=True)


def _date_payload_hash(df_day: pd.DataFrame) -> str:
    if df_day.empty:
        return hashlib.md5(b"").hexdigest()
    cols = ["symbol", "burst_pct_close", "source", "notes"]
    payload = df_day[cols].copy()
    payload["symbol"] = payload["symbol"].astype(str)
    payload["burst_pct_close"] = payload["burst_pct_close"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):.8f}"
    )
    payload["source"] = payload["source"].astype(str)
    payload["notes"] = payload["notes"].astype(str)
    payload = payload.sort_values("symbol")
    raw = "|".join(
        f"{r.symbol}:{r.burst_pct_close}:{r.source}:{r.notes}"
        for r in payload.itertuples(index=False)
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_close_pairs(prev_date: str, burst_date: str) -> pd.DataFrame:
    conn = dm.get_connection()
    query = """
        SELECT c.symbol, p.close AS close_prev, c.close AS close_burst
        FROM daily_prices c
        JOIN daily_prices p
          ON p.symbol = c.symbol
        WHERE c.date = ? AND p.date = ?
    """
    df = pd.read_sql_query(query, conn, params=(burst_date, prev_date))
    conn.close()
    return df


def _collect_auto_rows_for_date(pc, burst_date: str, threshold: float) -> pd.DataFrame:
    signal_date = _signal_date_for_burst_date(pc, burst_date)
    if signal_date is None:
        return pd.DataFrame(columns=BURSTS_LOG_COLUMNS)

    pairs = _get_close_pairs(signal_date, burst_date)
    if pairs.empty:
        return pd.DataFrame(columns=BURSTS_LOG_COLUMNS)

    pairs["burst_pct_close"] = (pairs["close_burst"] / pairs["close_prev"]) - 1.0
    bursts = pairs[pairs["burst_pct_close"] >= threshold].copy()
    if bursts.empty:
        return pd.DataFrame(columns=BURSTS_LOG_COLUMNS)

    now_iso = _now_iso()
    bursts["burst_date"] = burst_date
    bursts["source"] = "auto_db"
    bursts["notes"] = ""
    bursts["signal_date"] = signal_date
    bursts["last_updated_at"] = now_iso
    bursts["symbol"] = bursts["symbol"].map(_normalize_symbol)

    return bursts[
        [
            "burst_date",
            "symbol",
            "burst_pct_close",
            "source",
            "notes",
            "close_prev",
            "close_burst",
            "signal_date",
            "last_updated_at",
        ]
    ]


def _upsert_auto_rows(bursts_log: pd.DataFrame, auto_rows: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if auto_rows.empty:
        return bursts_log, 0, 0

    existing = bursts_log.copy()
    if existing.empty:
        existing = pd.DataFrame(columns=BURSTS_LOG_COLUMNS)

    existing = existing.copy()
    existing["__key"] = existing["burst_date"].astype(str) + "|" + existing["symbol"].astype(str)
    key_to_idx = {k: i for i, k in zip(existing.index, existing["__key"])}

    inserted_rows: list[dict] = []
    inserted = 0
    updated = 0
    for _, row in auto_rows.iterrows():
        key = f"{row['burst_date']}|{row['symbol']}"
        if key in key_to_idx:
            idx = key_to_idx[key]
            updated += 1
            existing.at[idx, "burst_pct_close"] = row["burst_pct_close"]
            existing.at[idx, "close_prev"] = row["close_prev"]
            existing.at[idx, "close_burst"] = row["close_burst"]
            existing.at[idx, "signal_date"] = row["signal_date"]
            existing.at[idx, "last_updated_at"] = row["last_updated_at"]
            if str(existing.at[idx, "source"]).strip().lower() != "manual":
                existing.at[idx, "source"] = "auto_db"
        else:
            inserted += 1
            inserted_rows.append({col: row.get(col) for col in BURSTS_LOG_COLUMNS})

    payload = existing.drop(columns=["__key"], errors="ignore")
    if inserted_rows:
        if payload.empty:
            payload = pd.DataFrame(inserted_rows)
        else:
            payload = pd.concat([payload, pd.DataFrame(inserted_rows)], ignore_index=True)
    payload = payload[BURSTS_LOG_COLUMNS]
    return payload, inserted, updated


def _ensure_daily_picks_for_signal_date(pc, signal_date: str, overwrite: bool = False) -> tuple[Path, Path]:
    csv_path = RESULTS_DIR / "daily_picks" / f"daily_picks_{signal_date}.csv"
    json_path = RESULTS_DIR / "daily_picks" / f"daily_picks_{signal_date}.json"
    if csv_path.exists() and json_path.exists() and not overwrite:
        return csv_path, json_path

    signals = [dict(s) for s in (pc.signals_by_date.get(signal_date, []) or [])]
    min_score = config.MIN_SCANNER_SCORE
    filtered = [s for s in signals if float(s.get("score", 0) or 0) >= min_score]
    filtered.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    _attach_entry_prices(pc, signal_date, filtered)

    max_jpy, lot_size = _get_budget_params()
    csv_out, json_out = _save_daily_picks(
        precomputed=pc,
        signal_date=signal_date,
        signals=filtered,
        min_score=min_score,
        stop_loss_pct=config.STOP_LOSS_PCT,
        rr_ratio=config.RISK_REWARD_RATIO,
        max_jpy=max_jpy,
        lot_size=lot_size,
    )

    if not csv_out.exists():
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DAILY_PICKS_COLUMNS)
            writer.writeheader()

    return csv_out, json_out


def _compute_shadow_signals_universe_min_volume(
    pc,
    signal_date: str,
    shadow_min_volume: int,
) -> list[dict]:
    """
    Shadow variant for Phase 2:
    widen only the min-volume universe filter.
    """
    baseline_universe = set(pc.universe_by_date.get(signal_date, []))
    shadow_universe = set(
        dm.build_liquid_universe(
            signal_date,
            top_n=config.UNIVERSE_TOP_N,
            min_volume=shadow_min_volume,
        )
    )
    if not shadow_universe:
        return []

    signals: list[dict] = []
    for sig in pc.signals_by_date.get(signal_date, []) or []:
        symbol = _normalize_symbol(sig.get("symbol"))
        if symbol in shadow_universe:
            clone = dict(sig)
            clone["symbol"] = symbol
            signals.append(clone)

    extra_symbols = shadow_universe - baseline_universe
    scanner_config = config.get_scanner_config()
    for symbol in extra_symbols:
        df = pc.symbol_data.get(symbol)
        if df is None:
            continue
        data_up_to = df[df.index.strftime("%Y-%m-%d") <= signal_date]
        if data_up_to is None or len(data_up_to) < 60:
            continue
        symbol_signals = sc.get_all_signals(
            symbol=symbol,
            data=data_up_to,
            jpx_data={},
            scanner_config=scanner_config,
            min_score=0,
            early_mode=config.EARLY_MODE_ENABLED,
        )
        for sig in symbol_signals:
            clone = dict(sig)
            clone["date"] = signal_date
            clone["symbol"] = _normalize_symbol(clone.get("symbol"))
            signals.append(clone)

    signals.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    return signals


def _signal_key(signal: dict) -> str:
    return _normalize_symbol(signal.get("symbol"))


def _is_single_scanner_signal(signal: dict) -> bool:
    try:
        count = int(signal.get("confluence_count", 1) or 1)
    except Exception:
        count = 1
    return count == 1


def _apply_single_signal_mix(
    signals: list[dict],
    *,
    top_n: int,
    min_single_count: int,
    min_single_score: float,
) -> list[dict]:
    """
    Phase 3 shadow variant:
    keep baseline order but ensure a minimum number of single-scanner names in top-N.
    """
    if not signals:
        return signals

    top_n = max(1, int(top_n))
    min_single_count = max(0, int(min_single_count))
    if min_single_count <= 0:
        return signals

    ordered = list(signals)
    top = list(ordered[:top_n])
    current_single = sum(1 for row in top if _is_single_scanner_signal(row))
    needed = max(0, min_single_count - current_single)
    if needed <= 0:
        return ordered

    eligible_pool = [
        row
        for row in ordered[top_n:]
        if _is_single_scanner_signal(row) and float(row.get("score", 0) or 0) >= float(min_single_score)
    ]
    if not eligible_pool:
        return ordered

    replace_slots = [idx for idx in range(len(top) - 1, -1, -1) if not _is_single_scanner_signal(top[idx])]
    if not replace_slots:
        return ordered

    promote_rows = eligible_pool[: min(needed, len(replace_slots))]
    for row, slot in zip(promote_rows, replace_slots):
        top[slot] = row

    top_keys = {_signal_key(row) for row in top}
    rest = []
    for row in ordered:
        key = _signal_key(row)
        if key in top_keys:
            continue
        rest.append(row)
    return top + rest


def _save_shadow_daily_picks(
    precomputed,
    signal_date: str,
    signals: list[dict],
    *,
    min_score: int,
    stop_loss_pct: float,
    rr_ratio: float,
    max_jpy: float,
    lot_size: int,
    variant: str,
    shadow_min_volume: int,
    ab_top_n: int,
    single_min_count: int,
    single_min_score: float,
) -> tuple[Path, Path]:
    entry_date = _get_next_trading_date(precomputed, signal_date)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: list[dict] = []
    for s in signals:
        symbol = s.get("symbol")
        if not symbol:
            continue

        entry = float(s.get("entry_price") or s.get("price") or 0)
        lot_cost = float(entry) * lot_size
        budget_ok = entry > 0 and lot_cost <= max_jpy

        rows.append(
            {
                "run_timestamp": ts,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "symbol": symbol,
                "strategy": s.get("strategy"),
                "score": s.get("score"),
                "close_price": s.get("price"),
                "entry_price": entry,
                "entry_source": s.get("entry_source", "close_est"),
                "stop_price": entry * (1 - stop_loss_pct) if entry else None,
                "target_price": entry * (1 + stop_loss_pct * rr_ratio) if entry else None,
                "lot_size": lot_size,
                "max_jpy_per_trade": max_jpy,
                "lot_cost": lot_cost,
                "budget_ok": budget_ok,
                "max_shares_within_budget": int(max_jpy // entry) if entry > 0 else 0,
                "confluence_count": s.get("confluence_count", 1),
                "confluence_scanners": "|".join(s.get("confluence_scanners", []) or []),
                "bucket": "lot_affordable" if budget_ok else "odd_lot",
            }
        )

    csv_path = CANDIDATES_AB_DIR / f"daily_picks_shadow_{signal_date}.csv"
    json_path = CANDIDATES_AB_DIR / f"daily_picks_shadow_{signal_date}.json"

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DAILY_PICKS_COLUMNS)
            writer.writeheader()

    payload = {
        "run_timestamp": ts,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "variant": variant,
        "min_score": min_score,
        "stop_loss_pct": stop_loss_pct,
        "risk_reward_ratio": rr_ratio,
        "ab_shadow_min_volume": shadow_min_volume,
        "ab_top_n": int(ab_top_n),
        "ab_single_min_count": int(single_min_count),
        "ab_single_min_score": float(single_min_score),
        "baseline_min_volume": config.MIN_AVG_DAILY_VOLUME,
        "budget": {"max_jpy_per_trade": max_jpy, "lot_size": lot_size},
        "picks": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def _ensure_shadow_picks_for_signal_date(
    pc,
    signal_date: str,
    variant: str,
    shadow_min_volume: int,
    ab_top_n: int,
    single_min_count: int,
    single_min_score: float,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    csv_path = CANDIDATES_AB_DIR / f"daily_picks_shadow_{signal_date}.csv"
    json_path = CANDIDATES_AB_DIR / f"daily_picks_shadow_{signal_date}.json"
    if csv_path.exists() and json_path.exists() and not overwrite:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if (
                str(payload.get("variant", "")) == str(variant)
                and int(payload.get("ab_shadow_min_volume", shadow_min_volume)) == int(shadow_min_volume)
                and int(payload.get("ab_top_n", ab_top_n)) == int(ab_top_n)
                and int(payload.get("ab_single_min_count", single_min_count)) == int(single_min_count)
                and float(payload.get("ab_single_min_score", single_min_score)) == float(single_min_score)
            ):
                return csv_path, json_path
        except Exception:
            pass

    min_score = int(config.MIN_SCANNER_SCORE)
    if variant == "universe_min_volume":
        shadow_signals = _compute_shadow_signals_universe_min_volume(pc, signal_date, shadow_min_volume)
        filtered = [s for s in shadow_signals if float(s.get("score", 0) or 0) >= min_score]
        filtered.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    elif variant == "single_signal_mix":
        baseline_signals = [dict(s) for s in (pc.signals_by_date.get(signal_date, []) or [])]
        baseline_signals.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
        filtered = [s for s in baseline_signals if float(s.get("score", 0) or 0) >= min_score]
        filtered = _apply_single_signal_mix(
            filtered,
            top_n=int(ab_top_n),
            min_single_count=int(single_min_count),
            min_single_score=float(single_min_score),
        )
    else:
        raise SystemExit(f"Unsupported A/B variant: {variant}")
    _attach_entry_prices(pc, signal_date, filtered)
    max_jpy, lot_size = _get_budget_params()
    return _save_shadow_daily_picks(
        precomputed=pc,
        signal_date=signal_date,
        signals=filtered,
        min_score=min_score,
        stop_loss_pct=config.STOP_LOSS_PCT,
        rr_ratio=config.RISK_REWARD_RATIO,
        max_jpy=max_jpy,
        lot_size=lot_size,
        variant=variant,
        shadow_min_volume=shadow_min_volume,
        ab_top_n=int(ab_top_n),
        single_min_count=int(single_min_count),
        single_min_score=float(single_min_score),
    )


def _load_candidates(signal_date: str, csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = RESULTS_DIR / "daily_picks" / f"daily_picks_{signal_date}.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=DAILY_PICKS_COLUMNS + ["candidate_rank"])

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=DAILY_PICKS_COLUMNS + ["candidate_rank"])
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df = df.reset_index(drop=True)
    df["candidate_rank"] = df.index + 1
    return df


def _compute_early_filter(df: pd.DataFrame) -> tuple[float | None, float | None, bool]:
    if df is None or df.empty or "rsi" not in df.columns:
        return None, None, False
    if len(df) < 11:
        return float(df["rsi"].iloc[-1]), None, False
    rsi = float(df["rsi"].iloc[-1])
    ret_10d = float((df["close"].iloc[-1] / df["close"].iloc[-11]) - 1.0)
    passes = (rsi <= config.EARLY_MODE_RSI_MAX) and (ret_10d < config.EARLY_MODE_10D_RETURN_MAX)
    return rsi, ret_10d, bool(passes)


def _top_signal(signals: list[dict]) -> dict | None:
    if not signals:
        return None
    return max(signals, key=lambda s: s.get("score", 0))


def _diagnose_miss(pc, signal_date: str, symbol: str, universe_set: set[str]) -> dict:
    out = {
        "miss_reason": "no_signal",
        "in_universe": False,
        "early_filter_pass": False,
        "rsi": None,
        "ret_10d": None,
        "early_score": None,
        "early_strategy": None,
        "legacy_score": None,
        "legacy_strategy": None,
    }

    df = pc.symbol_data.get(symbol)
    if df is None:
        out["miss_reason"] = "not_in_db"
        return out

    data_up_to = df[df.index.strftime("%Y-%m-%d") <= signal_date]
    if data_up_to is None or data_up_to.empty or len(data_up_to) < 60:
        out["miss_reason"] = "not_in_db"
        return out

    out["in_universe"] = symbol in universe_set
    if not out["in_universe"]:
        out["miss_reason"] = "not_in_universe"

    scanner_config = config.get_scanner_config()
    data_ind = ta.calculate_all_indicators(data_up_to, scanner_config)
    rsi, ret_10d, early_pass = _compute_early_filter(data_ind)
    out["rsi"] = rsi
    out["ret_10d"] = ret_10d
    out["early_filter_pass"] = early_pass

    legacy_signals = sc.get_all_signals(
        symbol=symbol,
        data=data_ind,
        jpx_data={},
        scanner_config=scanner_config,
        min_score=0,
        early_mode=False,
    )
    early_signals = sc.get_all_signals(
        symbol=symbol,
        data=data_ind,
        jpx_data={},
        scanner_config=scanner_config,
        min_score=0,
        early_mode=True,
    )

    legacy_top = _top_signal(legacy_signals)
    early_top = _top_signal(early_signals)
    legacy_score = float(legacy_top["score"]) if legacy_top else None
    early_score = float(early_top["score"]) if early_top else None
    legacy_strategy = legacy_top.get("strategy") if legacy_top else None
    early_strategy = early_top.get("strategy") if early_top else None

    out["legacy_score"] = legacy_score
    out["legacy_strategy"] = legacy_strategy
    out["early_score"] = early_score
    out["early_strategy"] = early_strategy

    min_score = config.MIN_SCANNER_SCORE
    if not out["in_universe"]:
        return out

    early_ok = early_score is not None and early_score >= min_score
    legacy_ok = legacy_score is not None and legacy_score >= min_score

    if early_ok:
        out["miss_reason"] = ""
    elif legacy_ok:
        if not early_top:
            if not early_pass:
                out["miss_reason"] = "early_filter_fail"
            elif legacy_strategy and legacy_strategy not in set(config.EARLY_MODE_SCANNERS):
                out["miss_reason"] = "excluded_by_early_scanner_subset"
            else:
                out["miss_reason"] = "no_signal"
        else:
            out["miss_reason"] = "early_score_below_min"
    else:
        if early_top and early_score is not None and early_score < min_score:
            out["miss_reason"] = "early_score_below_min"
        elif legacy_top and legacy_score is not None and legacy_score < min_score:
            out["miss_reason"] = "legacy_score_below_min"
        elif not early_top and not legacy_top:
            out["miss_reason"] = "no_signal"
        elif not early_pass:
            out["miss_reason"] = "early_filter_fail"
        else:
            out["miss_reason"] = "no_signal"

    return out


def _upsert_master(master_path: Path, day_df: pd.DataFrame, key_cols: list[str]) -> None:
    if day_df.empty:
        return
    if master_path.exists():
        master = pd.read_csv(master_path)
    else:
        master = pd.DataFrame(columns=day_df.columns)

    missing_cols = [c for c in day_df.columns if c not in master.columns]
    for col in missing_cols:
        master[col] = None
    master = master[day_df.columns]

    if not master.empty:
        key_series = master[key_cols].astype(str).agg("|".join, axis=1)
        drop_keys = set(day_df[key_cols].astype(str).agg("|".join, axis=1))
        master = master[~key_series.isin(drop_keys)]

    if master.empty:
        merged = day_df.copy()
    else:
        merged = pd.concat([master, day_df], ignore_index=True)
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    merged.to_csv(master_path, index=False, encoding="utf-8")


def run_collect(pc, *, dates: list[str], threshold: float, pending: bool, force: bool) -> list[str]:
    _ensure_dirs()
    bursts = _load_bursts_log()
    status = _load_collect_status()

    target_dates = list(dates)
    if pending and not force and not status.empty:
        processed = set(status["burst_date"].astype(str))
        target_dates = [d for d in dates if d not in processed]

    if not target_dates:
        print("Collect: no pending dates to process.")
        return []

    total_inserted = 0
    total_updated = 0
    for d in target_dates:
        auto_rows = _collect_auto_rows_for_date(pc, d, threshold)
        bursts, inserted, updated = _upsert_auto_rows(bursts, auto_rows)
        total_inserted += inserted
        total_updated += updated
        status = _upsert_collect_status(status, d, len(auto_rows))
        print(f"Collect {d}: auto_rows={len(auto_rows)} inserted={inserted} updated={updated}")

    _save_bursts_log(bursts)
    _save_collect_status(status)
    print(
        f"Collect done. Dates={len(target_dates)} inserted={total_inserted} updated={total_updated} "
        f"log={BURSTS_LOG_PATH}"
    )
    return target_dates


def run_audit(
    pc,
    *,
    dates: list[str],
    pending: bool,
    force: bool,
    ab_enabled: bool = False,
    ab_top_n: int = 10,
    ab_shadow_min_volume: int = 5_000,
    ab_variant: str = "single_signal_mix",
    ab_single_min_count: int = 3,
    ab_single_min_score: float = 70.0,
) -> list[str]:
    _ensure_dirs()
    bursts = _load_bursts_log()
    if bursts.empty:
        print("Audit: bursts_log is empty.")
        return []

    target = bursts[bursts["burst_date"].isin(dates)].copy()
    if target.empty:
        print("Audit: no burst rows in selected dates.")
        return []

    audit_status = _load_audit_status()
    target_dates = sorted(target["burst_date"].unique().tolist())
    if pending and not force:
        needed = []
        for d in target_dates:
            day_rows = target[target["burst_date"] == d]
            payload_hash = _date_payload_hash(day_rows)
            status_row = audit_status[audit_status["burst_date"] == d]
            if status_row.empty:
                needed.append(d)
                continue
            old_hash = str(status_row.iloc[0]["payload_hash"])
            old_rows = int(status_row.iloc[0]["row_count"])
            if old_hash != payload_hash or old_rows != len(day_rows):
                needed.append(d)
        target_dates = needed

    if not target_dates:
        print("Audit: no pending dates to process.")
        return []

    processed = []
    for burst_date in target_dates:
        signal_date = _signal_date_for_burst_date(pc, burst_date)
        if signal_date is None:
            print(f"Audit {burst_date}: skipped (no previous trading day in cache).")
            continue

        _ensure_daily_picks_for_signal_date(pc, signal_date)
        candidates = _load_candidates(signal_date)
        candidate_map = {
            row["symbol"]: row
            for _, row in candidates.iterrows()
        }

        baseline_top_map: dict[str, pd.Series] = {}
        shadow_top_map: dict[str, pd.Series] = {}
        if ab_enabled:
            top_n = max(1, int(ab_top_n))
            baseline_top = candidates[candidates["candidate_rank"] <= top_n].copy()
            baseline_top_map = {row["symbol"]: row for _, row in baseline_top.iterrows()}

            shadow_csv_path, _ = _ensure_shadow_picks_for_signal_date(
                pc,
                signal_date=signal_date,
                variant=str(ab_variant),
                shadow_min_volume=int(ab_shadow_min_volume),
                ab_top_n=int(ab_top_n),
                single_min_count=int(ab_single_min_count),
                single_min_score=float(ab_single_min_score),
            )
            shadow_candidates = _load_candidates(signal_date, csv_path=shadow_csv_path)
            shadow_top = shadow_candidates[shadow_candidates["candidate_rank"] <= top_n].copy()
            shadow_top_map = {row["symbol"]: row for _, row in shadow_top.iterrows()}

        universe_set = set(dm.build_liquid_universe(signal_date, top_n=config.UNIVERSE_TOP_N))
        day_rows = target[target["burst_date"] == burst_date].copy()

        records = []
        ab_records = []
        for _, row in day_rows.iterrows():
            symbol = _normalize_symbol(row["symbol"])
            candidate = candidate_map.get(symbol)

            base = {
                "burst_date": burst_date,
                "signal_date": signal_date,
                "symbol": symbol,
                "source": row.get("source", ""),
                "burst_pct_close": row.get("burst_pct_close"),
                "close_prev": row.get("close_prev"),
                "close_burst": row.get("close_burst"),
                "captured": bool(candidate is not None),
                "candidate_rank": None,
                "candidate_score": None,
                "candidate_strategy": None,
                "candidate_confluence_count": None,
                "candidate_confluence_scanners": None,
                "miss_reason": "",
                "in_universe": None,
                "early_filter_pass": None,
                "rsi": None,
                "ret_10d": None,
                "early_score": None,
                "early_strategy": None,
                "legacy_score": None,
                "legacy_strategy": None,
            }

            if candidate is not None:
                base["candidate_rank"] = int(candidate.get("candidate_rank", 0) or 0)
                base["candidate_score"] = float(candidate.get("score", 0) or 0)
                base["candidate_strategy"] = candidate.get("strategy")
                base["candidate_confluence_count"] = int(candidate.get("confluence_count", 0) or 0)
                base["candidate_confluence_scanners"] = candidate.get("confluence_scanners", "")
                base["in_universe"] = True
            else:
                diag = _diagnose_miss(pc, signal_date, symbol, universe_set)
                base["miss_reason"] = diag.get("miss_reason", "")
                base["in_universe"] = diag.get("in_universe")
                base["early_filter_pass"] = diag.get("early_filter_pass")
                base["rsi"] = diag.get("rsi")
                base["ret_10d"] = diag.get("ret_10d")
                base["early_score"] = diag.get("early_score")
                base["early_strategy"] = diag.get("early_strategy")
                base["legacy_score"] = diag.get("legacy_score")
                base["legacy_strategy"] = diag.get("legacy_strategy")

            records.append(base)

            if ab_enabled:
                a_row = baseline_top_map.get(symbol)
                b_row = shadow_top_map.get(symbol)
                captured_a = a_row is not None
                captured_b = b_row is not None
                ab_records.append(
                    {
                        "burst_date": burst_date,
                        "signal_date": signal_date,
                        "symbol": symbol,
                        "captured_a_topn": bool(captured_a),
                        "rank_a": int(a_row.get("candidate_rank")) if captured_a else None,
                        "score_a": float(a_row.get("score")) if captured_a else None,
                        "strategy_a": a_row.get("strategy") if captured_a else None,
                        "captured_b_topn": bool(captured_b),
                        "rank_b": int(b_row.get("candidate_rank")) if captured_b else None,
                        "score_b": float(b_row.get("score")) if captured_b else None,
                        "strategy_b": b_row.get("strategy") if captured_b else None,
                        "delta_capture_b_minus_a": int(captured_b) - int(captured_a),
                    }
                )

        day_df = pd.DataFrame(records)
        day_csv = AUDIT_DIR / f"audit_{burst_date}.csv"
        day_json = AUDIT_DIR / f"audit_summary_{burst_date}.json"
        day_df.to_csv(day_csv, index=False, encoding="utf-8")

        total = len(day_df)
        captured = int(day_df["captured"].sum()) if total else 0
        missed = total - captured
        reason_counts = (
            day_df[~day_df["captured"]]["miss_reason"]
            .fillna("")
            .replace("", "unknown")
            .value_counts()
            .to_dict()
        )
        summary = {
            "burst_date": burst_date,
            "signal_date": signal_date,
            "total_bursts": total,
            "captured": captured,
            "missed": missed,
            "capture_rate": (captured / total) if total else 0.0,
            "reason_counts": reason_counts,
            "generated_at": _now_iso(),
        }
        with open(day_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        _upsert_master(AUDIT_MASTER_PATH, day_df, key_cols=["burst_date", "symbol"])
        payload_hash = _date_payload_hash(day_rows)
        audit_status = _upsert_audit_status(audit_status, burst_date, len(day_rows), payload_hash)

        if ab_enabled:
            ab_df = pd.DataFrame(ab_records, columns=AB_AUDIT_COLUMNS)
            ab_day_csv = AUDIT_DIR / f"ab_audit_{burst_date}.csv"
            ab_day_json = AUDIT_DIR / f"ab_audit_summary_{burst_date}.json"
            ab_df.to_csv(ab_day_csv, index=False, encoding="utf-8")
            _upsert_master(AB_AUDIT_MASTER_PATH, ab_df, key_cols=["burst_date", "symbol"])

            ab_total = len(ab_df)
            ab_a = int(ab_df["captured_a_topn"].sum()) if ab_total else 0
            ab_b = int(ab_df["captured_b_topn"].sum()) if ab_total else 0
            ab_summary = {
                "burst_date": burst_date,
                "signal_date": signal_date,
                "top_n": int(max(1, int(ab_top_n))),
                "variant": str(ab_variant),
                "shadow_min_volume": int(ab_shadow_min_volume),
                "single_min_count": int(ab_single_min_count),
                "single_min_score": float(ab_single_min_score),
                "total_bursts": ab_total,
                "captured_a_topn": ab_a,
                "captured_b_topn": ab_b,
                "capture_rate_a_topn": (ab_a / ab_total) if ab_total else 0.0,
                "capture_rate_b_topn": (ab_b / ab_total) if ab_total else 0.0,
                "delta_capture_b_minus_a": int(ab_b - ab_a),
                "generated_at": _now_iso(),
            }
            with open(ab_day_json, "w", encoding="utf-8") as f:
                json.dump(ab_summary, f, ensure_ascii=False, indent=2)
            print(
                f"A/B {burst_date}: top{max(1, int(ab_top_n))} "
                f"A={ab_a}/{ab_total} B={ab_b}/{ab_total} "
                f"(variant={ab_variant}, shadow_min_volume={int(ab_shadow_min_volume):,}, "
                f"single_min_count={int(ab_single_min_count)}, single_min_score={float(ab_single_min_score):.0f})"
            )

        processed.append(burst_date)
        print(f"Audit {burst_date}: total={total} captured={captured} missed={missed}")

    _save_audit_status(audit_status)
    if processed:
        print(f"Audit done. Processed {len(processed)} date(s). Master: {AUDIT_MASTER_PATH}")
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Burst audit workflow")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_date_filters(p):
        p.add_argument("--date", type=str, default=None, help="Single burst date YYYY-MM-DD")
        p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
        p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
        p.add_argument("--pending", action="store_true", help="Process pending dates in the selected range")
        p.add_argument("--force", action="store_true", help="Reprocess dates even if status says done")

    p_collect = sub.add_parser("collect", help="Collect auto bursts from DB into bursts_log")
    add_date_filters(p_collect)
    p_collect.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Close-to-close burst threshold (default 0.10 = +10%%)",
    )

    p_audit = sub.add_parser("audit", help="Audit burst_log against previous-day daily picks")
    add_date_filters(p_audit)
    p_audit.add_argument("--ab", action="store_true", default=getattr(config, "BURST_AB_ENABLED", False), help="Run A/B top-N capture audit")
    p_audit.add_argument("--ab-top-n", type=int, default=getattr(config, "BURST_AB_TOP_N", 10), help="Top-N picks for A/B capture comparison")
    p_audit.add_argument("--ab-shadow-min-volume", type=int, default=getattr(config, "BURST_AB_SHADOW_MIN_VOLUME", 5_000), help="Shadow universe min volume for A/B")
    p_audit.add_argument("--ab-variant", type=str, default=getattr(config, "BURST_AB_VARIANT", "single_signal_mix"), choices=["universe_min_volume", "single_signal_mix"], help="A/B shadow variant type")
    p_audit.add_argument("--ab-single-min-count", type=int, default=getattr(config, "BURST_AB_SINGLE_MIN_COUNT", 3), help="For single_signal_mix: minimum single-scanner names in top-N")
    p_audit.add_argument("--ab-single-min-score", type=float, default=getattr(config, "BURST_AB_SINGLE_MIN_SCORE", 70), help="For single_signal_mix: minimum score for promoted single-scanner names")

    p_daily = sub.add_parser("daily", help="Run collect then audit")
    add_date_filters(p_daily)
    p_daily.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Close-to-close burst threshold for collect (default 0.10)",
    )
    p_daily.add_argument("--ab", action="store_true", default=getattr(config, "BURST_AB_ENABLED", False), help="Run A/B top-N capture audit")
    p_daily.add_argument("--ab-top-n", type=int, default=getattr(config, "BURST_AB_TOP_N", 10), help="Top-N picks for A/B capture comparison")
    p_daily.add_argument("--ab-shadow-min-volume", type=int, default=getattr(config, "BURST_AB_SHADOW_MIN_VOLUME", 5_000), help="Shadow universe min volume for A/B")
    p_daily.add_argument("--ab-variant", type=str, default=getattr(config, "BURST_AB_VARIANT", "single_signal_mix"), choices=["universe_min_volume", "single_signal_mix"], help="A/B shadow variant type")
    p_daily.add_argument("--ab-single-min-count", type=int, default=getattr(config, "BURST_AB_SINGLE_MIN_COUNT", 3), help="For single_signal_mix: minimum single-scanner names in top-N")
    p_daily.add_argument("--ab-single-min-score", type=float, default=getattr(config, "BURST_AB_SINGLE_MIN_SCORE", 70), help="For single_signal_mix: minimum score for promoted single-scanner names")

    args = parser.parse_args()
    _ensure_dirs()
    pc = _load_cache()

    dates = _resolve_dates(
        pc,
        date=args.date,
        start=args.start,
        end=args.end,
        pending=args.pending,
    )

    if args.cmd == "collect":
        run_collect(
            pc,
            dates=dates,
            threshold=float(args.threshold),
            pending=bool(args.pending),
            force=bool(args.force),
        )
        return 0

    if args.cmd == "audit":
        run_audit(
            pc,
            dates=dates,
            pending=bool(args.pending),
            force=bool(args.force),
            ab_enabled=bool(args.ab),
            ab_top_n=int(args.ab_top_n),
            ab_shadow_min_volume=int(args.ab_shadow_min_volume),
            ab_variant=str(args.ab_variant),
            ab_single_min_count=int(args.ab_single_min_count),
            ab_single_min_score=float(args.ab_single_min_score),
        )
        return 0

    # daily
    run_collect(
        pc,
        dates=dates,
        threshold=float(args.threshold),
        pending=bool(args.pending),
        force=bool(args.force),
    )
    run_audit(
        pc,
        dates=dates,
        pending=True if args.pending else False,
        force=bool(args.force),
        ab_enabled=bool(args.ab),
        ab_top_n=int(args.ab_top_n),
        ab_shadow_min_volume=int(args.ab_shadow_min_volume),
        ab_variant=str(args.ab_variant),
        ab_single_min_count=int(args.ab_single_min_count),
        ab_single_min_score=float(args.ab_single_min_score),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
