"""
Streamlit dashboard for browsing latest signals with interactive charts.

Run:
  streamlit run streamlit_dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    get_script_run_ctx = None

import config
import data_manager as dm
from precompute import load_precomputed


def _get_latest_date_with_signals(precomputed) -> str | None:
    for date in reversed(precomputed.trading_dates):
        if precomputed.signals_by_date.get(date):
            return date
    return None


def _get_dates_with_signals(precomputed) -> list[str]:
    return [d for d in precomputed.trading_dates if precomputed.signals_by_date.get(d)]


@st.cache_data(show_spinner=False)
def _load_cache():
    cache_path = Path("results/precomputed_cache.pkl")
    if not cache_path.exists():
        return None
    return load_precomputed(cache_path)


def _get_signals(precomputed, date: str, min_score: int):
    signals = precomputed.signals_by_date.get(date, [])
    return sorted([s for s in signals if s.get("score", 0) >= min_score], key=lambda x: x.get("score", 0), reverse=True)


def _plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )
    if "volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="Volume", yaxis="y2", opacity=0.3)
        )
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                title="Volume",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (JPY)",
        legend=dict(orientation="h"),
        height=500,
    )
    st.plotly_chart(fig, width="stretch")


def main():
    if get_script_run_ctx and get_script_run_ctx() is None:
        print("Please run this dashboard with:")
        print("  streamlit run streamlit_dashboard.py")
        return
    st.set_page_config(page_title="JP Stocks Signals Dashboard", layout="wide")
    st.title("JP Stocks Signals Dashboard")

    precomputed = _load_cache()
    if precomputed is None:
        st.error("Cache not found. Run: python precompute.py")
        return

    dates_with_signals = _get_dates_with_signals(precomputed)
    latest_date = _get_latest_date_with_signals(precomputed) or precomputed.trading_dates[-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        show_only_signal_dates = st.checkbox("Show only dates with signals", value=True)
        date_options = dates_with_signals if (show_only_signal_dates and dates_with_signals) else precomputed.trading_dates
        default_date = latest_date if latest_date in date_options else date_options[-1]
        date = st.selectbox("Signal date", date_options, index=date_options.index(default_date))
    with col2:
        min_score = st.slider("Min Score", 0, 150, config.MIN_SCANNER_SCORE)
    with col3:
        days = st.slider("Lookback Days", 30, 365, 180)

    signals = _get_signals(precomputed, date, min_score)
    if not signals:
        st.warning(f"No signals found for {date} at min score {min_score}.")
        if dates_with_signals:
            st.info(f"Latest date with signals: {dates_with_signals[-1]}")
        st.info("Try lowering the min score, or select a different date.")
        return

    symbols = [s["symbol"] for s in signals]
    selected = st.selectbox("Symbol", symbols)
    selected_signal = next(s for s in signals if s["symbol"] == selected)

    end_date = date
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    df = dm.get_daily_bars(selected, start_date=start_date, end_date=end_date)

    st.subheader(f"{selected} | Score {selected_signal.get('score', 0):.0f} | {selected_signal.get('strategy', '')}")
    if df is None or df.empty:
        st.info("No data found for selected symbol.")
        return

    _plot_candles(df, f"{selected} ({start_date} → {end_date})")

    with st.expander("Signal Details"):
        st.json(selected_signal)

    st.subheader("All Signals")
    st.dataframe(
        pd.DataFrame(signals)[["symbol", "strategy", "score", "price", "confluence_count"]]
    )


if __name__ == "__main__":
    main()
