import os
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "backtest_results.csv"
)

REGIME_PATH = os.path.join(
    BASE_DIR,
    "results",
    "regime_results.csv"
)

st.set_page_config(
    page_title="AI Stock Market Regime Detection",
    page_icon="ST",
    layout="wide"
)

st.title(
    "AI Stock Market Regime Detection & Forecasting"
)

st.write(
    "Machine-learning research platform for market-regime "
    "detection, return forecasting and strategy backtesting."
)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

regimes = pd.read_csv(REGIME_PATH)

regimes["date"] = pd.to_datetime(regimes["date"])

# =========================================================
# PERFORMANCE METRICS
# =========================================================

strategy_return = (
    df["strategy_equity"].iloc[-1] - 1
)

buy_hold_return = (
    df["buy_hold_equity"].iloc[-1] - 1
)

max_drawdown = (
    df["strategy_drawdown"].min()
)

sharpe = (
    df["strategy_return"].mean()
    / df["strategy_return"].std()
    * (252 ** 0.5)
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "AI Strategy Return",
        f"{strategy_return:.2%}"
    )

with c2:
    st.metric(
        "Buy & Hold",
        f"{buy_hold_return:.2%}"
    )

with c3:
    st.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2%}"
    )

with c4:
    st.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}"
    )

st.divider()

# =========================================================
# MARKET REGIMES
# =========================================================

st.subheader(
    "Market Regimes"
)

regime_chart = px.scatter(
    regimes,
    x="date",
    y="close",
    color="regime",
    title="AI-Detected Market Regimes"
)

st.plotly_chart(
    regime_chart,
    use_container_width=True
)

# =========================================================
# STRATEGY PERFORMANCE
# =========================================================

st.subheader(
    "Strategy Performance"
)

equity_chart = px.line(
    df,
    x="date",
    y=[
        "strategy_equity",
        "buy_hold_equity"
    ],
    title="AI Strategy vs Buy & Hold"
)

st.plotly_chart(
    equity_chart,
    use_container_width=True
)

# =========================================================
# DRAWDOWN
# =========================================================

st.subheader(
    "Risk Analysis"
)

drawdown_chart = px.area(
    df,
    x="date",
    y="strategy_drawdown",
    title="Strategy Drawdown"
)

st.plotly_chart(
    drawdown_chart,
    use_container_width=True
)

# =========================================================
# REGIME DISTRIBUTION
# =========================================================

st.subheader(
    "Regime Distribution"
)

counts = (
    regimes["regime"]
    .value_counts()
    .reset_index()
)

counts.columns = [
    "regime",
    "days"
]

bar = px.bar(
    counts,
    x="regime",
    y="days",
    title="Number of Trading Days per Regime"
)

st.plotly_chart(
    bar,
    use_container_width=True
)

st.divider()

st.caption(
    "AI Stock Market Regime Detection & Forecasting | "
    "Machine Learning + Quantitative Research"
)
