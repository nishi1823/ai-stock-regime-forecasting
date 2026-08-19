import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "forecast_results.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "results",
    "backtest_results.csv"
)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

# Strategy:
# Long when predicted return > 0
# Otherwise stay in cash.

df["signal"] = (
    df["predicted_return"] > 0
).astype(int)

df["strategy_return"] = (
    df["signal"]
    * df["actual_return"]
)

df["buy_hold_return"] = (
    df["actual_return"]
)

df["strategy_equity"] = (
    1
    + df["strategy_return"]
).cumprod()

df["buy_hold_equity"] = (
    1
    + df["buy_hold_return"]
).cumprod()

# Drawdown
peak = (
    df["strategy_equity"]
    .cummax()
)

df["strategy_drawdown"] = (
    df["strategy_equity"]
    / peak - 1
)

total_return = (
    df["strategy_equity"].iloc[-1]
    - 1
)

buy_hold_return = (
    df["buy_hold_equity"].iloc[-1]
    - 1
)

max_drawdown = (
    df["strategy_drawdown"].min()
)

daily_returns = (
    df["strategy_return"]
)

sharpe = (
    daily_returns.mean()
    /
    daily_returns.std()
    * np.sqrt(252)
)

win_rate = (
    (daily_returns > 0).sum()
    /
    (daily_returns != 0).sum()
)

print("=" * 65)
print("TRADING STRATEGY BACKTEST")
print("=" * 65)

print(
    f"\nStrategy Return: "
    f"{total_return:.2%}"
)

print(
    f"Buy & Hold Return: "
    f"{buy_hold_return:.2%}"
)

print(
    f"Sharpe Ratio: "
    f"{sharpe:.2f}"
)

print(
    f"Maximum Drawdown: "
    f"{max_drawdown:.2%}"
)

print(
    f"Trade Win Rate: "
    f"{win_rate:.2%}"
)

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

print("\nBACKTEST COMPLETE!")
