import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "backtest_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

# Equity curve
plt.figure(figsize=(14, 7))

plt.plot(
    df["date"],
    df["strategy_equity"],
    label="AI Strategy"
)

plt.plot(
    df["date"],
    df["buy_hold_equity"],
    label="Buy & Hold"
)

plt.title(
    "AI Strategy vs Buy & Hold"
)

plt.xlabel("Date")

plt.ylabel("Portfolio Value")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "equity_curve.png"
    ),
    dpi=250
)

plt.close()

# Drawdown
plt.figure(figsize=(14, 5))

plt.plot(
    df["date"],
    df["strategy_drawdown"]
)

plt.title(
    "AI Strategy Drawdown"
)

plt.xlabel("Date")

plt.ylabel("Drawdown")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "drawdown.png"
    ),
    dpi=250
)

plt.close()

print("=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)

print("\nCreated:")
print("results/equity_curve.png")
print("results/drawdown.png")
