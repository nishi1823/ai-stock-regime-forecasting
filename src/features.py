import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "data",
    "spy_daily.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "spy_features.csv"
)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# Returns
df["return_1d"] = df["close"].pct_change()

df["return_5d"] = (
    df["close"].pct_change(5)
)

df["return_20d"] = (
    df["close"].pct_change(20)
)

# Moving averages
df["sma_20"] = (
    df["close"].rolling(20).mean()
)

df["sma_50"] = (
    df["close"].rolling(50).mean()
)

df["sma_200"] = (
    df["close"].rolling(200).mean()
)

# Volatility
df["volatility_20"] = (
    df["return_1d"]
    .rolling(20)
    .std()
)

df["volatility_60"] = (
    df["return_1d"]
    .rolling(60)
    .std()
)

# Momentum
df["momentum_20"] = (
    df["close"] /
    df["close"].shift(20) - 1
)

# Volume features
df["volume_sma_20"] = (
    df["volume"].rolling(20).mean()
)

df["volume_ratio"] = (
    df["volume"] /
    df["volume_sma_20"]
)

# Drawdown
rolling_high = (
    df["close"]
    .rolling(252)
    .max()
)

df["drawdown"] = (
    df["close"] /
    rolling_high - 1
)

# RSI
delta = df["close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = (
    gain.rolling(14).mean()
)

avg_loss = (
    loss.rolling(14).mean()
)

rs = avg_gain / avg_loss

df["rsi_14"] = (
    100 - (100 / (1 + rs))
)

# Calendar
df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day_of_week"] = (
    df["date"].dt.dayofweek
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna().reset_index(
    drop=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print("=" * 65)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 65)

print("\nRows:", len(df))
print("Features:", len(df.columns))

print("\nSaved:")
print(OUTPUT)
