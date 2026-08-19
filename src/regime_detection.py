import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE_DIR,
    "data",
    "spy_features.csv"
)

OUTPUT = os.path.join(
    BASE_DIR,
    "results",
    "regime_results.csv"
)

PLOT = os.path.join(
    BASE_DIR,
    "results",
    "market_regimes.png"
)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

regime_features = [
    "return_20d",
    "volatility_20",
    "momentum_20",
    "drawdown"
]

X = df[regime_features].copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

df["regime_cluster"] = model.fit_predict(
    X_scaled
)

# Characterize clusters
summary = (
    df.groupby("regime_cluster")[
        regime_features
    ]
    .mean()
)

# Assign interpretable labels
score = (
    summary["return_20d"]
    + summary["momentum_20"]
    - summary["volatility_20"]
    + summary["drawdown"]
)

ordered = score.sort_values().index.tolist()

labels = {
    ordered[0]: "Bear / High Risk",
    ordered[1]: "Bear / Defensive",
    ordered[2]: "Bull / Moderate",
    ordered[3]: "Bull / Strong"
}

df["regime"] = (
    df["regime_cluster"]
    .map(labels)
)

df.to_csv(
    OUTPUT,
    index=False
)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))

for regime in df["regime"].unique():

    subset = df[
        df["regime"] == regime
    ]

    plt.scatter(
        subset["date"],
        subset["close"],
        s=8,
        label=regime
    )

plt.title(
    "AI-Detected Stock Market Regimes"
)

plt.xlabel("Date")

plt.ylabel("SPY Price")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT,
    dpi=250
)

plt.close()

print("=" * 65)
print("MARKET REGIME DETECTION COMPLETE")
print("=" * 65)

print("\nRegime distribution:")

print(
    df["regime"]
    .value_counts()
)

print("\nSaved:")
print(OUTPUT)
print(PLOT)
