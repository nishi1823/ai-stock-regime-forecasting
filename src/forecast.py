import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT = os.path.join(
    BASE_DIR,
    "data",
    "spy_features.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "return_forecast_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "forecast_features.pkl"
)

RESULT = os.path.join(
    BASE_DIR,
    "results",
    "forecast_results.csv"
)

print("=" * 65)
print("AI STOCK RETURN FORECASTING")
print("=" * 65)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(
    df["date"]
)

# =========================================================
# TARGET
# Future 5-day return
# =========================================================

df["target_return_5d"] = (
    df["close"].shift(-5)
    / df["close"]
    - 1
)

# =========================================================
# FEATURES
# =========================================================

features = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_20",
    "sma_50",
    "sma_200",
    "volatility_20",
    "volatility_60",
    "momentum_20",
    "volume_ratio",
    "drawdown",
    "rsi_14",
    "month",
    "day_of_week"
]

df = df.dropna().reset_index(
    drop=True
)

X = df[features]

y = df["target_return_5d"]

# =========================================================
# TIME-BASED SPLIT
# =========================================================

split = int(
    len(df) * 0.80
)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

# =========================================================
# MODEL
# =========================================================

print(
    "\nTraining Random Forest..."
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
    X_test
)

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

r2 = r2_score(
    y_test,
    prediction
)

# Direction accuracy

predicted_direction = (
    prediction > 0
)

actual_direction = (
    y_test.values > 0
)

direction_accuracy = (
    predicted_direction
    ==
    actual_direction
).mean()

# =========================================================
# RESULTS
# =========================================================

results = df.iloc[
    split:
].copy()

results["predicted_return"] = (
    prediction
)

results["actual_return"] = (
    y_test.values
)

results["predicted_direction"] = (
    predicted_direction
)

results["actual_direction"] = (
    actual_direction
)

print(
    "\n" + "=" * 65
)

print(
    "FORECAST PERFORMANCE"
)

print(
    "=" * 65
)

print(
    f"\nMAE: {mae:.6f}"
)

print(
    f"RMSE: {rmse:.6f}"
)

print(
    f"R2: {r2:.6f}"
)

print(
    f"Direction Accuracy: "
    f"{direction_accuracy:.2%}"
)

# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(RESULT),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    features,
    FEATURE_PATH
)

results.to_csv(
    RESULT,
    index=False
)

print(
    "\nModel saved:"
)

print(
    MODEL_PATH
)

print(
    "\nResults saved:"
)

print(
    RESULT
)

print(
    "\nFORECASTING COMPLETE!"
)
