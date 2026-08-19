import os
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TICKER = "SPY"

URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    f"{TICKER}?period1=1262304000&period2=1797724800"
    "&interval=1d&events=history&includeAdjustedClose=true"
)

print("=" * 65)
print("AI STOCK MARKET REGIME DETECTION")
print("=" * 65)
print("\nDownloading SPY market data...")

response = requests.get(
    URL,
    timeout=60,
    headers={"User-Agent": "Mozilla/5.0"}
)

response.raise_for_status()

result = response.json()["chart"]["result"][0]

df = pd.DataFrame({
    "date": pd.to_datetime(
        result["timestamp"],
        unit="s"
    ),
    "open": result["indicators"]["quote"][0]["open"],
    "high": result["indicators"]["quote"][0]["high"],
    "low": result["indicators"]["quote"][0]["low"],
    "close": result["indicators"]["quote"][0]["close"],
    "volume": result["indicators"]["quote"][0]["volume"]
})

df = df.dropna().reset_index(drop=True)

path = os.path.join(
    DATA_DIR,
    "spy_daily.csv"
)

df.to_csv(path, index=False)

print("\nRows:", len(df))
print("Start:", df["date"].min())
print("End:", df["date"].max())
print("\nSaved:", path)
print("\nDATA DOWNLOAD COMPLETE!")
