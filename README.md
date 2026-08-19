# AI Stock Market Regime Detection & Forecasting

A machine-learning research platform for detecting stock-market regimes, forecasting short-term returns, and evaluating systematic trading strategies.

## Overview

This project investigates whether machine-learning-based market regime detection and short-term return forecasting can support systematic trading decisions.

The system combines real market data, financial feature engineering, machine learning, regime detection, forecasting, backtesting, and risk analysis.

## Key Features

- Real SPY market data
- Financial time-series feature engineering
- K-Means market regime detection
- Random Forest return forecasting
- 5-day return prediction
- Systematic trading backtesting
- Sharpe ratio and drawdown analysis
- Interactive Streamlit dashboard

## Research Pipeline

Market Data  
↓  
Feature Engineering  
↓  
Market Regime Detection  
↓  
5-Day Return Forecasting  
↓  
Trading Signal  
↓  
Backtesting  
↓  
Risk Analysis  
↓  
Interactive Dashboard

## Dataset

**Asset:** SPY  
**Period:** 2010–2026  
**Frequency:** Daily

The raw market data is excluded from GitHub using `.gitignore`.

## Market Regime Detection

K-Means clustering is used to identify different market conditions using return, momentum, volatility, and drawdown characteristics.

Detected regimes:

| Regime | Trading Days |
|---|---:|
| Bull / Strong | 2,313 |
| Bear / Defensive | 1,075 |
| Bull / Moderate | 286 |
| Bear / High Risk | 256 |

## Return Forecasting

A Random Forest regression model predicts the future 5-day market return.

### Evaluation Metrics

- MAE
- RMSE
- R²
- Direction Accuracy

## Backtesting

The predicted return is converted into a simple trading signal:

- Predicted return > 0 → Long
- Predicted return ≤ 0 → Cash

The strategy is compared against a buy-and-hold benchmark.

### Current Backtest Results

| Metric | Result |
|---|---:|
| AI Strategy Return | 25.31% |
| Buy & Hold Return | 1634.56% |
| Maximum Drawdown | -27.40% |
| Sharpe Ratio | 0.77 |

These are historical backtest results and do not guarantee future performance.

## Dashboard

The project includes an interactive Streamlit dashboard showing:

- Market regimes
- Strategy performance
- Buy & hold comparison
- Maximum drawdown
- Sharpe ratio
- Regime distribution
- Equity curve

Run the dashboard:

```bash
streamlit run src/dashboard.py