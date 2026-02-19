# 🛢️ Crude Oil Systematic Walk-Forward Strategy

Brent Crude Total Return Index (2016–2024)  
Sharpe: **0.847** vs Baseline: **0.462**

---

## 📌 Project Overview

This project implements a systematic multi-signal trading framework on the Brent Crude Total Return Index using:

- Diversified signal construction (Trend, Mean Reversion, Volatility-aware, Orthogonal)
- Walk-forward adaptive strategy selection
- Volatility targeting (15% annualized)
- Transaction cost modeling (0.015% per trade)
- Regime attribution analysis

The primary objective is to maximize out-of-sample Sharpe ratio in a realistic setting, strictly avoiding look-ahead bias.

All strategy selection decisions are made using only information available at each rebalance date.

---

## ⚙️ Strategy Architecture

### 🔹 Signal Groups (11 Base Signals)

**Trend-Following**
- 1M Momentum
- 3M Momentum
- 200D Moving Average Crossover

**Mean Reversion**
- 20D Z-score Reversion
- 5D Reversal
- Percentile Reversion

**Volatility-Aware**
- Vol-Adjusted Momentum
- Volatility Breakout
- Volatility Trend

**Orthogonal**
- Carry Proxy (Momentum Spread)
- Momentum × Volatility Filter

Each base signal is extended with a volatility-scaled variant.

Final candidate pool:
- 11 base
- 11 vol-scaled
- 1 equal-weight ensemble  
= **23 competing strategies**

---

## 🔄 Walk-Forward Framework

Every 6 months:

1. Evaluate all 23 candidate strategies over the previous 18 months
2. Score using composite metric:0.6 × Sharpe + 0.4 × Sortino
3. Select best performer
4. Deploy out-of-sample until next rebalance

This process ensures:

- No look-ahead bias
- Adaptive strategy selection
- Reduced overfitting risk

Walk-forward logic mirrors the provided `WalkforwardBacktester` implementation.

---

## 🎯 Risk Management

- Transaction Cost: 0.015% per trade
- Volatility Target: 15% annualized
- Volatility Estimate: 63-day EWM (1-day lag)
- Leverage Scalar Clipped to: [0.25×, 4×]

Volatility targeting is applied **before** strategy selection so all candidates compete on equal risk footing.

---

## 📊 Performance Summary

| Metric | Baseline | Walkforward |
|--------|----------|-------------|
| Sharpe | 0.462 | 0.847 |
| Sortino | 0.581 | 1.096 |
| CAGR | 10.9% | 13.1% |
| Volatility | 39.5% | 16.1% |
| Max Drawdown | 78.2% | 22.1% |
| Calmar | 0.139 | 0.592 |

Improvement is driven primarily by disciplined volatility control and adaptive selection rather than aggressive directional forecasting.

The strategy demonstrates resilience during high-volatility regimes, particularly during the 2020 oil crash.

---

## 🌡️ Regime Attribution

Performance is decomposed across:

- High Volatility (>65th percentile)
- Low Volatility (<35th percentile)
- Neutral Regime

Results indicate consistent performance across regimes, with enhanced stability during extreme volatility periods.

---

## 💻 Interactive Dashboard

An interactive Streamlit dashboard allows:

- Strategy selection
- Walk-forward parameter tuning
- Volatility targeting adjustments
- Regime filtering
- Rolling Sharpe visualization
- Selection frequency diagnostics
- Full performance comparison

Run locally:

```bash
streamlit run app.py
📂 Repository Structure
app.py                         # Streamlit dashboard
backtest.py                    # Walkforward backtester class
crude_oil_strategy.ipynb       # Research notebook
crude_oil_strategy.html        # HTML export of notebook
brent_index.xlsx               # Data file
performance_summary_improved.csv
requirements.txt
README.md
📦 Installation
pip install -r requirements.txt
🧠 Design Philosophy
This project emphasizes:

Out-of-sample integrity

Adaptive systematic allocation

Risk-adjusted performance over raw returns

Clean, reproducible implementation

The objective is disciplined quantitative design rather than curve-fitting.

🚀 Future Extensions
Multi-commodity extension

True futures carry implementation

Cross-asset regime conditioning

Longer historical backtesting

👤 Author
Dev Sachan
Quantitative Strategy Project
2026


