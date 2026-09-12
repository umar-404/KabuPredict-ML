# KabuPredict ML
<img width="1813" height="853" alt="Screenshot from 2026-09-12 23-00-25" src="https://github.com/user-attachments/assets/06492d6f-33cb-48ed-940d-2026fa447107" />

> **Next-day price-direction forecasting for Suzuki Motor (7269.T)**
>
> An end-to-end machine-learning pipeline that engineers financial technical
> features, trains a leakage-free **XGBoost** classifier, back-tests a long-only
> strategy against Buy & Hold, and serves the results through an interactive
> **Streamlit** dashboard.

---

## Overview

KabuPredict ML turns daily OHLCV market data into a deployable ML workflow in
four stages:

| Stage | Responsibility | Module |
| --- | --- | --- |
| **1 — Preprocess** | Clean data, engineer technical indicators, build target | `src/preprocess.py` |
| **2 — Model** | Train XGBoost with leakage-free time-series CV | `src/model.py` |
| **3 — Backtest** | Simulate the long/flat strategy vs Buy & Hold | `src/backtest.py` |
| **4 — Dashboard** | Interactive tuning, KPIs & Plotly charts | `app.py` |

Every prediction is evaluated on data the model has **never seen**, and every
performance claim is backed by an out-of-sample back-test.

## Project Layout

```
suzuki-stock-project/
├── data/
│   └── suzuki_stock.csv      # Daily OHLCV for 7269.T (2015 → 2026)
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Loading, cleaning, feature engineering
│   ├── model.py              # XGBoost + TimeSeriesSplit training
│   └── backtest.py           # Strategy back-testing & analytics
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Requirements

- `pandas` — data wrangling
- `numpy` — numeric computing
- `scikit-learn` — `TimeSeriesSplit` + evaluation metrics
- `xgboost` — gradient-boosted tree classifier
- `streamlit` — web dashboard
- `plotly` — interactive charts

## Quick Start

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Each pipeline stage can also be run standalone for quick sanity checks:

```bash
python src/preprocess.py   # load + engineer features
python src/model.py        # train with time-series CV, print metrics
python src/backtest.py     # run the strategy back-test
```

## Methodology

### Feature Engineering

| Indicator | Definition |
| --- | --- |
| **RSI-14** | Relative Strength Index via Wilder exponential smoothing |
| **ROC-10** | 10-bar rate of change (%) |
| **ATR-14** | Average True Range from High / Low / Close |
| **EMA ratio (9/21)** | Short-term over long-term moving-average ratio |
| **Lagged returns** | 1-day and 3-day percentage returns |

The target is a binary label — `1` if `Close(t+1) > Close(t)`, else `0`.
Features use only information up to and including time `t`, so there is
**no look-ahead leakage**.

### Training

- The data is split chronologically into a **train block** and a held-out
  **test tail**.
- A `TimeSeriesSplit` inside the train block yields **out-of-fold** predictions
  for honest in-sample diagnostics.
- A final `XGBClassifier` (`max_depth=3`, `learning_rate=0.03`,
  `n_estimators=100`, `tree_method=hist`) is fit on the full train block and
  scored on the never-seen test block.

### Back-testing

- Predict **Up** → take a long position (earn that day's forward return).
- Predict **Down** → sit flat in cash (0% return).
- Compared against a **Buy & Hold** benchmark over the same window.

#### Reported Analytics

- Cumulative return
- Annualized return
- Sharpe ratio
- Maximum drawdown

## Dashboard

Tune everything from the **sidebar**:

- Train / test split ratio and number of CV folds
- XGBoost hyperparameters (depth, learning rate, trees, subsampling, …)

The main panel renders **KPI cards** (Sharpe, precision, cumulative return)
alongside raw / engineered data previews, feature distributions coloured by
target, an out-of-fold confusion matrix, and an interactive **Plotly** equity
curve comparing the ML strategy to Buy & Hold.

## Notes & Disclaimer

- Next-day direction is an inherently low-signal problem; classification metrics
  near 50 % are expected and reported honestly.
- The pipeline's value is the **robust workflow** — clean feature engineering,
  leakage-free cross-validation, and honest out-of-sample back-testing.
- All values are Yen-denominated. This is a research / simulation exercise and
  **is not financial advice**. Nothing here should be used to make investment
  decisions.
