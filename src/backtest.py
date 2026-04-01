"""Strategy back-testing engine and performance analytics.

When the model predicts ``1`` (next-day Up) we simulate holding a long
position from today's close to tomorrow's close, earning that day's forward
return; when it predicts ``0`` we sit flat in cash with a 0% daily return.
These per-day strategy returns are contrasted against a Buy-and-Hold
benchmark that is always long across the same out-of-sample window.

Performance analytics reported: cumulative return, annualized return,
Sharpe ratio and maximum drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _forward_returns(close: pd.Series) -> np.ndarray:
    """Per-row one-day-ahead simple return (Close[t+1] / Close[t] - 1)."""
    arr = close.to_numpy(dtype=float)
    fwd = np.full(arr.shape, np.nan, dtype=float)
    fwd[:-1] = arr[1:] / arr[:-1] - 1.0
    return fwd


def _equity_curve(daily_returns: np.ndarray) -> np.ndarray:
    """Cumulative product of `1 + daily_returns`, seeded at 1.0."""
    return np.cumprod(1.0 + daily_returns)


def annualized_return(equity: np.ndarray, periods: int,
                      trading_days: int = TRADING_DAYS) -> float:
    if periods <= 0 or equity[-1] <= 0:
        return 0.0
    return float(equity[-1] ** (trading_days / periods) - 1.0)


def sharpe_ratio(daily_returns: np.ndarray, periods: int,
                 risk_free_rate: float = 0.0,
                 trading_days: int = TRADING_DAYS) -> float:
    if periods == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / trading_days
    vol = float(excess.std(ddof=0))
    if vol == 0.0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(trading_days))


def max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(np.min(dd))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    strategy: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    equity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_size: int = 0

    def kpis(self) -> dict:
        return {
            "sharpe_strategy": self.strategy.get("sharpe_ratio", 0.0),
            "sharpe_benchmark": self.benchmark.get("sharpe_ratio", 0.0),
            "cumulative_strategy": self.strategy.get("cumulative_return", 0.0),
            "cumulative_benchmark": self.benchmark.get("cumulative_return", 0.0),
            "precision": self.strategy.get("precision", 0.0),
        }


def _summarize(name: str, daily_returns: np.ndarray,
               periods: int) -> dict:
    daily_returns = np.asarray(daily_returns, dtype=float)
    equity = _equity_curve(daily_returns)
    return {
        "name": name,
        "cumulative_return": float(equity[-1] - 1.0),
        "annualized_return": annualized_return(equity, periods),
        "sharpe_ratio": sharpe_ratio(daily_returns, periods),
        "max_drawdown": max_drawdown(equity),
        "daily_returns": daily_returns,
        "equity_curve": equity,
    }
def run_backtest(clean_df: pd.DataFrame, outcome) -> BacktestResult:
    """Simulate the ML strategy vs Buy-and-Hold over the held-out test block.

    Parameters
    ----------
    clean_df : pd.DataFrame
        The cleaned OHLCV frame (``preprocess.load_data`` output) indexed by
        date and holding a ``Close`` column.
    outcome
        A :class:`~src.model.ModelOutcome` with a ``test_df`` of hard
        predictions indexed by date. Predictions must align with rows of
        ``clean_df``.

    Returns
    -------
    BacktestResult
    """
    close = clean_df["Close"]
    fwd = _forward_returns(close)

    test_dates = list(outcome.test_df.index)
    test_pred = outcome.test_df["test_pred"].to_numpy(dtype=float)

    # Realised strategy return for each test bar:
    #   long (earning that day's +1 move) when pred == 1, flat in cash otherwise.
    strategy_daily = test_pred * fwd[-len(test_pred):]
    # Buy & Hold is always long across the same window.
    benchmark_daily = fwd[-len(test_pred):].copy()

    # Drop the terminal bar whose +1-day return is out of sample (NaN).
    valid = np.isfinite(strategy_daily) & np.isfinite(benchmark_daily)
    strategy_daily = strategy_daily[valid]
    benchmark_daily = benchmark_daily[valid]
    # The forward return realised at bar i covers (date_i -> date_{i+1}); we
    # label each move by its entry date so Strategy & Benchmark stay aligned.
    entry_dates = pd.DatetimeIndex(test_dates)[: len(strategy_daily)]
    periods = len(strategy_daily)

    strategy = _summarize("ML Strategy", strategy_daily, periods)
    benchmark = _summarize("Buy & Hold", benchmark_daily, periods)

    # Precision of the strategy's long calls (used as a dashboard KPI).
    holdout_true = outcome.test_df["test_true"].to_numpy()[: periods]
    holdout_pred = outcome.test_df["test_pred"].to_numpy()[: periods]
    tp = np.sum((holdout_pred == 1) & (holdout_true == 1))
    fp = np.sum((holdout_pred == 1) & (holdout_true == 0))
    strategy["precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0

    equity_df = pd.DataFrame(
        {
            "ML Strategy": _equity_curve(strategy_daily),
            "Buy & Hold": _equity_curve(benchmark_daily),
        },
        index=entry_dates,
    )
    daily_df = pd.DataFrame(
        {"ML Strategy": strategy_daily, "Buy & Hold": benchmark_daily},
        index=entry_dates,
    )

    return BacktestResult(
        strategy=strategy,
        benchmark=benchmark,
        equity_df=equity_df,
        daily_df=daily_df,
        window_size=periods,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import preprocess
    from src import model as model_mod

    fe, cols = preprocess.prepare_dataset()
    out = model_mod.train_and_evaluate(fe[cols], fe["target"])
    clean = preprocess.load_data()
    bt = run_backtest(clean, out)
    print(f"Backtest window: {bt.window_size} bars")
    keep = ("cumulative_return", "annualized_return",
            "sharpe_ratio", "max_drawdown", "precision")
    print("Strategy :", {k: v for k, v in bt.strategy.items() if k in keep})
    print("Benchmark:", {k: v for k, v in bt.benchmark.items() if k in keep})