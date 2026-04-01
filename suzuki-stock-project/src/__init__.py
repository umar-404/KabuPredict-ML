"""KabuPredict ML — source package.

Modules
-------
preprocess : data loading, cleaning and financial feature engineering.
model      : XGBoost classifier trained with time-series cross validation.
backtest   : strategy back-testing simulation and performance analytics.
"""

from . import preprocess, model, backtest

__all__ = ["preprocess", "model", "backtest"]
__version__ = "1.0.0"