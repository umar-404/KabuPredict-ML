"""XGBoost model training with leakage-free time-series cross validation.

The pipeline splits the data chronologically into an initial train block and a
held-out test block. Inside the train block a ``TimeSeriesSplit`` produces
out-of-fold (OOF) predictions used to report honest in-sample metrics, after
which a final model is fitted on the *whole* train block and evaluated on the
never-seen test block. The test block predictions are what feed the back-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_THRESHOLD = 0.5


def default_params() -> dict:
    """Sensible tabular-finance defaults for the XGBoost classifier."""
    return {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 100,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }


@dataclass
class ModelOutcome:
    """Container bundling OOF/test metrics, predictions and the fitted model."""

    oof_metrics: dict
    test_metrics: dict
    model: xgb.XGBClassifier
    feature_importance: pd.Series
    oof_df: pd.DataFrame          # indexed by date: oof_true, oof_pred, oof_prob
    test_df: pd.DataFrame         # indexed by date: test_true, test_pred, test_prob
    params: dict
    split_index: int              # row index separating train / test
    test_positions: np.ndarray    # integer row positions (into X) of the test block
    train_positions: np.ndarray   # integer row positions (into X) of the train block

    def as_dict(self):
        return {
            "oof_metrics": self.oof_metrics,
            "test_metrics": self.test_metrics,
            "model": self.model,
            "feature_importance": self.feature_importance,
            "oof_df": self.oof_df,
            "test_df": self.test_df,
            "params": self.params,
            "split_index": self.split_index,
            "test_positions": self.test_positions,
            "train_positions": self.train_positions,
        }


def make_classifier(params: Optional[dict] = None) -> xgb.XGBClassifier:
    """Build an XGBClassifier with merged (default + user) hyperparameters."""
    merged = {**default_params(), **(params or {})}
    return xgb.XGBClassifier(**merged)


def _metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def train_and_evaluate(X, y,
                       params: Optional[dict] = None,
                       test_size: float = 0.20,
                       n_splits: int = 5,
                       threshold: float = DEFAULT_THRESHOLD) -> ModelOutcome:
    """Chronologically split, time-series-CV train and evaluate an XGBoost model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, rows in chronological order.
    y : pd.Series
        Binary target aligned with ``X``.
    params : dict | None
        XGBoost hyperparameters; merged over :func:`default_params`.
    test_size : float
        Fraction of the *most recent* rows held out as the test block.
    n_splits : int
        Number of TimeSeriesSplit folds used for OOF metrics.
    threshold : float
        Probability cut-off used to turn probabilities into hard labels.

    Returns
    -------
    ModelOutcome
    """
    params = {**default_params(), **(params or {})}
    n = len(X)
    test_n = max(1, int(round(n * test_size)))
    split_at = n - test_n

    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    indices = np.arange(n)

    # ---- Out-of-fold predictions via TimeSeriesSplit ----------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.full(len(X_train), np.nan)
    for tr_idx, va_idx in tscv.split(X_train):
        fold_model = make_classifier(params)
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_prob[va_idx] = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
    oof_pred = (oof_prob >= threshold).astype(int)
    oof_true = y_train.to_numpy()
    oof_metrics = _metrics(oof_true, oof_pred)

    # ---- Final model trained on the entire train block ---------------------
    model = make_classifier(params)
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_true = y_test.to_numpy()
    test_metrics = _metrics(test_true, test_pred)

    feature_importance = pd.Series(
        model.feature_importances_, index=list(X.columns)
    ).sort_values(ascending=False)

    oof_df = pd.DataFrame(
        {"oof_true": oof_true, "oof_pred": oof_pred, "oof_prob": oof_prob},
        index=X_train.index,
    )
    test_df = pd.DataFrame(
        {"test_true": test_true, "test_pred": test_pred, "test_prob": test_prob},
        index=X_test.index,
    )

    return ModelOutcome(
        oof_metrics=oof_metrics,
        test_metrics=test_metrics,
        model=model,
        feature_importance=feature_importance,
        oof_df=oof_df,
        test_df=test_df,
        params=params,
        split_index=split_at,
        test_positions=indices[split_at:],
        train_positions=indices[:split_at],
    )


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import preprocess

    fe, cols = preprocess.prepare_dataset()
    out = train_and_evaluate(fe[cols], fe["target"])
    print("OOF  :", out.oof_metrics)
    print("TEST :", out.test_metrics)
    print("\nFeature importance:\n", out.feature_importance.round(4))