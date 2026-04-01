"""Data loading, cleaning and financial feature engineering.

This module reads the raw Suzuki Motor (7269.T) daily OHLCV CSV, cleans it,
engineers a set of technical indicators plus lagged returns, and builds the
supervised classification target: ``1`` if the next-day close is higher than
today's close, ``0`` otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Ordered candidate locations so the loader is robust to naming variations.
DEFAULT_DATA_PATH = DATA_DIR / "suzuki_stock.csv"
FALLBACK_PATHS = [
    DATA_DIR / "suzuki_stock.csv",
    DATA_DIR / "suzuki_stock_data.csv",
    PROJECT_ROOT / "suzuki_stock_data.csv",
    PROJECT_ROOT / "suzuki_stock.csv",
]


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def load_data(path=None) -> pd.DataFrame:
    """Load, parse, index and sort the OHLCV dataset chronologically.

    Parameters
    ----------
    path : str | PathLike | None
        Explicit path to the CSV. When ``None``, the default location
        ``data/suzuki_stock.csv`` is used, with a few fallbacks.

    Returns
    -------
    pd.DataFrame
        Cleaned frame indexed by a ``DatetimeIndex`` and sorted ascending.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = [
            p for p in [DEFAULT_DATA_PATH, *FALLBACK_PATHS] if p is not None
        ]

    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        raise FileNotFoundError(
            "Could not locate the Suzuki (7269.T) OHLCV CSV. Tried: "
            + ", ".join(str(p) for p in candidates)
        )

    df = pd.read_csv(resolved)

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date").sort_index()

    # Case-insensitive column normalisation for the OHLCV fields.
    expected = {"close": "Close", "high": "High", "low": "Low",
                "open": "Open", "volume": "Volume"}
    for lower_key, canonical in expected.items():
        match = [c for c in df.columns if c.lower() == lower_key]
        if match and match[0] != canonical:
            df = df.rename(columns={match[0]: canonical})

    # Coerce numerics, forward-fill missing prices and drop any leftovers.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill().dropna()

    return df
# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's exponential smoothing.

    RSI = 100 - 100 / (1 + avg_gain / avg_loss). ``avg_gain``/``avg_loss``
    are exponentially-weighted means of gains/losses with ``alpha = 1/period``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.clip(0.0, 100.0)


def compute_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change in percent over ``period`` bars."""
    return close.pct_change(periods=period) * 100.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder-smoothed true range)."""
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period).mean()


def compute_ema_ratio(close: pd.Series, short: int = 9, long: int = 21) -> pd.Series:
    """Ratio of a short-term EMA to a long-term EMA (trend/regime proxy)."""
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    return ema_short / ema_long


def compute_lagged_returns(close: pd.Series, lags=(1, 3)) -> pd.DataFrame:
    """Percentage returns over each requested lag, returned as a DataFrame."""
    return pd.concat(
        {f"ret_{lag}d": close.pct_change(periods=lag) * 100.0 for lag in lags},
        axis=1,
    )


def create_target(close: pd.Series) -> pd.Series:
    """Binary label: 1 if Close(t+1) > Close(t) else 0.

    Note
    ----
    The ``.shift(-1)`` looks *forward* on purpose; feature rows use only
    backwards-looking information so no future leak reaches the model.
    """
    return (close.shift(-1) > close).astype(int)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame,
                   rsi_period: int = 14,
                   roc_period: int = 10,
                   atr_period: int = 14,
                   ema_short: int = 9,
                   ema_long: int = 21,
                   lags=(1, 3)) -> pd.DataFrame:
    """Assemble the full feature matrix plus the classification target.

    Returns a DataFrame whose index is the original DatetimeIndex and which
    contains every engineered feature column together with the ``target``.
    All `NaN` rows arising from look-backs and the forward-shifted label are
    dropped and the frame is sorted chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV frame from :func:`load_data`.
    """
    close = df["Close"]

    features = pd.DataFrame(index=df.index)
    features["rsi_14"] = compute_rsi(close, rsi_period)
    features["roc_10"] = compute_roc(close, roc_period)
    features["atr_14"] = compute_atr(df, atr_period)
    features["ema_ratio_9_21"] = compute_ema_ratio(close, ema_short, ema_long)

    lagged = compute_lagged_returns(close, lags=lags)
    features = pd.concat([features, lagged], axis=1)

    # Informative helper (purely backward-looking).
    features["daily_return"] = close.pct_change() * 100.0

    # ---- Target -------------------------------------------------------------
    # For training row ``t`` the label says whether Close(t+1) > Close(t).
    # The feature frame above uses only information up to and including t, so
    # the forward-looking shift is restricted to the label: no leakage.
    features["target"] = create_target(close)

    # ---- Drop look-back NaN rows (incl. the final row for the label) --------
    features = features.dropna().sort_index()
    return features


def prepare_dataset(path=None, **fe_kwargs):
    """One-stop helper: load, clean and build the engineered feature frame.

    Returns ``(features, feature_columns)`` where ``features`` is the
    :func:`build_features` output and ``feature_columns`` is the ordered list
    of predictor column names (everything except ``target``).
    """
    df = load_data(path)
    fe = build_features(df, **fe_kwargs)
    feature_columns = [c for c in fe.columns if c != "target"]
    return fe, feature_columns


if __name__ == "__main__":  # pragma: no cover
    fe, cols = prepare_dataset()
    print(f"Loaded & engineered {fe.shape[0]:,} rows x {fe.shape[1]} cols")
    print("Feature columns:", cols)
    print(fe.head())