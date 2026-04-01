"""KabuPredict ML — interactive Streamlit dashboard.

Run with:  streamlit run app.py

The dashboard lets you tune the train/test split and XGBoost hyperparameters
in the sidebar, then live-refreshes data preview, feature distributions,
out-of-fold model diagnostics and a Plotly equity-curve comparison of the
ML long-only strategy versus a Buy-and-Hold benchmark.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import backtest, preprocess
from src import model as model_mod


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_assets():
    clean = preprocess.load_data()
    fe, cols = preprocess.prepare_dataset()
    return clean, fe, cols


@st.cache_resource(show_spinner=False)
def train_and_backtest(clean, fe, cols, params: dict, test_size: float,
                       n_splits: int, key: str):
    """Fit + evaluate on pushed params, then run the back-test."""
    outcome = model_mod.train_and_evaluate(
        fe[cols], fe["target"], params=params, test_size=test_size,
        n_splits=n_splits,
    )
    result = backtest.run_backtest(clean, outcome)
    return outcome, result


# --------------------------------------------------------------------------- #
# Page / sidebar configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="KabuPredict ML",
    page_icon="📈",
    layout="wide",
)

st.title("📈 KabuPredict ML")
st.caption("Next-day price-direction classifier for Suzuki Motor (7269.T) — "
           "XGBoost + time-series CV + strategy back-test.")

with st.sidebar:
    st.header("Configuration")

    st.subheader("Data split")
    test_size = st.slider(
        "Test set size (fraction, chronological tail)",
        min_value=0.10, max_value=0.40, value=0.20, step=0.05,
    )
    n_splits = st.slider("TimeSeriesSplit folds (train)", 3, 8, 5)

    st.subheader("XGBoost hyperparameters")
    max_depth = st.slider("max_depth", 1, 8, 3)
    learning_rate = st.slider(
        "learning_rate", 0.01, 0.30, 0.03, step=0.01, format="%.2f")
    n_estimators = st.slider("n_estimators", 50, 400, 100, step=50)
    subsample = st.slider("subsample", 0.5, 1.0, 0.9, step=0.05)
    colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.9, step=0.05)
    min_child_weight = st.slider(
        "min_child_weight", 1, 20, 1, help="1 = standard gain.")

    run = st.button("▶ Train & Back-test", type="primary",
                    width="stretch")

params = {
    "max_depth": int(max_depth),
    "learning_rate": float(learning_rate),
    "n_estimators": int(n_estimators),
    "subsample": float(subsample),
    "colsample_bytree": float(colsample_bytree),
    "min_child_weight": float(min_child_weight),
}
param_key = f"{max_depth}-{learning_rate}-{n_estimators}-{subsample}" \
            f"-{colsample_bytree}-{min_child_weight}-{test_size}-{n_splits}"
# --------------------------------------------------------------------------- #
# Main body
# --------------------------------------------------------------------------- #
clean, fe, cols = load_assets()
n_rows, n_cols = clean.shape
n_feat = fe.shape[0]

st.info(
    f"Loaded **{n_rows:,}** daily OHLCV rows "
    f"({clean.index.min().date()} → {clean.index.max().date()}) — "
    f"{n_feat:,} engineered rows usable for training."
)

if not run:
    st.warning("Adjust settings in the sidebar and press **▶ Train & Back-test**.")
    st.stop()

with st.spinner("☕ Fitting XGBoost and running the back-test…"):
    outcome, bt = train_and_backtest(clean, fe, cols, params, test_size,
                                     n_splits, param_key)

# ---- KPI cards ---------------------------------------------------------
kpi = bt.kpis()
c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Sharpe Ratio (ML)", f"{kpi['sharpe_strategy']:.3f}",
          f"B&H {kpi['sharpe_benchmark']:.3f}")
c2.metric("🎯 Precision (long calls)", f"{kpi['precision']:.1%}")
c3.metric("💹 Cumulative Return (ML)",
          f"{kpi['cumulative_strategy']:.2%}",
          f"B&H {kpi['cumulative_benchmark']:.2%}")
c4.metric("📅 Out-of-sample bars", f"{bt.window_size:,}")

# ---- Tabs ---------------------------------------------------------------
tab_raw, tab_feat, tab_oof, tab_equity, tab_io = st.tabs(
    ["Raw & Engineered Data", "Feature Distributions", "Out-of-Fold Performance",
     "Strategy Equity Curve", "Model / Importances"])

with tab_raw:
    st.subheader("Raw OHLCV preview")
    st.dataframe(clean.tail(15), width="stretch")
    st.subheader("Engineered features + target (tail)")
    st.dataframe(fe.tail(15), width="stretch")
    st.caption("Target = 1 if next-day Close is higher than today's.")

with tab_feat:
    st.subheader("Feature distributions (coloured by target)")
    sel = st.selectbox("Feature", cols, index=0)
    fe_col = fe[sel].astype(float)
    fe_color = fe["target"].map({1: "Up (1)", 0: "Down (0)"})
    chart = px.histogram(fe, x=fe_col, color=fe_color,
                         nbins=60, marginal="box", opacity=0.7,
                         title=f"Distribution of {sel} by next-day direction")
    st.plotly_chart(chart, width="stretch")

with tab_oof:
    st.subheader("Out-of-fold performance (TimeSeriesSplit on the train block)")
    oof_acc = outcome.oof_metrics["accuracy"]
    oof_prec = outcome.oof_metrics["precision"]
    oof_rec = outcome.oof_metrics["recall"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy", f"{oof_acc:.1%}")
    a2.metric("Precision", f"{oof_prec:.1%}")
    a3.metric("Recall", f"{oof_rec:.1%}")

    cm = np.array(outcome.oof_metrics["confusion_matrix"])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Predicted Down", "Predicted Up"],
        y=["Actual Down", "Actual Up"],
        colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}"))
    fig_cm.update_layout(title="OOF Confusion Matrix", height=320)
    st.plotly_chart(fig_cm, width="stretch")

with tab_equity:
    st.subheader("Out-of-sample equity comparison")
    eq = bt.equity_df
    fig_eq = go.Figure()
    for series_name, color in [("ML Strategy", "#e11d48"),
                               ("Buy & Hold", "#2563eb")]:
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq[series_name], mode="lines",
            name=series_name, line=dict(color=color, width=2)))
    fig_eq.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig_eq.update_layout(
        title="$1 invested → equity curve (start = 1.0)",
        xaxis_title="Date", yaxis_title="Equity growth",
        hovermode="x unified", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_eq, width="stretch")

    dd = bt.daily_df.melt(var_name="Series", value_name="Daily return")
    fig_ret = px.histogram(dd, x="Daily return", color="Series", nbins=80,
                           opacity=0.7, barmode="overlay", marginal="box",
                           title="Daily return distributions (test window)")
    st.plotly_chart(fig_ret, width="stretch")

with tab_io:
    st.subheader("Model diagnostic metrics")
    t = outcome.test_metrics
    st.write("**Held-out test block metrics**")
    d = pd.DataFrame([{
        "metric": "Accuracy", "train (OOF)": f"{oof_acc:.3f}",
        "test": f"{t['accuracy']:.3f}"},
        {"metric": "Precision", "train (OOF)": f"{oof_prec:.3f}",
         "test": f"{t['precision']:.3f}"},
        {"metric": "Recall", "train (OOF)": f"{oof_rec:.3f}",
         "test": f"{t['recall']:.3f}"},
        {"metric": "F1", "train (OOF)": f"{outcome.oof_metrics['f1']:.3f}",
         "test": f"{t['f1']:.3f}"}])
    st.dataframe(d, width="stretch")

    st.subheader("Feature importances")
    fi = outcome.feature_importance.reset_index()
    fi.columns = ["Feature", "Importance"]
    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    title="XGBoost gain-based feature importances")
    st.plotly_chart(fig_fi, width="stretch")

    with st.expander("Back-test performance detail"):
        keep = ("cumulative_return", "annualized_return",
                "sharpe_ratio", "max_drawdown")
        detail = pd.DataFrame([bt.strategy, bt.benchmark])[list(keep)].transpose()
        detail.index = ["Cumulative return", "Annualized return",
                        "Sharpe ratio", "Max drawdown"]
        st.dataframe(detail.style.format("{:.4f}"), width="stretch")

st.success("Done — back-test complete.")