"""
Smart Electricity Consumption - Analytics & Prediction dashboard.

Run:  python -m streamlit run dashboard.py
(If the model fails to load, run `python train_model.py` first.)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import (
    FEATURES, MIN_HISTORY, TARGET,
    chronological_split, load_series, make_features, next_hour_features,
)

# --------------------------------------------------------------------------
# Page setup and styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="Smart Electricity Consumption", page_icon="⚡", layout="wide")

NAVY, AMBER, TEAL, SLATE, RED = "#1E3A8A", "#F59E0B", "#0D9488", "#64748B", "#DC2626"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
      .hero {{
        background: linear-gradient(120deg, {NAVY} 0%, #2F5BD3 60%, {TEAL} 100%);
        padding: 1.4rem 1.8rem; border-radius: 14px; color: #fff; margin-bottom: 1.2rem;
      }}
      .hero h1 {{ margin: 0; font-size: 1.85rem; color: #fff; }}
      .hero p  {{ margin: .3rem 0 0 0; opacity: .9; font-size: 1rem; }}
      div[data-testid="stMetric"] {{
        background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid {AMBER};
        border-radius: 10px; padding: .8rem 1rem;
      }}
      div[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-weight: 600; }}
      h2, h3 {{ color: {NAVY}; }}
      section[data-testid="stSidebar"] {{ background: #F1F5F9; }}
    </style>
    <div class="hero">
      <h1>⚡ Smart Energy Analytics &amp; Prediction</h1>
      <p>Household electricity consumption analysis and next-hour forecasting with machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).parent


def show(fig: go.Figure) -> None:
    """Render a Plotly figure full-width (works on old and new Streamlit)."""
    fig.update_layout(
        template="plotly_white",
        colorway=[NAVY, AMBER, TEAL, RED, SLATE, "#7C3AED"],
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(family="Segoe UI, Arial, sans-serif", size=13),
        title=dict(font=dict(size=17, color=NAVY)),
        hovermode="x unified",
    )
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def need(name: str) -> Path:
    p = BASE / name
    if not p.exists():
        st.error(f"Missing **{name}**. Put it in the same folder as dashboard.py.")
        st.stop()
    return p


# --------------------------------------------------------------------------
# Data and model loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_series(path: Path, mtime: float) -> pd.Series:
    return load_series(path)


@st.cache_data(show_spinner=False)
def read_csv(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: Path, mtime: float):
    """Return (model, error_message, version_warnings). Never raises."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        vwarn = [str(w.message) for w in caught if "Inconsistent" in type(w.message).__name__]
        if not hasattr(model, "predict"):
            return None, "The file does not contain a scikit-learn model.", []
        if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
            return None, f"Model expects {model.n_features_in_} features, dashboard builds {len(FEATURES)}.", []
        names = getattr(model, "feature_names_in_", None)
        if names is not None and list(names) != FEATURES:
            return None, "Model feature names/order differ from the dashboard's 12 features.", []
        return model, None, vwarn
    except Exception as exc:  # e.g. ModuleNotFoundError: No module named '_loss'
        return None, f"{type(exc).__name__}: {exc}", []


def read_metadata() -> dict | None:
    p = BASE / "models" / "model_metadata.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def build_supervised(_series: pd.Series, key: str):
    data = make_features(_series)
    train, test = chronological_split(data)
    return data, train, test


@st.cache_data(show_spinner=False)
def evaluate_model(_model, _test: pd.DataFrame, key: str):
    pred = _model.predict(_test[FEATURES])
    mse = mean_squared_error(_test[TARGET], pred)
    return pred, {
        "MAE": mean_absolute_error(_test[TARGET], pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(_test[TARGET], pred),
    }


@st.cache_data(show_spinner=False)
def consistency_check(_series: pd.Series, key: str):
    """
    Rebuild the feature row of the LAST historical hour with next_hour_features()
    using only the data before it, and compare with the training table's row.
    """
    data = make_features(_series)
    last_ts = data.index[-1]
    pos = _series.index.get_loc(last_ts)
    t, X = next_hour_features(_series.iloc[:pos])
    cols = FEATURES if t == last_ts else [c for c in FEATURES if c.startswith(("lag", "rolling"))]
    ok = bool(np.allclose(X[cols].iloc[0].to_numpy(float), data.loc[last_ts, cols].to_numpy(float)))
    return ok, len(cols), last_ts


hourly_path = need("hourly_consumption.csv")
results_path = need("model_comparison_results.csv")
comparison_path = need("actual_vs_predicted.csv")
model_path = need("best_model.joblib")

series = read_series(hourly_path, hourly_path.stat().st_mtime)
results = read_csv(results_path, results_path.stat().st_mtime)
comparison = read_csv(comparison_path, comparison_path.stat().st_mtime).copy()
for c in ("datetime", "Unnamed: 0"):
    if c in comparison.columns:
        comparison["datetime"] = pd.to_datetime(comparison[c])
        break

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose page",
    ["Overview", "Consumption Analysis", "Model Comparison", "Prediction", "Energy Insights"],
)
st.sidebar.divider()
model, model_error, version_warnings = load_model(model_path, model_path.stat().st_mtime)
st.sidebar.markdown("**Model status**")
if model is not None:
    st.sidebar.success(f"Loaded: {type(model).__name__}")
else:
    st.sidebar.error("Model not loaded - see the Prediction page.")
st.sidebar.caption(f"{len(series):,} hourly readings · {series.index.min():%d %b %Y} → {series.index.max():%d %b %Y}")
st.sidebar.divider()
st.sidebar.caption("Machine Learning Mini Project | DSU")

# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Dashboard overview")
    a, b, c, d = st.columns(4)
    a.metric("Total recorded energy", f"{series.sum():,.1f} kWh")
    b.metric("Average hourly usage", f"{series.mean():.3f} kWh")
    c.metric("Peak hourly usage", f"{series.max():.3f} kWh")
    d.metric("Recorded hours", f"{len(series):,}")

    daily = series.resample("D").sum().rename("Daily consumption (kWh)").reset_index()
    daily.columns = ["Date", "Daily consumption (kWh)"]
    daily["7-day average"] = daily["Daily consumption (kWh)"].rolling(7, min_periods=1).mean()
    fig = px.line(daily, x="Date", y=["Daily consumption (kWh)", "7-day average"],
                  title="Daily electricity consumption")
    fig.data[0].update(line=dict(width=1), opacity=0.55)
    fig.data[1].update(line=dict(width=2.5))
    fig.update_layout(yaxis_title="kWh per day", legend_title_text="")
    show(fig)
    st.caption("Days with missing meter readings show lower totals; the dataset contains gaps in the original recording.")

    x, y = st.columns(2)
    by_hour = series.groupby(series.index.hour).mean().rename("Average kWh").reset_index()
    by_hour.columns = ["Hour", "Average kWh"]
    with x:
        show(px.bar(by_hour, x="Hour", y="Average kWh", title="Average consumption by hour"))
    by_day = (series.groupby(series.index.day_name()).mean().reindex(WEEKDAYS)
              .rename("Average kWh").reset_index())
    by_day.columns = ["Day", "Average kWh"]
    with y:
        show(px.bar(by_day, x="Day", y="Average kWh", title="Average consumption by weekday"))

elif page == "Consumption Analysis":
    st.subheader("Consumption analysis")
    min_d, max_d = series.index.min().date(), series.index.max().date()
    c1, c2 = st.columns([2, 1])
    dates = c1.date_input("Select date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    period = c2.selectbox("Aggregation", ["Hourly", "Daily", "Monthly"])

    selected = series
    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        selected = series.loc[(series.index >= pd.Timestamp(dates[0])) &
                              (series.index < pd.Timestamp(dates[1]) + pd.Timedelta(days=1))]
    else:
        st.info("Pick both a start and an end date to filter; showing the full period meanwhile.")
    if selected.empty:
        st.warning("No readings in the selected range.")
        st.stop()

    chart = selected if period == "Hourly" else selected.resample("D" if period == "Daily" else "MS").sum()
    fig = px.line(x=chart.index, y=chart.values, labels={"x": "Date/time", "y": "Consumption (kWh)"},
                  title=f"{period} consumption")
    fig.update_traces(line=dict(width=1.6))
    show(fig)
    if period == "Monthly":
        st.caption("The first and last months may be partial, so their totals can look lower.")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Summary statistics (hourly kWh)**")
        table(selected.describe().to_frame("Consumption (kWh)").round(4))
    with right:
        heat = (selected.groupby([selected.index.day_name(), selected.index.hour]).mean()
                .unstack().reindex(WEEKDAYS))
        fig = px.imshow(heat.values, x=list(heat.columns), y=WEEKDAYS, aspect="auto",
                        color_continuous_scale="YlOrRd",
                        labels=dict(x="Hour of day", y="", color="kWh"),
                        title="Average usage heatmap (weekday × hour)")
        fig.update_layout(hovermode="closest")
        show(fig)

elif page == "Model Comparison":
    st.subheader("Machine learning model comparison")
    st.caption("Original evaluation from the Google Colab notebook, on the chronological 20 % hold-out test set.")
    higher_better = {"R2"}
    best = results.sort_values("RMSE").iloc[0]
    base_row = results[results["Model"].str.contains("Persistence", case=False)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", str(best["Model"]))
    k2.metric("MAE", f"{best['MAE']:.4f} kWh")
    k3.metric("RMSE", f"{best['RMSE']:.4f} kWh")
    k4.metric("R²", f"{best['R2']:.4f}")
    if not base_row.empty:
        gain = (1 - best["MAE"] / base_row["MAE"].iloc[0]) * 100
        st.success(f"{best['Model']} reduces MAE by **{gain:.1f}%** compared with the persistence baseline "
                   f"(same value as one hour ago).")

    table(results.round(4), hide_index=True)
    metric = st.selectbox("Metric", [m for m in ["MAE", "MSE", "RMSE", "R2"] if m in results.columns])
    ordered = results.sort_values(metric, ascending=metric not in higher_better).copy()
    top = ordered.iloc[0]["Model"]
    ordered["Highlight"] = np.where(ordered["Model"] == top, "Best", "Other")
    fig = px.bar(ordered, x="Model", y=metric, color="Highlight", text=ordered[metric].round(4),
                 color_discrete_map={"Best": AMBER, "Other": NAVY}, title=f"Models compared by {metric}")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_traces(textposition="outside")
    show(fig)
    st.caption("Lower MAE, MSE and RMSE are better; higher R² is better. "
               "R² below zero means worse than predicting the mean.")

    if {"Actual", "Predicted"}.issubset(comparison.columns):
        n = st.slider("Number of most recent test hours to display", 100, min(2000, len(comparison)), 500, step=50)
        sample = comparison.tail(n)
        xs = sample["datetime"] if "datetime" in sample.columns else np.arange(len(sample))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=sample["Actual"], mode="lines", name="Actual", line=dict(color=NAVY, width=1.6)))
        fig.add_trace(go.Scatter(x=xs, y=sample["Predicted"], mode="lines", name="Predicted", line=dict(color=AMBER, width=1.6)))
        fig.update_layout(title="Actual vs predicted consumption (original Colab model)",
                          xaxis_title="Time", yaxis_title="kWh")
        show(fig)

    meta = read_metadata()
    if meta:
        with st.expander("Reproducibility: original Colab run vs local retrain"):
            loc, orig = meta["local_test_metrics"], meta["original_colab_metrics"]
            cmp_df = pd.DataFrame({
                "Metric": ["MAE", "RMSE", "R2"],
                "Original Colab (scikit-learn 1.6.1)": [orig["MAE"], orig["RMSE"], orig["R2"]],
                f"Local retrain (scikit-learn {meta['versions']['scikit-learn']})": [loc["MAE"], loc["RMSE"], loc["R2"]],
            })
            table(cmp_df.round(4), hide_index=True)
            st.caption("Same data, same 12 features, same chronological 80:20 split and same hyper-parameters. "
                       "Small differences come from scikit-learn's internal changes between versions "
                       "(e.g. early stopping). The table above is the original, unmodified Colab result.")

elif page == "Prediction":
    st.subheader("Next-hour electricity prediction")

    if model is None:
        st.error("The saved model could not be loaded in this Python environment.")
        st.code(model_error)
        st.markdown(
            "**Cause:** `best_model.joblib` was pickled with a different scikit-learn version "
            "(Colab) than the one installed here. Pickles are only guaranteed to load with the same version.\n\n"
            "**Fix:** retrain locally (about 10 seconds), then reload this page:"
        )
        st.code("python train_model.py", language="powershell")
        st.stop()
    if version_warnings:
        st.warning("The model was saved by a different scikit-learn version. Run `python train_model.py` "
                   "to retrain it for this environment.")
    if len(series) < MIN_HISTORY:
        st.error(f"At least {MIN_HISTORY} hourly observations are required.")
        st.stop()

    t, X = next_hour_features(series)
    prediction = float(model.predict(X)[0])
    last_val, avg24 = float(series.iloc[-1]), float(series.iloc[-24:].mean())

    a, b, c, d = st.columns(4)
    a.metric("Predicted next-hour consumption", f"{prediction:.3f} kWh", f"{prediction - last_val:+.3f} vs last reading")
    b.metric("Forecast timestamp", t.strftime("%d %b %Y, %H:%M"))
    c.metric("Last recorded reading", f"{last_val:.3f} kWh")
    d.metric("Last 24-hour average", f"{avg24:.3f} kWh")
    st.info("One-step-ahead forecast from the latest historical readings; not a live meter reading.")

    tab1, tab2, tab3 = st.tabs(["Forecast", "Feature audit (no leakage)", "Backtest on held-out data"])

    with tab1:
        recent = series.tail(48)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="History",
                                 line=dict(color=NAVY, width=2)))
        fig.add_trace(go.Scatter(x=[recent.index[-1], t], y=[last_val, prediction], mode="lines",
                                 line=dict(color=AMBER, dash="dot"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[t], y=[prediction], mode="markers+text", text=[f"{prediction:.3f} kWh"],
                                 textposition="top center", name="Forecast",
                                 marker=dict(size=13, color=AMBER, line=dict(width=2, color=NAVY))))
        fig.update_layout(title="Recent consumption and next-hour forecast", xaxis_title="Time", yaxis_title="kWh")
        show(fig)

    with tab2:
        desc = {
            "hour": "Hour of the forecast timestamp", "day_of_week": "0 = Monday … 6 = Sunday",
            "month": "Month of the forecast timestamp", "is_weekend": "1 if Saturday/Sunday",
            "lag_1": "Consumption 1 hour before", "lag_2": "Consumption 2 hours before",
            "lag_3": "Consumption 3 hours before", "lag_24": "Same hour yesterday",
            "lag_48": "Same hour two days ago", "lag_168": "Same hour last week",
            "rolling_mean_24": "Mean of the last 24 readings", "rolling_mean_168": "Mean of the last 168 readings",
        }
        feat = pd.DataFrame({"Feature": FEATURES, "Value": X.iloc[0].to_numpy(float).round(4),
                             "Meaning": [desc[f] for f in FEATURES]})
        st.markdown(f"**The {len(FEATURES)} inputs sent to the model for {t:%d %b %Y %H:%M}**")
        table(feat, hide_index=True)

        key = f"{len(series)}|{series.index[-1]}"
        data, train, test = build_supervised(series, key)
        ok, ncols, last_ts = consistency_check(series, key)
        names = getattr(model, "feature_names_in_", None)
        checks = [
            (names is None or list(names) == FEATURES, "Feature names and order match what the model was trained on."),
            (X.shape[1] == 12, "Exactly 12 features are passed to the model."),
            (train.index.max() < test.index.min(),
             f"Chronological split: training ends {train.index.max():%d %b %Y}, testing starts {test.index.min():%d %b %Y}."),
            (len(train) == int(len(data) * 0.8), f"80:20 split: {len(train):,} training rows, {len(test):,} test rows."),
            (ok, f"Live feature builder reproduces the training table for the last known hour ({last_ts:%d %b %Y %H:%M}) "
                 f"using only earlier readings ({ncols} features compared)."),
            (True, "Rolling means cover only past readings (the forecast hour is never inside its own window)."),
        ]
        for passed, text in checks:
            (st.success if passed else st.error)(("✅ " if passed else "❌ ") + text)

    with tab3:
        key = f"{len(series)}|{series.index[-1]}"
        data, train, test = build_supervised(series, key)
        pred_test, m = evaluate_model(model, test, f"{key}|{model_path.stat().st_mtime}")
        st.markdown("**Loaded model evaluated live on the held-out final 20 % of the data**")
        e1, e2, e3 = st.columns(3)
        e1.metric("MAE", f"{m['MAE']:.4f} kWh")
        e2.metric("RMSE", f"{m['RMSE']:.4f} kWh")
        e3.metric("R²", f"{m['R2']:.4f}")
        st.caption("Computed now from the model in best_model.joblib; the Model Comparison page shows the "
                   "original Colab figures.")
        n = st.slider("Hours to plot (end of test set)", 48, 1000, 336, step=24)
        bt = pd.DataFrame({"Actual": test[TARGET].to_numpy(), "Predicted": pred_test}, index=test.index).tail(n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Actual"], mode="lines", name="Actual", line=dict(color=NAVY, width=1.6)))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Predicted"], mode="lines", name="Predicted", line=dict(color=AMBER, width=1.6)))
        fig.update_layout(title="One-step-ahead predictions vs actual (held-out data)", xaxis_title="Time", yaxis_title="kWh")
        show(fig)

else:
    st.subheader("Energy usage insights")
    hour_avg = series.groupby(series.index.hour).mean()
    hi, lo = int(hour_avg.idxmax()), int(hour_avg.idxmin())
    daily = series.resample("D").sum()
    peak_day = daily.idxmax()
    a, b, c = st.columns(3)
    a.metric("Highest average usage hour", f"{hi:02d}:00")
    b.metric("Lowest average usage hour", f"{lo:02d}:00")
    c.metric("Highest-use day", peak_day.strftime("%d %b %Y"))

    left, right = st.columns([1, 1])
    with left:
        st.write(f"- Average hourly consumption: **{series.mean():.3f} kWh**.")
        st.write(f"- Highest average use occurs around **{hi:02d}:00**.")
        st.write(f"- Lowest average use occurs around **{lo:02d}:00**.")
        st.write(f"- Highest daily total: **{daily.max():.2f} kWh** on **{peak_day.date()}**.")
        wk = series[series.index.dayofweek < 5].mean()
        we = series[series.index.dayofweek >= 5].mean()
        st.write(f"- Average weekday use **{wk:.3f} kWh** vs weekend **{we:.3f} kWh** per hour.")
    with right:
        prof = (series.groupby([np.where(series.index.dayofweek >= 5, "Weekend", "Weekday"), series.index.hour])
                .mean().unstack(0))
        fig = px.line(prof, labels={"index": "Hour", "value": "Average kWh", "variable": ""},
                      title="Typical day: weekday vs weekend")
        fig.update_layout(xaxis_title="Hour of day")
        show(fig)
    st.warning("These are descriptive patterns from one household's historical dataset; they do not prove "
               "appliance-level causes or guaranteed savings.")