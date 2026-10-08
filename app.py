from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------------------------------------------
# App setup and data access
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "outputs" / "predictions"
ANOMALIES = ROOT / "outputs" / "anomalies"
METRICS = ROOT / "outputs" / "metrics"
INVERTER_METRICS = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="PV Digital Twin",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_COLUMNS = {
    "Persistence": "Persistence",
    "Linear Regression": "Linear_Regression",
    "Random Forest": "Random_Forest",
    "XGBoost": "XGBoost",
}
MODEL_COLORS = {
    "Actual power": "#f3b33d",
    "Persistence": "#778da9",
    "Linear Regression": "#56b4a9",
    "Random Forest": "#5687c8",
    "XGBoost": "#a476c4",
}

FORECAST_MODEL_FILES = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
FORECAST_FEATURES = [
    "hour", "minute", "day", "day_of_week", "day_of_year", "month",
    "hour_sin", "hour_cos",
    "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
    "AC_POWER_lag_1", "AC_POWER_lag_4", "AC_POWER_lag_8", "AC_POWER_lag_96",
    "AC_POWER_rolling_mean_4", "AC_POWER_rolling_mean_8",
]


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame | None:
    path = Path(path_string)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return pd.DataFrame({"_load_error": [str(exc)]})


def load_predictions(plant: int) -> pd.DataFrame | None:
    df = read_csv(str(PREDICTIONS / f"plant{plant}_test_predictions.csv"))
    if df is None or "_load_error" in df:
        return df
    if "DATE_TIME" in df:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    for col in ["AC_POWER", "TARGET_AC_POWER", *MODEL_COLUMNS.values()]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_anomaly_report(plant: int) -> pd.DataFrame | None:
    df = read_csv(str(ANOMALIES / f"plant{plant}_anomaly_report.csv"))
    if df is None or "_load_error" in df:
        return df
    if "DATE_TIME" in df:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    return df


def load_anomaly_scores(plant: int) -> pd.DataFrame | None:
    df = read_csv(str(ANOMALIES / f"plant{plant}_anomaly_scores.csv"))
    if df is None or "_load_error" in df:
        return df
    if "DATE_TIME" in df:
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    return df


def load_inverter_summary(plant: int) -> pd.DataFrame | None:
    return read_csv(str(ANOMALIES / f"plant{plant}_inverter_anomaly_summary.csv"))


def load_metrics(filename: str) -> pd.DataFrame | None:
    return read_csv(str(METRICS / filename))


@st.cache_resource(show_spinner="Loading the trained model...")
def load_forecast_model(plant_number: int, model_name: str):
    """Load a trained weather-aware model once per app process."""
    model_folder = FORECAST_MODEL_FILES[model_name]
    model_path = MODELS / model_folder / f"plant{plant_number}_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path.relative_to(ROOT)}. "
            "Make the trained model artifact available on this app host; "
            "if needed, rerun notebook 07_Model_Training_Comparison.ipynb."
        )
    model = joblib.load(model_path)
    expected_features = getattr(model, "n_features_in_", len(FORECAST_FEATURES))
    if expected_features != len(FORECAST_FEATURES):
        raise ValueError(
            f"This model expects {expected_features} features, but the app supplies "
            f"{len(FORECAST_FEATURES)} weather-aware features. Retrain it with notebook 07."
        )
    return model


def available_forecast_models(plant_number: int) -> list[str]:
    """Return models whose artifacts are present on this app host."""
    return [
        name for name, folder in FORECAST_MODEL_FILES.items()
        if (MODELS / folder / f"plant{plant_number}_model.pkl").is_file()
    ]


def build_forecast_features(
    timestamp: pd.Timestamp,
    irradiation: float,
    ambient_temperature: float,
    module_temperature: float,
    power_history: dict[str, float],
) -> pd.DataFrame:
    """Create the same ordered 17 predictors used by the training notebooks."""
    hour_fraction = timestamp.hour + timestamp.minute / 60
    values = {
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day": timestamp.day,
        "day_of_week": timestamp.dayofweek,
        "day_of_year": timestamp.dayofyear,
        "month": timestamp.month,
        "hour_sin": np.sin(2 * np.pi * hour_fraction / 24),
        "hour_cos": np.cos(2 * np.pi * hour_fraction / 24),
        "IRRADIATION": irradiation,
        "AMBIENT_TEMPERATURE": ambient_temperature,
        "MODULE_TEMPERATURE": module_temperature,
        **power_history,
    }
    return pd.DataFrame([[values[name] for name in FORECAST_FEATURES]], columns=FORECAST_FEATURES)


def show_load_problem(*files: Path) -> None:
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        st.warning("Some project outputs are missing. Expected files: " + ", ".join(missing))


def metrics_for(actual: pd.Series, predicted: pd.Series) -> tuple[float, float, float]:
    valid = actual.notna() & predicted.notna()
    y, yhat = actual[valid], predicted[valid]
    if y.empty:
        return np.nan, np.nan, np.nan
    return (
        mean_absolute_error(y, yhat),
        float(np.sqrt(mean_squared_error(y, yhat))),
        r2_score(y, yhat) if len(y) > 1 else np.nan,
    )


def power_chart(df: pd.DataFrame, model_names: list[str], title: str) -> go.Figure:
    plot_df = df.sort_values("DATE_TIME")
    # Each row's target is the next 15-minute observation, predicted from DATE_TIME.
    # Plot the target and its forecast at the horizon timestamp, not the input time.
    forecast_time = plot_df["DATE_TIME"] + pd.Timedelta(minutes=15)
    fig = go.Figure()
    if "TARGET_AC_POWER" in plot_df:
        fig.add_trace(go.Scatter(
            x=forecast_time, y=plot_df["TARGET_AC_POWER"],
            name="Actual power (target)", mode="lines",
            line={"color": MODEL_COLORS["Actual power"], "width": 2.2},
        ))
    for label in model_names:
        col = MODEL_COLUMNS[label]
        if col in plot_df:
            fig.add_trace(go.Scatter(
                x=forecast_time, y=plot_df[col], name=label,
                mode="lines", line={"color": MODEL_COLORS[label], "width": 1.5},
            ))
    fig.update_layout(
        title=title, xaxis_title="Timestamp", yaxis_title="AC power",
        hovermode="x unified", legend_title="Series", height=440,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return fig


def filter_dates(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if "DATE_TIME" not in df or df["DATE_TIME"].dropna().empty:
        return df
    lo, hi = df["DATE_TIME"].min().date(), df["DATE_TIME"].max().date()
    dates = st.sidebar.date_input("Test period", value=(lo, hi), min_value=lo, max_value=hi, key=key)
    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = dates
        return df[df["DATE_TIME"].dt.date.between(start, end)]
    return df


def format_number(value: float, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{value:,.{digits}f}"


# -----------------------------------------------------------------------------
# Navigation and shared filters
# -----------------------------------------------------------------------------
st.sidebar.title("☀️ PV Digital Twin")
st.sidebar.caption("Solar generation forecasting and inverter monitoring")
page = st.sidebar.radio(
    "Explore",
    ["Overview", "Make a forecast", "Power prediction", "Inverter performance", "Abnormal-performance screening", "Model comparison"],
    label_visibility="collapsed",
)
plant = st.sidebar.selectbox("Plant", [1, 2], format_func=lambda p: f"Plant {p}")
pred = load_predictions(plant)
summary = load_inverter_summary(plant)
report = load_anomaly_report(plant)
scores = load_anomaly_scores(plant)

st.title(page)
if page == "Make a forecast":
    st.caption(f"Plant {plant} · manual weather-aware forecast")
else:
    st.caption(f"Plant {plant} · historical test-period results")

if pred is not None and "_load_error" in pred:
    st.error("Could not read prediction data: " + pred["_load_error"].iloc[0])
    st.stop()


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Make a forecast":
    st.write(
        "Enter measurements available now for one inverter. The selected model "
        "will forecast its AC power 15 minutes after the timestamp."
    )
    st.info(
        "This form makes a forecast from manually entered values. It does not "
        "connect to live plant sensors. Use the same measurement units as the project data."
    )
    # XGBoost is a compact artifact and performed competitively on both plants.
    available_models = available_forecast_models(plant)
    if not available_models:
        st.error(
            f"No trained model artifacts are available for Plant {plant}. "
            "Add a trained artifact under the models folder and redeploy."
        )
        st.stop()
    default_model = "XGBoost" if "XGBoost" in available_models else available_models[0]
    model_name = st.selectbox(
        "Forecast model",
        available_models,
        index=available_models.index(default_model),
        help="Choose one of the weather-aware models trained in notebook 07.",
    )

    with st.form("manual_forecast_form"):
        st.subheader("Forecast timestamp and current weather")
        timestamp = st.datetime_input(
            "Timestamp of the current inverter reading",
            value=pd.Timestamp.now().floor("15min").to_pydatetime(),
            help="The forecast is for exactly 15 minutes after this timestamp.",
        )
        weather_cols = st.columns(3)
        irradiation = weather_cols[0].number_input(
            "Irradiation at this time", min_value=0.0, max_value=2.0,
            value=0.2, step=0.01, format="%.3f",
            help="Use the same irradiation scale as the plant weather CSV.",
        )
        ambient_temperature = weather_cols[1].number_input(
            "Ambient temperature", min_value=-50.0, max_value=80.0,
            value=25.0, step=0.5,
            help="Use the same temperature unit as the plant weather CSV.",
        )
        module_temperature = weather_cols[2].number_input(
            "Module temperature", min_value=-50.0, max_value=100.0,
            value=30.0, step=0.5,
            help="Use the same temperature unit as the plant weather CSV.",
        )

        st.subheader("Current and historical inverter power")
        st.caption(
            "Enter AC power in the same units as the training data. Lag inputs "
            "refer to earlier rows for this inverter; with missing timestamps, "
            "a row may be more than 15 minutes apart. Rolling means summarize "
            "the previous 4 or 8 rows."
        )
        power_cols = st.columns(4)
        current_power = power_cols[0].number_input(
            "Current AC power", min_value=0.0, value=0.0, step=10.0,
        )
        lag_1 = power_cols[1].number_input(
            "Power: previous row", min_value=0.0, value=0.0, step=10.0,
        )
        lag_4 = power_cols[2].number_input(
            "Power: 4 rows back", min_value=0.0, value=0.0, step=10.0,
        )
        lag_8 = power_cols[3].number_input(
            "Power: 8 rows back", min_value=0.0, value=0.0, step=10.0,
        )
        history_cols = st.columns(3)
        lag_96 = history_cols[0].number_input(
            "Power: 96 rows back", min_value=0.0, value=0.0, step=10.0,
        )
        rolling_4 = history_cols[1].number_input(
            "Mean power: previous 4 rows", min_value=0.0, value=0.0, step=10.0,
        )
        rolling_8 = history_cols[2].number_input(
            "Mean power: previous 8 rows", min_value=0.0, value=0.0, step=10.0,
        )
        submitted = st.form_submit_button("Forecast 15 minutes ahead", type="primary")

    if submitted:
        if timestamp.minute % 15 != 0 or timestamp.second != 0:
            st.error("Choose a timestamp on a 15-minute boundary (for example, 10:00 or 10:15).")
        else:
            history = {
                "AC_POWER_lag_1": lag_1,
                "AC_POWER_lag_4": lag_4,
                "AC_POWER_lag_8": lag_8,
                "AC_POWER_lag_96": lag_96,
                "AC_POWER_rolling_mean_4": rolling_4,
                "AC_POWER_rolling_mean_8": rolling_8,
            }
            model_input = build_forecast_features(
                pd.Timestamp(timestamp), irradiation, ambient_temperature,
                module_temperature, history,
            )
            try:
                trained_model = load_forecast_model(plant, model_name)
                forecast = float(trained_model.predict(model_input)[0])
            except Exception as exc:
                st.error(f"Could not load or run the trained model: {exc}")
            else:
                forecast_time = pd.Timestamp(timestamp) + pd.Timedelta(minutes=15)
                result_cols = st.columns(2)
                result_cols[0].metric(
                    f"{model_name} forecast at {forecast_time:%Y-%m-%d %H:%M}",
                    format_number(forecast),
                )
                result_cols[1].metric(
                    "Persistence reference",
                    format_number(current_power),
                    help="Persistence carries current AC power forward as its 15-minute forecast.",
                )
                st.caption(
                    "The forecast is an estimate. Compare it with the inverter's "
                    "actual reading once the forecast time arrives."
                )
                with st.expander("Inspect the exact model inputs"):
                    st.dataframe(model_input, hide_index=True, width="stretch")

elif page == "Overview":
    show_load_problem(
        PREDICTIONS / f"plant{plant}_test_predictions.csv",
        ANOMALIES / f"plant{plant}_inverter_anomaly_summary.csv",
    )
    if pred is None:
        st.info("Add the generated prediction CSVs under `outputs/predictions` to populate the dashboard.")
        st.stop()

    actual_col = "TARGET_AC_POWER" if "TARGET_AC_POWER" in pred else "AC_POWER"
    total_rows = len(pred)
    inverter_count = pred["SOURCE_KEY"].nunique() if "SOURCE_KEY" in pred else 0
    avg_power = pred[actual_col].mean() if actual_col in pred else np.nan
    anomaly_count = int(pd.to_numeric(summary.get("Anomalies", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if summary is not None else 0
    overall_rate = 100 * anomaly_count / summary["Total_Observations"].astype(float).sum() if summary is not None and "Total_Observations" in summary and summary["Total_Observations"].astype(float).sum() else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test observations", f"{total_rows:,}")
    k2.metric("Inverters", f"{inverter_count:,}")
    k3.metric("Mean target power", f"{format_number(avg_power)}")
    k4.metric("Screening flags", f"{anomaly_count:,}", f"{format_number(overall_rate)}% of evaluated active rows" if not pd.isna(overall_rate) else None)

    st.subheader("Actual and expected power")
    overview_df = pred.copy()
    if "DATE_TIME" in overview_df and not overview_df["DATE_TIME"].dropna().empty:
        start, end = overview_df["DATE_TIME"].min().date(), overview_df["DATE_TIME"].max().date()
        if (end - start).days > 3:
            overview_df = overview_df[overview_df["DATE_TIME"].dt.date >= end - pd.Timedelta(days=2)]
    if "SOURCE_KEY" in overview_df:
        overview_df = overview_df.groupby("DATE_TIME", as_index=False)[[actual_col, *[c for c in MODEL_COLUMNS.values() if c in overview_df]]].sum(min_count=1)
    st.plotly_chart(power_chart(overview_df, ["Persistence", "Random Forest"], "Plant aggregate · recent test window"), width="stretch")

    st.subheader("Forecast scores")
    metric_df = load_metrics("final_model_results.csv")
    if metric_df is not None and not metric_df.empty:
        plant_rows = metric_df[metric_df["Plant"].astype(str).str.lower() == f"plant {plant}".lower()]
        st.dataframe(plant_rows, hide_index=True, width="stretch")
    else:
        st.caption("Saved overall model metrics are not available.")
    st.info("Screening flags identify unusual statistical deviations; they are not confirmed equipment faults, and the flag rate is not a detection-accuracy score.")


# -----------------------------------------------------------------------------
# Power prediction
# -----------------------------------------------------------------------------
elif page == "Power prediction":
    if pred is None:
        show_load_problem(PREDICTIONS / f"plant{plant}_test_predictions.csv")
        st.stop()
    if "SOURCE_KEY" not in pred or "DATE_TIME" not in pred:
        st.error("Prediction data needs SOURCE_KEY and DATE_TIME columns.")
        st.stop()
    selected_source = st.sidebar.selectbox("Inverter", sorted(pred["SOURCE_KEY"].dropna().unique()))
    inverter_df = pred[pred["SOURCE_KEY"] == selected_source].copy()
    inverter_df = filter_dates(inverter_df, "prediction_dates")
    available = [name for name, col in MODEL_COLUMNS.items() if col in inverter_df]
    if not available:
        st.error("No model prediction columns were found in this CSV.")
        st.stop()
    selected_model = st.selectbox("Forecast model", available)
    actual_col = "TARGET_AC_POWER" if "TARGET_AC_POWER" in inverter_df else "AC_POWER"
    if actual_col == "AC_POWER":
        st.warning("TARGET_AC_POWER is absent; metrics below use AC_POWER as the available reference.")
    mae, rmse, r2 = metrics_for(inverter_df[actual_col], inverter_df[MODEL_COLUMNS[selected_model]])
    a, b, c = st.columns(3)
    a.metric("MAE", format_number(mae))
    b.metric("RMSE", format_number(rmse))
    c.metric("R²", format_number(r2, 4))
    st.plotly_chart(power_chart(inverter_df, [selected_model], f"{selected_source} · 15-minute-ahead forecast"), width="stretch")
    st.caption("Metrics compare predictions with TARGET_AC_POWER (the next-step target) when that column is available.")
    with st.expander("Show selected observations"):
        columns = [c for c in ["DATE_TIME", "SOURCE_KEY", "AC_POWER", "TARGET_AC_POWER", MODEL_COLUMNS[selected_model]] if c in inverter_df]
        st.dataframe(inverter_df[columns].sort_values("DATE_TIME", ascending=False), hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# Inverter performance
# -----------------------------------------------------------------------------
elif page == "Inverter performance":
    if pred is None:
        show_load_problem(PREDICTIONS / f"plant{plant}_test_predictions.csv")
        st.stop()
    inverter_mae_path = INVERTER_METRICS / f"plant{plant}_inverter_mae.csv"
    inverter_mae = read_csv(str(inverter_mae_path))
    if inverter_mae is None:
        show_load_problem(inverter_mae_path)
    else:
        st.subheader("Forecast error by inverter")
        score_cols = [c for c in inverter_mae.columns if c != "SOURCE_KEY"]
        if score_cols:
            chosen_score = st.selectbox("Score", score_cols, index=0)
            chart_df = inverter_mae.sort_values(chosen_score, ascending=False)
            fig = px.bar(chart_df, x="SOURCE_KEY", y=chosen_score, color=chosen_score,
                         color_continuous_scale="YlOrRd", title=f"{chosen_score} by inverter")
            fig.update_layout(xaxis_title="Inverter", yaxis_title="MAE", height=450, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(inverter_mae.sort_values(chosen_score, ascending=False), hide_index=True, width="stretch")
    if summary is not None:
        st.subheader("Screening results by inverter")
        sort_col = "Anomaly_Percentage" if "Anomaly_Percentage" in summary else "Anomalies"
        display_summary = summary.sort_values(sort_col, ascending=False)
        fig = px.bar(display_summary, x="SOURCE_KEY", y=sort_col, color=sort_col,
                     color_continuous_scale="OrRd", title="Existing flags by inverter")
        fig.update_layout(xaxis_title="Inverter", yaxis_title=sort_col.replace("_", " "), height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        if "Peer_Deviation_Pct" in display_summary:
            st.caption("Peer deviation is each inverter's mean signed percentage residual minus the plant-wide mean across inverters.")
            st.dataframe(
                display_summary[[c for c in ["SOURCE_KEY", "Mean_Deviation_Pct", "Peer_Deviation_Pct", "Anomalies", "Total_Observations", "Anomaly_Percentage"] if c in display_summary]],
                hide_index=True,
                width="stretch",
            )


# -----------------------------------------------------------------------------
# Anomaly detection
# -----------------------------------------------------------------------------
elif page == "Abnormal-performance screening":
    if report is None or scores is None:
        show_load_problem(
            ANOMALIES / f"plant{plant}_anomaly_scores.csv",
            ANOMALIES / f"plant{plant}_anomaly_report.csv",
        )
        st.stop()
    if "_load_error" in report or "_load_error" in scores:
        bad_file = report if "_load_error" in report else scores
        st.error("Could not read anomaly data: " + bad_file["_load_error"].iloc[0])
        st.stop()
    sources = sorted(scores["SOURCE_KEY"].dropna().unique()) if "SOURCE_KEY" in scores else []
    chosen = st.sidebar.multiselect("Inverters", sources, default=sources, key="anomaly_sources")
    score_view = scores[scores["SOURCE_KEY"].isin(chosen)].copy()
    score_view = filter_dates(score_view, "anomaly_dates")
    flagged_view = score_view[score_view["ANOMALY_FLAG"].astype(bool)]
    count = len(flagged_view)
    affected = flagged_view["SOURCE_KEY"].nunique() if "SOURCE_KEY" in flagged_view else 0
    avg_dev = pd.to_numeric(flagged_view.get("ABS_NORMALIZED_RESIDUAL", pd.Series(dtype=float)), errors="coerce").mean() * 100
    x, y, z = st.columns(3)
    x.metric("Flagged rows in view", f"{count:,}")
    y.metric("Inverters represented", f"{affected:,}")
    z.metric("Mean absolute deviation", f"{format_number(avg_dev)}%")
    if score_view.empty:
        st.info("No evaluated observations match these filters. Select at least one inverter or widen the date range.")
        st.stop()
    if count == 0:
        st.success("No observations crossed the calibrated thresholds for this selection.")
    if {"DATE_TIME", "ACTUAL_AC_POWER", "EXPECTED_AC_POWER"}.issubset(score_view.columns):
        st.subheader("Actual versus expected power with screening flags")
        selected = st.selectbox("Inspect inverter", sorted(score_view["SOURCE_KEY"].dropna().unique()), key="anomaly_chart_source")
        chart = score_view[score_view["SOURCE_KEY"] == selected].sort_values("DATE_TIME")
        fig = go.Figure()
        forecast_time = chart["DATE_TIME"] + pd.Timedelta(minutes=15)
        fig.add_trace(go.Scatter(x=forecast_time, y=chart["ACTUAL_AC_POWER"], name="Actual target", mode="lines+markers", line={"color": "#e76f51"}))
        fig.add_trace(go.Scatter(x=forecast_time, y=chart["EXPECTED_AC_POWER"], name="Expected target", mode="lines", line={"color": "#5687c8"}))
        marked = chart[chart["ANOMALY_FLAG"].astype(bool)]
        fig.add_trace(go.Scatter(x=marked["DATE_TIME"] + pd.Timedelta(minutes=15), y=marked["ACTUAL_AC_POWER"], name="Flagged", mode="markers", marker={"color": "#d62828", "size": 10, "symbol": "x"}))
        fig.update_layout(xaxis_title="Timestamp", yaxis_title="AC power", hovermode="x unified", height=400)
        st.plotly_chart(fig, width="stretch")
    st.subheader("Flagged screening observations")
    if count:
        st.dataframe(flagged_view.sort_values("DATE_TIME", ascending=False), hide_index=True, width="stretch")
    st.caption("Thresholds use each inverter’s first 60% of its chronological test rows for calibration; flags are measured on later active-power rows. These are statistical screening flags, not confirmed faults. The percentage is a flag rate, not detection accuracy; there are no ground-truth fault labels to calculate precision or recall.")


# -----------------------------------------------------------------------------
# Model comparison
# -----------------------------------------------------------------------------
elif page == "Model comparison":
    results = load_metrics("final_model_results.csv")
    if results is None:
        show_load_problem(METRICS / "final_model_results.csv")
        st.stop()
    if "_load_error" in results:
        st.error("Could not read model metrics: " + results["_load_error"].iloc[0])
        st.stop()
    results = results[results["Plant"].astype(str).str.lower() == f"plant {plant}".lower()].copy()
    if results.empty:
        st.info(f"No saved model comparison rows for Plant {plant}.")
        st.stop()
    for col in ["MAE", "RMSE", "R2"]:
        if col in results:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    available_metrics = [c for c in ["MAE", "RMSE", "R2"] if c in results and results[c].notna().any()]
    if not available_metrics:
        st.error("The saved results contain no numeric MAE, RMSE, or R² values.")
        st.stop()
    chosen_metric = st.selectbox("Compare by", available_metrics,
                                 format_func=lambda c: {"MAE": "MAE · lower is better", "RMSE": "RMSE · lower is better", "R2": "R² · higher is better"}[c])
    higher_better = chosen_metric == "R2"
    fig = px.bar(results.sort_values(chosen_metric, ascending=not higher_better), x="Model", y=chosen_metric,
                 color="Model", color_discrete_map={name: MODEL_COLORS.get(name, "#9aa0a6") for name in results["Model"].unique()},
                 title=f"Plant {plant} · {chosen_metric} by model")
    fig.update_layout(xaxis_title="Model", yaxis_title=chosen_metric, height=420, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(results, hide_index=True, width="stretch")
    best = results.loc[results[chosen_metric].idxmax() if higher_better else results[chosen_metric].idxmin()]
    st.info(f"Best recorded {chosen_metric}: **{best['Model']}** ({format_number(best[chosen_metric], 4)}). Scores are from the saved chronological test evaluation.")
