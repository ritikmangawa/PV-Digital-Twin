import streamlit as st
import pandas as pd
import os

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_data
def load_predictions(plant_id):
    """Loads prediction data for a specific plant."""
    file_path = f"outputs/predictions/plant{plant_id}_test_predictions.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        return df
    return None

@st.cache_data
def load_anomaly_summary(plant_id):
    """Loads the inverter-level anomaly summary."""
    file_path = f"outputs/anomalies/plant{plant_id}_inverter_anomaly_summary.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_anomaly_report(plant_id):
    """Loads the detailed anomaly report."""
    file_path = f"outputs/anomalies/plant{plant_id}_anomaly_report.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        return df
    return None

@st.cache_data
def load_model_metrics():
    """Loads the final model comparison metrics."""
    file_path = "outputs/metrics/final_model_results.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("☀️ Solar Intelligence")
st.sidebar.markdown("---")

# Navigation radio buttons
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", 
     "Power Prediction", 
     "Inverter Performance", 
     "Anomaly Detection", 
     "Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.info("Capstone Project: Solar Power Generation Prediction & Inverter Performance Intelligence System.")

# -----------------------------------------------------------------------------
# PAGE ROUTING (Placeholders for now)
# -----------------------------------------------------------------------------
if page == "Dashboard Overview":
    st.title("Dashboard Overview")
    st.write("Data loading successful. Page content will go here.")
    
elif page == "Power Prediction":
    st.title("Power Prediction")
    st.write("Page content will go here.")
    
elif page == "Inverter Performance":
    st.title("Inverter Performance")
    st.write("Page content will go here.")
    
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.write("Page content will go here.")
    
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("Page content will go here.")
