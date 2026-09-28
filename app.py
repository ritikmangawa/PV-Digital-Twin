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
st.sidebar.info("Solar Power Generation Prediction & Inverter Performance Intelligence System.")

# -----------------------------------------------------------------------------
# PAGE ROUTING (Placeholders for now)
# -----------------------------------------------------------------------------
if page == "Dashboard Overview":
    st.title("Dashboard Overview")
    
    st.markdown("### Select Plant")
    plant_choice = st.radio("Plant ID", [1, 2], horizontal=True)
    
    predictions_df = load_predictions(plant_choice)
    anomaly_summary_df = load_anomaly_summary(plant_choice)
    
    if predictions_df is not None and anomaly_summary_df is not None:
        # Calculate metrics
        total_obs = anomaly_summary_df["Total_Observations"].sum()
        num_inverters = anomaly_summary_df["SOURCE_KEY"].nunique()
        avg_ac_power = predictions_df["AC_POWER"].mean()
        avg_anomaly_pct = anomaly_summary_df["Anomaly_Percentage"].mean()
        
        # Determine available models
        model_cols = [col for col in predictions_df.columns if col not in ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'AC_POWER', 'TARGET_AC_POWER']]
        
        # Display KPIs
        st.markdown("### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Total Observations", value=f"{total_obs:,}")
        with col2:
            st.metric(label="Inverters Monitored", value=num_inverters)
        with col3:
            st.metric(label="Average AC Power", value=f"{avg_ac_power:.2f} kW")
        with col4:
            st.metric(label="Avg Anomaly %", value=f"{avg_anomaly_pct:.2f}%")
            
        st.markdown("---")
        st.markdown("### Models Evaluated")
        st.write(", ".join(model_cols).replace("_", " "))
        
        st.info("💡 **Tip:** Use the sidebar to navigate to deeper analysis pages for Power Prediction, Inverter Performance, and Anomaly Detection.")
    else:
        st.warning("Data files not found. Please ensure the output CSVs exist in the correct directories.")
    
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
