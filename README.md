# ☀️ PV Digital Twin — Solar Power Forecasting & Inverter Performance Intelligence

### Data-Driven Photovoltaic Power Forecasting and Inverter Abnormal-Performance Screening

A classical Machine Learning based **Photovoltaic (PV) Digital Twin** that combines solar power forecasting, expected-vs-actual power analysis, inverter-level performance monitoring, residual analysis, and abnormal-performance detection into an interactive Streamlit dashboard.

---

## 📌 Overview

Solar photovoltaic plants generate large amounts of operational data from individual inverters and environmental sensors. Simply predicting solar power is not enough for effective plant monitoring.

This project builds a **data-driven monitoring prototype** that forecasts expected AC power from timestamp and historical-power features, then compares forecasts with observed generation. Weather data is included in preprocessing and EDA, but is not currently fed into the forecasting models.

The system follows the pipeline:

```text
PV Generation Data + Weather Data
              ↓
       Data Preprocessing
              ↓
              EDA
              ↓
      Feature Engineering
              ↓
      Time-Series Data Split
              ↓
       Forecasting Models
              ↓
      Expected AC Power
              ↓
     Actual vs Expected Power
              ↓
      Residual / Deviation
              ↓
   Inverter-Level Monitoring
              ↓
 Abnormal-Performance Detection
              ↓
       Streamlit Dashboard
```

The project focuses on **classical Machine Learning**, using Persistence, Linear Regression, Random Forest, and XGBoost rather than deep learning approaches.

---

# 🎯 Project Objectives

The major objectives of this project are:

1. Predict short-term photovoltaic AC power.
2. Establish a simple Persistence forecasting baseline.
3. Compare classical ML models for solar power forecasting.
4. Evaluate models using MAE, RMSE, and R².
5. Generate expected AC power for the plant.
6. Compare actual power against expected power.
7. Analyze prediction residuals and deviations.
8. Monitor individual inverter/source performance.
9. Detect observations showing abnormal performance.
10. Provide an interactive dashboard for plant and inverter monitoring.

---

# 🏭 Dataset

The project uses photovoltaic generation and weather sensor data from two solar plants.

Each plant contains:

- Generation data
- Weather sensor data

The generation data contains information such as:

```text
DATE_TIME
PLANT_ID
SOURCE_KEY
DC_POWER
AC_POWER
DAILY_YIELD
TOTAL_YIELD
```

The weather data contains:

```text
DATE_TIME
PLANT_ID
SOURCE_KEY
AMBIENT_TEMPERATURE
MODULE_TEMPERATURE
IRRADIATION
```

`SOURCE_KEY` identifies an individual inverter/source, allowing the project to perform **inverter-level performance analysis** rather than only plant-level analysis.

---

# 🔑 Prediction Target and Model Inputs

The raw power measurement is `AC_POWER`. Feature engineering creates `TARGET_AC_POWER` from the same inverter’s observation exactly 15 minutes after the feature timestamp. Rows without an exact +15-minute observation are excluded from model training and evaluation:

```text
TARGET_AC_POWER = same inverter AC_POWER at DATE_TIME + 15 minutes
```

AC power represents power delivered on the AC side of the photovoltaic system. `TARGET_AC_POWER` is the next observed AC power value, while features are read from the current timestamp.

Conceptually:

```text
Time and cyclic features
Current and lagged AC power
Rolling AC power features
        ↓
   ML Model
        ↓
Predicted AC Power
```

The weather observations are merged and explored in the analysis, but **the current forecasting models do not use weather columns as predictors**. The trained feature list is time features plus historical/rolling AC power. This distinction matters when describing the current model.

---

# 🧹 Data Preprocessing

The preprocessing stage includes:

- Loading plant generation datasets
- Loading weather sensor datasets
- Converting timestamps to datetime
- Sorting observations chronologically
- Checking data types
- Checking missing values
- Checking duplicate records
- Aligning generation and weather observations
- Merging relevant plant-level information
- Preparing data for time-series modeling

The original raw datasets are kept separate from processed datasets.

---

# 📊 Exploratory Data Analysis

EDA was performed to understand the behavior of the photovoltaic system.

Major analyses include:

### Power Analysis

- AC power distribution
- DC power distribution
- Daily generation behavior
- Power variation over time

### Weather Analysis

- Irradiation
- Ambient temperature
- Module temperature

### Relationship Analysis

- Irradiation vs AC power
- Temperature vs power
- Daily generation patterns
- Inverter-level behavior

### Time-Series Analysis

The project examines the strong daily cycle of photovoltaic generation:

```text
Night
  ↓
Zero / near-zero generation
  ↓
Morning increase
  ↓
Peak generation
  ↓
Evening decrease
  ↓
Night
```

This temporal structure is important for short-term forecasting.

---

# ⚙️ Feature Engineering

Feature engineering was designed specifically for the time-series nature of solar power generation.

## Time Features

Examples include:

```text
Hour
Day
Day of Week
Weekend Indicator
```

## Cyclic Features

To represent the cyclical nature of time:

```text
hour_sin
hour_cos
```

This avoids treating hour values such as 23 and 0 as completely distant values.

## Historical Power Features

Lag features were created to capture recent and historical generation behavior:

```text
AC_POWER_lag_1
AC_POWER_lag_4
AC_POWER_lag_8
AC_POWER_lag_96
```

At 15-minute resolution:

```text
lag_1  → 15 minutes
lag_4  → 1 hour
lag_8  → 2 hours
lag_96 → 24 hours
```

These features allow the model to learn temporal dependencies from historical power generation.

## Rolling Features

Rolling statistics were also created to represent recent power behavior, such as:

```text
AC_POWER_rolling_mean_4
AC_POWER_rolling_mean_8
```

All time-dependent features were created with attention to chronological ordering and leakage prevention.

---

# ⏱️ Time-Series Train/Test Strategy

Because solar power is a time-dependent forecasting problem, the data was split chronologically rather than randomly. Rows are excluded from training or validation when their +15-minute target would cross the next split boundary.

Conceptually:

```text
Past Data
    ↓
Training Set
    ↓
Future Data
    ↓
Testing Set
```

Random train-test splitting is avoided because it can allow information from future observations to influence training.

This provides a more realistic evaluation of how the model would behave when forecasting future solar generation. The project specification also emphasizes chronological or rolling-window evaluation for time-series observations.

---

# 🤖 Machine Learning Models

Four forecasting approaches were evaluated.

## 1. Persistence Baseline

The Persistence model predicts the next power value using the most recent observed value:

```text
Predicted Power(t+1) = Power(t)
```

This is an important benchmark because solar generation has strong temporal continuity.

A complex ML model should be evaluated against this simple baseline rather than being assumed to be better simply because it is more sophisticated.

---

## 2. Linear Regression

Linear Regression was used as an interpretable ML baseline.

It models the relationship between input features and AC power using a linear combination of the features.

Advantages:

- Simple
- Interpretable
- Fast to train
- Useful as a baseline

---

## 3. Random Forest

Random Forest was used to capture nonlinear relationships between:

- Historical power
- Temporal features

It combines multiple decision trees and averages their predictions.

---

## 4. XGBoost

XGBoost was evaluated as a powerful gradient-boosting model for nonlinear regression.

It can capture complex interactions between historical-power and temporal features.

The project keeps XGBoost within the classical ML pipeline and does not rely on deep learning.

---

# 📈 Model Evaluation

The models were evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted power.

```text
MAE = average(|Actual - Predicted|)
```

Lower MAE is better.

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly.

```text
RMSE = sqrt(mean((Actual - Predicted)²))
```

Lower RMSE is better.

### R² — Coefficient of Determination

Measures how much variation in the target is explained by the model.

Higher R² generally indicates better explanatory performance.

---

# 📊 Final Model Comparison

The final chronological test-set comparison obtained from the project is:

| Plant | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Plant 1 | Persistence | 60.15 | 113.23 | 0.9113 |
| Plant 1 | Linear Regression | 85.73 | 131.52 | 0.8803 |
| Plant 1 | Random Forest | 74.73 | 141.51 | 0.8614 |
| Plant 1 | XGBoost | 74.51 | 144.27 | 0.8560 |
| Plant 2 | Persistence | 64.05 | 132.34 | 0.8029 |
| Plant 2 | Linear Regression | 86.63 | 149.63 | 0.7480 |
| Plant 2 | Random Forest | 74.00 | 149.39 | 0.7488 |
| Plant 2 | XGBoost | 72.09 | 148.23 | 0.7527 |

### Important observation

In the latest rerun after enforcing the exact 15-minute target and split-boundary exclusions, Persistence produced the lowest MAE for both plants. None of the tested ML models beat Persistence on MAE in this evaluation.

This is an important result rather than something to hide.

It demonstrates why a forecasting project should always compare sophisticated models against a strong baseline.

The project therefore does **not** claim that XGBoost or Random Forest automatically outperformed Persistence.

---

# 🔬 Digital Twin Concept

The central idea of the project is the comparison between:

```text
Expected AC Power
        vs
Actual AC Power
```

The ML forecasting system provides an estimate of expected plant behavior.

Then:

```text
Residual = Actual Power - Expected Power
```

For example:

```text
Expected Power = 1000 kW
Actual Power   = 850 kW

Residual = 850 - 1000
         = -150 kW
```

A persistent negative residual can indicate abnormal performance, although it does not automatically prove that a physical hardware fault exists.

This expected-vs-actual layer forms the foundation of the digital-twin monitoring system.

---

# 🚨 Inverter Abnormal-Performance Detection

The project performs monitoring at the individual inverter/source level using `SOURCE_KEY`.

For each inverter, the system analyzes:

```text
Actual Power
Expected Power
Residual
Deviation %
Normalized Residual
Absolute Normalized Residual
Peer-Inverter Deviation
Observation Count
```

This allows the system to identify inverters whose observed behavior differs from expected behavior.

---

# ⚠️ Why This Is Not Called Fault Classification

The dataset does not contain ground-truth labels such as:

```text
Hardware Fault
Sensor Fault
Shading
Inverter Failure
```

Therefore, the project does not train a supervised fault classifier or claim that an anomaly definitely represents a specific physical failure.

Instead, the system performs:

### Inverter Abnormal-Performance Detection

This distinction is important because an anomaly may have multiple possible causes.

Possible causes could include:

- Environmental changes
- Sensor issues
- Temporary operating conditions
- Inverter behavior
- Data quality problems
- Other operational factors

The project identifies **unusual behavior**, which can then be investigated further.

---

# 📊 Inverter-Level Analysis

For each inverter, summary statistics are generated, including:

```text
Mean Actual Power
Mean Expected Power
Mean Residual
Mean Deviation %
Median Absolute Normalized Residual
P90 Absolute Normalized Residual
P95 Absolute Normalized Residual
Observation Count
Peer Deviation %
```

This allows the dashboard to move from:

```text
Plant-level monitoring
```

to:

```text
Individual inverter monitoring
```

---

# 🔍 Anomaly Detection Results

Thresholds are calibrated on the first 60% of each inverter's chronological test rows and applied to active-power rows in the later 40%. Current screening results are:

Plant 1:

```text
Evaluated active observations = 2,022
Inverters                      = 22
```

The inverter-level screening counts sum to:

```text
113 flagged observations
```

which corresponds to approximately:

```text
5.59% of evaluated active observations
```

Plant 2:

```text
Evaluated active observations = 1,998
Anomalies                     = 78
Anomaly percentage            ≈ 3.90%
```

These values are screening flag rates, not fault-detection accuracy. Because the dataset has no confirmed fault labels, precision and recall cannot be measured from this project, and a flagged observation does not establish an equipment failure.

---

# 📉 Actual vs Expected Power Visualization

The project visualizes:

```text
Actual AC Power
Expected AC Power
Anomaly Points
```

on the same time-series plot.

This makes it possible to visually inspect whether flagged observations correspond to deviations between actual and expected behavior.

Example interpretation:

```text
Actual ≈ Expected
        ↓
Normal expected behavior

Actual << Expected
        ↓
Potential abnormal performance

Repeated deviation
        ↓
Requires further investigation
```

---

# 🖥️ Streamlit Digital Twin Dashboard

The final system is deployed as an interactive Streamlit application.

### Live Application

[Open the PV Digital Twin Dashboard](https://pv-digital-twin-fefzanku23cmjalelgubiu.streamlit.app/?utm_source=chatgpt.com)

The dashboard provides an interactive interface for exploring:

- Solar power behavior
- Expected vs actual power
- Model predictions
- Model metrics
- Inverter performance
- Residual/deviation behavior
- Anomaly information
- Plant-level and inverter-level analysis

---

# 🧩 Application Architecture

The final project follows:

```text
                   ┌──────────────────────┐
                   │   PV Generation Data  │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │    Weather Data      │
                   └──────────┬───────────┘
                              │
                              ▼
                    Data Preprocessing
                              │
                              ▼
                     Feature Engineering
                              │
                              ▼
                 Chronological Train/Test
                              │
                              ▼
                ┌────────────────────────┐
                │     ML Forecasting     │
                │                        │
                │ Persistence            │
                │ Linear Regression      │
                │ Random Forest          │
                │ XGBoost                │
                └───────────┬────────────┘
                            │
                            ▼
                    Expected AC Power
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Actual vs Expected       Residual
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 Inverter Monitoring
                            │
                            ▼
             Abnormal-Performance Detection
                            │
                            ▼
                  Streamlit Dashboard
```

---

# 📁 Project Structure

```text
PV-Digital-Twin/
├── app.py
├── requirements.txt
├── README.md
├── data/                         # local raw and processed data (not needed by dashboard)
├── models/                       # large local trained model artifacts
├── notebooks/                    # analysis and model workflow (01–10)
├── src/
│   └── anomaly_detection.py      # calibrated inverter screening and export pipeline
└── outputs/                      # dashboard data, included in the repository
    ├── anomalies/
    ├── metrics/
    ├── predictions/
    └── tables/
```

The dashboard reads generated CSV files from `outputs/`, which are included so deployments have the required data. Raw datasets and large serialized models remain local-only and are ignored by Git. Recreate the anomaly outputs from saved test predictions with `python src/anomaly_detection.py`.

---

# 📓 Notebook Workflow

## 01 — Data Inspection

Understand the original datasets:

- Shape
- Columns
- Data types
- Missing values
- Duplicates
- Unique plants
- Unique inverters
- Date range
- Basic statistics

---

## 02 — Data Preprocessing

Clean and prepare the datasets for further analysis.

---

## 03 — EDA

Explore:

- Solar generation
- Irradiation
- Temperature
- Time patterns
- Inverter behavior
- Relationships between variables

---

## 04 — Data Merging & Synchronization

Merge generation and weather information using the relevant temporal and plant/source identifiers.

---

## 05 — Feature Engineering

Create:

- Time features
- Cyclic features
- Lag features
- Rolling features
- Forecasting target

---

## 06 — Forecasting Baseline

Implement Persistence forecasting and calculate:

- MAE
- RMSE
- R²

---

## 07 — Model Training & Comparison

Train and compare:

- Linear Regression
- Random Forest
- XGBoost

against the Persistence baseline.

---

## 08 — Model Evaluation

Analyze:

- Actual vs predicted power
- Prediction errors
- Residuals
- MAE
- RMSE
- R²
- Time-dependent performance
- Inverter-level performance

---

## 09 — Digital Twin

Generate:

```text
Expected Power
Actual Power
Residual
Deviation %
```

This establishes the expected-vs-observed monitoring layer.

---

## 10 — Anomaly Detection

Perform inverter-level abnormal-performance analysis using:

- Residuals
- Normalized residuals
- Deviation
- Persistence
- Peer-inverter behavior

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Streamlit visualizations

### Machine Learning

- Scikit-learn
- XGBoost

### Development

- Jupyter Notebook
- Git
- GitHub

### Deployment

- Streamlit Community Cloud

---

# 📦 Installation

From the project root, create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

# 📊 Key Results

The project demonstrates that:

- Solar AC power has strong temporal behavior that can be exploited for short-term forecasting.
- A Persistence baseline can be surprisingly strong for short-horizon solar forecasting.
- The current classical models use time and historical-power features; weather columns are available for analysis but are not in the trained feature set.
- Expected-vs-actual comparison provides a useful basis for operational monitoring.
- Inverter-level analysis provides more granular information than plant-level averages.
- Residual and normalized-residual analysis can identify observations requiring further investigation.
- Anomaly detection should be interpreted as abnormal-performance detection when ground-truth fault labels are unavailable.

---

# ⚠️ Limitations

### 1. No Ground-Truth Fault Labels

The dataset does not contain confirmed fault labels.

Therefore, the system detects **abnormal performance**, not confirmed physical faults.

### 2. Strong Persistence Baseline

Persistence performed strongly in the current evaluation.

This means more complex models did not automatically provide better forecasting performance.

### 3. Dataset Time Period

The model is trained and evaluated on the available historical dataset. Performance may change under different weather conditions, seasons, plants, or operating environments.

### 4. Anomaly Interpretation

An anomaly is an indication that observed behavior differs from expected behavior.

It does not directly identify the physical cause.

### 5. External Validation

The anomaly-detection system would benefit from operational maintenance records or labeled fault events for future validation.

---

# 🚀 Future Improvements

Possible future improvements include:

- Multi-horizon forecasting
- 1-hour, 4-hour and 24-hour forecasting
- More robust rolling-window validation
- Hyperparameter optimization
- Improved anomaly threshold calibration
- Maintenance-event integration
- Confirmed fault labels
- Energy-loss estimation
- Interactive scenario simulation
- Inverter degradation modeling
- Weather forecast integration
- Automated alerts
- Model retraining pipeline
- Production monitoring

The current implementation focuses on a one-step (15-minute) forecasting horizon. Scenario simulation and multi-horizon forecasting are not implemented.

---

# 🎓 Interview / Viva Highlights

This project demonstrates practical knowledge of:

### Machine Learning

- Regression
- Linear Regression
- Random Forest
- XGBoost
- Baseline modeling
- Model evaluation

### Time-Series ML

- Chronological splitting
- Lag features
- Rolling features
- Temporal features
- Leakage prevention
- Persistence forecasting

### Model Evaluation

- MAE
- RMSE
- R²
- Actual vs predicted analysis
- Residual analysis

### Anomaly Detection

- Residual-based monitoring
- Normalized residuals
- Deviation analysis
- Peer-inverter comparison
- Unsupervised/label-free abnormal-performance detection

### ML Engineering

- Reproducible project structure
- Saved model outputs
- CSV-based prediction pipeline
- Streamlit deployment

---

# 💡 Why This Project Is Different From a Basic Solar Prediction Project

A typical solar ML project ends at:

```text
Weather Data
     ↓
ML Model
     ↓
Predicted Solar Power
```

This project extends the pipeline:

```text
Weather + Historical Generation
             ↓
       ML Forecasting
             ↓
       Expected AC Power
             ↓
    Actual vs Expected
             ↓
       Residual Analysis
             ↓
   Inverter-Level Monitoring
             ↓
Abnormal-Performance Detection
             ↓
      Digital Twin Dashboard
```

Therefore, the project combines **forecasting and operational intelligence** rather than treating solar prediction as an isolated regression problem.

---

# 👨‍💻 Author

**Ritik Mangawa**

B.Tech — Computer Science Engineering

VIT Bhopal University

---

# ⭐ Project Summary

> **PV Digital Twin is a classical Machine Learning based photovoltaic monitoring system that forecasts expected AC power, compares it with actual generation, analyzes inverter-level deviations, and identifies abnormal performance through residual-based intelligence. The complete system is exposed through an interactive Streamlit dashboard.**

---

## 🔗 Project Links

The Streamlit app entry point is `app.py`. For hosted deployment, set the repository's main file to `app.py` and ensure the contents of `outputs/` are committed.

---

## 📜 Disclaimer

This project is developed for educational, research, and demonstration purposes.

Anomaly indicators represent **statistical abnormal-performance signals** and should not be interpreted as confirmed physical equipment faults without additional operational or maintenance data.
