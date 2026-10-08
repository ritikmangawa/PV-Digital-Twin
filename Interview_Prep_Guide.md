# PV Digital Twin — Interview Study Workbook

This guide is built around the current repository and saved results. Learn the ideas and sequence first; memorize the short answers only after you can explain them in your own words. Your strongest interview quality is honest reasoning, not pretending the project has no limitations.

## 1. The project in one sentence

I built a historical-data prototype for two photovoltaic plants that forecasts inverter AC power 15 minutes ahead, compares simple and machine-learning models, screens for unusual inverter behavior using forecast residuals, and presents the results in a Streamlit dashboard.

Call it a **data-driven PV monitoring prototype** or **digital-twin-style prototype**. It is not a live operational twin: it does not ingest live plant telemetry or simulate plant physics.

## 2. 60-second introduction (practice aloud)

“My first end-to-end machine-learning project is a PV plant monitoring prototype using historical generation and weather records from two plants. I cleaned and aligned the timestamped data, engineered time features and historical AC-power features, and created a target from the same inverter’s reading exactly 15 minutes later. I used chronological train, validation, and test periods, excluding examples whose target crosses a split boundary. I compared Persistence, Linear Regression, Random Forest, and XGBoost. Persistence had the lowest test MAE on both plants—60.15 on Plant 1 and 64.05 on Plant 2—so the tested ML models did not improve on that baseline. I then used expected-versus-actual residuals to produce per-inverter statistical abnormal-performance flags, and built a Streamlit dashboard to inspect predictions, scores, and flags. The flags are for investigation, not confirmed faults, because there are no fault labels. One limitation I would address next is making every lag timestamp-aware when observations are missing.”

If asked what you personally did, say what you actually implemented and can explain. Don’t claim a team contribution or production deployment unless it is true.

## 3. Dataset: what it contains

The repository has generation and weather CSV data for **Plant 1 and Plant 2**. Generation records are inverter-level: `SOURCE_KEY` identifies the inverter, and each plant has 22 inverter keys. Weather is plant/sensor-level, with fields including ambient temperature, module temperature, and irradiation. The data spans roughly May 15 through June 17, 2020, at a nominal 15-minute cadence; the timestamps are not perfectly regular.

Common generation fields:

- `DATE_TIME`: observation timestamp.
- `PLANT_ID`: plant identifier.
- `SOURCE_KEY`: inverter/source identifier.
- `DC_POWER`: measured DC-side power.
- `AC_POWER`: measured AC-side power; this is the forecast quantity.
- `DAILY_YIELD`, `TOTAL_YIELD`: energy-yield counters, distinct from instantaneous power.

Common weather fields:

- `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`: temperature readings.
- `IRRADIATION`: measured solar irradiation.
- Weather `SOURCE_KEY`: sensor/source identifier.

**Important distinction:** weather data is loaded, aligned/merged for analysis, and explored in EDA. The current forecasting models do **not** use weather as a predictor. Avoid saying that weather improves the forecasts. Confirm power units from the original dataset metadata before naming a unit in an interview; the dashboard/results may leave units unspecified.

**Why have multiple tables?** Generation is per inverter while weather is a plant-level environmental signal. Timestamps and plant identifiers are used to align them. A join must avoid multiplying inverter rows if the weather table has duplicate keys for the join columns.

**What does a row represent?** One inverter’s measured generation at a timestamp, not necessarily one whole plant’s total output. The plant contains many inverters.

## 4. Problem definition and target

At a feature timestamp `t`, the model receives information available at `t` and predicts the same inverter’s `AC_POWER` at `t + 15 minutes`:

```text
TARGET_AC_POWER(t) = AC_POWER for the same SOURCE_KEY at timestamp t + 15 minutes
```

The engineered target accepts an observation only when its timestamp is exactly 15 minutes later. It does not blindly call the next available row a 15-minute target. Rows without that exact future observation cannot be used as labeled examples.

The baseline called **Persistence** predicts that power stays at its current value:

```text
Persistence prediction at t+15 = observed AC_POWER at t
```

This makes the problem a one-step-ahead regression task. It does not mean the model forecasts an entire day or predicts future weather.

## 5. End-to-end workflow and architecture

Know this sequence. The notebooks implement the research workflow; the dashboard reads saved artifacts rather than retraining models on every page load.

```text
Raw generation + weather CSVs
        ↓
Inspect, parse timestamps, sort, check data quality
        ↓
Explore plant, weather, time, and inverter behavior
        ↓
Align/merge weather for analysis; engineer target and features
        ↓
Chronological train / validation / test periods
        ↓
Persistence baseline + Linear Regression + Random Forest + XGBoost
        ↓
Test predictions and metrics saved to outputs
        ↓
Residual-based inverter screening results saved to outputs
        ↓
Streamlit + Plotly dashboard reads and displays those results
```

Repository map (confirm exact names in the current checkout if they change):

- `data/`: raw and processed input data.
- `notebooks/01...10...`: data inspection through modeling and abnormal-performance screening.
- `src/`: reusable project code where present.
- `models/`: saved fitted model artifacts where present.
- `outputs/metrics/`: model score tables.
- `outputs/predictions/` and related output folders: test predictions, inverter summaries, and screening results.
- `app.py` / `app/`: Streamlit dashboard code.
- `requirements.txt`: Python package dependencies.
- `README.md`: project overview, methods, results, and limitations.

A useful notebook-by-notebook explanation is: inspect raw inputs; clean and prepare; explore patterns; align generation/weather; engineer target and predictors; establish persistence; train candidate models; evaluate only on later test data; create the expected-versus-actual monitoring view; calculate and export screening flags. If asked for a specific notebook’s exact operations, open it and describe the code rather than guessing from its title.

## 6. Features and the caveat you must know

Current engineered predictors include calendar/time information (including cyclical hour encodings), current/historical AC power, lagged power, and rolling means. For 15-minute regularly spaced data, row lags 1, 4, 8, and 96 would nominally mean 15 minutes, 1 hour, 2 hours, and 24 hours. Rolling windows summarize recent rows; they are shifted so the current target is not included in its own historical summary.

Cyclical encoding represents hour as a circle:

```text
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

That lets 23:00 and 00:00 have similar values instead of treating them as far apart. Calendar features represent predictable daily/seasonal structure.

**Remaining important limitation:** lag and rolling features are row-based. When a timestamp is missing, the previous row may be farther back than 15 minutes, so `lag_1` is not guaranteed to mean exactly 15 minutes and a multi-row window may span more clock time than intended. The target and split-boundary corrections do not fix this feature-timing issue. Be transparent: “The target is exact +15 minutes, but some historical features still refer to the previous available row across gaps. I would build those features with timestamp joins or reindex to a regular 15-minute grid, mask missing history, then rerun the full evaluation.”

Never describe lag 96 as definitely “yesterday at the same time” for every row under irregular sampling. Say “96 prior rows, nominally 24 hours on a regular grid.”

## 7. Evaluation design and results

The data is split in chronological order, approximately 70% training, 15% validation, 15% test. Shuffling would mix later conditions into training and make the result less like forecasting the future. Examples whose 15-minute target crosses a train/validation or validation/test cutoff are removed at that boundary. The held-out final test period is used for the reported comparison.

Metrics:

- **MAE** = mean of `|actual − predicted|`. Easy to interpret as the average absolute miss in the target’s units. Lower is better.
- **RMSE** = square root of the mean squared error. Large misses count more heavily. Lower is better.
- **R²** = relative fit compared with predicting the test-set mean. It can be negative; higher is generally better, but it does not tell you the typical error in the target’s units.

Saved final test results (rounded):

| Plant | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| 1 | Persistence | 60.15 | 113.23 | 0.9113 |
| 1 | Linear Regression | 85.73 | 131.52 | 0.8803 |
| 1 | Random Forest | 74.73 | 141.51 | 0.8614 |
| 1 | XGBoost | 74.51 | 144.27 | 0.8560 |
| 2 | Persistence | 64.05 | 132.34 | 0.8029 |
| 2 | Linear Regression | 86.63 | 149.63 | 0.7480 |
| 2 | Random Forest | 74.00 | 149.39 | 0.7488 |
| 2 | XGBoost | 72.09 | 148.23 | 0.7527 |

**Interpretation:** Persistence had the best MAE, RMSE, and R² among the listed models on both plant test sets. The ML models did not beat it in this evaluation. This is an outcome, not proof that ML is useless or that persistence always wins. Likely explanations to investigate include short forecast horizon, smooth power persistence, lack of future weather forecast inputs, feature timing gaps, and test-period characteristics. Don’t say any one explanation is proven by these results.

Do not compare these metric values with another project unless the target units, plant, time period, test rows, and metric calculation are the same. The CSV values are more reliable than memory; re-open `outputs/metrics/final_model_results.csv` before quoting if the project is rerun.

## 8. Abnormal-performance screening

The screening layer uses the Persistence expected value and the observed next-step value. In plain language, it asks: “Was the observed power much different from the simple expected power for this inverter?”

```text
residual = actual target AC power − expected (Persistence) AC power
normalized residual = residual / expected power
screening score = absolute normalized residual
```

To avoid unstable ratios around zero/nighttime production, the calculation is limited to observations where expected power is at least 50 (in the data’s supplied power units). For each inverter, the P95 (95th percentile) of absolute normalized residuals is calculated from the first 60% of that inverter’s chronological test rows. That inverter-specific threshold is applied to its later 40%. A score above threshold is flagged.

Latest saved rates: Plant 1: **113 / 2,022 = 5.59%**; Plant 2: **78 / 1,998 = 3.90%** of evaluated active observations. These are screening rates, not fault prevalence or model accuracy. The denominator is eligible/evaluated observations, not all raw rows.

There are no confirmed fault labels. Therefore precision, recall, F1, and “fault detection accuracy” cannot be computed from this data. A flag means unusual deviation under this statistical rule; an operator would need to inspect it. It does not diagnose equipment failure, prove underperformance, or identify root cause.

## 9. Models: explain them at interview depth

- **Persistence:** a baseline, not a trained ML model. Copies the most recent measured power to the next step. Fast and surprisingly strong at a short horizon.
- **Linear Regression:** estimates a linear weighted relationship between predictors and target. Interpretable baseline; cannot automatically represent every nonlinear relationship.
- **Random Forest:** averages many decision trees, allowing nonlinear rules and feature interactions. It can be less interpretable and can be expensive for large data.
- **XGBoost:** builds trees sequentially, with each new tree correcting prior errors. Strong on many tabular tasks, but sophistication does not guarantee better generalization.

Hyperparameters in the repository are configured in the modeling notebook. If asked for exact settings, check the notebook/model code before the interview rather than memorizing an unverified value. A strong answer is to explain the algorithm and compare fairly on the same chronological test set.

## 10. Dashboard and deployment explanation

The Streamlit app is a presentation layer for saved historical outputs. It gives the user plant/model summaries and views for power predictions, inverter performance, abnormal-performance screening, and model comparison. Plotly charts make the time series and comparisons interactive.

Be precise: the deployed app is not a live SCADA connection, does not collect real-time telemetry, and does not retrain a model for each visitor. It visualizes the project’s saved results. If asked how you’d productionize it: add a validated data-ingestion service, timestamp/quality checks, scheduled inference, monitoring and alert delivery, access controls, model/version tracking, and operational review with plant engineers. Don’t claim those components exist today.

## 11. Resume wording

**PV Plant Forecasting & Inverter Monitoring Prototype | Python, pandas, scikit-learn, XGBoost, Streamlit, Plotly**

- Built a historical-data workflow for two PV plants, engineering exact 15-minute-ahead same-inverter AC-power targets and evaluating models with chronological splits and boundary filtering.
- Compared Persistence, Linear Regression, Random Forest, and XGBoost on held-out periods; Persistence achieved the lowest MAE on both plants (60.15 and 64.05 in dataset units).
- Developed per-inverter residual-based abnormal-performance screening and an interactive Streamlit dashboard; documented that flags are unvalidated screening signals because confirmed fault labels were unavailable.

Only use this if you can explain each bullet. “Developed” can mean you built it yourself; if not, adjust authorship. Avoid claiming an accuracy percentage, live prediction, fault diagnosis, or that ML improved the baseline.

## 12. Interview questions and sample answers

### Project and motivation

**Q: Walk me through your project.**
A: “I start with historical generation and weather records for two plants. After inspecting and aligning timestamps, I engineer a same-inverter +15-minute AC-power target and historical features. I split chronologically, compare persistence with three regression models, and evaluate on the later test period. Then I use per-inverter deviations from persistence to create statistical screening flags and show the saved outputs in Streamlit.”

**Cross-question: Is it really a digital twin?**
A: “It is a digital-twin-style monitoring prototype, not a complete live twin. It represents historical plant behavior and compares expected with observed power, but it has no live telemetry loop or physics-based simulation.”

**Q: What was your role?**
A: State accurately which stages you personally completed. Then explain one difficult decision, one result, and one limitation. Do not claim to have authored code you cannot walk through.

### Data and preparation

**Q: What is an inverter and why analyze it separately?**
A: “An inverter converts the PV array’s DC electricity to AC electricity. Since each `SOURCE_KEY` identifies an inverter, grouping by it lets us inspect unit-level behavior that a plant-wide total could hide.”

**Cross-question: Why not just aggregate plant power?**
A: “Plant totals are useful for forecasting overall output, but they can conceal one inverter deviating while others perform normally. This project keeps inverter identity for unit-level analysis.”

**Q: How did you align weather and generation?**
A: “I parsed timestamps and plant identifiers, checked the timestamp keys, then joined matching observations for analysis. I checked the join keys because duplicate weather timestamps could multiply generation rows. Weather is not currently a model input.”

**Q: What did you do with missing values?**
A: “I inspected missingness and removed rows that could not provide the required target or model features. A missing future +15-minute target is not fabricated. I would describe exact imputation only after checking the preprocessing notebook; the important modeling choice is not to invent target labels.”

**Cross-question: Doesn’t dropping rows bias the data?**
A: “It can. If gaps are systematic, the retained sample may differ from the full operating period. I’d report data coverage by plant, time, and inverter, and test a regular-grid approach with explicit missingness indicators.”

**Q: Why use `AC_POWER`, not `DC_POWER` or yield?**
A: “The goal is to forecast delivered AC-side power, so AC power matches the target. DC power could be a useful predictor if available at prediction time. Yield fields are cumulative/energy counters and represent a different quantity.”

### Target, leakage, and time series

**Q: How do you know the target is 15 minutes ahead?**
A: “The target is matched to the same inverter at exactly timestamp plus 15 minutes. Rows without that exact timestamp are excluded. The target is not simply the next available row.”

**Cross-question: What if rows are missing?**
A: “The target still requires exact +15 minutes, so the label horizon remains correct. However, row-based lag features can span longer than their nominal interval. That limitation remains and I would make lags timestamp-aware, then rerun metrics.”

**Q: Why chronological split?**
A: “The intended use predicts future from past. A random split can put later conditions in training and earlier conditions in testing, making evaluation optimistic. Chronological splitting better mimics deployment.”

**Cross-question: What is boundary leakage?**
A: “A feature row immediately before a cutoff might have a target timestamp inside the next period. I exclude those crossing examples from the earlier split so their labels don’t reach across the boundary.”

**Cross-question: Can using lagged values from before the test cutoff be leakage?**
A: “No, if those values were genuinely observed by the forecast time; historical context is allowed. Leakage would be using information unavailable at that time, such as a future target or future-derived statistic.”

**Q: Why not random train/test split?**
A: “It breaks the order needed to simulate forecasting and can mix temporal regimes. For a more robust estimate, I’d also use rolling-origin/backtesting across several future windows.”

### Features and models

**Q: Why encode hour with sine and cosine?**
A: “Hour is cyclic. Encoding 23 and 0 as ordinary integers makes them appear far apart; sine/cosine places them close on a circle.”

**Q: What does lag 4 represent?**
A: “Four prior rows; on a complete 15-minute grid that is one hour. Because timestamps can be missing, I avoid claiming that it is always exactly one clock hour in this version.”

**Q: Why did you include a persistence baseline?**
A: “It sets a simple benchmark. If complex models can’t beat copying current power for the next short interval, that is valuable evidence about the data, horizon, and features.”

**Q: Why did ML not beat persistence?**
A: “That’s what this test comparison showed, not a universal conclusion. The 15-minute horizon may be highly persistent; the models lacked future weather forecasts; and row-based history across gaps may weaken features. I would diagnose these with timestamp-aware lags, backtesting, and weather forecasts available at prediction time.”

**Cross-question: Why keep the ML models then?**
A: “They provide a fair comparison and show whether extra complexity adds value. Since they did not win here, persistence is the best tested choice by these metrics. The comparison is still useful.”

**Q: What would you try next to improve forecasting?**
A: “First fix time alignment in all lag/rolling features and rerun a backtest. Then add weather forecasts—not future observed weather unless it would truly be available at inference—and compare daylight/irradiance regimes. I’d also examine errors by inverter and forecast horizon.”

### Metrics and interpretation

**Q: Explain MAE, RMSE, and R².**
A: “MAE is average absolute error; RMSE gives more weight to large errors; R² compares the predictions with a mean-only reference in terms of explained variation. I report them together because no single metric tells the whole story.”

**Cross-question: Which metric mattered most here?**
A: “MAE is easiest to interpret as a typical absolute miss and was the primary comparison I discuss. RMSE highlights occasional large misses, while R² summarizes fit relative to a baseline. The project does not define a business cost function, so I shouldn’t pretend one metric maps directly to operational cost.”

**Q: What do the results mean?**
A: “Persistence had the lowest MAE on both plants: 60.15 and 64.05. It also had the best RMSE and R² in the saved comparison. Thus, the tested ML models did not improve the forecast on this split.”

**Cross-question: Is R² of 0.91 91% accuracy?**
A: “No. R² is not classification accuracy and should not be described as 91% accurate. It is a measure of fit relative to predicting the mean under its usual definition.”

**Cross-question: Could high R² hide bad predictions?**
A: “Yes. R² can look good when the target has strong daily variation, while absolute errors or peak-period errors remain important. That is why I also report MAE/RMSE and would inspect error plots and operating regimes.”

### Abnormal screening

**Q: How are flags generated?**
A: “For eligible active-power observations, I calculate the absolute normalized difference between actual next-step power and persistence expected power. For each inverter, I use the 95th percentile from the first 60% of its chronological test rows as its threshold and apply it to the last 40%.”

**Cross-question: Why normalize residuals?**
A: “A fixed absolute difference has different meaning at low and high output. Dividing by expected power expresses deviation relative to expected output. We exclude very low expected power because division near zero is unstable.”

**Cross-question: Why P95? Why 60/40?**
A: “P95 is a chosen statistical threshold that flags the upper tail, not a threshold validated against faults. The 60/40 chronological calibration/evaluation split preserves order within the test period. These are design choices; I would validate their stability with expert review and labeled cases.”

**Q: What do 5.59% and 3.90% mean?**
A: “They are the fractions of eligible evaluated observations flagged by the rule in Plant 1 and Plant 2. They are not failure rates or accuracy.”

**Cross-question: What are precision and recall? Can you report them?**
A: “Precision is the share of flags that are true faults; recall is the share of true faults we flag. We have no confirmed fault labels, so neither can be estimated from this project.”

**Q: How would you validate anomaly detection?**
A: “Get work-order or engineer-confirmed event labels, align them to the inverter/time window, agree on what counts as an event, then measure precision, recall, alert delay, and false alarms. I’d also have plant experts review a sample of flagged and unflagged windows.”

### Deployment and engineering

**Q: What does Streamlit do here?**
A: “It presents the saved test predictions, metrics, inverter summaries, and screening outputs interactively. It is a dashboard over historical artifacts, not a live control system.”

**Cross-question: What if an output CSV is missing?**
A: “The app should fail with a clear message rather than show misleading blank results. I’d validate required files and columns at startup and document how to regenerate them.”

**Q: What would production require?**
A: “Reliable telemetry ingestion, schema and timestamp validation, missing-data handling, scheduled prediction, storage/versioning, monitoring, alert routing, security, and human review. The current dashboard is a prototype, not that production service.”

**Q: What was the hardest part?**
A: “The most important challenge was making the evaluation match the forecasting claim: the target must be exactly 15 minutes ahead, and chronological split boundaries must not be crossed by labels. I also learned that row-based lags need special care when timestamps are missing.”

**Q: What did you learn?**
A: “A more complex algorithm is not automatically better; baselines and leakage-safe evaluation matter. I also learned to separate a statistical alert from a verified diagnosis and to state limitations clearly.”

## 13. Questions you should ask the interviewer

- “How does the team validate forecasting and monitoring models once they are in use?”
- “What data-quality and monitoring problems are most common in your time-series projects?”
- “How do data scientists work with domain experts to confirm alerts?”

## 14. Claims to avoid

Do not say:

- “The system detects inverter faults accurately.” There are no fault labels.
- “Machine learning beat the baseline.” The saved comparison says otherwise.
- “Weather features improved the model.” Weather is not a current predictor.
- “Every lag is exactly 15 minutes / 24 hours.” Row gaps break that guarantee.
- “It is a live digital twin.” It displays saved historical outputs.
- “R² means prediction accuracy.” It does not.
- “Persistence proved current power is always the best predictor.” It won on this tested split only.

## 15. One-page recall sheet

Memorize these anchors, then explain them without reading:

1. **Goal:** next-step PV AC power + inverter-level abnormal-performance screening.
2. **Data:** two plants, generation by inverter, weather sensor table, nominal 15-minute cadence with gaps.
3. **Target:** same inverter at exactly `t + 15 min`; invalid target rows excluded.
4. **Predictors:** time/cyclic features, current and row-based historical/rolling AC power; weather not used by current models.
5. **Split:** chronological 70/15/15; remove labels that cross split boundaries; no shuffle.
6. **Models:** Persistence, Linear Regression, Random Forest, XGBoost.
7. **Result:** Persistence best on both plants; MAE 60.15 / 64.05.
8. **Screening:** abs normalized residual; per-inverter P95 from first 60% of test rows; apply to last 40%; expected power at least 50.
9. **Flags:** 5.59% / 3.90% of eligible observations; not confirmed faults; no precision/recall without labels.
10. **App:** Streamlit/Plotly dashboard of saved historical results, not live telemetry.
11. **Caveat:** row-based lags can span longer intervals across missing timestamps; fix with timestamp alignment and rerun.
12. **Next:** timestamp-aware features, rolling-origin backtest, real forecast weather, labeled/expert-reviewed anomaly validation.

## 16. How to study this as a beginner

**Session 1 — Understand the story:** read sections 1–5. Draw the workflow from raw CSV to dashboard by hand. Explain what one generation row means.

**Session 2 — Learn the modeling:** read sections 6–9. On paper, define target, leakage, lag, MAE, RMSE, R², persistence, and chronological split. Explain why the baseline winning is still a useful result.

**Session 3 — Defend the limitations:** read sections 8 and 12–14. Practice saying “screening signal, not confirmed fault” and explaining the lag-gap caveat without becoming defensive.

**Session 4 — Mock interview:** give the 60-second pitch, then answer the cross-questions aloud. Keep each first answer to 2–4 sentences. If you don’t know, say what you know, what you would inspect, and how you would test it. Don’t bluff.

**Before the interview:** open the current metric CSV, app, README, and notebooks. Check any exact hyperparameter, unit, missing-value rule, or implementation detail you plan to mention. Reruns can change metrics; always use the current saved outputs.
