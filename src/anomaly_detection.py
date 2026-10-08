"""Generate calibrated, inverter-specific performance screening outputs.

Thresholds are fitted on an early chronological calibration window for each
inverter, then applied only to its later evaluation window. This avoids using
the same observations both to set a threshold and report its flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
ANOMALIES_DIR = ROOT / "outputs" / "anomalies"
CALIBRATION_FRACTION = 0.60
MIN_EXPECTED_POWER = 50.0
THRESHOLD_QUANTILE = 0.95


def score_plant(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"DATE_TIME", "SOURCE_KEY", "TARGET_AC_POWER", "Persistence"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction data is missing required columns: {', '.join(missing)}")

    data = predictions.copy()
    data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"], errors="coerce")
    data["ACTUAL_AC_POWER"] = pd.to_numeric(data["TARGET_AC_POWER"], errors="coerce")
    data["EXPECTED_AC_POWER"] = pd.to_numeric(data["Persistence"], errors="coerce")
    data = data.dropna(subset=["DATE_TIME", "SOURCE_KEY", "ACTUAL_AC_POWER", "EXPECTED_AC_POWER"])
    data["RESIDUAL"] = data["ACTUAL_AC_POWER"] - data["EXPECTED_AC_POWER"]
    data["NORMALIZED_RESIDUAL"] = np.where(
        data["EXPECTED_AC_POWER"] >= MIN_EXPECTED_POWER,
        data["RESIDUAL"] / data["EXPECTED_AC_POWER"],
        np.nan,
    )
    data["ABS_NORMALIZED_RESIDUAL"] = data["NORMALIZED_RESIDUAL"].abs()

    scored_groups: list[pd.DataFrame] = []
    calibration_counts: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    for source, group in data.sort_values(["SOURCE_KEY", "DATE_TIME"]).groupby("SOURCE_KEY", sort=False):
        group = group.reset_index(drop=True)
        cut = int(np.floor(len(group) * CALIBRATION_FRACTION))
        calibration = group.iloc[:cut]
        evaluation = group.iloc[cut:].copy()
        calibration_active = calibration.loc[calibration["EXPECTED_AC_POWER"] >= MIN_EXPECTED_POWER,
                                              "ABS_NORMALIZED_RESIDUAL"].dropna()
        if calibration_active.empty:
            continue
        threshold = float(calibration_active.quantile(THRESHOLD_QUANTILE))
        evaluation = evaluation[evaluation["EXPECTED_AC_POWER"] >= MIN_EXPECTED_POWER].copy()
        evaluation["P95_Abs_Norm_Residual"] = threshold
        evaluation["CALIBRATION_OBSERVATIONS"] = int(len(calibration_active))
        evaluation["ANOMALY_FLAG"] = evaluation["ABS_NORMALIZED_RESIDUAL"] > threshold
        evaluation["STATUS"] = np.where(evaluation["ANOMALY_FLAG"], "FLAGGED", "WITHIN_THRESHOLD")
        scored_groups.append(evaluation)
        calibration_counts[str(source)] = int(len(calibration_active))
        thresholds[str(source)] = threshold

    if not scored_groups:
        raise ValueError("No inverter had enough active calibration observations to calculate thresholds.")
    scores = pd.concat(scored_groups, ignore_index=True).sort_values(["DATE_TIME", "SOURCE_KEY"])

    inverter = scores.groupby("SOURCE_KEY", as_index=False).agg(
        Total_Observations=("ANOMALY_FLAG", "size"),
        Anomalies=("ANOMALY_FLAG", "sum"),
        Mean_Residual=("RESIDUAL", "mean"),
        Mean_Deviation_Pct=("NORMALIZED_RESIDUAL", lambda x: x.mean() * 100),
        P95_Abs_Norm_Residual=("P95_Abs_Norm_Residual", "first"),
    )
    inverter["Calibration_Observations"] = inverter["SOURCE_KEY"].map(calibration_counts)
    inverter["Normal"] = inverter["Total_Observations"] - inverter["Anomalies"]
    inverter["Anomaly_Percentage"] = inverter["Anomalies"] / inverter["Total_Observations"] * 100
    inverter["Peer_Deviation_Pct"] = inverter["Mean_Deviation_Pct"] - inverter["Mean_Deviation_Pct"].mean()
    inverter["ABS_Peer_Deviation_Pct"] = inverter["Peer_Deviation_Pct"].abs()
    inverter = inverter.sort_values("Anomaly_Percentage", ascending=False).reset_index(drop=True)

    report_columns = [
        "DATE_TIME", "SOURCE_KEY", "ACTUAL_AC_POWER", "EXPECTED_AC_POWER",
        "RESIDUAL", "NORMALIZED_RESIDUAL", "ABS_NORMALIZED_RESIDUAL",
        "P95_Abs_Norm_Residual", "CALIBRATION_OBSERVATIONS",
    ]
    report = scores.loc[scores["ANOMALY_FLAG"], report_columns].sort_values(
        "ABS_NORMALIZED_RESIDUAL", ascending=False
    ).reset_index(drop=True)
    return scores.reset_index(drop=True), report, inverter


def main() -> None:
    ANOMALIES_DIR.mkdir(parents=True, exist_ok=True)
    for plant in (1, 2):
        path = PREDICTIONS_DIR / f"plant{plant}_test_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required prediction file not found: {path}")
        scores, report, summary = score_plant(pd.read_csv(path))
        scores.to_csv(ANOMALIES_DIR / f"plant{plant}_anomaly_scores.csv", index=False)
        report.to_csv(ANOMALIES_DIR / f"plant{plant}_anomaly_report.csv", index=False)
        summary.to_csv(ANOMALIES_DIR / f"plant{plant}_inverter_anomaly_summary.csv", index=False)
        print(
            f"Plant {plant}: {len(report):,} flags across {len(scores):,} evaluated active rows "
            f"({len(summary):,} inverters)."
        )


if __name__ == "__main__":
    main()
