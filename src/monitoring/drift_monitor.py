import pandas as pd
import mlflow
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from datetime import datetime

mlflow.set_tracking_uri("http://localhost:5001")

def monitor_drift(ref_path="data/processed/processed.parquet",
                  cur_path="data/processed/current_batch.parquet",
                  drift_threshold=0.3):
    """Compare reference vs current data distributions."""

    ref = pd.read_parquet(ref_path).sample(2000, random_state=42)
    cur = pd.read_parquet(cur_path).sample(min(2000, len(pd.read_parquet(cur_path))),
                                           random_state=42)

    # Drop target
    for df in [ref, cur]:
        if "isFraud" in df.columns:
            df.drop(columns=["isFraud"], inplace=True)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)

    report_path = f"/tmp/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(report_path)

    # Extract drift score
    result = report.as_dict()
    drifted_features = result["metrics"][0]["result"].get("number_of_drifted_columns", 0)
    total_features   = result["metrics"][0]["result"].get("number_of_columns", 1)
    drift_score      = drifted_features / total_features

    print(f"Drift score: {drift_score:.3f} ({drifted_features}/{total_features} features drifted)")

    with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.log_metric("drift_score", drift_score)
        mlflow.log_metric("drifted_features", drifted_features)
        mlflow.log_artifact(report_path)

    if drift_score > drift_threshold:
        print(f"ALERT: Drift {drift_score:.3f} > threshold {drift_threshold}. Retraining needed!")
        return True, drift_score

    return False, drift_score

if __name__ == "__main__":
    monitor_drift()
