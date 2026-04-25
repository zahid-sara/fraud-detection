import pandas as pd
import mlflow
import json

mlflow.set_tracking_uri("http://localhost:5001")

REQUIRED_COLS  = ["TransactionID", "isFraud", "TransactionAmt", "ProductCD"]
MAX_MISSING    = 0.90   # drop columns with >90% missing
MIN_FRAUD_RATE = 0.005
MAX_FRAUD_RATE = 0.20

def validate(parquet="data/processed/raw_merged.parquet"):
    df = pd.read_parquet(parquet)
    errors = []

    # Schema check
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Fraud rate check
    fr = df["isFraud"].mean()
    if not (MIN_FRAUD_RATE <= fr <= MAX_FRAUD_RATE):
        errors.append(f"Fraud rate {fr:.4f} outside [{MIN_FRAUD_RATE},{MAX_FRAUD_RATE}]")

    # High-missing columns
    high_miss = df.columns[df.isnull().mean() > MAX_MISSING].tolist()

    passed = len(errors) == 0
    report = {
        "passed": passed,
        "errors": errors,
        "high_missing_cols": high_miss,
        "fraud_rate": float(fr),
        "shape": list(df.shape)
    }

    with open("data/processed/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with mlflow.start_run(run_name="data_validation"):
        mlflow.log_param("validation_passed", passed)
        mlflow.log_param("fraud_rate", round(fr, 5))
        mlflow.log_param("high_missing_cols_count", len(high_miss))
        mlflow.log_artifact("data/processed/validation_report.json")

    if not passed:
        raise ValueError(f"Validation failed: {errors}")

    print("Validation PASSED.")
    return report

if __name__ == "__main__":
    validate()
