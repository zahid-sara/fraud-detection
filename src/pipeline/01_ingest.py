import pandas as pd
import mlflow
import os, json
from datetime import datetime

MLFLOW_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_URI)

def ingest_data(raw_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    train_tx   = pd.read_csv(f"{raw_dir}/train_transaction.csv")
    train_id   = pd.read_csv(f"{raw_dir}/train_identity.csv")
    df = train_tx.merge(train_id, on="TransactionID", how="left")

    # Basic stats for logging
    stats = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "fraud_rate": float(df["isFraud"].mean()),
        "missing_pct": float(df.isnull().mean().mean()),
        "ingest_time": datetime.now().isoformat()
    }

    df.to_parquet(f"{out_dir}/raw_merged.parquet", index=False)

    with mlflow.start_run(run_name="data_ingestion"):
        mlflow.log_params(stats)
        mlflow.log_artifact(f"{out_dir}/raw_merged.parquet")

    print(f"Ingested {stats['rows']} rows. Fraud rate: {stats['fraud_rate']:.4f}")
    return df, stats

if __name__ == "__main__":
    ingest_data()
