import pandas as pd
import numpy as np
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

def simulate_temporal_drift(parquet="data/processed/processed.parquet"):
    """
    Simulate train-on-early / test-on-late distribution.
    Introduce new fraud patterns and feature shifts.
    """
    df = pd.read_parquet(parquet)

    # Sort by TransactionAmt as a proxy for time ordering
    df = df.sort_values("TransactionAmt").reset_index(drop=True)

    n = len(df)
    split = int(n * 0.7)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    # --- Simulate distribution shift in later data ---
    # 1. Shift numeric mean (feature drift)
    num_cols = test_df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "isFraud"]
    for col in num_cols[:10]:   # shift top 10 features
        test_df[col] = test_df[col] * np.random.uniform(1.1, 1.4)

    # 2. New fraud pattern: inject high-amount frauds
    n_new_fraud = int(len(test_df) * 0.02)
    fraud_idx = np.random.choice(test_df[test_df["isFraud"]==0].index, n_new_fraud, replace=False)
    test_df.loc[fraud_idx, "isFraud"] = 1
    test_df.loc[fraud_idx, "TransactionAmt"] *= 5   # high-value fraud

    train_df.to_parquet("data/processed/train_early.parquet", index=False)
    test_df.to_parquet("data/processed/test_late.parquet", index=False)

    with mlflow.start_run(run_name="drift_simulation"):
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("new_fraud_injected", n_new_fraud)
        mlflow.log_param("features_shifted", 10)
        mlflow.log_metric("train_fraud_rate", float(train_df["isFraud"].mean()))
        mlflow.log_metric("test_fraud_rate",  float(test_df["isFraud"].mean()))

    print(f"Train fraud rate: {train_df['isFraud'].mean():.4f}")
    print(f"Test fraud rate:  {test_df['isFraud'].mean():.4f}")
    return train_df, test_df

if __name__ == "__main__":
    simulate_temporal_drift()
