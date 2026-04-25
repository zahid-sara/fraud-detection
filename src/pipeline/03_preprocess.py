import pandas as pd
import numpy as np
import mlflow
import pickle
from sklearn.preprocessing import LabelEncoder

mlflow.set_tracking_uri("http://localhost:5001")

def preprocess(parquet="data/processed/raw_merged.parquet",
               out_dir="data/processed"):
    df = pd.read_parquet(parquet)

    # Drop columns with >90% missing
    threshold = 0.90
    keep_cols = df.columns[df.isnull().mean() <= threshold]
    df = df[keep_cols]

    # Separate target
    y = df["isFraud"]
    df = df.drop(columns=["isFraud", "TransactionID"])

    # Split numeric / categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Advanced imputation ---
    # Numeric: median impute
    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Categorical: mode impute
    for col in cat_cols:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if len(mode) else "UNKNOWN")

    # --- High-cardinality categorical: target encode ---
    # (Using mean fraud rate per category, fitted only on training data)
    # For simplicity here we label-encode; Task 4 does target encoding
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Save encoders
    with open(f"{out_dir}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    df["isFraud"] = y.values
    df.to_parquet(f"{out_dir}/processed.parquet", index=False)

    with mlflow.start_run(run_name="preprocessing"):
        mlflow.log_param("num_features", len(num_cols))
        mlflow.log_param("cat_features", len(cat_cols))
        mlflow.log_param("missing_strategy", "median+mode")
        mlflow.log_artifact(f"{out_dir}/processed.parquet")
        mlflow.log_artifact(f"{out_dir}/encoders.pkl")

    print(f"Preprocessed. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    preprocess()
