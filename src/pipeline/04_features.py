import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle

mlflow.set_tracking_uri("http://localhost:5001")

def engineer_and_balance(parquet="data/processed/processed.parquet",
                         strategy="smote",   # "smote" | "undersample" | "class_weight"
                         out_dir="data/processed"):

    df = pd.read_parquet(parquet)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Before balancing — fraud: {y_train.sum()}, non-fraud: {(y_train==0).sum()}")

    with mlflow.start_run(run_name=f"feature_engineering_{strategy}"):
        mlflow.log_param("balance_strategy", strategy)
        mlflow.log_param("original_fraud_count", int(y_train.sum()))
        mlflow.log_param("original_nonfraud_count", int((y_train==0).sum()))

        if strategy == "smote":
            sampler = SMOTE(random_state=42, k_neighbors=5)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            mlflow.log_param("sampler", "SMOTE")

        elif strategy == "undersample":
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            mlflow.log_param("sampler", "RandomUnderSampler")

        else:  # class_weight — no resampling, weights passed to model
            X_res, y_res = X_train, y_train
            mlflow.log_param("sampler", "class_weight_only")

        mlflow.log_param("resampled_fraud_count", int(y_res.sum()))
        mlflow.log_param("resampled_nonfraud_count", int((y_res==0).sum()))

        # Save splits
        splits = {"X_train": X_res, "X_test": X_test,
                  "y_train": y_res, "y_test": y_test}
        with open(f"{out_dir}/splits_{strategy}.pkl", "wb") as f:
            pickle.dump(splits, f)

        mlflow.log_artifact(f"{out_dir}/splits_{strategy}.pkl")

    print(f"After {strategy} — fraud: {y_res.sum()}, non-fraud: {(y_res==0).sum()}")
    return splits

if __name__ == "__main__":
    for s in ["smote", "undersample", "class_weight"]:
        engineer_and_balance(strategy=s)
