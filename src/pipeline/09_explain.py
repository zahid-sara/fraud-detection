import pickle, mlflow
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud_detection")

def explain_model():
    with open("data/processed/splits_smote.pkl", "rb") as f:
        splits = pickle.load(f)
    X_test  = splits["X_test"]
    y_test  = splits["y_test"]

    # Load best run
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        ["1"],
        filter_string="tags.mlflow.runName = 'xgboost_standard'",
        max_results=1
    )
    run_id = runs[0].info.run_id
    model  = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    sample    = X_test.iloc[:500]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)

    with mlflow.start_run(run_name="shap_explainability_report"):

        # 1. Summary dot plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, sample, show=False, max_display=20)
        plt.title("SHAP Summary — Why is the model predicting fraud?", fontsize=13)
        plt.tight_layout()
        plt.savefig("/tmp/shap_01_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("/tmp/shap_01_summary.png", "explainability")

        # 2. Bar importance plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, sample, plot_type="bar",
                         show=False, max_display=20)
        plt.title("Top 20 Features by Mean SHAP Value", fontsize=13)
        plt.tight_layout()
        plt.savefig("/tmp/shap_02_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("/tmp/shap_02_importance.png", "explainability")

        # 3. Waterfall for top fraud case
        fraud_idx = np.where(y_test.values == 1)[0][0]
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals[fraud_idx],
                base_values=explainer.expected_value,
                data=sample.iloc[fraud_idx],
                feature_names=sample.columns.tolist()
            ), show=False, max_display=15
        )
        plt.title("Why this transaction was flagged as FRAUD", fontsize=12)
        plt.tight_layout()
        plt.savefig("/tmp/shap_03_waterfall_fraud.png", dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("/tmp/shap_03_waterfall_fraud.png", "explainability")

        # 4. Waterfall for legit case
        legit_idx = np.where(y_test.values == 0)[0][0]
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals[legit_idx],
                base_values=explainer.expected_value,
                data=sample.iloc[legit_idx],
                feature_names=sample.columns.tolist()
            ), show=False, max_display=15
        )
        plt.title("Why this transaction was classified as LEGITIMATE", fontsize=12)
        plt.tight_layout()
        plt.savefig("/tmp/shap_04_waterfall_legit.png", dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("/tmp/shap_04_waterfall_legit.png", "explainability")

        # 5. Top features table
        mean_shap = pd.DataFrame({
            "feature": sample.columns,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).head(20)

        print("\nTop 20 features driving fraud predictions:")
        print(mean_shap.to_string(index=False))
        mean_shap.to_csv("/tmp/shap_feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/shap_feature_importance.csv", "explainability")

        mlflow.log_param("sample_size", len(sample))
        mlflow.log_param("top_feature", mean_shap.iloc[0]["feature"])
        mlflow.log_metric("mean_shap_top_feature",
                         float(mean_shap.iloc[0]["mean_abs_shap"]))

        print("\nAll SHAP artifacts logged to MLflow.")
        print(f"View at: http://localhost:5001/#/experiments/1/runs/{mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    import mlflow.sklearn
    explain_model()
