import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5001")
client = MlflowClient()

AUC_THRESHOLD    = 0.85
RECALL_THRESHOLD = 0.70
MODEL_NAME       = "fraud_xgboost"   # change to lgb/rf as needed

def get_best_run(experiment_name="fraud_detection"):
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        order_by=["metrics.auc_roc DESC"],
        max_results=1
    )
    return runs[0] if runs else None

def conditional_deploy():
    run = get_best_run()
    if not run:
        print("No runs found. Aborting.")
        return

    auc    = run.data.metrics.get("auc_roc", 0)
    recall = run.data.metrics.get("recall", 0)

    print(f"Best run: AUC={auc:.4f}, Recall={recall:.4f}")

    if auc >= AUC_THRESHOLD and recall >= RECALL_THRESHOLD:
        print("Thresholds met. Promoting model to Production.")
        # Register model version from best run
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=f"{run.info.artifact_uri}/model",
            run_id=run.info.run_id
        )
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model v{mv.version} is now in Production.")
    else:
        print(f"Thresholds NOT met (AUC≥{AUC_THRESHOLD}, Recall≥{RECALL_THRESHOLD}). Skipping deploy.")

if __name__ == "__main__":
    conditional_deploy()
