import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud_detection")

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }

def save_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit','Fraud'],
                yticklabels=['Legit','Fraud'])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13)

    # Add TN/FP/FN/TP labels
    tn, fp, fn, tp = cm.ravel()
    fig.text(0.15, 0.02,
             f"TN={tn}  FP={fp}  FN={fn}  TP={tp} | "
             f"Recall={tp/(tp+fn):.3f}  FPR={fp/(fp+tn):.3f}",
             fontsize=9, ha='left')

    plt.tight_layout()
    path = f"/tmp/cm_{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path, tn, fp, fn, tp

def save_shap(model, X_test, name):
    print(f"  Computing SHAP for {name}...")
    explainer = shap.TreeExplainer(model)
    sample = X_test.iloc[:300]
    shap_vals = explainer.shap_values(sample)

    # Summary dot plot
    shap.summary_plot(shap_vals, sample, show=False, max_display=20)
    path1 = f"/tmp/shap_summary_{name}.png"
    plt.savefig(path1, bbox_inches="tight", dpi=150)
    plt.close()

    # Bar plot (feature importance)
    shap.summary_plot(shap_vals, sample, plot_type="bar", show=False, max_display=20)
    path2 = f"/tmp/shap_bar_{name}.png"
    plt.savefig(path2, bbox_inches="tight", dpi=150)
    plt.close()

    return path1, path2

def train_xgboost(splits, cost_sensitive=False):
    X_train = splits["X_train"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_test  = splits["y_test"]

    # Cost-sensitive: weight minority class heavily
    scale = float((y_train==0).sum()) / float(y_train.sum())
    
    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": scale if cost_sensitive else 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc"
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Cost-sensitive uses LOWER threshold to maximize recall
    threshold = 0.3 if cost_sensitive else 0.5
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["threshold"] = threshold

    run_name = f"xgboost_{'cost_sensitive' if cost_sensitive else 'standard'}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("cost_sensitive", cost_sensitive)
        mlflow.log_metrics(metrics)

        # Business impact
        _, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fraud_loss_saved   = tp * 500    # avg fraud = $500
        false_alarm_cost   = fp * 10     # investigation cost
        net_benefit        = fraud_loss_saved - false_alarm_cost
        mlflow.log_metric("fraud_loss_saved_usd",  fraud_loss_saved)
        mlflow.log_metric("false_alarm_cost_usd",  false_alarm_cost)
        mlflow.log_metric("net_benefit_usd",        net_benefit)
        mlflow.log_metric("false_negatives",        fn)
        mlflow.log_metric("false_positives",        fp)

        # Confusion matrix
        cm_path, tn, fp2, fn2, tp2 = save_confusion_matrix(y_test, y_pred, run_name)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # SHAP
        p1, p2 = save_shap(model, X_test, run_name)
        mlflow.log_artifact(p1, artifact_path="shap")
        mlflow.log_artifact(p2, artifact_path="shap")

        mlflow.sklearn.log_model(model, "model",
            registered_model_name="fraud_xgboost")

        print(f"{run_name} (threshold={threshold}):")
        print(f"  AUC={metrics['auc_roc']:.4f} Recall={metrics['recall']:.4f} "
              f"Precision={metrics['precision']:.4f} F1={metrics['f1']:.4f}")
        print(f"  FP={fp2} FN={fn2} TP={tp2} | Net benefit=${net_benefit:,}")

    return model, metrics

def train_lightgbm(splits, cost_sensitive=False):
    X_train = splits["X_train"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_test  = splits["y_test"]

    scale = float((y_train==0).sum()) / float(y_train.sum())

    params = {
        "n_estimators": 500,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "scale_pos_weight": scale if cost_sensitive else 1.0,
        "is_unbalance": cost_sensitive,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)])

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.3 if cost_sensitive else 0.5
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["threshold"] = threshold

    run_name = f"lightgbm_{'cost_sensitive' if cost_sensitive else 'standard'}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("cost_sensitive", cost_sensitive)
        mlflow.log_metrics(metrics)

        _, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fraud_loss_saved = tp * 500
        false_alarm_cost = fp * 10
        net_benefit      = fraud_loss_saved - false_alarm_cost
        mlflow.log_metric("net_benefit_usd",       net_benefit)
        mlflow.log_metric("false_negatives",        fn)
        mlflow.log_metric("false_positives",        fp)

        cm_path, *_ = save_confusion_matrix(y_test, y_pred, run_name)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        p1, p2 = save_shap(model, X_test, run_name)
        mlflow.log_artifact(p1, artifact_path="shap")
        mlflow.log_artifact(p2, artifact_path="shap")

        mlflow.sklearn.log_model(model, "model",
            registered_model_name="fraud_lightgbm")

        print(f"{run_name} (threshold={threshold}):")
        print(f"  AUC={metrics['auc_roc']:.4f} Recall={metrics['recall']:.4f} "
              f"Precision={metrics['precision']:.4f} F1={metrics['f1']:.4f}")
        print(f"  FP={fp} FN={fn} TP={tp} | Net benefit=${net_benefit:,}")

    return model, metrics

def train_hybrid_rf(splits):
    X_train = splits["X_train"]
    X_test  = splits["X_test"]
    y_train = splits["y_train"]
    y_test  = splits["y_test"]

    with mlflow.start_run(run_name="rf_feature_selection_hybrid"):
        # Stage 1: feature selection
        selector_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        selector_rf.fit(X_train, y_train)
        sel = SelectFromModel(selector_rf, prefit=True, threshold="mean")
        X_tr_sel = sel.transform(X_train)
        X_te_sel = sel.transform(X_test)
        n_sel = X_tr_sel.shape[1]
        mlflow.log_param("features_selected", n_sel)
        mlflow.log_param("features_original", X_train.shape[1])

        # Stage 2: final model on selected features
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                       class_weight="balanced", random_state=42)
        model.fit(X_tr_sel, y_train)

        y_prob = model.predict_proba(X_te_sel)[:, 1]
        y_pred = (y_prob >= 0.4).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        cm_path, *_ = save_confusion_matrix(y_test, y_pred, "rf_hybrid")
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        print(f"RF hybrid: AUC={metrics['auc_roc']:.4f} "
              f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

    return model, metrics, sel

def print_comparison(results):
    print("\n" + "="*75)
    print(f"{'Model':<30} {'AUC':>7} {'Recall':>8} {'Precision':>10} "
          f"{'F1':>7} {'Threshold':>10}")
    print("="*75)
    for name, m in results.items():
        print(f"{name:<30} {m['auc_roc']:>7.4f} {m['recall']:>8.4f} "
              f"{m['precision']:>10.4f} {m['f1']:>7.4f} "
              f"{m.get('threshold', 0.5):>10.2f}")
    print("="*75)
    print("\nKey insight: Cost-sensitive models use threshold=0.3 → higher recall,")
    print("lower precision — catches more fraud at cost of more false alarms.")

if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "smote"

    with open(f"data/processed/splits_{strategy}.pkl", "rb") as f:
        splits = pickle.load(f)

    results = {}
    for cs in [False, True]:
        _, m = train_xgboost(splits, cost_sensitive=cs)
        results[f"xgb_cs{cs}"] = m
        _, m = train_lightgbm(splits, cost_sensitive=cs)
        results[f"lgb_cs{cs}"] = m

    train_hybrid_rf(splits)
    print_comparison(results)
