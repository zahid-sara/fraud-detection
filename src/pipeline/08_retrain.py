import time, pickle, mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import recall_score, roc_auc_score
import xgboost as xgb

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("retraining_strategies")

def quick_train(splits):
    X_tr, X_te = splits["X_train"], splits["X_test"]
    y_tr, y_te = splits["y_train"], splits["y_test"]
    scale = float((y_tr==0).sum()) / float(y_tr.sum())
    model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale,
                               random_state=42, n_jobs=-1, verbosity=0)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)
    return model, recall_score(y_te, y_pred), roc_auc_score(y_te, y_prob), train_time

def strategy_threshold(splits, recall_threshold=0.70, max_rounds=3):
    """Retrain only when recall drops below threshold."""
    print("\n--- Strategy 1: Threshold-based ---")
    results = []
    with mlflow.start_run(run_name="strategy_threshold_based"):
        for round_n in range(max_rounds):
            model, recall, auc, t = quick_train(splits)
            triggered = recall < recall_threshold
            mlflow.log_metric("recall",      recall, step=round_n)
            mlflow.log_metric("auc",         auc,    step=round_n)
            mlflow.log_metric("train_time",  t,      step=round_n)
            mlflow.log_metric("retrain_triggered", int(triggered), step=round_n)
            results.append({"round": round_n, "recall": recall,
                           "auc": auc, "time": t, "triggered": triggered})
            print(f"  Round {round_n}: Recall={recall:.4f} AUC={auc:.4f} "
                  f"Time={t:.1f}s Triggered={triggered}")
            if not triggered:
                print("  Recall OK — skipping retrain")
                break
        mlflow.log_param("strategy", "threshold")
        mlflow.log_param("recall_threshold", recall_threshold)
        mlflow.log_metric("total_rounds", len(results))
        mlflow.log_metric("total_train_time", sum(r["time"] for r in results))
    return results

def strategy_periodic(splits, interval_seconds=2, rounds=3):
    """Retrain on fixed schedule regardless of performance."""
    print("\n--- Strategy 2: Periodic ---")
    results = []
    with mlflow.start_run(run_name="strategy_periodic"):
        for round_n in range(rounds):
            model, recall, auc, t = quick_train(splits)
            mlflow.log_metric("recall",     recall, step=round_n)
            mlflow.log_metric("auc",        auc,    step=round_n)
            mlflow.log_metric("train_time", t,      step=round_n)
            results.append({"round": round_n, "recall": recall,
                           "auc": auc, "time": t})
            print(f"  Round {round_n}: Recall={recall:.4f} AUC={auc:.4f} Time={t:.1f}s")
            if round_n < rounds - 1:
                time.sleep(interval_seconds)
        mlflow.log_param("strategy", "periodic")
        mlflow.log_param("interval_seconds", interval_seconds)
        mlflow.log_metric("total_train_time", sum(r["time"] for r in results))
    return results

def strategy_hybrid(splits, recall_threshold=0.70,
                    interval_seconds=2, rounds=3):
    """Retrain if recall drops OR on schedule — whichever comes first."""
    print("\n--- Strategy 3: Hybrid ---")
    results = []
    last_retrain = time.time()
    with mlflow.start_run(run_name="strategy_hybrid"):
        for round_n in range(rounds):
            model, recall, auc, t = quick_train(splits)
            time_since = time.time() - last_retrain
            triggered_recall = recall < recall_threshold
            triggered_time   = time_since >= interval_seconds
            triggered = triggered_recall or triggered_time
            if triggered:
                last_retrain = time.time()
            mlflow.log_metric("recall",           recall, step=round_n)
            mlflow.log_metric("auc",              auc,    step=round_n)
            mlflow.log_metric("retrain_triggered",int(triggered), step=round_n)
            results.append({"round": round_n, "recall": recall,
                           "auc": auc, "time": t, "triggered": triggered})
            print(f"  Round {round_n}: Recall={recall:.4f} AUC={auc:.4f} "
                  f"Triggered={triggered} "
                  f"(recall={triggered_recall}, schedule={triggered_time})")
            time.sleep(interval_seconds)
        mlflow.log_param("strategy", "hybrid")
        mlflow.log_metric("total_train_time", sum(r["time"] for r in results))
    return results

def compare_strategies(r_threshold, r_periodic, r_hybrid):
    print("\n" + "="*65)
    print(f"{'Strategy':<20} {'Rounds':>7} {'Avg Recall':>11} "
          f"{'Avg AUC':>9} {'Total Time':>11}")
    print("="*65)
    for name, results in [("Threshold", r_threshold),
                          ("Periodic",  r_periodic),
                          ("Hybrid",    r_hybrid)]:
        rounds     = len(results)
        avg_recall = np.mean([r["recall"] for r in results])
        avg_auc    = np.mean([r["auc"]    for r in results])
        total_time = sum(r["time"] for r in results)
        print(f"{name:<20} {rounds:>7} {avg_recall:>11.4f} "
              f"{avg_auc:>9.4f} {total_time:>10.1f}s")
    print("="*65)
    print("\nAnalysis:")
    print("  Threshold: lowest compute cost, only retrains when needed")
    print("  Periodic:  predictable schedule, may retrain unnecessarily")
    print("  Hybrid:    best coverage, highest compute cost")

if __name__ == "__main__":
    with open("data/processed/splits_smote.pkl", "rb") as f:
        splits = pickle.load(f)

    r1 = strategy_threshold(splits)
    r2 = strategy_periodic(splits)
    r3 = strategy_hybrid(splits)
    compare_strategies(r1, r2, r3)
