import mlflow.pyfunc
import numpy as np
import pandas as pd
import time
import psutil
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from prometheus_client import (Counter, Histogram, Gauge,
                                Summary, generate_latest, CONTENT_TYPE_LATEST)
from fastapi.responses import Response

MLFLOW_URI  = "http://localhost:5001"
MODEL_NAME  = "fraud_xgboost"

mlflow.set_tracking_uri(MLFLOW_URI)
app = FastAPI(title="Fraud Detection API")

# ── System metrics ──────────────────────────────────────
REQUEST_COUNT   = Counter("api_requests_total", "Total API requests", ["endpoint","status"])
REQUEST_LATENCY = Histogram("api_latency_seconds", "Request latency",
                            ["endpoint"], buckets=[.01,.05,.1,.25,.5,1,2,5])
ERROR_COUNT     = Counter("api_errors_total", "Total errors", ["endpoint"])
CPU_USAGE       = Gauge("system_cpu_percent", "CPU usage percent")
MEMORY_USAGE    = Gauge("system_memory_percent", "Memory usage percent")

# ── Model metrics ───────────────────────────────────────
FRAUD_PRED_COUNT   = Counter("fraud_predictions_total", "Total fraud predictions")
LEGIT_PRED_COUNT   = Counter("legit_predictions_total", "Total legit predictions")
CONFIDENCE_HIST    = Histogram("prediction_confidence", "Prediction confidence distribution",
                               buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
FRAUD_PROB_GAUGE   = Gauge("last_fraud_probability", "Last fraud probability score")
FALSE_POS_RATE     = Gauge("false_positive_rate", "Estimated false positive rate")
RECALL_GAUGE       = Gauge("model_fraud_recall", "Current model fraud recall")
PRECISION_GAUGE    = Gauge("model_fraud_precision", "Current model fraud precision")

# ── Data metrics ────────────────────────────────────────
MISSING_VALUES     = Gauge("input_missing_values_total", "Missing values in input")
DRIFT_SCORE        = Gauge("data_drift_score", "Current data drift score")
ANOMALY_COUNT      = Counter("input_anomalies_total", "Anomalous inputs detected")
HIGH_AMT_COUNT     = Counter("high_amount_transactions_total", "High amount transactions")

# Sliding window for recall estimation
_predictions = []   # (true_label_if_known, predicted, prob)
_fp = 0
_tp = 0
_fn = 0
_total = 0

model = None

def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}@production"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model: {model_uri}")
        # Set initial metric values from MLflow
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(["1"], order_by=["metrics.auc_roc DESC"], max_results=1)
        if runs:
            RECALL_GAUGE.set(runs[0].data.metrics.get("recall", 0.86))
            PRECISION_GAUGE.set(runs[0].data.metrics.get("precision", 0.40))
    except Exception as e:
        print(f"Model load error: {e}")
        model = None

@app.on_event("startup")
def startup():
    load_model()

def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

class TransactionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence: str
    anomaly_detected: bool

@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest):
    start = time.time()
    update_system_metrics()

    try:
        df = pd.DataFrame([req.features])

        # ── Data-level monitoring ──
        missing = int(df.isnull().sum().sum())
        MISSING_VALUES.set(missing)

        anomaly = False
        amt = req.features.get("TransactionAmt", 0)
        if amt and float(amt) > 10000:
            ANOMALY_COUNT.inc()
            HIGH_AMT_COUNT.inc()
            anomaly = True

        # ── Prediction ──
        proba = model.predict(df)
        prob  = float(proba[0]) if hasattr(proba, "__len__") else float(proba)
        is_fraud = prob >= 0.5

        # ── Model-level metrics ──
        CONFIDENCE_HIST.observe(prob)
        FRAUD_PROB_GAUGE.set(prob)

        if is_fraud:
            FRAUD_PRED_COUNT.inc()
        else:
            LEGIT_PRED_COUNT.inc()

        # Estimate FPR from running totals (simplified)
        global _total, _fp
        _total += 1
        if not is_fraud and prob > 0.3:
            _fp += 1
        if _total > 0:
            FALSE_POS_RATE.set(_fp / max(_total, 1))

        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()

        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            confidence="high" if (prob > 0.8 or prob < 0.2) else "low",
            anomaly_detected=anomaly
        )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(payload: Dict[str, Any]):
    """Receive ground truth labels to update recall/precision metrics."""
    global _tp, _fn
    predicted = payload.get("predicted_fraud", False)
    actual    = payload.get("actual_fraud", False)
    if actual and predicted:
        _tp += 1
    elif actual and not predicted:
        _fn += 1
    recall = _tp / max(_tp + _fn, 1)
    RECALL_GAUGE.set(recall)
    return {"recall_updated": round(recall, 4)}

@app.post("/update_drift")
def update_drift(payload: Dict[str, Any]):
    """Called by drift_monitor.py to push drift score into Prometheus."""
    score = float(payload.get("drift_score", 0))
    DRIFT_SCORE.set(score)
    return {"drift_score_updated": score}

@app.get("/health")
def health():
    update_system_metrics()
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    update_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



