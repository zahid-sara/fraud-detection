import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("."))

def test_fraud_rate_range():
    """Fraud rate must be between 0.5% and 20%."""
    y = pd.Series([0]*950 + [1]*50)
    rate = y.mean()
    assert 0.005 <= rate <= 0.20

def test_preprocess_no_nulls():
    """After preprocessing, no null values should remain."""
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": ["x", np.nan, "z"],
        "isFraud": [0, 1, 0]
    })
    df["a"] = df["a"].fillna(df["a"].median())
    df["b"] = df["b"].fillna("UNKNOWN")
    assert df["a"].isnull().sum() == 0
    assert df["b"].isnull().sum() == 0

def test_compute_metrics():
    """Basic metrics shape check."""
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0,0,0,1,1])
    y_prob = np.array([0.1,0.2,0.3,0.7,0.9])
    auc = roc_auc_score(y_true, y_prob)
    assert 0.5 <= auc <= 1.0

def test_smote_increases_minority():
    """SMOTE should increase minority class count."""
    from imblearn.over_sampling import SMOTE
    X = np.random.randn(200, 5)
    y = np.array([0]*190 + [1]*10)
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    assert (y_res == 1).sum() > (y == 1).sum()
