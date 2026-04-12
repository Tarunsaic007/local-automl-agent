"""Unit tests for the ML trainer module."""

import pandas as pd
import pytest

from app.trainer import _compute_metrics, _build_preprocessing, run_training


SAMPLE_STRATEGY = {
    "task_type": "classification",
    "target_col": "churn",
    "models": ["random_forest", "logistic_regression"],
    "preprocessing": ["impute_mean", "standard_scaler"],
}


def make_df(n=100):
    import numpy as np
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 60, n),
        "salary": np.random.randint(30000, 150000, n),
        "churn": np.random.randint(0, 2, n),
    })


def test_compute_metrics_classification():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    result = _compute_metrics("classification", y_true, y_pred)
    assert "accuracy" in result
    assert "f1" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_compute_metrics_regression():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.1]
    result = _compute_metrics("regression", y_true, y_pred)
    assert "rmse" in result
    assert "r2" in result


def test_build_preprocessing_standard_scaler():
    steps = _build_preprocessing(["standard_scaler"])
    names = [s[0] for s in steps]
    assert "scaler" in names


def test_build_preprocessing_impute_mean():
    steps = _build_preprocessing(["impute_mean"])
    names = [s[0] for s in steps]
    assert "imputer" in names


def test_run_training_returns_best_model():
    df = make_df(200)
    result = run_training(df, SAMPLE_STRATEGY)
    assert "best_model" in result
    assert "best_score" in result
    assert result["best_model"] in ["random_forest", "logistic_regression"]
    assert 0.0 <= result["best_score"] <= 1.0


def test_run_training_all_results():
    df = make_df(200)
    result = run_training(df, SAMPLE_STRATEGY)
    assert len(result["all_results"]) >= 1