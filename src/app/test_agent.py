"""Unit tests for agent analysis and LLM strategy parsing."""

import pytest

from app.agent import _analyze_dataset, _build_prompt
from app.utils import parse_llm_strategy
import pandas as pd
from io import StringIO


CSV_DATA = """age,salary,churn
25,50000,0
35,80000,1
45,120000,0
"""


def get_df():
    return pd.read_csv(StringIO(CSV_DATA))


def test_analyze_dataset_shape():
    df = get_df()
    result = _analyze_dataset(df)
    assert result["rows"] == 3
    assert result["cols"] == 3


def test_analyze_detects_target_column():
    df = get_df()
    result = _analyze_dataset(df)
    assert result["target_col"] == "churn"


def test_analyze_detects_task_type_classification():
    df = get_df()
    result = _analyze_dataset(df)
    assert result["task_type"] == "classification"


def test_analyze_detects_task_type_regression():
    # Large number of unique target values = regression
    import numpy as np
    df = pd.DataFrame({
        "feature": np.random.rand(100),
        "price": np.random.rand(100) * 1000,
    })
    result = _analyze_dataset(df)
    assert result["task_type"] == "regression"


def test_build_prompt_contains_task_type():
    df = get_df()
    analysis = _analyze_dataset(df)
    prompt = _build_prompt(analysis)
    assert "classification" in prompt
    assert "churn" in prompt


def test_parse_llm_strategy_valid_json():
    raw = '{"models": ["random_forest"], "preprocessing": ["standard_scaler"], "reason": "good"}'
    result = parse_llm_strategy(raw)
    assert result["models"] == ["random_forest"]


def test_parse_llm_strategy_with_markdown_fence():
    raw = '```json\n{"models": ["xgboost"], "preprocessing": [], "reason": "ok"}\n```'
    result = parse_llm_strategy(raw)
    assert "xgboost" in result["models"]


def test_parse_llm_strategy_fallback_on_garbage():
    raw = "Sorry, I cannot help with that."
    result = parse_llm_strategy(raw)
    # Should return fallback defaults, not crash
    assert "models" in result
    assert isinstance(result["models"], list)