"""
Trainer module -ML Pipeline execution.

Runs preprocessing and model training based on LLM-chosen strategy.
REturns matrics for comparision acress models and best model details.
"""

import time
from typing import Any, Dict
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor


#Model registry - Maps LLM strategy names to sklearn classes  
MODEL_REGISTRY = {
    "classification": {
        "random_forest": RandomForestClassifier (n_estimators=100, random_state=42),
        "xgboost": XGBClassifier (n_estimators=100, random_state=42, eval_metric='logloss'),
        "logistic_regression": LogisticRegression()
},
}

def run_trainng(df: pd.DataFrame, strategy: dict) -> dict:
    """
    Main training entry point.
    Runs preprocessing + trainng for each model in strategy.
    Returns comparison results and best model info.
    """
    task = strategy.get("task_type", "classification")
    target_col = strategy.get("target_col")
    model_names = strategy.get("models", ["random_forest"])
    preprocessing = strategy.get("preprocessing", ["standard_scaler"])

    #split features and target
    x= df.drop(columns=[target_col])
    y= df[target_col]

    #Encode target if classification
    label_enc = None
    if task == "classification" and y.dtype == object:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)

        #train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    #building preprocessing pipeline steps
    preprocessing_steps = _build_preprocessing(preprocessing)

    results = []
    for model_name in model_names:
        model = MODEL_REGISTRY.get(task, {}).get(model_name)
        if model is None:
            logger.warning(f"Model {model_name} not found for task {task}. Skipping.")
            continue

        start = time.time()
        Pipeline = Pipeline([*preprocessing_steps,("model", model)])

        try:
            Pipeline.fit(x_train, y_train)
            preds = Pipeline.predict(x_test)
            elapsed = round(time.time() - start, 2)

            metrics = _compute_metrics(task, y_test, preds)
            metrics["model"] = model_name
            metrics["train_time_sec"] = elapsed
            results.append(metrics)
            logger.info(f"Trained {model_name} in {elapsed} sec with metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")

    if not results:
        raise RuntimeError("All models failed duuring trainng.")
    #Pick best model
    score_key = "accuracy" if task == "classification" else "r2_score"
    best = max(results, key=lambda x: x.get(score_key, 0))

    return {
        "task_type": task,
        "target_col": target_col,
        "best_model": best["model"],
        "best_score": best.get(score_key, 0),
        "all_results": results
    }


def _build_preprocessing(steps: list) -> list[tuple]:
    """ Convert LLM-chosen preprocessing step names into sklearn transformers tuples.  """
    pipeline_steps = []

    if "impute_mean" in steps:
        pipeline_steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif "impute_median" in steps:
        pipeline_steps.append(("imputer", SimpleImputer(strategy="median")))
    
    if "standard_scaler" in steps:
        pipeline_steps.append(("scaler", StandardScaler()))
    elif "minmax_scaler" in steps:
        pipeline_steps.append(("scaler", MinMaxScaler()))

    return pipeline_steps

def _compute_metrics(task: str, y_true: Any, y_pred: Any) -> dict:
    """ Compute relevant metrics based on task type. """
    if task == "classification":
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0),4),
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        return{
            "rmse": round(np.sqrt(mse), 4),
            "r2_score": round(r2_score(y_true, y_pred), 4)
        }