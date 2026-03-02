"""
Agent module - LLM orchestration layer.

Flow:
 start_job(csv_bytes) -> job_id
    ->analyze_dataset(df)
    ->llm_decides_strategya(analysis)
    ->run_pipeline_async(job_id, df, strategy)
"""
import asyncio
import uuid
from io import StringIO
import pandas as pd
from loggur import logger
from app.store import JobStore,JobStatus
from app.trainer import run_training 
from app.utils import call_llm, parse_llm_strategy


def start_job(csv_bytes: bytes) -> str:
    """
    Entry point: accepts raw CSV bytes, registers a job,
    and kicks off the async pipeline.
    Returns:
        job_id (str): Unique identifier for the job.
 
    """
    job_id = str(uuid.uuid4()[:8])  # Shorten for readability
    store = JobStore.get_instance()
    store.create_job(job_id)

    # READ CSV
    try:
        df = pd.read_csv(StringIO(csv_bytes.decode('utf-8')))
    except Exception as e:
        store.fail(job_id, f"CSV parsing error: {str(e)}")
        return job_id
    
    logger.info(f"Job [{job_id}]: CSV parsed successfully with shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Start async pipeline in background 
    asyncio.create_task(_pipeline(job_id, df))

    return job_id

async def _pipeline(job_id: str, df: pd.DataFrame) ->None:
    """ Full async pipeline: analyze -> LLM decides -> train -> store results. """
    store - JobStore.get_instance()
    try:
        # Step 1: Analyze dataset
        store.update(job_id, JobStatus.ANALYZING, progress=10)
        analysis = analyze_dataset(df)
        logger.info(f"Job [{job_id}]: Dataset analysis: {analysis}")

        # Step 2: LLM decides strategy
        store.update(job_id, JobStatus.STRATEGIZING, progress=25)
        raw_strategy = await call_llm(_build_prompt(analysis))
        strategy = parse_llm_strategy(raw_strategy)
        logger.info(f"Job [{job_id}]: LLM strategy: {strategy}")

        # Step 3: Run training pipeline
        store.update(job_id, JobStatus.TRAINING, progress=50)
        results = await asyncio.to_thread(run_training, df, strategy)
        logger.info(f"Job [{job_id}]: Training completed with results: {results}")

        # Step 4: Store results
        store.complete(job_id, results)
        logger.success(f"Job [{job_id}]: Completed Best model: {results['best_model']}, "
                       f"Score: {results['best_score']:.4f}")
    except Exception as e:
        store.fail(job_id, f"Pipeline error: {str(e)}")
        logger.error(f"Job [{job_id}]: Pipeline failed with error: {str(e)}")

def analyze_dataset(df: pd.DataFrame) -> dict:
    """ Light weight dataset profiler.
    Detects : target column, task type (classification/regression), numeric/categorical features, null rates.
    """    
    #Heuristic: target is last column
    target_col = df.columns[-1]
    unique_col = df[target_col].nunique()
    task_type = "classification" if unique_col <= 20 else "regression"
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    null_rates = (df.isnull().sum() / len(df)).to_dict()

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "target_col": target_col,
        "task_type": task_type,
        "unique_target_vals": unique_vals,
        "numeric_features": [c for c in numeric_cols if c != target_col],
        "categorical_features": [c for c in categorical_cols if c != target_col],
        "null_rates": null_rates,
    }

def _build_prompt(analysis: dict) -> str:
    """ Constructs the LLM prompt from dataset analysis. """
    return f"""
You are an AutoML assistant. Given the following dataset analysis, 
recommend the best ML strategy. Be concise and structured.

Dataset Info:
- Rows: {analysis['rows']}
- Columns: {analysis['cols']}
- Task type: {analysis['task_type']}
- Target column: {analysis['target_col']}
- Numeric features: {analysis['numeric_features']}
- Categorical features: {analysis['categorical_features']}
- Null rates (high nulls > 0.3 need imputation): {analysis['null_rates']}

Respond ONLY in this JSON format:
{{
  "models": ["model1", "model2"],  
  "preprocessing": ["step1", "step2"],
  "reason": "one sentence explanation"
}}

Valid model names: random_forest, xgboost, logistic_regression, linear_regression, svm
Valid preprocessing steps: standard_scaler, minmax_scaler, label_encoder, 
                           one_hot_encoder, impute_mean, impute_median, drop_nulls
""".strip()

