"""
Utilities - LLM client, response parser, and shared helpers.
"""

import json
import re
from typing import Any 
import httpx
from loguru import logger

async def call_llm(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    call ollama's local inference API with a prompt. 
    Fall back to a default strategy if ollama is unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},

            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.warning(f"Ollama API is unavailable: {e}. Using fallback strategy.")
        return _fallback_strategy()
    

def parse_llm_strategy(raw: str) -> dict:
    """
    ExExtract JSON from LLM response.
    Handles markdown code fences, extra whitespace, and partial responses.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting the first JSON-like block
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response, using fallback.")
                parsed = json.loads(_fallback_strategy())
        else:
            logger.warning("No JSON found in LLM response, using fallback.")
            parsed = json.loads(_fallback_strategy())

    return parsed


def _fallback_strategy() -> str:
    """Default strategy when LLM is unavailable or fails to parse."""
    return json.dumps({
        "models": ["random_forest", "xgboost"],
        "preprocessing": ["impute_mean", "standard_scaler"],
        "reason": "Fallback strategy: balanced defaults for unknown datasets.",
    })


class AutoMLException(Exception):
    """Base exception for local-automl-agent errors."""
    pass


class DatasetError(AutoMLException):
    """Raised when the uploaded dataset is invalid or unreadable."""
    pass


class TrainingError(AutoMLException):
    """Raised when all model training attempts fail."""
    pass
    