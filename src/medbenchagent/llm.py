"""Small OpenAI-compatible client for the deployed DeepSeek model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelUnavailable(RuntimeError):
    """Raised when the planning model is not configured or returns invalid JSON."""


def load_project_env() -> None:
    """Load the repository's optional .env without overriding process-level settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


load_project_env()


def complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Return a JSON object from the configured DeepSeek-compatible endpoint."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL")
    if not api_key or not model:
        raise ModelUnavailable("DEEPSEEK_API_KEY and DEEPSEEK_MODEL must be configured.")

    client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL"))
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ModelUnavailable("DeepSeek returned an empty response.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ModelUnavailable("DeepSeek did not return valid JSON.") from exc
    if not isinstance(result, dict):
        raise ModelUnavailable("DeepSeek JSON response must be an object.")
    return result
