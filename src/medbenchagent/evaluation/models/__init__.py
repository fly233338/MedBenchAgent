"""Explicit registry of supported Medical VLM adapters."""

from __future__ import annotations

from typing import Any

from medbenchagent.evaluation.models.base import ModelAdapter
from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


ADAPTERS = {"qwen2_5_vl": Qwen25VLAdapter}


def load_model_adapter(name: str, model_path: str, **kwargs: Any) -> ModelAdapter:
    try:
        adapter = ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model adapter {name!r}; choose one of: {', '.join(sorted(ADAPTERS))}."
        ) from exc
    return adapter(model_path, **kwargs)
