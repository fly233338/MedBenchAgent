"""Explicit registry of supported Medical VLM adapters."""

from __future__ import annotations

from typing import Any

from medbenchagent.evaluation.models.base import ModelAdapter
from medbenchagent.evaluation.models.chiron_o1_8b import ChironO1_8BAdapter
from medbenchagent.evaluation.models.hulu_med_7b import HuluMed7BAdapter
from medbenchagent.evaluation.models.internvl3_8b import InternVL3_8BAdapter
from medbenchagent.evaluation.models.lingshu_7b import Lingshu7BAdapter
from medbenchagent.evaluation.models.qoq_med_vl_7b import QoQMedVL7BAdapter
from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter
from medbenchagent.evaluation.models.qwen2_5_vl_7b_instruct import Qwen25VL7BInstructAdapter
from medbenchagent.evaluation.models.qwen3_vl_8b_instruct import Qwen3VL8BInstructAdapter
from medbenchagent.evaluation.models.qwen3_vl_8b_thinking import Qwen3VL8BThinkingAdapter


ADAPTERS = {
    "qwen2_5_vl": Qwen25VLAdapter,
    "qwen2_5_vl_7b_instruct": Qwen25VL7BInstructAdapter,
    "internvl3_8b": InternVL3_8BAdapter,
    "qwen3_vl_8b_instruct": Qwen3VL8BInstructAdapter,
    "qwen3_vl_8b_thinking": Qwen3VL8BThinkingAdapter,
    "lingshu_7b": Lingshu7BAdapter,
    "hulu_med_7b": HuluMed7BAdapter,
    "qoq_med_vl_7b": QoQMedVL7BAdapter,
    "chiron_o1_8b": ChironO1_8BAdapter,
}


def load_model_adapter(name: str, model_path: str, **kwargs: Any) -> ModelAdapter:
    try:
        adapter = ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model adapter {name!r}; choose one of: {', '.join(sorted(ADAPTERS))}."
        ) from exc
    return adapter(model_path, **kwargs)
