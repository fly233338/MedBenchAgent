"""Adapter for Qwen3-VL-8B-Thinking."""

from medbenchagent.evaluation.models.qwen3_vl import Qwen3VLAdapter


class Qwen3VL8BThinkingAdapter(Qwen3VLAdapter):
    """Qwen3-VL-8B-Thinking with official think-boundary answer extraction."""

    def extract_answer(self, text: str) -> str:
        marker = "</think>"
        return text.rpartition(marker)[2].strip() if marker in text else text
