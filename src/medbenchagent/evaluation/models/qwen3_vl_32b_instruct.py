"""Adapter for Qwen3-VL-32B-Instruct."""

from medbenchagent.evaluation.models.qwen3_vl import Qwen3VLAdapter


class Qwen3VL32BInstructAdapter(Qwen3VLAdapter):
    """Qwen3-VL-32B-Instruct using the shared Qwen3-VL implementation."""
