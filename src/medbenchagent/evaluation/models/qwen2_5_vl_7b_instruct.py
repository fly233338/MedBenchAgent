"""Adapter registration class for Qwen2.5-VL-7B-Instruct."""

from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


class Qwen25VL7BInstructAdapter(Qwen25VLAdapter):
    """Qwen2.5-VL-7B-Instruct using the shared Qwen2.5-VL implementation."""
