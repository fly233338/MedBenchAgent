"""Adapter for the Qwen2.5-VL-compatible Lingshu-7B checkpoint."""

from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


class Lingshu7BAdapter(Qwen25VLAdapter):
    """Lingshu-7B using the shared Qwen2.5-VL implementation."""
