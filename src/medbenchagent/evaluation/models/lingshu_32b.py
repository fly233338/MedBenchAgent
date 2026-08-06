"""Adapter for the Qwen2.5-VL-compatible Lingshu-32B checkpoint."""

from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


class Lingshu32BAdapter(Qwen25VLAdapter):
    """Lingshu-32B using the shared Qwen2.5-VL implementation."""
