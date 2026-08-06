"""Adapter for the Qwen2.5-VL-compatible QoQ-Med-VL-32B checkpoint."""

from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


class QoQMedVL32BAdapter(Qwen25VLAdapter):
    """QoQ-Med-VL-32B using the shared Qwen2.5-VL implementation."""
