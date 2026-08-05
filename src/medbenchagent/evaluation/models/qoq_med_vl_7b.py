"""Adapter for the Qwen2.5-VL-compatible QoQ-Med-VL-7B checkpoint."""

from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


class QoQMedVL7BAdapter(Qwen25VLAdapter):
    """QoQ-Med-VL-7B using the shared Qwen2.5-VL implementation."""
