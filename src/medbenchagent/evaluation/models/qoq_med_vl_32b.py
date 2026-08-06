"""Adapter for the Qwen2.5-VL-compatible QoQ-Med-VL-32B checkpoint."""

from medbenchagent.evaluation.models.qoq_med_vl import QoQMedVLAdapter


class QoQMedVL32BAdapter(QoQMedVLAdapter):
    """QoQ-Med-VL-32B using the native QoQ reasoning protocol."""
