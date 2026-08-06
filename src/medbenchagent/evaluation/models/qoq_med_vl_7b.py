"""Adapter for the Qwen2.5-VL-compatible QoQ-Med-VL-7B checkpoint."""

from medbenchagent.evaluation.models.qoq_med_vl import QoQMedVLAdapter


class QoQMedVL7BAdapter(QoQMedVLAdapter):
    """QoQ-Med-VL-7B using the native QoQ reasoning protocol."""
