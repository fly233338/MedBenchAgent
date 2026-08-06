"""Adapter for Hulu-Med-32B."""

from medbenchagent.evaluation.models.hulu_med_7b import HuluMed7BAdapter


class HuluMed32BAdapter(HuluMed7BAdapter):
    """Hulu-Med-32B using the shared Hulu-Med processor protocol."""
