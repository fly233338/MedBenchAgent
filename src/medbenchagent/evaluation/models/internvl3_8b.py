"""Adapter for InternVL3-8B."""

from medbenchagent.evaluation.models.internvl import InternVLAdapter


class InternVL3_8BAdapter(InternVLAdapter):
    """InternVL3-8B with a bounded dynamic-image tile budget."""

    max_dynamic_tiles = 6
