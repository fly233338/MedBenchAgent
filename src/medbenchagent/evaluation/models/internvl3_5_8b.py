"""Adapter for InternVL3.5-8B."""

from medbenchagent.evaluation.models.internvl import InternVLAdapter


class InternVL3_5_8BAdapter(InternVLAdapter):
    """InternVL3.5-8B with think-boundary answer extraction."""

    def extract_answer(self, text: str) -> str:
        marker = "</think>"
        return text.rpartition(marker)[2].strip() if marker in text else text
