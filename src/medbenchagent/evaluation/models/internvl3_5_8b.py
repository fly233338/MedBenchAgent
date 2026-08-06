"""Adapter for InternVL3.5-8B."""

from medbenchagent.evaluation.contracts import InferenceSample
from medbenchagent.evaluation.models.internvl import InternVLAdapter


class InternVL3_5_8BAdapter(InternVLAdapter):
    """InternVL3.5-8B with think-boundary answer extraction."""

    def instruction_for(self, sample: InferenceSample) -> str:
        instruction = super().instruction_for(sample)
        final_only = (
            "Do not include reasoning or <think> tags. Return only the final answer and no "
            "other text."
        )
        return f"{instruction} {final_only}" if instruction else final_only

    def extract_answer(self, text: str) -> str:
        marker = "</think>"
        return text.rpartition(marker)[2].strip() if marker in text else text
