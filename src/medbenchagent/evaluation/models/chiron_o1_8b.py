"""Adapter for the InternVL3-compatible Chiron-o1-8B checkpoint."""

from medbenchagent.evaluation.contracts import InferenceSample
from medbenchagent.evaluation.models.internvl import InternVLAdapter


class ChironO1_8BAdapter(InternVLAdapter):
    """Chiron-o1-8B with its official final-answer boundary."""

    def instruction_for(self, _sample: InferenceSample) -> str:
        return self.model.system_message

    def extract_answer(self, text: str) -> str:
        marker = "### The final answer is:"
        return text.rpartition(marker)[2].strip() if marker in text else text
