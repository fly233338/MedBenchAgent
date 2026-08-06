"""Adapter for InternVL3.5-8B."""

from medbenchagent.evaluation.contracts import InferenceSample
from medbenchagent.evaluation.models.internvl import InternVLAdapter


class InternVL3_5_8BAdapter(InternVLAdapter):
    """InternVL3.5-8B with think-boundary answer extraction."""

    def prompt_for(self, sample: InferenceSample) -> str:
        return (
            f"{super().prompt_for(sample)}\n\n"
            "Do not include reasoning or <think> tags. Return only the final answer and no "
            "other text."
        )

    def generation_config(self, max_new_tokens: int) -> dict[str, object]:
        config = super().generation_config(max_new_tokens)
        config["pad_token_id"] = self.tokenizer.eos_token_id
        return config

    def extract_answer(self, text: str) -> str:
        marker = "</think>"
        return text.rpartition(marker)[2].strip() if marker in text else text
