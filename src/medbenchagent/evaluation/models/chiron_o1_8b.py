"""Adapter for the InternVL3-compatible Chiron-o1-8B checkpoint."""

from medbenchagent.evaluation.contracts import InferenceSample
from medbenchagent.evaluation.models.internvl import InternVLAdapter


class ChironO1_8BAdapter(InternVLAdapter):
    """Chiron-o1-8B with its official final-answer boundary."""

    def instruction_for(self, _sample: InferenceSample) -> str:
        return self.model.system_message

    def prompt_for(self, sample: InferenceSample) -> str:
        final_format = {
            "letter": "one valid option letter and nothing else",
            "free_text": "the concise final answer and nothing else",
            "bbox": (
                'only a JSON object containing the single bounding box under the "bbox" key '
                "as four xyxy coordinates normalized to the fixed [0, 1000] range on each axis"
            ),
        }[sample.task.response_format]
        return (
            f"{super().prompt_for(sample)}\n\n"
            "Let's reason step-by-step to answer the above question. End the response with the "
            f"exact marker ### The final answer is: followed by {final_format}."
        )

    def generation_config(self, max_new_tokens: int) -> dict[str, object]:
        config = super().generation_config(max_new_tokens)
        config["pad_token_id"] = self.tokenizer.eos_token_id
        return config

    def extract_answer(self, text: str) -> str:
        # Some Chiron checkpoints omit the requested Markdown heading prefix.
        marker = "The final answer is:"
        return text.rpartition(marker)[2].strip() if marker in text else text
