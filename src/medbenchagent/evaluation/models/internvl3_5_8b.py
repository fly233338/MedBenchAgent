"""Adapter for InternVL3.5-8B."""

from medbenchagent.evaluation.contracts import InferenceSample
from medbenchagent.evaluation.models.internvl import InternVLAdapter


INTERNVL35_REASONING_SYSTEM_PROMPT = (
    "You are an AI assistant that follows this response protocol exactly. First, reason "
    "through the problem and enclose the entire reasoning process within <think> and "
    "</think> tags. Then provide the final answer within <answer> and </answer> tags. Do "
    "not output any text outside these tags."
)


class InternVL3_5_8BAdapter(InternVLAdapter):
    """InternVL3.5-8B with explicit reasoning and final-answer boundaries."""

    def instruction_for(self, _sample: InferenceSample) -> str:
        return INTERNVL35_REASONING_SYSTEM_PROMPT

    def prompt_for(self, sample: InferenceSample) -> str:
        if sample.task.response_format == "letter":
            options = sample.item["options"]
            assert isinstance(options, list)
            final_format = (
                "exactly one valid option letter from A through "
                f"{chr(ord('A') + len(options) - 1)}"
            )
        elif sample.task.response_format == "free_text":
            final_format = "only the concise final answer"
        else:
            final_format = (
                'only a JSON object containing the single bounding box under the "bbox" key '
                "as four absolute xyxy coordinates"
            )
        return (
            f"{super().prompt_for(sample)}\n\n"
            f"Inside <answer> and </answer>, put {final_format}. Always complete both the "
            "<think>...</think> and <answer>...</answer> sections."
        )

    def generation_config(self, max_new_tokens: int) -> dict[str, object]:
        config = super().generation_config(max_new_tokens)
        config["pad_token_id"] = self.tokenizer.eos_token_id
        return config

    def extract_answer(self, text: str) -> str:
        answer_start = text.rfind("<answer>")
        if answer_start >= 0:
            payload_start = answer_start + len("<answer>")
            answer_end = text.find("</answer>", payload_start)
            if answer_end < 0 or text[answer_end + len("</answer>") :].strip():
                return ""
            return text[payload_start:answer_end].strip()
        if "</answer>" in text:
            return ""
        marker = "</think>"
        return text.rpartition(marker)[2].strip() if marker in text else text
