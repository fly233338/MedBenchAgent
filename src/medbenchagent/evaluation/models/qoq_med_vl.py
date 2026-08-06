"""Shared native reasoning protocol for QoQ-Med-VL checkpoints."""

from medbenchagent.evaluation.contracts import InferenceSample
from medbenchagent.evaluation.models.qwen2_5_vl import Qwen25VLAdapter


QOQ_REASONING_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide "
    "the final answer. Before analyzing medical images, identify and outline all objects of "
    'interest relevant to diagnosis as JSON with a list of bounding boxes under the "bbox_2d" '
    "key in [x1, y1, x2, y2] format, wrapped in ```json ... ``` tags. The reasoning process "
    "MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in "
    "\\boxed{}."
)


def extract_boxed_answer(text: str) -> str:
    """Extract the final balanced LaTeX boxed payload, including nested JSON braces."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text
    payload_start = start + len(marker)
    depth = 1
    for index in range(payload_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[payload_start:index].strip()
    return ""


class QoQMedVLAdapter(Qwen25VLAdapter):
    """Qwen2.5-VL-compatible QoQ inference with its native reasoning format."""

    def instruction_for(self, _sample: InferenceSample) -> None:
        return None

    def prompt_for(self, sample: InferenceSample) -> str:
        final_format = {
            "letter": "Inside \\boxed{}, put exactly one valid option letter.",
            "free_text": "Inside \\boxed{}, put only the final free-text answer.",
            "bbox": (
                'Inside \\boxed{}, put only a JSON object with the single "bbox" key and four '
                "absolute xyxy coordinates."
            ),
        }[sample.task.response_format]
        return f"{super().prompt_for(sample)}\n\n{QOQ_REASONING_PROMPT} {final_format}"

    def extract_answer(self, text: str) -> str:
        return extract_boxed_answer(text)
