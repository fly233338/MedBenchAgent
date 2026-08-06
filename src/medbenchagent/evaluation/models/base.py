"""Common model adapter interface and response-format prompts."""

from __future__ import annotations

from typing import Protocol, Sequence

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput


class ModelAdapter(Protocol):
    def generate(
        self,
        samples: Sequence[InferenceSample],
        *,
        max_new_tokens: int,
    ) -> list[ModelOutput]:
        """Generate one raw model output per input sample."""


def system_prompt(sample: InferenceSample) -> str | None:
    if sample.task.response_format == "letter":
        return (
            "Answer with only the correct option letter. Do not include reasoning, "
            "punctuation, tags, or any other text."
        )
    if sample.task.response_format == "bbox":
        return (
            'Answer with only a JSON object containing the single bounding box under the "bbox" '
            "key as four absolute pixel coordinates in x1, y1, x2, y2 order."
        )
    return None


def user_prompt(sample: InferenceSample) -> str:
    question = sample.item["question"]
    if sample.task.response_format == "letter":
        options = sample.item["options"]
        assert isinstance(options, list)
        rendered = "\n".join(
            f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
        )
        return f"{question}\n\n{rendered}"
    return question
