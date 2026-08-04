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


def system_prompt(sample: InferenceSample) -> str:
    if sample.task.response_format == "letter":
        return "Answer with the correct single letter answer."
    return "Answer the medical image benchmark question using only the requested JSON."


def user_prompt(sample: InferenceSample) -> str:
    question = sample.item["question"]
    if sample.task.response_format == "letter":
        options = sample.item["options"]
        assert isinstance(options, list)
        rendered = "\n".join(
            f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
        )
        return f"{question}\n\n{rendered}"
    if sample.task.response_format == "bbox":
        return (
            f"{question}\n\nReturn the single target box as absolute pixel xyxy coordinates "
            'in the image shown. Return JSON only in the form {"answer":[x1,y1,x2,y2]}.'
        )
    return (
        f"{question}\n\nAnswer the question directly. "
        'Return JSON only in the form {"answer":"your answer"}.'
    )
