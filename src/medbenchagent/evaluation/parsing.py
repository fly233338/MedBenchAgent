"""Constrained parsing of model answers into benchmark response contracts."""

from __future__ import annotations

import json
import math

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput


class PredictionParseError(ValueError):
    """Raised when a model response cannot be safely normalized."""


def _bbox_answer(raw_text: str) -> object:
    try:
        payload = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise PredictionParseError("bbox response is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"bbox"}:
        raise PredictionParseError("bbox response must contain exactly one bbox field")
    return payload["bbox"]


def parse_prediction(output: ModelOutput, sample: InferenceSample) -> str | list[float]:
    """Parse one adapter output and return the benchmark's canonical prediction type."""
    response_format = sample.task.response_format
    answer_text = output.answer_text if output.answer_text is not None else output.raw_text
    if response_format == "letter":
        options = sample.item["options"]
        assert isinstance(options, list)
        raw_answer = answer_text.strip()
        if len(raw_answer) >= 2 and raw_answer[1] == ".":
            candidate = raw_answer[0]
        else:
            candidate = raw_answer
        if len(candidate) == 1 and "A" <= candidate.upper() <= "Z":
            letter = candidate.upper()
            option_index = ord(letter) - ord("A")
            if option_index in range(len(options)):
                return letter
        raise PredictionParseError(
            "response must be one valid option letter, optionally followed by a period and text"
        )
    if response_format == "free_text":
        answer = answer_text.strip()
        if not answer:
            raise PredictionParseError("response must be non-empty text")
        return answer
    answer = _bbox_answer(answer_text)
    if (
        not isinstance(answer, list)
        or len(answer) != 4
        or not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in answer
        )
    ):
        raise PredictionParseError("answer must be four finite xyxy coordinates")
    source_width, source_height = output.coordinate_size or (
        sample.image.width,
        sample.image.height,
    )
    if source_width <= 0 or source_height <= 0:
        raise PredictionParseError("adapter returned an invalid coordinate size")
    box = [float(value) for value in answer]
    if not (0 <= box[0] < box[2] <= source_width and 0 <= box[1] < box[3] <= source_height):
        raise PredictionParseError("bbox is outside the adapter coordinate space")
    x_scale = sample.image.width / source_width
    y_scale = sample.image.height / source_height
    return [
        box[0] * x_scale,
        box[1] * y_scale,
        box[2] * x_scale,
        box[3] * y_scale,
    ]
