"""Constrained parsing of model answers into benchmark response contracts."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput


class PredictionParseError(ValueError):
    """Raised when a model response cannot be safely normalized."""


def _json_answer(raw_text: str) -> Any:
    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PredictionParseError("response is not valid answer JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        raise PredictionParseError("response must contain exactly one answer field")
    return payload["answer"]


def parse_prediction(output: ModelOutput, sample: InferenceSample) -> str | list[float]:
    """Parse one adapter output and return the benchmark's canonical prediction type."""
    response_format = sample.task.response_format
    if response_format == "letter":
        options = sample.item["options"]
        assert isinstance(options, list)
        raw_answer = output.raw_text.strip()
        if len(raw_answer) == 1 and raw_answer.isalpha():
            letter = raw_answer.upper()
            if ord(letter) - ord("A") in range(len(options)):
                return letter
        raise PredictionParseError("response must be exactly one valid option letter")
    answer = _json_answer(output.raw_text)
    if response_format == "free_text":
        if not isinstance(answer, str) or not answer.strip():
            raise PredictionParseError("answer must be a non-empty string")
        return answer.strip()
    if (
        not isinstance(answer, list)
        or len(answer) != 4
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in answer)
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
