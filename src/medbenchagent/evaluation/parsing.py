"""Constrained parsing of model answers into benchmark response contracts."""

from __future__ import annotations

import math
import re

from medbenchagent.evaluation.contracts import InferenceSample, ModelOutput


class PredictionParseError(ValueError):
    """Raised when a model response cannot be safely normalized."""


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_BBOX_FIELD = re.compile(
    r'(?i)(?<![A-Za-z0-9_])(?:["\']?bbox(?:_2d)?["\']?)(?![A-Za-z0-9_])'
    r"\s*[:=]\s*(\[[^\[\]]*\]|\([^()]*\))"
)
_JSON_FENCE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.IGNORECASE | re.DOTALL)
_PARENTHESIZED_LETTER = re.compile(
    r"\A\s*(?:\(([A-Za-z])(?:[.．]\s*.*?)?\)|（([A-Za-z])(?:[.．]\s*.*?)?）)(?:\s|\Z)",
    re.DOTALL,
)


def _coordinate_candidate(value: str) -> tuple[float, float, float, float] | None:
    stripped = value.strip()
    if len(stripped) < 2 or (stripped[0], stripped[-1]) not in {("[", "]"), ("(", ")")}:
        return None
    parts = [part.strip() for part in stripped[1:-1].split(",")]
    if len(parts) != 4 or any(re.fullmatch(_NUMBER, part) is None for part in parts):
        return None
    coordinates = tuple(float(part) for part in parts)
    if not all(math.isfinite(part) for part in coordinates):
        return None
    return coordinates


def _bbox_answer(raw_text: str) -> list[float]:
    candidates = {
        candidate
        for match in _BBOX_FIELD.finditer(raw_text)
        if (candidate := _coordinate_candidate(match.group(1))) is not None
    }
    stripped = raw_text.strip()
    fence = _JSON_FENCE.fullmatch(stripped)
    if fence is not None:
        stripped = fence.group(1).strip()
    bare_candidate = _coordinate_candidate(stripped)
    if bare_candidate is not None:
        candidates.add(bare_candidate)
    if len(candidates) != 1:
        raise PredictionParseError("bbox response must contain exactly one distinct bounding box")
    return list(candidates.pop())


def _letter_candidate(raw_answer: str) -> str:
    """Extract the optional wrapper used by some letter-answer models."""
    wrapped = _PARENTHESIZED_LETTER.match(raw_answer)
    if wrapped is not None:
        return wrapped.group(1) or wrapped.group(2) or ""
    if len(raw_answer) >= 2 and raw_answer[1] == ".":
        return raw_answer[0]
    return raw_answer


def parse_prediction(output: ModelOutput, sample: InferenceSample) -> str | list[float]:
    """Parse one adapter output and return the benchmark's canonical prediction type."""
    response_format = sample.task.response_format
    answer_text = output.answer_text if output.answer_text is not None else output.raw_text
    if response_format == "letter":
        options = sample.item["options"]
        assert isinstance(options, list)
        raw_answer = answer_text.strip()
        candidate = _letter_candidate(raw_answer)
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
        raise PredictionParseError(
            f"bbox {box} is outside the adapter coordinate space "
            f"{source_width}x{source_height}"
        )
    x_scale = sample.image.width / source_width
    y_scale = sample.image.height / source_height
    return [
        min(box[0] * x_scale, float(sample.image.width)),
        min(box[1] * y_scale, float(sample.image.height)),
        min(box[2] * x_scale, float(sample.image.width)),
        min(box[3] * y_scale, float(sample.image.height)),
    ]
