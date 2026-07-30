"""Deterministic item and task metrics for generated benchmark responses."""

from __future__ import annotations

from collections.abc import Sequence

import sacrebleu

from medbenchagent.evaluation.contracts import InferenceSample


def _bleu_tokenizer(language: str) -> str:
    return "zh" if language == "zh" else "13a"


def sentence_bleu(prediction: str, reference: str, language: str) -> float:
    return sacrebleu.sentence_bleu(
        prediction,
        [reference],
        tokenize=_bleu_tokenizer(language),
    ).score / 100.0


def corpus_bleu(predictions: Sequence[str], references: Sequence[str], language: str) -> float:
    if not predictions:
        return 0.0
    return sacrebleu.corpus_bleu(
        list(predictions),
        [list(references)],
        tokenize=_bleu_tokenizer(language),
    ).score / 100.0


def bbox_iou(prediction: Sequence[float], reference: Sequence[float]) -> float:
    left = max(prediction[0], reference[0])
    top = max(prediction[1], reference[1])
    right = min(prediction[2], reference[2])
    bottom = min(prediction[3], reference[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    reference_area = (reference[2] - reference[0]) * (reference[3] - reference[1])
    union = prediction_area + reference_area - intersection
    return intersection / union if union > 0 else 0.0


def option_value(item: dict[str, object], letter: object) -> str | None:
    options = item.get("options")
    if (
        not isinstance(options, list)
        or not isinstance(letter, str)
        or len(letter) != 1
        or not letter.isalpha()
    ):
        return None
    index = ord(letter.upper()) - ord("A")
    if index not in range(len(options)) or not isinstance(options[index], str):
        return None
    return options[index]


def item_score(
    sample: InferenceSample,
    prediction: str | list[float] | None,
    language: str,
) -> float:
    if prediction is None:
        return 0.0
    if sample.task.response_format == "letter":
        return float(option_value(sample.item, prediction) == option_value(sample.item, sample.item["answer"]))
    if sample.task.response_format == "free_text":
        assert isinstance(prediction, str) and isinstance(sample.item["answer"], str)
        return sentence_bleu(prediction, sample.item["answer"], language)
    assert isinstance(prediction, list) and isinstance(sample.item["answer"], list)
    return bbox_iou(prediction, sample.item["answer"])


def item_correct(
    sample: InferenceSample,
    score: float,
    *,
    bleu_threshold: float,
    bbox_iou_threshold: float,
) -> bool:
    if sample.task.response_format == "letter":
        return score == 1.0
    if sample.task.response_format == "free_text":
        return score >= bleu_threshold
    return score >= bbox_iou_threshold


def macro_f1(gold: Sequence[str], predicted: Sequence[str | None]) -> float:
    labels = sorted(set(gold))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)
