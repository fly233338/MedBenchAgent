"""Aggregate a self-contained Medical VLM inference file into an evaluation report."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medbenchagent.evaluation.contracts import ImageEntry, InferenceSample, TaskSpec
from medbenchagent.evaluation.data import EvaluationDataError, METRICS_BY_FORMAT, file_sha256
from medbenchagent.evaluation.io import atomic_write_json
from medbenchagent.evaluation.scoring import (
    corpus_bleu,
    item_correct,
    item_score,
    macro_f1,
    option_value,
)


def _load_inference(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationDataError(f"Inference file could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"Inference file is not valid JSON: {path}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "metadata", "items"}
        or payload["schema_version"] != "medbench-inference.v1"
        or not isinstance(payload["metadata"], dict)
        or not isinstance(payload["items"], list)
    ):
        raise EvaluationDataError("Inference file does not match medbench-inference.v1.")
    return payload


def _sample_from_result(result: dict[str, Any]) -> InferenceSample:
    item, task, image = result.get("item"), result.get("task"), result.get("image")
    if not isinstance(item, dict) or not isinstance(task, dict) or not isinstance(image, dict):
        raise EvaluationDataError("Inference result is missing item, task, or image metadata.")
    task_id = item.get("task_id")
    category, response_format, metric = (
        task.get("category"),
        task.get("response_format"),
        task.get("metric"),
    )
    width, height = image.get("width"), image.get("height")
    if (
        not isinstance(task_id, str)
        or not isinstance(category, str)
        or response_format not in {"letter", "free_text", "bbox"}
        or metric not in METRICS_BY_FORMAT[response_format]
        or not isinstance(width, int)
        or width <= 0
        or not isinstance(height, int)
        or height <= 0
    ):
        raise EvaluationDataError("Inference result has invalid task or image metadata.")
    return InferenceSample(
        index=result.get("index"),
        item=item,
        task=TaskSpec(task_id, category, response_format, metric),
        image=ImageEntry(
            image_id=str(item.get("image_id", "")),
            png_path=Path("."),
            relative_png_path=str(image.get("png_path", "")),
            width=width,
            height=height,
            sha256="",
        ),
    )


def _prediction(result: dict[str, Any], sample: InferenceSample) -> str | list[float] | None:
    prediction, error = result.get("prediction"), result.get("error")
    if error is not None:
        if (
            prediction is not None
            or not isinstance(error, dict)
            or not isinstance(error.get("code"), str)
            or not isinstance(error.get("message"), str)
        ):
            raise EvaluationDataError("Inference error result has an invalid contract.")
        return None
    if sample.task.response_format in {"letter", "free_text"}:
        if not isinstance(prediction, str):
            raise EvaluationDataError("Inference result has an invalid text prediction.")
        if sample.task.response_format == "letter" and option_value(sample.item, prediction) is None:
            raise EvaluationDataError("Inference result has an invalid option prediction.")
        return prediction
    if (
        not isinstance(prediction, list)
        or len(prediction) != 4
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in prediction)
        or not (0 <= prediction[0] < prediction[2] <= sample.image.width)
        or not (0 <= prediction[1] < prediction[3] <= sample.image.height)
    ):
        raise EvaluationDataError("Inference result has an invalid bbox prediction.")
    return [float(value) for value in prediction]


def _thresholds(
    metadata: dict[str, Any],
    bleu_override: float | None,
    bbox_override: float | None,
) -> tuple[float, float, bool]:
    correctness = metadata.get("correctness")
    if not isinstance(correctness, dict):
        raise EvaluationDataError("Inference metadata is missing correctness thresholds.")
    embedded_bleu = correctness.get("bleu_threshold")
    embedded_bbox = correctness.get("bbox_iou_threshold")
    if (
        not isinstance(embedded_bleu, (int, float))
        or not 0 <= embedded_bleu <= 1
        or not isinstance(embedded_bbox, (int, float))
        or not 0 <= embedded_bbox <= 1
    ):
        raise EvaluationDataError("Inference metadata has invalid correctness thresholds.")
    if bleu_override is not None and not 0 <= bleu_override <= 1:
        raise EvaluationDataError("BLEU threshold override must be between 0 and 1.")
    if bbox_override is not None and not 0 <= bbox_override <= 1:
        raise EvaluationDataError("BBox IoU threshold override must be between 0 and 1.")
    overridden = bleu_override is not None or bbox_override is not None
    return (
        float(embedded_bleu if bleu_override is None else bleu_override),
        float(embedded_bbox if bbox_override is None else bbox_override),
        overridden,
    )


def _task_score(
    metric: str,
    rows: list[dict[str, Any]],
    language: str,
) -> float:
    if metric in {"accuracy", "miou"}:
        return sum(row["score"] for row in rows) / len(rows)
    if metric == "macro_f1":
        score = _task_macro_f1(rows)
        if score is None:
            raise EvaluationDataError("Macro-F1 requires a letter task.")
        return score
    if metric == "bleu":
        predictions = [
            row["prediction"] if isinstance(row["prediction"], str) else "" for row in rows
        ]
        references = [str(row["sample"].item["answer"]) for row in rows]
        return corpus_bleu(predictions, references, language)
    raise EvaluationDataError(f"Unsupported metric implementation: {metric}")


def _task_macro_f1(rows: list[dict[str, Any]]) -> float | None:
    if rows[0]["sample"].task.response_format != "letter":
        return None
    gold = [option_value(row["sample"].item, row["sample"].item["answer"]) for row in rows]
    predicted = [option_value(row["sample"].item, row["prediction"]) for row in rows]
    if not all(isinstance(value, str) for value in gold):
        raise EvaluationDataError("Letter task has an invalid gold option.")
    return macro_f1(gold, predicted)


def evaluate_inference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    bleu_threshold: float | None = None,
    bbox_iou_threshold: float | None = None,
) -> str:
    source_path = Path(input_path)
    payload = _load_inference(source_path)
    metadata, raw_results = payload["metadata"], payload["items"]
    language = metadata.get("language")
    if language not in {"en", "zh"}:
        raise EvaluationDataError("Inference metadata has an invalid language.")
    expected_count = metadata.get("inputs", {}).get("item_count")
    if expected_count != len(raw_results):
        raise EvaluationDataError("Inference item count does not match metadata.")
    bleu, bbox, thresholds_overridden = _thresholds(
        metadata,
        bleu_threshold,
        bbox_iou_threshold,
    )

    rows = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    error_counts: Counter[str] = Counter()
    for result in raw_results:
        if not isinstance(result, dict) or not isinstance(result.get("index"), int):
            raise EvaluationDataError("Inference result must be an indexed JSON object.")
        sample = _sample_from_result(result)
        item_id = sample.item.get("id")
        if (
            sample.index in seen_indices
            or not isinstance(item_id, str)
            or item_id in seen_ids
        ):
            raise EvaluationDataError("Inference results contain duplicate indices or item ids.")
        prediction = _prediction(result, sample)
        raw_prediction = result.get("raw_prediction")
        if not isinstance(raw_prediction, str):
            raise EvaluationDataError(f"{item_id}: raw prediction must be text.")
        score = item_score(sample, prediction, language)
        correct = item_correct(
            sample,
            score,
            bleu_threshold=bleu,
            bbox_iou_threshold=bbox,
        )
        if result.get("error") is not None:
            error_counts[result["error"]["code"]] += 1
        recorded_score = result.get("item_score")
        recorded_correct = result.get("correct")
        if (
            not isinstance(recorded_score, (int, float))
            or isinstance(recorded_score, bool)
            or not math.isfinite(recorded_score)
            or not isinstance(recorded_correct, bool)
        ):
            raise EvaluationDataError(f"{item_id}: recorded item score or correctness is invalid.")
        if not thresholds_overridden:
            if (
                not math.isclose(recorded_score, score, rel_tol=1e-9, abs_tol=1e-9)
                or recorded_correct is not correct
            ):
                raise EvaluationDataError(
                    f"{item_id}: recorded item score or correctness is inconsistent."
                )
        rows.append(
            {
                "sample": sample,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "score": score,
                "correct": correct,
                "error": result.get("error"),
            }
        )
        seen_indices.add(sample.index)
        seen_ids.add(item_id)
    if seen_indices != set(range(len(rows))):
        raise EvaluationDataError("Inference result indices are not contiguous.")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["sample"].task.task_id].append(row)
    tasks = {}
    metric_scores: dict[str, list[float]] = defaultdict(list)
    for task_id in sorted(by_task):
        task_rows = by_task[task_id]
        task = task_rows[0]["sample"].task
        if any(row["sample"].task != task for row in task_rows):
            raise EvaluationDataError(f"{task_id}: inconsistent task metadata in inference results.")
        score = _task_score(task.metric, task_rows, language)
        task_macro_f1 = _task_macro_f1(task_rows)
        correct_count = sum(row["correct"] for row in task_rows)
        tasks[task_id] = {
            "category": task.category,
            "response_format": task.response_format,
            "metric": task.metric,
            "count": len(task_rows),
            "score": score,
            "macro_f1": task_macro_f1,
            "correct": correct_count,
            "correct_rate": correct_count / len(task_rows),
            "errors": sum(row["error"] is not None for row in task_rows),
        }
        metric_scores[task.metric].append(score)

    categories: dict[str, dict[str, int | float | None]] = {}
    for category in sorted({row["sample"].task.category for row in rows}):
        category_rows = [row for row in rows if row["sample"].task.category == category]
        correct_count = sum(row["correct"] for row in category_rows)
        category_task_f1s = [
            task["macro_f1"]
            for task in tasks.values()
            if task["category"] == category and task["macro_f1"] is not None
        ]
        categories[category] = {
            "count": len(category_rows),
            "correct": correct_count,
            "correct_rate": correct_count / len(category_rows),
            "macro_f1": (
                sum(category_task_f1s) / len(category_task_f1s) if category_task_f1s else None
            ),
        }
    correct_total = sum(row["correct"] for row in rows)
    task_f1s = [task["macro_f1"] for task in tasks.values() if task["macro_f1"] is not None]
    report = {
        "schema_version": "medbench-evaluation.v1",
        "metadata": {
            "inference_sha256": file_sha256(source_path),
            "benchmark_id": metadata.get("benchmark_id"),
            "dataset_id": metadata.get("dataset_id"),
            "model": metadata.get("model"),
            "thresholds": {
                "bleu": bleu,
                "bbox_iou": bbox,
                "overridden": thresholds_overridden,
            },
        },
        "summary": {
            "total": len(rows),
            "correct": correct_total,
            "incorrect": len(rows) - correct_total,
            "correct_rate": correct_total / len(rows) if rows else 0.0,
            "macro_f1": sum(task_f1s) / len(task_f1s) if task_f1s else None,
            "errors": sum(error_counts.values()),
            "error_counts": dict(sorted(error_counts.items())),
        },
        "metric_groups": {
            metric: {
                "task_count": len(scores),
                "task_macro_score": sum(scores) / len(scores),
            }
            for metric, scores in sorted(metric_scores.items())
        },
        "categories": categories,
        "tasks": tasks,
        "items": [
            {
                "index": row["sample"].index,
                "item_id": row["sample"].item["id"],
                "task_id": row["sample"].task.task_id,
                "prediction": row["prediction"],
                "raw_prediction": row["raw_prediction"],
                "item_score": row["score"],
                "correct": row["correct"],
                "error": row["error"],
            }
            for row in rows
        ],
    }
    atomic_write_json(output_path, report)
    return str(output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--bleu-threshold", type=float)
    parser.add_argument("--bbox-iou-threshold", type=float)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    for name in ("bleu_threshold", "bbox_iou_threshold"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")
    print(
        evaluate_inference(
            args.input_path,
            args.output_path,
            bleu_threshold=args.bleu_threshold,
            bbox_iou_threshold=args.bbox_iou_threshold,
        )
    )


if __name__ == "__main__":
    main()
