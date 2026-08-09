"""Print task-level accuracy and F1 rows from one evaluation report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# 修改这里即可
JSON_PATH = Path(
    "/public/home/rxa-jj-wzk/projects/fyl/medbenchagent/results/evaluation/"
    "qwen3_vl_8b_instruct/BrEaST.report.json"
)


def normalize_task(task_name: str, item: dict[str, Any]) -> str:
    """Collapse question types to their parent benchmark task."""
    parent_task_id = item.get("parent_task_id")
    if isinstance(parent_task_id, str) and parent_task_id:
        return parent_task_id
    for decomposable in ("anatomy_attribute", "finding_attribute"):
        if task_name.startswith(f"{decomposable}."):
            return decomposable
    return task_name


def task_metrics(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"accuracy": [], "f1": []}
    )
    for task_name, item in data["tasks"].items():
        task = normalize_task(task_name, item)
        grouped[task]["accuracy"].append(float(item["correct_rate"]))
        grouped[task]["f1"].append(float(item["score"]))

    return {
        task: {
            metric: sum(scores) / len(scores)
            for metric, scores in values.items()
        }
        for task, values in grouped.items()
    }


def report_sections(data: dict[str, Any]) -> list[tuple[str | None, dict[str, Any]]]:
    """Accept either one legacy report or response-format keyed reports."""
    if isinstance(data.get("tasks"), dict):
        return [(None, data)]
    sections = []
    for task_type, report in data.items():
        if task_type not in {"letter", "bbox", "free_text"} or not isinstance(report, dict):
            raise ValueError("Evaluation report has an unsupported grouped structure.")
        if not isinstance(report.get("tasks"), dict):
            raise ValueError(f"{task_type}: evaluation report is missing tasks.")
        sections.append((task_type, report))
    if not sections:
        raise ValueError("Evaluation report contains no task sections.")
    return sections


def summary_lines(data: dict[str, Any]) -> list[str]:
    lines = []
    for task_type, report in report_sections(data):
        if lines:
            lines.append("")
        if task_type is not None:
            lines.append(task_type)
        results = task_metrics(report)
        tasks = list(results)
        lines.append(",".join(tasks))
        first_metric, second_metric = (
            ("f1", "accuracy") if task_type == "bbox" else ("accuracy", "f1")
        )
        lines.append(",".join(f"{results[task][first_metric]:.4f}" for task in tasks))
        lines.append(",".join(f"{results[task][second_metric]:.4f}" for task in tasks))
    return lines


def main() -> None:
    with JSON_PATH.open("r", encoding="utf-8") as source:
        data = json.load(source)
    print("\n".join(summary_lines(data)))


if __name__ == "__main__":
    main()
