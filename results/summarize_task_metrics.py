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


def main() -> None:
    with JSON_PATH.open("r", encoding="utf-8") as source:
        data = json.load(source)
    results = task_metrics(data)
    tasks = list(results)
    print(",".join(tasks))
    print(",".join(f"{results[task]['accuracy']:.4f}" for task in tasks))
    print(",".join(f"{results[task]['f1']:.4f}" for task in tasks))


if __name__ == "__main__":
    main()
