"""Fail-open, task-level semantic screening of final audited benchmark items."""

from __future__ import annotations

import random
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medbenchagent.runtime.construction import AUDIT_AGENT, event_context, record_event
from medbenchagent.runtime.llm import ModelUnavailable, complete_json
from medbenchagent.settings import get_settings

SAMPLES_PER_TASK = 10
VIOLATIONS = {
    "task_drift",
    "answer_leakage",
    "ambiguity_or_option_overlap",
    "semantic_duplication_or_template_collapse",
}

SEMANTIC_AUDIT_PROMPT = """
Screen a small sample of final benchmark items from one medical VLM task.
This is a deliberately permissive, observation-only audit.

Flag only a clear, substantial, high-confidence problem in one of these categories:
- task_drift: the question plainly tests a different capability than the task definition.
- answer_leakage: the question directly reveals or strongly hints at its supplied answer.
- ambiguity_or_option_overlap: the wording is clearly ambiguous, or multiple options are
  plainly equivalent for the question as written.
- semantic_duplication_or_template_collapse: sampled questions are obvious semantic
  near-duplicates or show severe mechanical template collapse.

Ordinary wording variation, reasonable structural similarity, repeated medical terms, stylistic
differences, and uncertain cases must pass. Do not judge annotation evidence, expected-answer
correctness, medical truth, image contents, or issues that require seeing the image.

Return JSON only:
{"items":[{"item_id":"exact supplied id","decision":"pass|flag","violations":[]}]}
Return every supplied item exactly once and no other item. A pass has no violations. A flag has
one or more values from the four categories above.
""".strip()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample(items: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["task_id"])].append(item)
    for task_id, task_items in grouped.items():
        task_items.sort(key=lambda item: item["id"])
        random.Random(seed + int.from_bytes(task_id.encode("utf-8"), "big")).shuffle(task_items)
        grouped[task_id] = task_items[:SAMPLES_PER_TASK]
    return dict(sorted(grouped.items()))


def _answer_value(item: dict[str, Any]) -> object:
    options, answer = item.get("options"), item.get("answer")
    if (
        isinstance(options, list)
        and isinstance(answer, str)
        and len(answer) == 1
        and 0 <= ord(answer.upper()) - ord("A") < len(options)
    ):
        return options[ord(answer.upper()) - ord("A")]
    return answer


def _validate_response(response: object, expected_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or set(response) != {"items"}:
        raise ValueError("Semantic audit response must contain only items.")
    items = response["items"]
    if not isinstance(items, list) or len(items) != len(expected_ids):
        raise ValueError("Semantic audit must return every sampled item exactly once.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"item_id", "decision", "violations"}:
            raise ValueError("Each semantic audit item has an invalid shape.")
        item_id, decision, violations = item["item_id"], item["decision"], item["violations"]
        if not isinstance(item_id, str) or item_id not in expected_ids or item_id in seen:
            raise ValueError("Semantic audit returned an unknown or duplicate item ID.")
        if not isinstance(decision, str) or decision not in {"pass", "flag"}:
            raise ValueError("Semantic audit decision must be pass or flag.")
        if (
            not isinstance(violations, list)
            or not all(isinstance(value, str) and value in VIOLATIONS for value in violations)
            or len(violations) != len(set(violations))
            or (decision == "pass" and violations)
            or (decision == "flag" and not violations)
        ):
            raise ValueError("Semantic audit violations do not match the decision.")
        seen.add(item_id)
        normalized.append(
            {"item_id": item_id, "decision": decision, "violations": violations}
        )
    if seen != expected_ids:
        raise ValueError("Semantic audit omitted sampled item IDs.")
    return normalized


def semantic_audit_run(
    *,
    audited_path: str | Path,
    report_path: str | Path,
    blueprint_path: str | Path,
    events_path: str | Path,
    output_directory: str | Path,
    run_id: str,
    dataset_id: str,
    seed: int,
) -> dict[str, str]:
    audited = _rows(Path(audited_path))
    blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
    tasks = blueprint["tasks"]
    sampled = _sample(audited, seed)
    records: list[dict[str, Any]] = []
    flagged_ids: set[str] = set()
    settings = get_settings()

    for task_id, items in sampled.items():
        task = tasks[task_id]
        item_payload = [
            {
                "item_id": item["id"],
                "question": item["question"],
                "options": item.get("options"),
                "answer": item.get("answer"),
                "answer_value": _answer_value(item),
            }
            for item in items
        ]
        user = json.dumps(
            {
                "task": {
                    key: task[key]
                    for key in (
                        "description",
                        "require",
                        "concept_id",
                        "question_templates",
                        "protocol",
                    )
                    if key in task
                },
                "items": item_payload,
            },
            ensure_ascii=False,
        )
        with event_context(
            construction_events_path=events_path,
            run_id=run_id,
            dataset_id=dataset_id,
            task_id=task_id,
            agent=AUDIT_AGENT,
            operation="semantic_screen",
            model=settings.llm.model,
        ) as call_ids:
            try:
                decisions = _validate_response(
                    complete_json(SEMANTIC_AUDIT_PROMPT, user),
                    {str(item["id"]) for item in items},
                )
                by_id = {decision["item_id"]: decision for decision in decisions}
                for item in items:
                    decision = by_id[str(item["id"])]
                    if decision["decision"] == "flag":
                        flagged_ids.add(str(item["id"]))
                    records.append(
                        {
                            "item_id": item["id"],
                            "task_id": task_id,
                            "audit_status": "completed",
                            "decision": decision["decision"],
                            "violations": decision["violations"],
                            "call_id": call_ids[0] if call_ids else None,
                        }
                    )
            except (ModelUnavailable, ValueError) as exc:
                status = (
                    "model_unavailable"
                    if isinstance(exc, ModelUnavailable)
                    else "invalid_response"
                )
                for item in items:
                    records.append(
                        {
                            "item_id": item["id"],
                            "task_id": task_id,
                            "audit_status": status,
                            "decision": None,
                            "violations": [],
                            "call_id": call_ids[0] if call_ids else None,
                        }
                    )

    flagged = [item for item in audited if str(item.get("id")) in flagged_ids]

    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    completed = [record for record in records if record["audit_status"] == "completed"]
    violation_counts = Counter(
        violation for record in completed for violation in record["violations"]
    )
    report["semantic_records"] = records
    report["semantic_audit"] = {
        "eligible": len(audited),
        "sampled": len(records),
        "completed": len(completed),
        "passed": sum(record["decision"] == "pass" for record in completed),
        "flagged": len(flagged),
        "not_run": len(records) - len(completed),
        "not_sampled": len(audited) - len(records),
        "violation_counts": dict(violation_counts),
    }
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_event(
        events_path,
        "semantic_audit_summary",
        run_id=run_id,
        dataset_id=dataset_id,
        agent=AUDIT_AGENT,
        operation="semantic_screen",
        status="complete",
        **report["semantic_audit"],
    )
    return {
        "report_path": str(report_path),
    }
