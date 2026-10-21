"""Freeze validated prepared candidates and bind the user's dataset choice."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.contracts import DatasetAccessReceipt
from medbenchagent.data.dataset_catalog import load_dataset_catalog, load_prepared, resolve_path
from medbenchagent.data.readiness import prepared_directory_entries
from medbenchagent.data.bundle_catalog import card_summary
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.planning.bir import BIRStore, business_dependencies


def _snapshot_inputs(context):
    if context.get("task_catalog_snapshot_path"):
        catalog = json.loads(Path(context["task_catalog_snapshot_path"]).read_text(encoding="utf-8"))["entries"]
    else:
        catalog = load_dataset_catalog(context.get("registration_catalog_path") or context.get("catalog_path"))
    direct, direct_errors = prepared_directory_entries(context)
    entries = {entry["dataset_id"]: entry for entry in catalog}
    for entry in direct:
        entries.setdefault(entry["dataset_id"], entry)
    if entry := context.get("prepared_entry"):
        entries[entry["dataset_id"]] = entry
    return list(entries.values()), direct_errors


def _candidate_snapshot(context, path):
    requirement_version = BIRStore(context["bir_path"]).digest("benchmark_info.json")
    prepared_version = json.dumps(context.get("prepared_entry"), sort_keys=True)
    if path.is_file():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        changed = (snapshot.get("schema_version") != "dataset-candidates"
                   or snapshot["requirement_version"] != requirement_version
                   or snapshot["prepared_version"] != prepared_version)
        if not changed:
            return snapshot

    catalog, direct_errors = _snapshot_inputs(context)
    snapshot = {
        "schema_version": "dataset-candidates",
        "requirement_version": requirement_version,
        "prepared_version": prepared_version,
    }
    frozen, excluded = {}, dict(direct_errors)
    for entry in catalog:
        identifier = entry["dataset_id"]
        if context.get("binding_dataset_id") and identifier != context["binding_dataset_id"]:
            continue
        if not entry.get("available", True):
            excluded[identifier] = "Dataset is disabled"
            continue
        try:
            original = resolve_path(entry["dataset_card_path"])
            card_path = path.parent / "candidate_cards" / f"{identifier}.json"
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_bytes(original.read_bytes())
            frozen_entry = {**entry, "dataset_card_path": str(card_path)}
            card, _ = load_prepared(frozen_entry)
        except (ValueError, OSError, KeyError) as exc:
            excluded[identifier] = str(exc)
            continue
        frozen[identifier] = {
            "card_digest": digest_file(card_path),
            "summary": card_summary(card),
            "entry": frozen_entry,
        }
    order = list(frozen)
    snapshot.update(
        candidates=frozen,
        candidate_order=order,
        excluded_candidates=excluded,
    )
    atomic_write_json(path, snapshot)
    return snapshot


def discover_dataset_candidates(context):
    """Freeze validated Prepared candidates for Supervisor review; never selects a dataset."""
    folder = Path(context["operation_directory"])
    path = folder / "dataset_candidates.json"
    snapshot = _candidate_snapshot(context, path)
    candidates = [snapshot["candidates"][key]["summary"] for key in snapshot["candidate_order"]]
    return result(
        "success",
        json.dumps({"candidates": candidates, "excluded": snapshot["excluded_candidates"]}, ensure_ascii=False),
        candidate_snapshot_path=str(path),
    )


def _selection_issue_updates(context, issue_id):
    if not issue_id:
        return {}
    issue = context.get("global_issue")
    if not issue or issue.get("issue_id") != issue_id:
        raise ValueError("selection_issue_id must reference the pending dataset selection question")
    if issue.get("purpose") != "dataset_selection" or issue.get("status") != "answered":
        raise ValueError("Dataset selection question must have an answer before binding")
    resolved = {**issue, "kind": "dataset_selection", "status": "resolved"}
    return {
        "issues": {**context.get("issues", {}), issue_id: resolved},
        "decisions": {**context.get("decisions", {}), issue_id: issue.get("answer_reference", "")},
        "global_issue": None,
        "clarification": resolved,
    }


def bind_dataset(context, *, dataset_id: str, reason: str, selection_issue_id=""):
    """Bind one verified dataset from current frozen candidates."""
    store = BIRStore(context["bir_path"])
    current = store.current("dataset_binding.json")
    if current and current["access"]["dataset_id"] != dataset_id:
        raise ValueError("A compilation cannot switch datasets")
    if context.get("requested_dataset_id") and context["requested_dataset_id"] != dataset_id:
        raise ValueError("Binding differs from the explicitly requested dataset")
    path = Path(context["operation_directory"]) / "dataset_candidates.json"
    snapshot = _candidate_snapshot({**context, "binding_dataset_id": dataset_id}, path)
    if dataset_id not in snapshot["candidates"]:
        raise ValueError("Dataset is outside the verified candidate snapshot")
    candidate = snapshot["candidates"][dataset_id]
    if digest_file(resolve_path(candidate["entry"]["dataset_card_path"])) != candidate["card_digest"]:
        raise ValueError("Frozen Dataset Card changed")
    _, access = load_prepared(candidate["entry"])
    parsed = DatasetAccessReceipt.model_validate(access).verify()
    if parsed.dataset_id != dataset_id:
        raise ValueError("Dataset access belongs to another dataset")
    store.update("dataset_binding.json", {"schema_version": "dataset-binding", "access": parsed.model_dump(exclude_none=True),
        "entry": candidate["entry"], "selection_reason": reason, "candidate_snapshot_digest": digest_file(path)},
        run_id=context["run_id"], actor="dataset_selection",
        dependencies=business_dependencies(context, "benchmark_info.json"))
    return result("success", "Dataset bound", candidate_snapshot_path=str(path),
                  **_selection_issue_updates(context, selection_issue_id))
