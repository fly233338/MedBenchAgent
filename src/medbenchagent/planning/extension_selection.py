"""Freeze and bind human-selected Prompt extensions for selected datasets."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.planning.bir import BIRStore
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.data.contracts import Contract, Text
from medbenchagent.data.sources import digest_file
from pydantic import Field


class ExtensionSelection(Contract):
    selected_extension_ids: list[Text] = Field(default_factory=list)
    selection_reason: dict[str, Text] = Field(default_factory=dict)


def _catalog(context):
    return BIRStore(context["bir_path"]).registry().planning_extensions()


def _validate_selection(selection: ExtensionSelection, candidates: dict[str, dict]):
    ids = selection.selected_extension_ids
    if len(ids) != len(set(ids)) or set(ids) != set(selection.selection_reason):
        raise ValueError("Extension IDs/reasons must match unique selected extension IDs")
    if not set(ids) <= set(candidates):
        raise ValueError("Extension selection contains an ID outside the frozen candidate snapshot")


def _snapshot(context, *, refresh: bool = False):
    dataset_id = BIRStore(context["bir_path"]).load("dataset_binding.json")["access"]["dataset_id"]
    directory = Path(context["operation_directory"]) / "extensions"
    path = directory / "extension_candidates.json"
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8")), path
    catalog = _catalog(context)
    metadata = catalog.metadata()
    candidates = {item["id"]: item for item in metadata}
    snapshot = {
        "schema_version": "planning-extension-candidates",
        "dataset_id": dataset_id,
        "registry_digest": BIRStore(context["bir_path"]).registry().digest,
        "candidates": candidates,
        "candidate_order": sorted(candidates),
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, snapshot)
    return snapshot, path


def discover_extension_candidates(context):
    snapshot, path = _snapshot(context, refresh=True)
    if not snapshot["candidate_order"]:
        selected = select_extensions(
            context,
            extension_ids=[],
            selection_reason={},
        )
        return result("success", "No planning extensions are registered", **selected["updates"],
                      extension_candidates=[], candidate_snapshot_path=str(path))
    return result(
        "success",
        json.dumps({"dataset_id": snapshot["dataset_id"], "candidates": [snapshot["candidates"][key] for key in snapshot["candidate_order"]]}, ensure_ascii=False),
        extension_candidates=[snapshot["candidates"][key] for key in snapshot["candidate_order"]],
        extension_candidate_order=snapshot["candidate_order"],
        extension_candidate_snapshot_path=str(path),
    )


def select_extensions(context, *, extension_ids: list[str],
                      selection_reason: dict[str, str], selection_issue_id: str = ""):
    snapshot, snapshot_path = _snapshot(context)
    selection = ExtensionSelection(
        selected_extension_ids=extension_ids,
        selection_reason=selection_reason,
    )
    _validate_selection(selection, snapshot["candidates"])
    catalog = _catalog(context)
    digest = catalog.selection_digest(extension_ids)
    selection_value = {
        **selection.model_dump(mode="json"),
        "dataset_id": snapshot["dataset_id"],
        "candidate_snapshot_digest": digest_file(snapshot_path),
        "selected_extension_digest": digest,
        "selection_issue_id": selection_issue_id,
    }
    store = BIRStore(context["bir_path"])
    store.update("planning_extensions.json", selection_value, run_id=context["run_id"], actor="extension_selection",
                 dependencies={"registry": store.registry().digest})
    issue_updates = {}
    if selection_issue_id:
        issue = context.get("global_issue")
        if not issue or issue.get("issue_id") != selection_issue_id:
            raise ValueError("selection_issue_id must reference the pending extension selection question")
        if issue.get("purpose") != "planning_extension_selection" or issue.get("status") != "answered":
            raise ValueError("Extension selection question must have an answer before binding")
        resolved = {**issue, "status": "resolved"}
        issue_updates = {
            "issues": {**context.get("issues", {}), selection_issue_id: resolved},
            "decisions": {**context.get("decisions", {}), selection_issue_id: issue.get("answer_reference", "")},
            "global_issue": None,
            "clarification": resolved,
        }
    return result(
        "success",
        "Planning extensions selected",
        extension_selection_path=str(store.root / "planning_extensions.json"),
        extension_candidate_snapshot_path=str(snapshot_path),
        **issue_updates,
    )
