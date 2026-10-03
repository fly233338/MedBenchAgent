"""Freeze validated prepared candidates and bind the user's dataset choice."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.contracts import Contract, Text
from medbenchagent.data.dataset_catalog import load_dataset_catalog, load_prepared, resolve_path
from medbenchagent.data.readiness import prepared_directory_entries
from medbenchagent.data.bundle_catalog import card_summary
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.planning.bir import BIRStore


class Selection(Contract):
    selected_dataset_ids: list[Text]
    selection_reason: dict[str, Text]


def _validate_selection(selection, candidates):
    ids = selection.selected_dataset_ids
    if (not ids or len(ids) != len(set(ids)) or set(ids) != set(selection.selection_reason)
            or not set(ids) <= set(candidates)):
        raise ValueError("Selection IDs/reasons must match one or more unique prepared candidates")


def _snapshot_inputs(context):
    if context.get("task_catalog_snapshot_path"):
        catalog = json.loads(Path(context["task_catalog_snapshot_path"]).read_text(encoding="utf-8"))["entries"]
    else:
        catalog = load_dataset_catalog(context.get("registration_catalog_path") or context.get("catalog_path"))
    direct, direct_errors = prepared_directory_entries(context)
    entries = {entry["dataset_id"]: entry for entry in catalog}
    for entry in direct:
        entries.setdefault(entry["dataset_id"], entry)
    entries.update(context.get("prepared_entries", {}))
    return list(entries.values()), direct_errors


def _candidate_snapshot(context, path, *, refresh=False):
    requirement_version = require(context, "Requirement")["Requirement"]["id"]
    prepared_version = context.get("artifact_versions", {}).get("PreparedCandidates")
    if path.is_file():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        changed = (snapshot.get("schema_version") != "dataset-candidates"
                   or snapshot["requirement_version"] != requirement_version
                   or snapshot["requirement"] != context["user_requirement"]
                   or snapshot["prepared_version"] != prepared_version)
        if changed and not refresh:
            raise ValueError("Selection inputs changed; start a new candidate discovery operation")
        if not changed:
            return snapshot

    catalog, direct_errors = _snapshot_inputs(context)
    from medbenchagent.settings import get_settings
    raw_sources = list(get_settings().dataset_sources)
    snapshot = {
        "schema_version": "dataset-candidates",
        "requirement": context["user_requirement"],
        "requirement_version": requirement_version,
        "benchmark_info": context.get("benchmark_info", {}),
        "prepared_version": prepared_version,
        "catalog_entries": catalog,
        "configured_sources": raw_sources,
        "dataset_mentions": context.get("benchmark_info", {}).get("dataset_mentions", []),
        "direct_bundle_errors": direct_errors,
    }
    frozen, excluded = {}, dict(direct_errors)
    for entry in catalog:
        identifier = entry["dataset_id"]
        if not entry.get("available", True):
            excluded[identifier] = "Dataset is disabled"
            continue
        try:
            original = resolve_path(entry["dataset_card_path"])
            card_path = path.parent / "candidate_cards" / f"{identifier}.json"
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_bytes(original.read_bytes())
            frozen_entry = {**entry, "dataset_card_path": str(card_path)}
            card, access = load_prepared(frozen_entry)
        except (ValueError, OSError, KeyError) as exc:
            excluded[identifier] = str(exc)
            continue
        frozen[identifier] = {
            "card_digest": digest_file(card_path),
            "summary": card_summary(card),
            "original_card_path": str(original),
            "entry": frozen_entry,
        }
    order = list(frozen)
    snapshot.update(
        candidates=frozen,
        candidate_order=order,
        excluded_candidates=excluded,
        preparation_candidates=sorted({source["dataset_id"] for source in raw_sources
                                       if source["dataset_id"] not in frozen}),
        selection_input={
            "requirement": snapshot["requirement"],
            "benchmark_info": snapshot["benchmark_info"],
            "cards": [frozen[key]["summary"] for key in order],
        },
    )
    atomic_write_json(path, snapshot)
    return snapshot


def discover_dataset_candidates(context):
    """Freeze validated Prepared candidates for Supervisor review; never selects a dataset."""
    folder = Path(context["operation_directory"])
    path = folder / "dataset_candidates.json"
    snapshot = _candidate_snapshot(context, path, refresh=True)
    deps = ("Requirement", "PreparedCandidates") if "PreparedCandidates" in context.get("artifact_versions", {}) else ("Requirement",)
    versions = publish(context, "DatasetCandidates", {"path": str(path)}, deps)
    candidates = [snapshot["candidates"][key]["summary"] for key in snapshot["candidate_order"]]
    return result(
        "success",
        json.dumps({"candidates": candidates, "excluded": snapshot["excluded_candidates"]}, ensure_ascii=False),
        candidate_snapshot_path=str(path),
        dataset_candidates=candidates,
        candidate_order=snapshot["candidate_order"],
        preparation_candidates=snapshot["preparation_candidates"],
        **versions,
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


def select_datasets(context, *, dataset_ids, selection_reason, selection_issue_id=""):
    """Bind one or more Supervisor/user-selected IDs from the frozen candidate snapshot."""
    folder = Path(context["operation_directory"])
    snapshot_path = Path(context.get("candidate_snapshot_path", folder / "dataset_candidates.json"))
    snapshot = _candidate_snapshot(context, snapshot_path)
    selection = Selection(selected_dataset_ids=dataset_ids, selection_reason=selection_reason)
    _validate_selection(selection, snapshot["candidates"])
    path = folder / "dataset_selection.json"
    atomic_write_json(path, {**snapshot, **selection.model_dump(), "selection_issue_id": selection_issue_id})
    assets = {}
    for identifier in selection.selected_dataset_ids:
        candidate = snapshot["candidates"][identifier]
        if digest_file(candidate["entry"]["dataset_card_path"]) != candidate["card_digest"]:
            raise ValueError("Frozen candidate Card changed; start a new candidate discovery operation")
        card, access = load_prepared(candidate["entry"])
        asset = {"entry": candidate["entry"], "access": access, "card_summary": card_summary(card)}
        assets[identifier] = asset
        child = dict(context, dataset_scope=identifier)
        for kind, references in (
            ("Dataset", {"cases": asset["entry"]["cases_path"], "access": access}),
            ("DatasetProfile", {"path": asset["entry"]["profile_path"]}),
            ("DatasetCard", {"path": asset["entry"]["dataset_card_path"]}),
        ):
            deps = ("DataConfig",) if kind == "Dataset" else ("Dataset",) if kind == "DatasetProfile" else ("DatasetProfile",)
            context.update(publish(child, kind, references, deps))
    if context.get("bir_path"):
        store = BIRStore(context["bir_path"])
        changes = {"benchmark_info.json": store.load("benchmark_info.json")}
        for identifier in selection.selected_dataset_ids:
            asset = assets[identifier]
            entry = snapshot["candidates"][identifier]["entry"]
            access = asset["access"]
            receipt = {
                "schema_version": "dataset-access",
                "source_schema_version": access.get("schema_version", "annotation"),
                "source_access": access,
                "dataset_id": identifier,
                "revision": str(access.get("revision", "1")),
                "cases_path": access.get("cases_path", entry["cases_path"]),
                "asset_manifest_path": access.get("asset_manifest_path", entry.get("asset_manifest_path", "")),
                "profile_path": access.get("profile_path", entry.get("profile_path", "")),
                "duckdb_path": access.get("duckdb_path", ""),
                "annotation_digest": access.get("annotation_digest", ""),
                "asset_manifest_digest": access.get("asset_manifest_digest", ""),
                "profile_digest": access.get("profile_digest", ""),
                "duckdb_digest": access.get("duckdb_digest", ""),
                "case_count": int(access.get("case_count", 0)),
                "patient_count": int(access.get("patient_count", 0)),
                "patient_identity": access.get("patient_identity", "unknown"),
                "dataset_card_path": entry.get("dataset_card_path"),
                "dataset_card_digest": snapshot["candidates"][identifier].get("card_digest"),
            }
            changes[f"datasets/{identifier}/selection.json"] = {
                "schema_version": "dataset-selection",
                "dataset_id": identifier,
                "selected": True,
                "selection_reason": selection.selection_reason,
                "candidate_snapshot_digest": digest_file(snapshot_path),
                "dataset_card_digest": snapshot["candidates"][identifier].get("card_digest"),
                "entry": entry,
            }
            changes[f"datasets/{identifier}/access.json"] = receipt
        commit = store.transaction({key: value for key, value in changes.items() if key != "benchmark_info.json"},
                                   run_id=context.get("run_id", "unknown"), actor="dataset_selection")
        context.update(bir_revision=commit.revision, bir_manifest_digest=commit.manifest_digest)
    issue_updates = _selection_issue_updates(context, selection_issue_id)
    versions = publish(context, "DatasetSelection", {"path": str(path)},
                       ("Requirement", "DatasetCandidates",
                        *(f"dataset/{key}/DatasetCard" for key in selection.selected_dataset_ids)))
    return result(
        "success",
        "Dataset selection published",
        selected_dataset_ids=selection.selected_dataset_ids,
        bir_revision=context.get("bir_revision"), bir_manifest_digest=context.get("bir_manifest_digest"),
        dataset_assets=assets,
        selection_path=str(path),
        candidate_snapshot_path=str(snapshot_path),
        warnings=list(snapshot["excluded_candidates"].values()),
        **issue_updates,
        **versions,
    )

