"""High recall selection with immutable snapshots of every model-visible candidate."""

import json
from pathlib import Path
from pydantic import Field
from medbenchagent.data.contracts import Contract, Text
from medbenchagent.data.dataset_catalog import dataset_by_id, load_prepared, load_dataset_catalog, resolve_path
from medbenchagent.data.dataset_card import card_summary
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.harness import result


class ExplicitDatasets(Contract):
    dataset_ids: list[Text] | None


class Selection(Contract):
    selected_dataset_ids: list[Text] = Field(min_length=1)
    selection_reason: dict[str, Text]


def _candidate_snapshot(context, path):
    requirement_version = require(context, "Requirement")["Requirement"]["id"]
    prepared_version = context.get("artifact_versions", {}).get("PreparedCandidates")
    if path.is_file():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if (snapshot["requirement_version"] != requirement_version or
                snapshot["requirement"] != context["user_requirement"] or
                snapshot["prepared_version"] != prepared_version or
                snapshot["explicit_dataset_ids"] != context.get("requested_dataset_ids")):
            raise ValueError("Selection inputs changed; start a new selection operation")
    else:
        catalog = load_dataset_catalog(context.get("registration_catalog_path") or context.get("catalog_path"))
        entries = {e["dataset_id"]: e for e in catalog}
        entries.update(context.get("prepared_entries", {}))
        catalog = list(entries.values())
        snapshot = {
            "schema_version": "dataset-candidates.v1",
            "requirement": context["user_requirement"], "requirement_version": requirement_version,
            "benchmark_info": context.get("benchmark_info", {}),
            "prepared_version": prepared_version, "explicit_dataset_ids": context.get("requested_dataset_ids"),
            "catalog_entries": catalog,
            "name_resolution_input": {
                "requirement": context["user_requirement"],
                "datasets": [{"dataset_id": e["dataset_id"], "name": e["name"]} for e in catalog],
            },
        }
        atomic_write_json(path, snapshot)
    if "requested_dataset_ids" not in snapshot:
        requested = context.get("requested_dataset_ids")
        if requested is None:
            requested = ExplicitDatasets.model_validate(complete_json(
                'Extract only datasets explicitly named by the user. Return {"dataset_ids":null} if none. '
                'Otherwise return their IDs in dataset_ids. Resolve supplied names to IDs; preserve unknown names '
                'as unknown IDs so validation can report them. Do not recommend datasets.',
                json.dumps(snapshot["name_resolution_input"], ensure_ascii=False),
            )).dataset_ids
        if requested is not None and (not requested or len(requested) != len(set(requested))):
            raise ValueError("Explicit dataset selection must be nonempty and unique")
        snapshot["requested_dataset_ids"] = requested
        atomic_write_json(path, snapshot)
    if "candidates" in snapshot:
        return snapshot
    requested = snapshot["requested_dataset_ids"]
    catalog = snapshot["catalog_entries"]
    excluded = {e["dataset_id"]: "Dataset is not available" for e in catalog if not e["available"]}
    if requested is not None:
        excluded.update({e["dataset_id"]: "Not explicitly requested" for e in catalog
                         if e["available"] and e["dataset_id"] not in requested})
        candidates = []
        for identifier in requested:
            try:
                candidates.append(dataset_by_id(catalog, identifier))
            except ValueError as exc:
                excluded[identifier] = str(exc)
                atomic_write_json(path, {**snapshot, "excluded_candidates": excluded})
                raise
    else:
        candidates = [e for e in catalog if e["available"]]
    frozen = {}
    for entry in candidates:
        identifier = entry["dataset_id"]
        try:
            original = resolve_path(entry["dataset_card_path"])
            card_path = path.parent / "candidate_cards" / f"{identifier}.json"
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_bytes(original.read_bytes())
            frozen_entry = {**entry, "dataset_card_path": str(card_path)}
            card, access = load_prepared(frozen_entry)
        except (ValueError, OSError) as exc:
            excluded[identifier] = str(exc)
            if requested is not None:
                atomic_write_json(path, {**snapshot, "excluded_candidates": excluded})
                raise
            continue
        frozen[identifier] = {"card_digest": digest_file(card_path), "summary": card_summary(card),
                              "original_card_path": str(original), "entry": frozen_entry, "access": access}
    order = list(frozen)
    snapshot.update(candidates=frozen, candidate_order=order, excluded_candidates=excluded,
                    selection_method="explicit" if requested is not None else "only_candidate" if len(order) == 1 else "model" if order else "no_candidates",
                    selection_input={"requirement": snapshot["requirement"], "benchmark_info": snapshot["benchmark_info"],
                                     "cards": [frozen[k]["summary"] for k in order]})
    atomic_write_json(path, snapshot)
    return snapshot


def select_datasets(context):
    """Choose prepared datasets from frozen inputs; a new operation creates a new universe snapshot."""
    folder = Path(context["operation_directory"])
    snapshot_path = folder / "dataset_candidates.json"
    snapshot = _candidate_snapshot(context, snapshot_path)
    candidate_deps = ("Requirement", "PreparedCandidates") if "PreparedCandidates" in context.get("artifact_versions", {}) else ("Requirement",)
    publish(context, "DatasetCandidates", {"path": str(snapshot_path)}, candidate_deps)
    path = folder / "dataset_selection.json"
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        response = {key: document[key] for key in ("selected_dataset_ids", "selection_reason")}
    else:
        candidates = snapshot["candidate_order"]
        method = snapshot["selection_method"]
        if method == "no_candidates":
            response = {"selected_dataset_ids": [], "selection_reason": {}}
        elif method in {"explicit", "only_candidate"}:
            response = {"selected_dataset_ids": candidates,
                        "selection_reason": {k: "Explicitly requested" if method == "explicit" else "Only prepared candidate" for k in candidates}}
        else:
            response = complete_json(
                "Select relevant medical datasets with high recall; selection only authorizes deeper task planning, "
                "Use the benchmark clinical scope, explicit disease/modality restrictions and evaluation focus. "
                "not guaranteed sample contribution. Never invent IDs. Return selected_dataset_ids and selection_reason "
                "keyed by exactly the selected IDs. Return an empty selection if none match.",
                json.dumps(snapshot["selection_input"], ensure_ascii=False),
            )
    if response == {"selected_dataset_ids": [], "selection_reason": {}}:
        atomic_write_json(path, {**snapshot, **response})
        return result("needs_input", "No relevant prepared datasets; prepare/register datasets or clarify the request",
                      global_issue={"question": "Which prepared datasets should support this benchmark?", "excluded": snapshot["excluded_candidates"]},
                      selection_path=str(path), candidate_snapshot_path=str(snapshot_path),
                      artifact_versions=context["artifact_versions"])
    selection = Selection.model_validate(response)
    ids = selection.selected_dataset_ids
    if len(ids) != len(set(ids)) or set(ids) != set(selection.selection_reason) or not set(ids) <= set(snapshot["candidates"]):
        raise ValueError("Selection IDs/reasons must match unique prepared candidates")
    atomic_write_json(path, {**snapshot, **selection.model_dump()})
    assets = {}
    for identifier in ids:
        candidate = snapshot["candidates"][identifier]
        # Only selected data is revalidated; other candidates are historical decision evidence.
        if digest_file(candidate["entry"]["dataset_card_path"]) != candidate["card_digest"]:
            raise ValueError("Frozen candidate Card changed; start a new selection operation")
        card, access = load_prepared(candidate["entry"])
        asset = {"entry": candidate["entry"], "access": access, "card_summary": card_summary(card)}
        assets[identifier] = asset
        child = dict(context, dataset_scope=identifier)
        for kind, references in (("Dataset", {"annotation": asset["entry"]["annotation_path"], "access": access}),
                                 ("DatasetProfile", {"path": asset["entry"]["profile_path"]}),
                                 ("DatasetCard", {"path": asset["entry"]["dataset_card_path"]})):
            deps = ("DataConfig",) if kind == "Dataset" else ("Dataset",) if kind == "DatasetProfile" else ("DatasetProfile",)
            context.update(publish(child, kind, references, deps))
    return result("success", "Dataset selection published", selected_dataset_ids=ids,
                  dataset_assets=assets, selection_path=str(path), candidate_snapshot_path=str(snapshot_path),
                  warnings=list(snapshot["excluded_candidates"].values()),
                  **publish(context, "DatasetSelection", {"path": str(path)},
                            ("Requirement", "DatasetCandidates", *(f"dataset/{k}/DatasetCard" for k in ids))))
