"""High recall selection over prepared Card summaries, outside task planning."""

import json
from pathlib import Path
from pydantic import Field
from medbenchagent.data.contracts import Contract, Text
from medbenchagent.data.dataset_catalog import dataset_by_id, load_prepared, load_dataset_catalog
from medbenchagent.data.dataset_card import card_summary
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.harness import result


class ExplicitDatasets(Contract):
    dataset_ids: list[Text] | None


class Selection(Contract):
    selected_dataset_ids: list[Text] = Field(min_length=1)
    selection_reason: dict[str, Text]


def select_datasets(context):
    """Select explicitly requested datasets or recommend relevant prepared candidates."""
    require(context, "Requirement")
    catalog = load_dataset_catalog(context.get("catalog_path"))
    entries = {e["dataset_id"]: e for e in catalog}
    entries.update(context.get("prepared_entries", {}))
    catalog = list(entries.values())
    requested = ExplicitDatasets.model_validate(complete_json(
        'Extract only datasets explicitly named by the user. Return {"dataset_ids":null} if none. '
        'Otherwise return their IDs in dataset_ids. Resolve supplied names to IDs; preserve unknown names '
        'as unknown IDs so validation can report them. Do not recommend datasets.',
        json.dumps({"requirement": context["user_requirement"],
                    "datasets": [{"dataset_id": e["dataset_id"], "name": e["name"]} for e in catalog]}),
    )).dataset_ids
    if requested is not None:
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Explicit dataset selection must be nonempty and unique")
        candidates = [dataset_by_id(catalog, identifier) for identifier in requested]
    else:
        candidates = [e for e in catalog if e["available"]]
    summaries, assets, unavailable = [], {}, {}
    for entry in candidates:
        try:
            card, access = load_prepared(entry)
        except (ValueError, OSError) as exc:
            if requested is not None:
                raise
            unavailable[entry["dataset_id"]] = str(exc)
            continue
        summaries.append(card_summary(card))
        assets[entry["dataset_id"]] = {"entry": entry, "access": access, "card_summary": card_summary(card)}
    if not assets:
        return result("needs_input", "No relevant prepared datasets; prepare/register datasets or clarify the request",
                      global_issue={"unprepared": unavailable}, warnings=list(unavailable.values()))
    if requested is not None or len(assets) == 1:
        identifiers = requested if requested is not None else list(assets)
        selection = Selection(selected_dataset_ids=identifiers,
                              selection_reason={k: "Explicitly requested" if requested else "Only prepared candidate" for k in identifiers})
    else:
        response = complete_json(
            "Select relevant medical datasets with high recall; selection only authorizes deeper task planning, "
            "not guaranteed sample contribution. Never invent IDs. Return selected_dataset_ids and selection_reason "
            "keyed by exactly the selected IDs. Return an empty selection if none match.",
            json.dumps({"requirement": context["user_requirement"], "cards": summaries}, ensure_ascii=False),
        )
        if response.get("selected_dataset_ids") == []:
            return result("needs_input", "No relevant dataset; clarify the requirement", global_issue={"question": "Which datasets should support this benchmark?"})
        selection = Selection.model_validate(response)
    ids = selection.selected_dataset_ids
    if len(ids) != len(set(ids)) or set(ids) != set(selection.selection_reason) or not set(ids) <= set(assets):
        raise ValueError("Selection IDs/reasons must match unique prepared candidates")
    path = Path(context["operation_directory"]) / "dataset_selection.json"
    atomic_write_json(path, selection.model_dump())
    for identifier in ids:
        asset = assets[identifier]
        child = dict(context, dataset_scope=identifier)
        for kind, references in (("Dataset", {"annotation": asset["entry"]["annotation_path"], "access": asset["access"]}),
                                 ("DatasetProfile", {"path": asset["entry"]["profile_path"]}),
                                 ("DatasetCard", {"path": asset["entry"]["dataset_card_path"]})):
            deps = ("DataConfig",) if kind == "Dataset" else ("Dataset",) if kind == "DatasetProfile" else ("DatasetProfile",)
            context.update(publish(child, kind, references, deps))
    return result("success", "Dataset selection published", selected_dataset_ids=ids,
                  dataset_assets={k: assets[k] for k in ids}, selection_path=str(path),
                  warnings=list(unavailable.values()),
                  **publish(context, "DatasetSelection", {"path": str(path)},
                            ("Requirement", *(f"dataset/{k}/DatasetCard" for k in ids))))
