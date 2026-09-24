"""Observe configured sources and bind validated prepared versions to a run."""

import json
from pathlib import Path

from medbenchagent.data.dataset_catalog import load_dataset_catalog, load_prepared, resolve_path
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


def catalog_entries(context):
    if context.get("task_catalog_snapshot_path"):
        path = Path(context["task_catalog_snapshot_path"])
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    else:
        entries = load_dataset_catalog(context.get("registration_catalog_path") or context.get("catalog_path"))
    by_id = {entry["dataset_id"]: entry for entry in entries}
    by_id.update(context.get("prepared_entries", {}))
    return list(by_id.values())


def configured_source(dataset_id):
    return next((item for item in get_settings().dataset_sources if item["dataset_id"] == dataset_id), None)


def source_errors(source):
    if source is None:
        return ["Missing dataset_sources entry; configure dataset_id and sources before starting a new run"]
    errors = []
    for item in source["sources"]:
        root = Path(item["root"])
        if not root.is_dir():
            errors.append(f"Source {item['id']} root is unavailable: {root}")
        elif item["kind"] == "annotation" and not (root / item["annotation"]).is_file():
            errors.append(f"Source {item['id']} annotation is unavailable: {root / item['annotation']}")
    return errors


def list_datasets(context, *, offset=0, limit=20, query=""):
    """Page registered datasets and configured raw sources; listing does not validate or select data."""
    entries = {item["dataset_id"]: {key: item[key] for key in ("dataset_id", "name", "description", "available")}
               for item in catalog_entries(context)}
    for item in get_settings().dataset_sources:
        entries.setdefault(item["dataset_id"], {key: item[key] for key in ("dataset_id", "name", "description")})
        entries[item["dataset_id"]]["configured_sources"] = True
    rows = [entries[key] for key in sorted(entries)
            if query.casefold() in json.dumps(entries[key], ensure_ascii=False).casefold()]
    return result("success", json.dumps({"total": len(rows), "datasets": rows[offset:offset + limit]}, ensure_ascii=False))


def dataset_status(context, dataset_id):
    entry = next((item for item in catalog_entries(context) if item["dataset_id"] == dataset_id), None)
    source = configured_source(dataset_id)
    errors = source_errors(source)
    status = {"dataset_id": dataset_id, "prepared": False, "available": entry["available"] if entry else None,
              "configured_sources": source is not None, "source_errors": errors, "issues": []}
    if entry:
        if not entry["available"]:
            status["issues"].append("Catalog entry is disabled; ordinary preparation must not replace it")
        else:
            try:
                load_prepared(entry)
                status["prepared"] = True
            except (ValueError, OSError, KeyError) as exc:
                status["issues"].append(str(exc))
    status["next_action"] = ("reuse" if status["prepared"] else "repair_catalog_or_explicit_rebuild" if entry
                             else "fix_source_configuration" if errors else "prepare")
    return status


def inspect_dataset_status(context, *, dataset_id):
    """Validate one dataset's prepared assets and inspect configured source paths without publishing artifacts."""
    return result("success", json.dumps(dataset_status(context, dataset_id), ensure_ascii=False))


def reuse_prepared(context, dataset_id):
    """Freeze and validate a registered Card before binding its prepared data to this run."""
    folder = Path(context["operation_directory"])
    path = folder / "reused_dataset.json"
    if path.is_file():
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["dataset_id"] != dataset_id:
            raise ValueError("Reused dataset identity changed")
    else:
        entry = next(item for item in catalog_entries(context) if item["dataset_id"] == dataset_id)
        if not entry["available"]:
            raise ValueError("Cannot reuse a disabled dataset")
        load_prepared(entry)
        card_path = folder / "dataset_card.json"
        folder.mkdir(parents=True, exist_ok=True)
        card_path.write_bytes(resolve_path(entry["dataset_card_path"]).read_bytes())
        entry = {**entry, "dataset_card_path": str(card_path)}
        load_prepared(entry)
        atomic_write_json(path, entry)
    _, access = load_prepared(entry)
    child = dict(context, dataset_scope=dataset_id)
    publish(child, "Dataset", {"annotation": entry["annotation_path"], "access": access}, ("DataConfig",))
    publish(child, "DatasetProfile", {"path": entry["profile_path"]}, ("Dataset",))
    publish(child, "DatasetCard", {"path": entry["dataset_card_path"]}, ("DatasetProfile",))
    entries = {**context.get("prepared_entries", {}), dataset_id: entry}
    child.pop("dataset_scope")
    publish(child, "PreparedCandidates", {"entries": entries}, ())
    return result("success", "Reused validated prepared dataset", prepared_entries=entries,
                  artifact_versions=child["artifact_versions"])
