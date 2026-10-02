"""Resumable source inventory, annotation mapping and image preparation harness."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.validation import source_image_path
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.inspection import inspected

from medbenchagent.data.execution import run as _run


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def _inventory(context, key, value, *, operation=None, arguments=None):
    entries = _read(context["inventory_path"]) if context.get("inventory_path") else {}
    entries[key] = value
    path = Path(context["operation_directory"]) / "source_inventory.json"
    atomic_write_json(path, entries)
    versions = publish(context, "Inventory", {"path": str(path)}, ("Source",))
    return inspected(
        context, operation or key.split(":", 1)[0], arguments or {"key": key}, value,
        ("Source",), source_id=context.get("source_scope"),
        inventory_path=str(path),
        **versions,
    )


def inspect_files(context, *, directory="", offset=0):
    """Page source files; snapshots are readonly inputs to subsequent preparation stages."""
    root = Path(context["source_root"])
    folder = (root / directory).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise ValueError("Select a source directory")
    files = sorted(folder.iterdir())
    entries = [
        {
            "path": p.relative_to(root).as_posix(),
            "directory": p.is_dir(),
            "bytes": p.stat().st_size if p.is_file() else None,
        }
        for p in files[offset : offset + 100]
    ]
    return _inventory(
        context, f"directory:{directory}:{offset}", {"total": len(files), "entries": entries}
    )


def read_text(context, *, path, offset=0, source_id=None, page=1):
    """Read a bounded portion of source documentation or a saved preparation script/plan."""
    if source_id is not None:
        source = next((s for s in context["sources"] if s["id"] == source_id), None)
        if source is None:
            raise ValueError("Select a configured source document")
        require(context, f"source/{source_id}/Source")
        scoped = {**context, "source_scope": source_id, "source_root": source["root"],
                  "source_manifest": str(Path(context["bundle_root"]) / "source_manifests" / source_id)}
        return read_text(scoped, path=path, offset=offset, page=page)
    candidates = {
        str(context.get(key)): context.get(key)
        for key in (
            "annotation_script_path",
            "image_script_path",
            "fingerprint_path",
            "semantic_spec_path",
            "execution_plan_path",
            "candidate_path",
            "preparation_validation_path",
            "inventory_path",
            "asset_manifest_path",
            "presentation_plan_path",
            "validation_path",
        )
    }
    source = (
        Path(candidates[path])
        if path in candidates
        else source_image_path(context["source_root"], path)
    )
    if path not in candidates and context.get("source_manifest"):
        from medbenchagent.data.sources import track_files
        track_files(context["source_manifest"], [path])
    if path in candidates:
        with source.open(encoding="utf-8") as handle:
            handle.read(offset)
            return result("success", handle.read(6000))
    from medbenchagent.data.documentation import freeze_document, read_document
    freeze_document(context["bundle_root"], context["source_scope"], context["source_root"], path)
    value = read_document(context["bundle_root"], context["source_scope"], path, page=page, offset=offset)
    return inspected(context, "read_text", {"path": path, "page": page, "offset": offset}, value,
                     ("Source",), source_id=context["source_scope"])


def inspect_table(context, *, path, sheet=None, delimiter=","):
    """Measure table/sheet row counts, types, nulls and key uniqueness."""
    if context.get("source_manifest"):
        from medbenchagent.data.sources import track_files
        track_files(context["source_manifest"], [path])
    value = _run(
        context,
        Path(context["operation_directory"]) / "inspection",
        request={"operation": "table", "path": path, "sheet": sheet, "delimiter": delimiter},
    )
    return _inventory(context, f"table:{path}:{sheet}:{delimiter}", value,
                      operation="table", arguments={"path": path, "sheet": sheet, "delimiter": delimiter})


def inspect_join(context, **arguments):
    """Measure an exact source join's duplicates and unmatched rows."""
    if context.get("source_manifest"):
        from medbenchagent.data.sources import track_files
        track_files(context["source_manifest"], [arguments["left"], arguments["right"]])
    value = _run(
        context,
        Path(context["operation_directory"]) / "inspection",
        request={"operation": "join", **arguments},
    )
    key = "join:" + json.dumps(arguments, sort_keys=True)
    return _inventory(context, key, value, operation="join", arguments=arguments)


def save_image_plan(context, *, overrides):
    """Record a v5 presentation plan against the Asset manifest."""
    manifest_path = context.get("asset_manifest_path")
    if not manifest_path:
        raise ValueError("Presentation plans require a v5 asset_manifest.jsonl")
    from medbenchagent.data.asset_manifest import iter_asset_manifest
    from medbenchagent.data.v5_models import PresentationSpec
    asset_ids = {entry.asset_id for entry in iter_asset_manifest(manifest_path)}
    policies = {
        key: (value.model_dump() if hasattr(value, "model_dump") else value)
        for key, value in overrides.items()
    }
    unknown = set(policies) - asset_ids
    if unknown:
        raise ValueError(f"Presentation plan references unknown assets: {sorted(unknown)}")
    for asset_id, policy in policies.items():
        PresentationSpec.model_validate(policy)
    path = Path(context["operation_directory"]) / "presentation_plan.json"
    atomic_write_json(path, policies)
    return result("success", "Presentation plan recorded", presentation_plan_path=str(path))


def compile_data_processing_agent(*, checkpointer=None):
    from medbenchagent.agents.dataset_agent import compile_dataset_agent
    return compile_dataset_agent(checkpointer=checkpointer)
