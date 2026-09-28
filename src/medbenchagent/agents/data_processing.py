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
            "image_inventory_path",
            "image_plan_path",
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
    """Choose image encodings; default to unchanged PNG/JPEG or lossless PNG representation."""
    require(context, "ImageInventory")
    from medbenchagent.data.image_inventory import inventory_entries
    details = inventory_entries(context["image_inventory_path"])
    policies = {
        key: (value.model_dump() if hasattr(value, "model_dump") else value)
        for key, value in overrides.items()
    }
    unmatched = set(policies)
    for key, info in details:
        if "error" in info:
            continue
        unmatched.discard(key)
        policy = policies.get(key, {"encoding": "preserve"})
        if key not in policies and not info["preserve_supported"] and info.get("window"):
            policy = {"encoding": "window", "low": info["window"][0], "high": info["window"][1],
                      "invert": info["invert"], "decision_reference": ""}
        if info["copy"]:
            if policy["encoding"] != "preserve":
                raise ValueError("PNG/JPEG must be copied unchanged")
            continue
        if policy["encoding"] == "preserve":
            if not info["preserve_supported"]:
                raise ValueError(f"{key}: choose a source-supported window or ask_user")
        else:
            low, high = policy["low"], policy["high"]
            if low is None or high is None or low >= high:
                raise ValueError("Window requires low < high")
            supported = info["window"] == [low, high] and info["invert"] == policy["invert"]
            if not supported:
                issue = context.get("issues", {}).get(policy.get("decision_reference"), {})
                if (issue.get("status") != "resolved" or issue.get("kind") != "semantic"
                        or issue.get("role") != "DataProcessingAgent"
                        or issue.get("dataset_id") != context.get("dataset_scope")
                        or issue.get("source_ids") != [context.get("source_scope")]
                        or key not in issue.get("image_ids", [])):
                    raise ValueError(f"{key}: display policy requires an answered user decision for this image/source")
                # Already parsed decisions are checked without another model call.
                from medbenchagent.runtime.dependencies import require as current_artifacts
                current = current_artifacts(context, *issue["input_versions"])
                if any(current[k]["id"] != v for k, v in issue["input_versions"].items()):
                    raise ValueError("Display decision is stale; ask again")
                selected = {k: policy[k] for k in ("encoding", "low", "high", "invert")}
                if issue.get("display_policies", {}).get(key) != selected:
                    raise ValueError("Display parameters differ from the resolved user decision")
    if unmatched:
        raise ValueError("Image plan contains unreferenced or invalid images")
    path = Path(context["operation_directory"]) / "image_plan.json"
    atomic_write_json(path, policies)
    versions = publish(context, "ImagePlan", {"path": str(path)}, ("ImageInventory",))
    return result("success", "Image encoding plan recorded", image_plan_path=str(path), **versions)


def compile_data_processing_agent(*, checkpointer=None):
    from medbenchagent.agents.dataset_agent import compile_dataset_agent
    return compile_dataset_agent(checkpointer=checkpointer)
