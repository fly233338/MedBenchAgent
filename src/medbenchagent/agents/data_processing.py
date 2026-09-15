"""Resumable source inventory, annotation mapping and image preparation harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import uuid4

from medbenchagent.data.validation import source_image_path
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json

from medbenchagent.settings import get_settings


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(context, output, *, script=None, request=None, annotation=None):
    """Execute generated or fixed code with readonly source, code and annotation mounts."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if context.get("source_manifest"):
        from medbenchagent.data.sources import verify_source
        verify_source(context["source_manifest"])
    settings = get_settings().data_processing
    inspected = subprocess.run(["docker", "image", "inspect", "--format={{.Id}}", settings["image"]],
                               capture_output=True, text=True, check=False, timeout=30)
    if inspected.returncode:
        raise ValueError("Preparation container is unavailable; build the configured image first")
    image_id = inspected.stdout.strip()
    kind = "script" if script else request["operation"]
    receipt = output.parent / f"{output.name}-{kind}.complete.json"
    identity = {
        "image_id": image_id,
        "script": str(script) if script else None,
        "request": request,
        "annotation": annotation,
        "dependencies": context.get("artifact_versions", {}),
    }
    if receipt.is_file():
        if _read(receipt) != identity:
            raise ValueError(
                "Prepared operation dependencies changed; use a new operation directory"
            )
        expected = output / ("result.json" if request else "annotation.jsonl")
        if expected.is_file():
            return _read(output / "result.json") if request else None
        # Repeat-verification outputs may have been cleaned after completion.
        receipt.unlink()
    name = "medbench-data-" + uuid4().hex
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        f"--memory={settings['container_memory']}",
        f"--cpus={settings['threads']}",
        "--pids-limit=128",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,size=256m",
        "--env",
        "PYTHONPATH=/project",
        "--env",
        f"MEDBENCH_DATA_MEMORY={settings['memory_limit']}",
        "--env",
        f"MEDBENCH_DATA_THREADS={settings['threads']}",
        "--env",
        "MEDBENCH_DATA_SCRATCH=/work/scratch",
        "--mount",
        f"type=bind,source={Path(context['source_root']).resolve()},target=/source,readonly",
        "--mount",
        f"type=bind,source={Path(__file__).resolve().parents[2]},target=/project,readonly",
        "--mount",
        f"type=bind,source={output},target=/work",
        "--workdir",
        "/work",
    ]
    if annotation:
        command += [
            "--mount",
            f"type=bind,source={Path(annotation).resolve()},target=/annotation/annotation.jsonl,readonly",
        ]
    if script:
        command += [
            "--mount",
            f"type=bind,source={Path(script).resolve()},target=/script.py,readonly",
        ]
        arguments = ["python", "/script.py"]
    else:
        request_path = output.parent / (output.name + "-request.json")
        atomic_write_json(request_path, request)
        command += ["--mount", f"type=bind,source={request_path},target=/request.json,readonly"]
        arguments = ["python", "-m", "medbenchagent.data.preparation_ops", "/request.json"]
    try:
        completed = subprocess.run(
            command + [image_id] + arguments,
            capture_output=True,
            text=True,
            timeout=settings["timeout_seconds"],
            check=False,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False, timeout=30)
        raise ValueError("Preparation exceeded its execution budget") from None
    if completed.returncode:
        raise ValueError((completed.stdout + "\n" + completed.stderr)[-5000:])
    if context.get("source_manifest"):
        verify_source(context["source_manifest"])
    atomic_write_json(receipt, identity)
    return _read(output / "result.json") if request else None


def _inventory(context, key, value):
    entries = _read(context["inventory_path"]) if context.get("inventory_path") else {}
    entries[key] = value
    path = Path(context["operation_directory"]) / "source_inventory.json"
    atomic_write_json(path, entries)
    versions = publish(context, "Inventory", {"path": str(path)}, ("Source",))
    return result(
        "success",
        json.dumps({"reference": key, "inspection": value}, ensure_ascii=False)[:6000],
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


def read_text(context, *, path, offset=0):
    """Read a bounded portion of source documentation or a saved preparation script/plan."""
    candidates = {
        str(context.get(key)): context.get(key)
        for key in (
            "annotation_script_path",
            "image_script_path",
            "preparation_plan_path",
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
    with source.open(encoding="utf-8") as handle:
        handle.seek(offset)
        text = handle.read(6000)
    return result("success", text)


def inspect_table(context, *, path, sheet=None):
    """Measure table/sheet row counts, types, nulls and key uniqueness."""
    if context.get("source_manifest"):
        from medbenchagent.data.sources import track_files
        track_files(context["source_manifest"], [path])
    value = _run(
        context,
        Path(context["operation_directory"]) / "inspection",
        request={"operation": "table", "path": path, "sheet": sheet},
    )
    return _inventory(context, f"table:{path}:{sheet}", value)


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
    return _inventory(context, key, value)


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
    decisions = {json.loads(key)[1] for key in context.get("decisions", {})}
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
            if not supported and policy["decision_reference"] not in decisions:
                raise ValueError(
                    f"{key}: window/bit-depth/color change requires an answered user decision"
                )
    if unmatched:
        raise ValueError("Image plan contains unreferenced or invalid images")
    path = Path(context["operation_directory"]) / "image_plan.json"
    atomic_write_json(path, policies)
    versions = publish(context, "ImagePlan", {"path": str(path)}, ("ImageInventory",))
    return result("success", "Image encoding plan recorded", image_plan_path=str(path), **versions)


def compile_data_processing_agent(*, checkpointer=None):
    from medbenchagent.agents.data_pipeline import compile_pipeline
    return compile_pipeline(checkpointer=checkpointer)
