"""Resumable source inventory, annotation mapping and image preparation harness."""

from __future__ import annotations

import filecmp
import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from medbenchagent.data.validation import load_annotation_jsonl, source_image_path, validate_dataset
from medbenchagent.data.image_assets import load_image_manifest
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import (
    InspectFilesArguments,
    ReadTextArguments,
    TableArguments,
    JoinArguments,
    MappingArguments,
    ScriptArguments,
    ImagePlanArguments,
)
from medbenchagent.settings import get_settings


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(context, output, *, script=None, request=None, annotation=None):
    """Execute generated or fixed code with readonly source, code and annotation mounts."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kind = "script" if script else request["operation"]
    receipt = output.parent / f"{output.name}-{kind}.complete.json"
    identity = {
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
        return _read(output / "result.json") if request else None
    settings = get_settings().data_processing
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
        "--memory=4g",
        "--cpus=2",
        "--pids-limit=128",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,size=256m",
        "--env",
        "PYTHONPATH=/project",
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
            f"type=bind,source={Path(annotation).parent.resolve()},target=/annotation,readonly",
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
            command + [settings["image"]] + arguments,
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
    with source.open(encoding="utf-8") as handle:
        handle.seek(offset)
        text = handle.read(6000)
    return result("success", text)


def inspect_table(context, *, path, sheet=None):
    """Measure table/sheet row counts, types, nulls and key uniqueness."""
    value = _run(
        context,
        Path(context["operation_directory"]) / "inspection",
        request={"operation": "table", "path": path, "sheet": sheet},
    )
    return _inventory(context, f"table:{path}:{sheet}", value)


def inspect_join(context, **arguments):
    """Measure an exact source join's duplicates and unmatched rows."""
    value = _run(
        context,
        Path(context["operation_directory"]) / "inspection",
        request={"operation": "join", **arguments},
    )
    key = "join:" + json.dumps(arguments, sort_keys=True)
    return _inventory(context, key, value)


def save_mapping(context, **mapping):
    """Record source-supported entity/field/coordinate mappings and reconciliation rules."""
    require(context, "Inventory")
    if mapping["unresolved"]:
        raise ValueError("Resolve mapping ambiguities with sources or ask_user before proceeding")
    for path in mapping["sources"]:
        source_image_path(context["source_root"], path)
    inventory = _read(context["inventory_path"])
    if not set(mapping["joins"]) <= set(inventory):
        raise ValueError("Every join must reference an existing measured join inspection")
    known_references = {json.loads(key)[1] for key in context.get("decisions", {})}
    if not set(mapping["decision_references"]) <= known_references:
        raise ValueError("Mapping references an unanswered user decision")
    path = Path(context["operation_directory"]) / "preparation_plan.json"
    atomic_write_json(path, mapping)
    versions = publish(context, "Mapping", {"path": str(path)}, ("Inventory",))
    return result(
        "success",
        "Mapping recorded; implement its deterministic annotation script",
        preparation_plan_path=str(path),
        **versions,
    )


def write_annotation_script(context, *, code):
    """Write only the annotation converter; source is /source and output /work/annotation.jsonl."""
    require(context, "Mapping")
    try:
        compile(code, "prepare_annotations.py", "exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python at line {exc.lineno}: {exc.msg}") from exc
    path = Path(context["operation_directory"]) / "prepare_annotations.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    versions = publish(context, "AnnotationScript", {"path": str(path)}, ("Mapping",))
    return result(
        "success", "Annotation script saved", annotation_script_path=str(path), **versions
    )


def execute_annotations(context):
    """Run the annotation converter twice and validate schema, references and planned count."""
    require(context, "AnnotationScript")
    folder = Path(context["operation_directory"])
    first, second = folder / "annotation", folder / "repeat"
    for output in (first, second):
        _run(context, output, script=context["annotation_script_path"])
    if not filecmp.cmp(first / "annotation.jsonl", second / "annotation.jsonl", shallow=False):
        raise ValueError("Annotation converter is not deterministic")
    records = load_annotation_jsonl(first / "annotation.jsonl")
    plan = _read(context["preparation_plan_path"])
    if len(records) != plan["expected_images"]:
        raise ValueError("Output image population differs from the authoritative mapping count")
    for row in records:
        source_image_path(context["source_root"], row["image"]["image_path"])
    versions = publish(
        context,
        "Annotation",
        {"path": str(first / "annotation.jsonl"), "records": len(records)},
        ("Mapping", "AnnotationScript"),
    )
    return result(
        "success",
        f"Validated {len(records)} deterministic annotation rows",
        prepared_annotation_path=str(first / "annotation.jsonl"),
        **versions,
    )


def inspect_images(context):
    """Inspect only annotation-referenced image content, dimensionality and display requirements."""
    require(context, "Annotation")
    output = Path(context["operation_directory"]) / "inspection"
    details = _run(
        context,
        output,
        request={"operation": "inspect_images"},
        annotation=context["prepared_annotation_path"],
    )
    versions = publish(
        context, "ImageInventory", {"path": str(output / "result.json")}, ("Annotation",)
    )
    formats = {}
    unresolved = []
    for key, item in details.items():
        formats[item["format"]] = formats.get(item["format"], 0) + 1
        if not item["copy"] and not item["preserve_supported"]:
            unresolved.append({"image_id": key, **item})
    return result(
        "success",
        json.dumps(
            {
                "formats": formats,
                "needs_display": unresolved[:20],
                "needs_display_count": len(unresolved),
                "reference": str(output / "result.json"),
            }
        ),
        image_inventory_path=str(output / "result.json"),
        **versions,
    )


def save_image_plan(context, *, overrides):
    """Choose image encodings; default to unchanged PNG/JPEG or lossless PNG representation."""
    require(context, "ImageInventory")
    details = _read(context["image_inventory_path"])
    policies = {
        key: (value.model_dump() if hasattr(value, "model_dump") else value)
        for key, value in overrides.items()
    }
    if set(policies) - set(details):
        raise ValueError("Image plan contains unreferenced images")
    decisions = {json.loads(key)[1] for key in context.get("decisions", {})}
    for key, info in details.items():
        policy = policies.get(key, {"encoding": "preserve"})
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
    path = Path(context["operation_directory"]) / "image_plan.json"
    atomic_write_json(path, policies)
    versions = publish(context, "ImagePlan", {"path": str(path)}, ("ImageInventory",))
    return result("success", "Image encoding plan recorded", image_plan_path=str(path), **versions)


def execute_images(context):
    """Execute the fixed per-dataset image script twice with annotation mounted readonly."""
    require(context, "ImagePlan")
    folder = Path(context["operation_directory"])
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "prepare_images.py"
    policies = _read(context["image_plan_path"])
    code = r"""import json
from pathlib import Path
from medbenchagent.data.validation import load_annotation_jsonl, source_image_path
from medbenchagent.data.image_conversion import inspect_image, write_image
policies = json.loads(POLICY_TEXT)
manifest = []
for index, row in enumerate(load_annotation_jsonl('/annotation/annotation.jsonl')):
    key = row['image']['image_id']
    source = source_image_path('/source', row['image']['image_path'])
    info = inspect_image(source)
    extension = '.jpg' if info['format'] == 'raster:jpeg' else '.png'
    relative = f'images/{index:08d}{extension}'
    write_image(source, Path('/work') / relative, policies.get(key))
    manifest.append({'image_id': key, 'png_path': relative, 'width': info['width'], 'height': info['height']})
Path('/work/image_manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in manifest), encoding='utf-8')
""".replace("POLICY_TEXT", repr(json.dumps(policies)))
    script.write_text(code, encoding="utf-8")
    first, second = folder / "images", folder / "repeat"
    for output in (first, second):
        _run(context, output, script=script, annotation=context["prepared_annotation_path"])
    files = {
        path.relative_to(first)
        for path in first.rglob("*")
        if path.is_file() and path.name not in {"annotation.jsonl", "result.json"}
    }
    if files != {path.relative_to(second) for path in second.rglob("*") if path.is_file()} or any(
        not filecmp.cmp(first / path, second / path, shallow=False) for path in files
    ):
        raise ValueError("Image preparation is not deterministic")
    report = _run(
        context,
        first,
        request={"operation": "validate_images"},
        annotation=context["prepared_annotation_path"],
    )
    # Publish the annotation beside its image manifest only after the readonly image stage completes.
    shutil.copyfile(context["prepared_annotation_path"], first / "annotation.jsonl")
    report.update(
        annotation_deterministic=True,
        images_deterministic=True,
        mapping=context["preparation_plan_path"],
        image_plan=context["image_plan_path"],
        annotation_script=context["annotation_script_path"],
        image_script=str(script),
    )
    report_path = folder / "validation.json"
    atomic_write_json(report_path, report)
    versions = publish(
        context,
        "PreparedDataset",
        {"annotation": str(first / "annotation.jsonl"), "validation": str(report_path)},
        ("Annotation", "ImagePlan"),
    )
    return result(
        "success",
        "Images, annotation population, geometry and reproducibility validated",
        prepared_dataset_path=str(first / "annotation.jsonl"),
        validation_path=str(report_path),
        image_script_path=str(script),
        **versions,
    )


def finish(context):
    """Publish Dataset only after every preparation stage and joint validation succeeds."""
    require(context, "PreparedDataset")
    from medbenchagent.data.dataset import prepare_annotation_dataset

    path = Path(context["prepared_dataset_path"])
    validate_dataset(
        load_annotation_jsonl(path), load_image_manifest(path.parent / "image_manifest.jsonl")
    )
    publish(
        context,
        "DatasetSelection",
        {"path": str(path), "dataset_id": context["dataset_id"]},
        ("PreparedDataset",),
    )
    _, access = prepare_annotation_dataset(path, Path(context["operation_directory"]) / "cache")
    versions = publish(
        context,
        "Dataset",
        {"annotation": str(path), "access": access},
        ("DataConfig", "DatasetSelection"),
    )
    return result(
        "complete",
        "Prepared Dataset published",
        selected_dataset_id=context["dataset_id"],
        annotation_path=str(path),
        dataset_access_manifest=access,
        dataset_entry={
            "id": context["dataset_id"],
            "name": context["dataset_id"],
            "description": "Prepared local dataset",
            "annotation_path": str(path),
            "available": True,
        },
        **versions,
    )


def describe(context):
    return {
        "requirement": context["user_requirement"][:2000],
        "feedback": context.get("feedback"),
        "versions": context.get("artifact_versions", {}),
        "clarification": context.get("clarification"),
        "references": {
            key: context.get(key)
            for key in (
                "inventory_path",
                "preparation_plan_path",
                "annotation_script_path",
                "prepared_annotation_path",
                "image_inventory_path",
                "image_plan_path",
                "validation_path",
            )
        },
    }


def compile_data_processing_agent(*, checkpointer=None):
    tools = {
        function.__name__: function
        for function in (
            inspect_files,
            read_text,
            inspect_table,
            inspect_join,
            save_mapping,
            write_annotation_script,
            execute_annotations,
            inspect_images,
            save_image_plan,
            execute_images,
        )
    }
    return compile_agent(
        "DataProcessingAgent",
        "Prepare source-supported image-centric annotations and 2D image assets. Inventory all tables/sheets, "
        "measure candidate keys/joins, document image population, entity ownership, conflicts and unmatched rows. "
        "Save a structured mapping; resolve ambiguous entities, joins, bbox conventions or derivations with "
        "local ask_user. Never infer labels from filenames, images or medical knowledge. One-to-many findings "
        "remain grouped under one image. Annotation script outputs /work/annotation.jsonl and reads /source only; "
        "use medbenchagent.data.validation.load_annotation_jsonl for validation. Canonical fields: patient "
        "{patient_id,annotation}, case {case_id,annotation}, study {study_id,annotation}, image "
        "{image_id,image_path,modality,anatomy,annotation}; nullable absent entity IDs and modality/anatomy. "
        "BBox is absolute pixel xyxy on its source finding. Execute annotations, inspect only referenced images, "
        "choose image encoding, execute images, finish. Do not request routine approval; ask when reliable "
        "display/bit-depth/color conversion evidence is absent. Never choose volume slices or slide regions. "
        "Read saved scripts/plan portions on demand; do not rewrite completed stages unnecessarily.",
        tools,
        describe,
        finish,
        args_schemas={
            "inspect_files": InspectFilesArguments,
            "read_text": ReadTextArguments,
            "inspect_table": TableArguments,
            "inspect_join": JoinArguments,
            "save_mapping": MappingArguments,
            "write_annotation_script": ScriptArguments,
            "save_image_plan": ImagePlanArguments,
        },
        checkpointer=checkpointer,
    )


def preparation_context(context, source, identifier):
    if context.get("original_source_root") == source and context.get("dataset_id") == identifier:
        return dict(context)
    destination = Path(context["operation_directory"]) / "source"
    if destination.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError("Source directory must not contain this run's output directory")
    marker = destination.parent / "source_ready.json"
    if not marker.is_file():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        atomic_write_json(marker, {"source": source})
    prepared = dict(
        context,
        source_root=str(destination.resolve()),
        original_source_root=source,
        dataset_id=identifier,
    )
    publish(prepared, "Source", {"path": str(destination.resolve())}, ())
    for key in (
        "inventory_path",
        "preparation_plan_path",
        "annotation_script_path",
        "prepared_annotation_path",
        "image_inventory_path",
        "image_plan_path",
        "validation_path",
        "prepared_dataset_path",
    ):
        prepared.pop(key, None)
    return prepared
