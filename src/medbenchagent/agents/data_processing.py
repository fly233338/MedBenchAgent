"""Optional raw-data preparation through isolated, reproducible Python scripts."""
from __future__ import annotations

import json
import subprocess
from uuid import uuid4
from pathlib import Path

from medbenchagent.audit import _image_errors
from medbenchagent.dataset import load_annotation_jsonl
from medbenchagent.dataset_catalog import DATASET_ID_PATTERN
from medbenchagent.evaluation.data import _load_image_manifest
from medbenchagent.harness import compile_agent, result
from medbenchagent.settings import get_settings


def _workspace(context):
    path = Path(context["data_workspace"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def inspect_files(context, arguments):
    """List source files by relative directory: {directory?: string, offset?: integer}. Returns up to 100 entries."""
    root = Path(context["source_root"])
    folder = (root / arguments.get("directory", "")).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise ValueError("Inspect a directory inside the source root")
    files = sorted(folder.iterdir())
    offset = int(arguments.get("offset", 0))
    listing = [{"path": p.relative_to(root).as_posix(), "directory": p.is_dir(),
                "bytes": p.stat().st_size if p.is_file() else None} for p in files[offset:offset + 100]]
    return result("success", json.dumps({"total": len(files), "entries": listing}, ensure_ascii=False))


def read_text(context, arguments):
    """Read source documentation or text annotations: {path: string, offset?: integer}. Up to 12000 characters."""
    root = Path(context["source_root"])
    path = (root / arguments["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Read a file inside the source root")
    with path.open(encoding="utf-8") as source:
        source.seek(int(arguments.get("offset", 0)))
        text = source.read(12000)
    return result("success", text)


def write_script(context, arguments):
    """Write a reproducible preparation or inspection script: {code: string, mapping: string}. No execution yet."""
    code, mapping = arguments.get("code"), arguments.get("mapping")
    if not isinstance(code, str) or not code.strip() or not isinstance(mapping, str) or not mapping.strip():
        raise ValueError("Both Python code and a mapping explanation are required")
    compile(code, "prepare.py", "exec")
    work = _workspace(context)
    (work / "prepare.py").write_text(code, encoding="utf-8")
    (work / "mapping.md").write_text(mapping, encoding="utf-8")
    return result("success", "Preparation script saved", script_ready=True, script_executed=False)


def execute_script(context, arguments):
    """Execute the saved script in the configured local Docker image: {}. Source is /source, outputs /work."""
    if not context.get("script_ready"):
        raise ValueError("Write a script first")
    settings = get_settings().data_processing
    work = _workspace(context).resolve()
    container_name = "medbench-data-" + uuid4().hex
    project = Path(__file__).resolve().parents[3]
    validator = Path(__file__).resolve().parents[1] / "preparation_validation.py"
    converter = project / ".codex/skills/prepare-medical-images/references/medical_image_conversion.py"
    command = ["docker", "run", "--name", container_name, "--rm", "--pull=never", "--network=none", "--read-only",
               "--memory=4g", "--cpus=2", "--pids-limit=128", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--tmpfs", "/tmp:rw,size=256m",
               "--mount", f"type=bind,source={Path(context['source_root']).resolve()},target=/source,readonly",
               "--mount", f"type=bind,source={work},target=/work", "--workdir", "/work",
               "--mount", f"type=bind,source={validator},target=/validation.py,readonly",
               "--mount", f"type=bind,source={converter},target=/conversion.py,readonly",
               settings["image"], "python", "prepare.py"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=settings["timeout_seconds"], check=False)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False, timeout=30)
        raise ValueError("Preparation exceeded its execution budget") from exc
    if completed.returncode == 0 and (work / "annotation.jsonl").is_file() and (work / "image_manifest.jsonl").is_file():
        completed = subprocess.run(command[:-1] + ["/validation.py"], capture_output=True, text=True,
                                   timeout=settings["timeout_seconds"], check=False)
    summary = (completed.stdout + "\n" + completed.stderr)[-12000:]
    return result("success" if completed.returncode == 0 else "failed", summary,
                  script_executed=completed.returncode == 0)


def finish(context, arguments):
    if not context.get("script_executed"):
        raise ValueError("Execute and validate a preparation script first")
    work = _workspace(context)
    annotation = work / "annotation.jsonl"
    records = load_annotation_jsonl(annotation)
    images = _load_image_manifest(work / "image_manifest.jsonl")
    if {r["image"]["image_id"] for r in records} != set(images):
        raise ValueError("Annotations and image manifest must contain exactly the same images")
    paths = [entry.png_path for entry in images.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("Image manifest paths must be unique")
    for entry in images.values():
        errors = _image_errors({"png_path": entry.relative_png_path, "width": entry.width, "height": entry.height}, work)
        if errors:
            raise ValueError(f"{entry.image_id}: {errors}")
    identifier = context["dataset_id"]
    return result("complete", f"Prepared {len(records)} images", selected_dataset_id=identifier,
                  annotation_path=str(annotation), dataset_profile=None, dataset_access_manifest=None,
                  dataset_entry={"id": identifier, "name": identifier, "description": "Prepared local dataset",
                                 "annotation_path": str(annotation), "available": True})


def describe(context):
    schema = Path(__file__).parents[1] / "schemas" / "annotation.v2.schema.json"
    return {"requirement": context["user_requirement"], "feedback": context.get("feedback"),
            "schema": json.loads(schema.read_text(encoding="utf-8")),
            "script_ready": context.get("script_ready"), "script_executed": context.get("script_executed")}


def compile_data_processing_agent(*, checkpointer=None):
    agent = compile_agent("DataProcessingAgent",
        "Prepare heterogeneous raw annotations and 2D images. Inspect sources and documentation before writing code. "
        "Use source-supported joins, entity scopes and bbox conventions only. Ask for clarification when ambiguous. "
        "Never infer labels from image appearance or filenames. Preserve source facts, account for unmatched rows. "
        "Source /source is read-only. Save annotation.jsonl, image_manifest.jsonl and images under /work. "
        "One annotation and one manifest row per image; manifest keys exactly image_id,png_path,width,height. "
        "annotation image.image_path must be the relative original source path beneath /source, never the output path. "
        "png_path is relative to /work. Canonical boxes are absolute xyxy. Preserve image geometry. "
        "Only 2D raster or single-frame medical images; reject volumes, multiframe, pyramids and whole slides. "
        "Copy PNG/JPEG unchanged. For medical display conversion use documented transforms; ask when unclear. "
        "Available Python packages: pandas, openpyxl, Pillow, pydicom, numpy, nibabel, pynrrd, SimpleITK. "
        "Scripts must be deterministic and rerunnable. No network, external commands or changes outside /work.",
        {"inspect_files": inspect_files, "read_text": read_text, "write_script": write_script,
         "execute_script": execute_script}, describe, finish, checkpointer=checkpointer)
    return agent


def preparation_context(context, source: str, identifier: str) -> dict:
    if not isinstance(identifier, str) or not DATASET_ID_PATTERN.fullmatch(identifier):
        raise ValueError("dataset_id must be a lowercase slug")
    work = context.get("data_workspace") if context.get("source_root") == source and context.get("dataset_id") == identifier else None
    return dict(context, source_root=source, dataset_id=identifier,
                data_workspace=work or str(Path(context["operation_directory"]) / "prepared"))
