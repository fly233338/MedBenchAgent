"""Interactive discovery of a raw source using the current v5 Source contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict

from medbenchagent.data.contracts import Source, SourcesArguments
from medbenchagent.runtime.llm import complete_validated_json
from medbenchagent.runtime.run_artifacts import atomic_write_json


class RootProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    root: str | None = None
    dataset_name: str | None = None


def _inventory(root: Path) -> dict:
    counts = {"files": 0, "images": 0, "tables": 0}
    sample = []
    image_suffixes = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".nii", ".nii.gz", ".nrrd", ".mha"}
    table_suffixes = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}
    for directory, directories, files in os.walk(root):
        directories.sort()
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            counts["files"] += 1
            lower = name.lower()
            if any(lower.endswith(suffix) for suffix in image_suffixes):
                counts["images"] += 1
            if path.suffix.lower() in table_suffixes:
                counts["tables"] += 1
            if len(sample) < 80:
                sample.append(relative)
    return {"root": str(root), "counts": counts, "sample_files": sample}


def _answer_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("root", "path", "answer", "text", "value"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


def discover_source(context: dict, dataset_id: str) -> dict:
    """Resolve one source directory and persist the decision for run recovery."""
    folder = Path(context["run_directory"]) / "discovery" / "sources" / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    decision_path = folder / "source_decision.json"
    if decision_path.is_file():
        return json.loads(decision_path.read_text(encoding="utf-8"))["entry"]

    proposal_path = folder / "proposal.json"
    if proposal_path.is_file():
        proposal = RootProposal.model_validate_json(proposal_path.read_text(encoding="utf-8"))
    else:
        proposal = complete_validated_json(
            "Extract the raw dataset directory and optional dataset name from the request. "
            "Return JSON with root and dataset_name; use null when absent and never invent a path.",
            context.get("user_requirement", ""), RootProposal,
        )
        atomic_write_json(proposal_path, proposal.model_dump())

    root_text = proposal.root
    while not root_text:
        answer = interrupt({"kind": "clarification", "role": "DataProcessingAgent",
                            "question": f"Where is the raw directory for {dataset_id}?"})
        root_text = _answer_text(answer)
    root = Path(root_text).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Raw dataset directory is unavailable: {root}")
    run_directory = Path(context["run_directory"]).resolve()
    if run_directory.is_relative_to(root):
        raise ValueError("Raw dataset directory must not contain the run output directory")
    inventory = _inventory(root)
    if inventory["counts"]["files"] == 0:
        raise ValueError(f"Raw dataset directory is empty: {root}")
    atomic_write_json(folder / "source_inventory.json", inventory)

    annotation = root / "annotation.jsonl"
    source = Source(id="original", root=str(root), kind="annotation" if annotation.is_file() else "raw")
    entry = {"dataset_id": dataset_id, "name": proposal.dataset_name or root.name,
             "description": "", "sources": [source.model_dump(exclude_none=True)]}
    SourcesArguments.model_validate({"dataset_id": dataset_id, "sources": entry["sources"]})
    atomic_write_json(decision_path, {"entry": entry, "inventory_path": str(folder / "source_inventory.json")})
    experiment_root = context.get("experiments_root")
    experiment_name = context.get("experiment_name")
    if experiment_root and experiment_name:
        from medbenchagent.runtime.sessions import upsert_dataset_source
        upsert_dataset_source(experiment_name, entry, root=Path(experiment_root))
    return entry
