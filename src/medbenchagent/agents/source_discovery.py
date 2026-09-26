"""Inspect a user-supplied raw directory before preparing a new dataset."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict

from medbenchagent.data.contracts import Source, SourcesArguments
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.sessions import upsert_dataset_source


_TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".nii", ".nrrd", ".tif", ".tiff"}


class RootProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    root: str | None
    dataset_name: str | None


def _inventory(root: Path) -> dict:
    files: list[str] = []
    counts = {"tables": 0, "images": 0, "other": 0}
    table_headers: dict[str, list[str]] = {}
    visited = 0
    for directory, subdirs, names in os.walk(root):
        subdirs.sort()
        names.sort()
        visited += 1
        if visited > 2000:
            break
        for name in names:
            relative = str((Path(directory) / name).relative_to(root))
            suffix = Path(name).suffix.lower()
            if suffix in _IMAGE_SUFFIXES:
                counts["images"] += 1
            elif suffix in _TABLE_SUFFIXES:
                counts["tables"] += 1
                if suffix in {".csv", ".tsv"} and len(table_headers) < 8:
                    try:
                        with (root / relative).open("r", encoding="utf-8-sig", newline="") as stream:
                            table_headers[relative] = next(csv.reader(stream, delimiter="\t" if suffix == ".tsv" else ","))[:40]
                    except (OSError, UnicodeError, StopIteration):
                        table_headers[relative] = []
            else:
                counts["other"] += 1
            if len(files) < 80:
                files.append(relative)
    return {"root": str(root), "counts": counts, "sample_files": files,
            "table_headers": table_headers, "directory_scan_limited": visited > 2000}


def _missing_inputs(root: Path, inventory: dict) -> list[str]:
    missing = []
    if not (root / "annotation.jsonl").is_file() and not inventory["counts"]["tables"]:
        missing.append("annotation or source table")
    if not inventory["counts"]["images"] and not (root / "image_manifest.jsonl").is_file():
        missing.append("images or image manifest")
    return missing


def discover_source(context: dict, dataset_id: str) -> dict:
    """Persist a validated source decision before synchronizing experiment TOML."""
    folder = Path(context["run_directory"]) / ".runtime" / "source_discovery" / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    decision_path = folder / "source_decision.json"
    if decision_path.is_file():
        entry = json.loads(decision_path.read_text(encoding="utf-8"))["entry"]
    else:
        proposal_path = folder / "proposal.json"
        if proposal_path.is_file():
            proposal = RootProposal.model_validate_json(proposal_path.read_text(encoding="utf-8"))
        else:
            if context.get("intake_source_root") or context.get("intake_dataset_name"):
                proposal = RootProposal(root=context.get("intake_source_root"),
                                        dataset_name=context.get("intake_dataset_name"))
            else:
                proposal = RootProposal.model_validate(complete_json(
                    "Extract the raw dataset root path and dataset name from the user's request. "
                    "Return exactly root and dataset_name; use null when absent. Do not invent paths or medical facts.",
                    context["user_requirement"], RootProposal,
                ))
            atomic_write_json(proposal_path, proposal.model_dump())
        from medbenchagent.runtime.preparation_intake import parse_answer
        root_text = proposal.root
        question = f"Where is the raw directory for {dataset_id}?"
        while True:
            if not root_text:
                answer = interrupt({"kind": "clarification", "role": "DataProcessingAgent",
                                    "question": question})
                try:
                    root_text = parse_answer("source_root", answer)
                except ValueError as exc:
                    question = f"Could not understand the directory: {exc}. Provide the raw dataset directory."
                    continue
            root = Path(root_text).expanduser().resolve()
            if not root.is_dir():
                question = f"The raw directory is unavailable: {root}. Provide an accessible directory."
            elif Path(context["run_directory"]).resolve().is_relative_to(root):
                question = "Raw directory must not contain this run's output directory. Provide a separate source directory."
            else:
                inventory = _inventory(root)
                atomic_write_json(folder / "source_inventory.json", inventory)
                missing = _missing_inputs(root, inventory)
                if not missing:
                    break
                question = f"{root} lacks {', '.join(missing)}. Provide the directory containing the dataset tables and images."
            root_text = None
        annotation = (root / "annotation.jsonl").is_file()
        source = Source(id="original", root=str(root), kind="annotation" if annotation else "raw")
        proposed_name = proposal.dataset_name or ""
        name = (proposed_name if context.get("intake_dataset_name") else
                proposed_name if proposed_name and
                proposed_name.casefold() in (context["user_requirement"] + root.name).casefold()
                else root.name)
        entry = {"dataset_id": dataset_id, "name": name,
                 "description": "", "sources": [source.model_dump(exclude_none=True)]}
        SourcesArguments.model_validate({"dataset_id": dataset_id, "sources": entry["sources"]})
        atomic_write_json(decision_path, {"entry": entry, "inventory_path": str(folder / "source_inventory.json")})
    name = context.get("experiment_name")
    if name:
        upsert_dataset_source(name, entry, root=Path(context["experiments_root"]))
    return entry
