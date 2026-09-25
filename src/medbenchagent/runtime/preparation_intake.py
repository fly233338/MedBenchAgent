"""Collect the minimum preparation inputs before creating a run."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from medbenchagent.agents.source_discovery import _inventory, _missing_inputs
from medbenchagent.data.dataset_catalog import load_prepared, validate_dataset_id
from medbenchagent.runtime.llm import complete_validated_json
from medbenchagent.runtime.session_supervisor import SessionDecision, SessionTask


class IntakeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str


def parse_answer(field: str, text: str) -> str:
    parsed = complete_validated_json(
        f"Extract the user's answer to the requested {field} field. Return exactly {{\"value\":\"...\"}}. "
        "Preserve names and paths verbatim. Do not invent information.", text, IntakeAnswer)
    if not parsed.value.strip():
        raise ValueError(f"Provide a value for {field}")
    return parsed.value.strip()


def _slug(value: str) -> str | None:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug if slug and all(ord(char) < 128 for char in value) else None


def _question(field: str, prompt: str) -> dict:
    return {"field": field, "prompt": prompt, "answer": None, "status": "pending"}


def resolve_intake(decision: SessionDecision, catalog: list[dict], sources: tuple[dict, ...]) -> tuple[list[SessionTask], list[dict]]:
    """Return executable tasks or the complete set of currently missing fields."""
    drafts = [task.model_dump() for task in decision.tasks]
    questions = []
    if drafts and drafts[0]["kind"] == "prepare_dataset":
        draft = drafts[0]
        name = (draft.get("dataset_name") or "").strip()
        raw_id = (draft.get("dataset_id") or "").strip()
        if not name and raw_id:
            known = next((item for item in (*catalog, *sources) if item["dataset_id"].casefold() == raw_id.casefold()), None)
            name = known["name"] if known else raw_id
        if not name:
            questions.append(_question("dataset_name", "What is the dataset name?"))
        identifier = _slug(raw_id or name) if raw_id or name else None
        if identifier is None and (raw_id or name):
            questions.append(_question("dataset_id", "What lowercase ASCII dataset ID should be used?"))
        elif identifier is not None:
            validate_dataset_id(identifier)
            existing = next((item for item in (*catalog, *sources) if item["dataset_id"] == identifier), None)
            if existing and existing["name"].casefold() != name.casefold():
                questions.append(_question("dataset_id", f"ID {identifier} belongs to {existing['name']}. Provide another ID."))
        draft["dataset_name"] = name or None
        draft["dataset_id"] = identifier
        source = next((item for item in sources if item["dataset_id"] == identifier), None)
        entry = next((item for item in catalog if item["dataset_id"] == identifier), None)
        supplied_root = (draft.get("source_root") or "").strip()
        if source and supplied_root:
            roots = {str(Path(item["root"]).expanduser().resolve()) for item in source["sources"]}
            if str(Path(supplied_root).expanduser().resolve()) not in roots:
                questions.append(_question("dataset_id", "This ID has a different configured source. Provide a new dataset ID."))
        prepared = False
        if entry and entry["available"] and not draft["rebuild"]:
            try:
                load_prepared(entry)
                prepared = True
            except (ValueError, OSError, KeyError):
                pass
        if not prepared and source is None:
            root_text = (draft.get("source_root") or "").strip()
            if not root_text:
                questions.append(_question("source_root", "Where is the raw dataset directory?"))
            else:
                root = Path(root_text).expanduser().resolve()
                if not root.is_dir():
                    questions.append(_question("source_root", f"Raw directory is unavailable: {root}. Provide an accessible directory."))
                else:
                    missing = _missing_inputs(root, _inventory(root))
                    if missing:
                        questions.append(_question("source_root", f"{root} lacks {', '.join(missing)}. Provide the complete directory."))
                    else:
                        draft["source_root"] = str(root)
    if questions:
        return [], questions
    return [SessionTask.model_validate(item) for item in drafts], []
