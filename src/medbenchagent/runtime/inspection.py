"""Run-local evidence payloads with scoped, version-bound, paged access."""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from pydantic import Field
from medbenchagent.runtime.dependencies import require
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty


class EvidencePage(NoArguments):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)
    kind: str | None = None
    source_id: str | None = None
    current_only: bool = True


class EvidenceRead(NoArguments):
    reference: NonEmpty
    pointer: list[str] = Field(default_factory=list)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)
    text_offset: int = Field(default=0, ge=0)


def _scope(context):
    return (context.get("dataset_scope", context.get("selected_dataset_id")),
            context.get("agent_role", "DataProcessingAgent" if context.get("source_scope") else "TaskPlanningAgent"))


def _folder(context):
    return Path(context["run_directory"]) / ".runtime" / "evidence"


def _index(context):
    folder = _folder(context)
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "index.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
    return db


def save_evidence(context, operation, arguments, value, dependencies, *, source_ids=(), kind="inspection", reference=None):
    records = require(context, *dependencies)
    dataset, role = _scope(context)
    reference = reference or str(uuid4())
    metadata = {"reference": reference, "dataset_id": dataset, "role": role, "source_ids": sorted(source_ids),
                "source_id": next(iter(source_ids)) if len(source_ids) == 1 else None,
                "kind": kind, "operation": operation, "arguments": arguments,
                "dependencies": {r["kind"]: r["id"] for r in records.values()}}
    path = _folder(context) / (reference + ".json")
    payload = {**metadata, "value": value}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Evidence identity cannot be overwritten")
    else:
        atomic_write_json(path, payload)
    with _index(context) as db:
        db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?)", (reference, json.dumps(metadata)))
    db.close()
    return reference


def _validate(context, metadata):
    if (metadata["dataset_id"], metadata["role"]) != _scope(context):
        raise ValueError("Evidence belongs to another dataset or agent")
    current = require(context, *metadata["dependencies"])
    if any(current[key]["id"] != version for key, version in metadata["dependencies"].items()):
        raise ValueError("Evidence is stale; inspect the current inputs")


def evidence(context, reference):
    with _index(context) as db:
        row = db.execute("SELECT metadata FROM evidence WHERE id=?", (reference,)).fetchone()
    db.close()
    if row is None:
        raise ValueError("Unknown evidence reference")
    metadata = json.loads(row[0])
    _validate(context, metadata)
    return json.loads((_folder(context) / (reference + ".json")).read_text(encoding="utf-8"))


def inspected(context, operation, arguments, value, dependencies, *, source_id=None, **updates):
    from medbenchagent.runtime.harness import result
    kind = "document" if operation in {"read_text", "read_dataset_document"} else "inspection"
    ref = save_evidence(context, operation, arguments, value, dependencies,
                        source_ids=[source_id] if source_id else [], kind=kind)
    return result("success", json.dumps({"inspection_reference": ref, "inspection": value}, ensure_ascii=False)[:6000],
                  evidence_index_path=str(_folder(context) / "index.sqlite"), **updates)


def inspection(context, reference, *, source_id=None):
    receipt = evidence(context, reference)
    if receipt["source_ids"] != ([source_id] if source_id else []):
        raise ValueError("Inspection reference belongs to another source")
    return receipt


def evidence_page(context, *, offset=0, limit=20, kind=None, source_id=None, current_only=True):
    dataset, role = _scope(context)
    rows, total = [], 0
    with _index(context) as db:
        for (text,) in db.execute("SELECT metadata FROM evidence ORDER BY rowid"):
            item = json.loads(text)
            if (item["dataset_id"], item["role"]) != (dataset, role):
                continue
            if kind is not None and item["kind"] != kind or source_id is not None and source_id not in item["source_ids"]:
                continue
            try:
                _validate(context, item)
                valid = True
            except ValueError:
                valid = False
            if current_only and not valid:
                continue
            if offset <= total < offset + limit:
                rows.append({"reference": item["reference"], "kind": item["kind"], "operation": item["operation"],
                             "source_ids": item["source_ids"], "current": valid})
            total += 1
    db.close()
    return {"entries": rows, "total": total, "next_offset": offset + limit if offset + limit < total else None}


def list_evidence(context, **arguments):
    """Page available evidence identities; use read_evidence to recall facts after observations expire."""
    from medbenchagent.runtime.harness import result
    return result("success", json.dumps(evidence_page(context, **arguments), ensure_ascii=False))


def read_evidence(context, *, reference, pointer=None, offset=0, limit=20, text_offset=0):
    """Read current evidence; pointer selects nested keys/list indices, offset pages containers or text."""
    from medbenchagent.runtime.harness import result
    pending = context.get("global_issue")
    if pending and pending.get("answer_reference") == reference:
        from medbenchagent.runtime.clarifications import issue_context
        receipt = evidence(issue_context(context, pending), reference)
    else:
        receipt = evidence(context, reference)
    value = receipt["value"]
    try:
        for part in pointer or []:
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Unknown evidence pointer") from exc
    if isinstance(value, str):
        page = {"text": value[offset:offset + 4000], "next_offset": offset + 4000 if offset + 4000 < len(value) else None}
    elif isinstance(value, (dict, list)):
        entries = list(value.items()) if isinstance(value, dict) else value
        page = {"entries": entries[offset:offset + limit], "total": len(entries),
                "next_offset": offset + limit if offset + limit < len(entries) else None}
    else:
        page = {"value": value}
    text = json.dumps(page, ensure_ascii=False)
    if len(text) > 5000 or text_offset:
        page = {"serialized_page": text[text_offset:text_offset + 4000],
                "next_text_offset": text_offset + 4000 if text_offset + 4000 < len(text) else None}
    return result("success", json.dumps({"reference": reference, "kind": receipt["kind"], **page}, ensure_ascii=False))


def inspection_index(context):
    return evidence_page(context, limit=10)
