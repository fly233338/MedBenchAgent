"""Model decisions with bounded, inspectable context owned by the Supervisor."""

import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field

from medbenchagent.runtime.llm import (
    RequestContextBudgetError, complete_validated_json, json_request_chars,
)
from medbenchagent.settings import SupervisorSettings

from .artifacts import preview_artifact
from .contracts import AskUser, Contract, Delegate, Finish, Identifier, Text, WorkRecord, WorkResult
from .interfaces import SupervisorView, record_summary


class _ModelDelegation(Contract):
    action: Literal["delegate"]
    agent_id: Identifier
    workstream_id: Identifier | None = None
    objective: Text
    scope: list[Text] = Field(default_factory=list)
    input_artifact_ids: list[Identifier] = Field(default_factory=list)
    constraints: list[Text] = Field(default_factory=list)


class _Inspection(Contract):
    action: Literal["inspect"]
    target: Literal["artifacts", "workstreams", "work", "preview"]
    ids: list[Identifier] = Field(default_factory=list, description="IDs of the selected target type")
    kind: Text | None = None
    request_id: Identifier | None = None
    workstream_id: Identifier | None = None
    agent_id: Identifier | None = None
    feedback_only: bool = False
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)


class SupervisorResponse(Contract):
    decision: Annotated[_ModelDelegation | _Inspection | AskUser | Finish, Field(discriminator="action")]


class SupervisorLookupLimit(RuntimeError):
    """Reading is bounded; exhausting the budget is not evidence that work is finished."""


def _result_view(result: WorkResult) -> dict:
    return {
        "request_id": result.request_id, "outcome": result.outcome, "summary": result.summary,
        "artifact_ids": [ref.artifact_id for ref in result.artifacts],
        "artifact_count": len(result.artifacts), "issue_count": len(result.issues), "need_count": len(result.needs),
        "issues": [issue.model_dump() for issue in result.issues],
        "needs": [{"need_id": need.need_id, "kind": need.kind, "summary": need.summary,
                   "detail_artifact_id": need.detail_ref.artifact_id if need.detail_ref else None}
                  for need in result.needs],
    }


def _work_view(record: WorkRecord) -> dict:
    request = record.request
    return {
        "agent_id": request.agent_id, "workstream_id": request.workstream_id,
        "objective": request.objective, "scope": list(request.scope), "constraints": list(request.constraints),
        "input_artifact_ids": [ref.artifact_id for ref in request.inputs],
        **_result_view(record.result),
    }


def _excerpt(value: Any, text_limit: int, item_limit: int, field: str = "") -> tuple[Any, bool]:
    """A Supervisor-only projection. Identifiers and query parameters retain their meaning."""
    if field == "query":
        return value, False
    if isinstance(value, str):
        if field in {"kind", "outcome", "ids"} or field.endswith(("_id", "_ids")):
            return value, False
        return value[:text_limit], len(value) > text_limit
    if isinstance(value, list):
        items = [_excerpt(item, text_limit, item_limit, field) for item in value[:item_limit]]
        return [item for item, _ in items], len(value) > item_limit or any(flag for _, flag in items)
    if isinstance(value, dict):
        entries = {key: _excerpt(item, text_limit, item_limit, key) for key, item in value.items()}
        truncated = any(flag for _, flag in entries.values())
        result = {key: item for key, (item, _) in entries.items()}
        if truncated:
            result["truncated"] = True
        return result, truncated
    return value, False


class ModelSupervisor:
    def __init__(self, directory: Path, settings: SupervisorSettings, *, instructions: str):
        if not settings.llm.model:
            raise ValueError("Resolve the Supervisor model before starting a run")
        self.directory = Path(directory).resolve()
        self.settings = settings
        self.instructions = instructions

    @staticmethod
    def _artifact_catalog(view: SupervisorView) -> list[dict]:
        producers = {ref.artifact_id: row.request for row in view.history for ref in row.result.artifacts}
        rows = []
        for ref in view.artifacts:
            request = producers.get(ref.artifact_id)
            parts = PurePosixPath(ref.path).parts
            initial_request = parts[1] if len(parts) > 2 and parts[0] == "artifacts" else None
            rows.append({"artifact_id": ref.artifact_id, "kind": ref.kind,
                         "request_id": request.request_id if request else initial_request,
                         "agent_id": request.agent_id if request else None,
                         "workstream_id": request.workstream_id if request else None})
        return rows

    def _validate_inspection(self, view: SupervisorView, query: _Inspection) -> None:
        ceiling = self.settings.max_artifacts if query.target == "preview" else self.settings.catalog_page_size
        if query.limit is not None and query.limit > ceiling:
            raise ValueError(f"Inspection limit cannot exceed {ceiling}")
        if len(query.ids) > self.settings.catalog_page_size or len(query.ids) != len(set(query.ids)):
            raise ValueError("Inspection IDs must be unique and fit catalog_page_size")
        known = ({ref.artifact_id for ref in view.artifacts} if query.target in {"artifacts", "preview"}
                 else {item.workstream_id for item in view.workstreams} if query.target == "workstreams"
                 else {row.request.request_id for row in view.history})
        if unknown := set(query.ids) - known:
            raise ValueError(f"Unknown inspection IDs: {sorted(unknown)}")
        if query.agent_id is not None and query.agent_id not in view.agents:
            raise ValueError(f"Unregistered Agent: {query.agent_id}")
        if (query.workstream_id is not None
                and query.workstream_id not in {item.workstream_id for item in view.workstreams}):
            raise ValueError(f"Unknown Workstream ID: {query.workstream_id}")
        if query.kind is not None and query.target != "artifacts":
            raise ValueError("kind filters the artifact catalog only")
        if query.feedback_only and query.target != "work":
            raise ValueError("feedback_only filters reported work needs/issues only")
        if query.target == "preview" and (not query.ids or any(
                value is not None for value in (query.kind, query.request_id, query.workstream_id, query.agent_id))):
            raise ValueError("preview requires artifact IDs and accepts no provenance filters")
        if query.target == "workstreams" and query.request_id is not None:
            raise ValueError("Use work inspection to find a request's Workstream")

    def _inspect(self, view: SupervisorView, query: _Inspection) -> dict:
        self._validate_inspection(view, query)
        if query.target in {"artifacts", "preview"}:
            rows, key = self._artifact_catalog(view), "artifact_id"
        elif query.target == "workstreams":
            rows, key = view.workstream_summaries(), "workstream_id"
        else:
            rows, key = [_work_view(row) for row in view.history], "request_id"
        matching = [row for row in rows if (not query.ids or row[key] in query.ids)
                    and all(expected is None or row.get(name) == expected
                            for name, expected in (("kind", query.kind), ("request_id", query.request_id),
                                                   ("workstream_id", query.workstream_id), ("agent_id", query.agent_id)))
                    and (not query.feedback_only or row.get("need_count") or row.get("issue_count"))]
        if query.offset > len(matching):
            raise ValueError(f"Inspection offset exceeds {len(matching)} matching records")
        limit = query.limit or (self.settings.max_artifacts if query.target == "preview"
                                else self.settings.catalog_page_size)
        selected = matching[query.offset:query.offset + limit]
        if query.target == "preview":
            known = {ref.artifact_id: ref for ref in view.artifacts}
            selected = [preview_artifact(self.directory, known[row["artifact_id"]], limit=self.settings.artifact_chars)
                        for row in selected]
        end = query.offset + len(selected)
        return {"query": query.model_dump(mode="json", exclude_none=True), "items": selected,
                "total": len(matching), "offset": query.offset, "next_offset": end if end < len(matching) else None}

    def _context(self, view: SupervisorView, inspections: tuple[dict, ...] = ()) -> dict:
        result = view.last_result
        candidates = ([need.detail_ref for need in result.needs if need.detail_ref is not None]
                      + list(result.artifacts)) if result else list(view.artifacts)
        known = {ref.artifact_id: ref for ref in view.artifacts}
        for ref in candidates:
            if known.get(ref.artifact_id) != ref:
                raise ValueError("Preview references an unregistered artifact")
        unique = {ref.artifact_id: ref for ref in candidates}
        priority = list(unique)
        if view.history:
            priority += [ref.artifact_id for ref in view.history[-1].request.inputs]
        priority += [ref.artifact_id for ref in reversed(view.artifacts)]
        priority = list(dict.fromkeys(priority))[:self.settings.catalog_page_size]
        catalog = {row["artifact_id"]: row for row in self._artifact_catalog(view)}
        previews = [preview_artifact(self.directory, ref, limit=self.settings.artifact_chars)
                    for ref in list(unique.values())[:self.settings.max_artifacts]]
        # These facts are never truncated; an oversized mandatory core is an explicit error.
        required = {"compilation_id": view.compilation_id, "goal": view.goal, "agents": dict(view.agents),
                    "workstream_agents": list(view.workstream_agents), "user_answers": list(view.answers),
                    "catalog_totals": {"artifacts": len(view.artifacts), "workstreams": len(view.workstreams),
                                       "work": len(view.history), "feedback_records": sum(
                                           bool(row.result.needs or row.result.issues) for row in view.history)},
                    "lookup_limits": {"remaining": self.settings.max_lookups - len(inspections),
                                      "catalog_page_size": self.settings.catalog_page_size,
                                      "preview_items": self.settings.max_artifacts,
                                      "preview_chars": self.settings.artifact_chars}}
        optional = {
            "last_result": _result_view(result) if result else None,
            "history": [dict(record_summary(row), need_count=len(row.result.needs), issue_count=len(row.result.issues))
                        for row in view.history[-self.settings.history_items:]],
            "workstreams": view.workstream_summaries()[-self.settings.history_items:],
            "available_artifacts": [catalog[key] for key in priority if key in catalog],
            "artifact_previews": previews,
            "inspections": [dict(item, items=list(item["items"])) for item in inspections],
        }
        text_limit, item_limit = self.settings.artifact_chars, self.settings.catalog_page_size
        while True:
            context = dict(required)
            clipped = []
            for name, value in optional.items():
                if isinstance(value, list):
                    projected = [_excerpt(item, text_limit, item_limit) for item in value]
                    context[name] = [item for item, _ in projected]
                    truncated = any(flag for _, flag in projected)
                else:
                    context[name], truncated = _excerpt(value, text_limit, item_limit)
                if truncated:
                    clipped.append(name)
            for page in context["inspections"]:
                end = page["offset"] + len(page["items"])
                page["next_offset"] = end if end < page["total"] else None
            context["omitted_counts"] = {
                "history": len(view.history) - len(context["history"]),
                "workstreams": len(view.workstreams) - len(context["workstreams"]),
                "artifacts": len(view.artifacts) - len(context["available_artifacts"]),
                "previews": len(unique) - len(context["artifact_previews"]),
                "inspections": len(inspections) - len(context["inspections"]),
            }
            context["truncated_sections"] = clipped
            size = json_request_chars(self.instructions, json.dumps(context, ensure_ascii=False),
                                      SupervisorResponse, self.settings.llm)
            if size <= self.settings.context_chars:
                return context
            # Drop less important projections before reducing detail in the current feedback/read.
            for name in ("history", "workstreams", "available_artifacts", "artifact_previews"):
                if optional[name]:
                    optional[name].pop(0 if name in {"history", "workstreams"} else -1)
                    break
            else:
                if len(optional["inspections"]) > 1:
                    optional["inspections"].pop(0)
                elif optional["inspections"] and len(optional["inspections"][0]["items"]) > 1:
                    optional["inspections"][0]["items"].pop()
                elif text_limit > 1 or item_limit > 1:
                    text_limit, item_limit = max(1, text_limit // 2), max(1, item_limit // 2)
                else:
                    raise RequestContextBudgetError(size, self.settings.context_chars)

    def __call__(self, view: SupervisorView):
        known = {ref.artifact_id: ref for ref in view.artifacts}
        inspections, reads = [], {}

        def resolve(response: SupervisorResponse):
            decision = response.decision
            if isinstance(decision, _Inspection):
                self._validate_inspection(view, decision)
                key = decision.model_dump_json()
                if len(inspections) < self.settings.max_lookups and key not in reads:
                    # Validate offsets against the filtered catalog inside the existing repair path.
                    reads[key] = self._inspect(view, decision)
                return decision
            if not isinstance(decision, _ModelDelegation):
                return decision
            if decision.agent_id not in view.agents:
                raise ValueError(f"Unregistered Agent: {decision.agent_id}")
            view.validate_workstream(decision.agent_id, decision.workstream_id)
            identifiers = decision.input_artifact_ids
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Input artifact IDs must be unique")
            if unknown := set(identifiers) - known.keys():
                raise ValueError(f"Unknown artifact IDs: {sorted(unknown)}")
            return Delegate(agent_id=decision.agent_id, objective=decision.objective,
                            workstream_id=decision.workstream_id, scope=decision.scope,
                            constraints=decision.constraints, inputs=[known[key] for key in identifiers])

        while True:
            response = complete_validated_json(
                self.instructions, json.dumps(self._context(view, tuple(inspections)), ensure_ascii=False),
                SupervisorResponse, validate=resolve, request_char_budget=self.settings.context_chars,
                llm_settings=self.settings.llm,
            )
            decision = resolve(response)
            if not isinstance(decision, _Inspection):
                return decision
            if len(inspections) >= self.settings.max_lookups:
                raise SupervisorLookupLimit(f"Supervisor exhausted max_lookups={self.settings.max_lookups}")
            inspections.append(reads[decision.model_dump_json()])
