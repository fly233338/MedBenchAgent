"""Data Processing tools."""

import hashlib
import json
from pathlib import Path
from langchain.tools import tool, ToolRuntime
from medbenchagent.compilation.artifacts import read_json_artifact, register_file
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.data.dataset import file_digest
from medbenchagent.data.task_data import (
    SourceManifest,
    TaskDataReader,
    snapshot_task_data,
)
from medbenchagent.generation.backends import backend_catalog
from medbenchagent.agents._shared.handoff import (
    get_ref,
    inspect_artifact,
    publish_value,
    tool_result,
)

from .state import DataProcessingContext, _candidate, data_contracts, validate_candidate


def data_processing_tools():
    @tool
    def describe_data_contracts() -> dict:
        """Get exact Profile/TaskData/Case/Asset schemas and implemented backend capabilities."""
        return {"schemas": data_contracts(), "backends": backend_catalog()}

    @tool
    def inspect_sources(
        artifact_id: str, runtime: ToolRuntime[DataProcessingContext], offset: int = 0
    ) -> dict:
        """Inspect a delegated SourceManifest and a bounded top-level listing; not a full scan or Profile."""
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        ref = get_ref(runtime, artifact_id, "source_manifest")
        source = SourceManifest.model_validate(read_json_artifact(runtime.context.directory, ref))
        rows = []
        for entry in source.sources:
            root = Path(entry.root)
            if not root.is_absolute() or not root.is_dir():
                raise ValueError("Delegated source roots must be existing absolute directories")
            from medbenchagent.data.sources import inspect_source
            rows.append({"source_id": entry.source_id, **inspect_source(
                root, offset=offset, limit=runtime.context.settings.page_size)})
        return {"dataset": source.dataset.model_dump(), "sources": rows}

    @tool
    def invoke_coding(
        objective: str,
        input_artifact_ids: list[str],
        constraints: list[str],
        runtime: ToolRuntime[DataProcessingContext],
        workstream_id: str | None = None,
    ):
        """Execute or repair a data program. Returns candidates only; validation and submission are separate."""
        context = runtime.context
        sessions = list(runtime.state.get("coding_sessions", []))
        identity = hashlib.sha256(
            f"{context.invocation_id}/{runtime.tool_call_id}".encode()
        ).hexdigest()[:24]
        if workstream_id is None:
            workstream_id = f"dp-{identity}"
            if workstream_id not in sessions:
                sessions.append(workstream_id)
        elif workstream_id not in sessions:
            raise ValueError("Coding Workstream is not owned by this Data Processing agent")
        inputs = [get_ref(runtime, key) for key in input_artifact_ids]
        request = WorkRequest(
            request_id=f"call-{identity}",
            compilation_id=context.compilation_id,
            agent_id="coding", operation="execute",
            workstream_id=workstream_id,
            objective=objective,
            inputs=inputs,
            constraints=[
                *context.request.constraints,
                *constraints,
                context.instructions["coding_handoff"],
                "Parent operation: " + context.request.operation,
                "Contracts: "
                + json.dumps(data_contracts(), ensure_ascii=False)
                + " Backend reference contracts: "
                + json.dumps(backend_catalog(), ensure_ascii=False),
            ],
        )
        calls = dict(runtime.state.get("coding_calls", {}))
        if request.request_id in calls:
            if calls[request.request_id]["request"] != request.model_dump(mode="json"):
                raise ValueError("A Coding call identity cannot change its instructions")
            return tool_result(
                runtime, calls[request.request_id], updates={"coding_sessions": sessions}
            )
        candidate, files = context.coding.execute_candidate(
            context.directory, request, context.goal
        )
        receipt = {
            "call_id": request.request_id,
            "request": request.model_dump(mode="json"),
            "outcome": candidate.outcome,
            "summary": candidate.summary,
            "issues": [v.model_dump() for v in candidate.issues],
            "needs": [v.model_dump() for v in candidate.needs],
            "files": files,
        }
        calls[request.request_id] = receipt
        return tool_result(
            runtime, receipt, updates={"coding_sessions": sessions, "coding_calls": calls}
        )

    @tool
    def inspect_coding(
        runtime: ToolRuntime[DataProcessingContext], call_id: str | None = None
    ) -> dict:
        """Inspect this agent's Coding call receipts and private Workstream IDs, without executing code."""
        calls = runtime.state.get("coding_calls", {})
        if call_id is not None:
            return calls[call_id]
        return {
            "workstreams": runtime.state.get("coding_sessions", []),
            "calls": [
                {"call_id": key, "outcome": value["outcome"], "summary": value["summary"]}
                for key, value in list(calls.items())[-runtime.context.settings.page_size :]
            ],
        }

    @tool
    def inspect_data_candidate(
        call_id: str, path: str, runtime: ToolRuntime[DataProcessingContext], text_offset: int = 0
    ):
        """Read a candidate Manifest or report without accepting it."""
        _, _, actual = _candidate(runtime, call_id, path)
        if text_offset < 0:
            raise ValueError("Invalid text offset")
        text = actual.read_text(encoding="utf-8")
        end = text_offset + runtime.context.settings.preview_chars
        return {"text": text[text_offset:end], "next_text_offset": end if end < len(text) else None}

    @tool
    def validate_data_candidate(
        call_id: str, path: str, runtime: ToolRuntime[DataProcessingContext]
    ):
        """Check actual members, source permissions, Case structure and coverage without formal publication."""
        value, _, kind = validate_candidate(runtime, call_id, path)
        return {
            "valid": True,
            "kind": kind,
            "dataset": value.dataset.model_dump(),
            "members": len(value.members),
            "case_count": getattr(value, "case_count", None),
        }

    @tool
    def inspect_candidate_cases(
        call_id: str,
        path: str,
        runtime: ToolRuntime[DataProcessingContext],
        offset: int = 0,
        limit: int = 5,
        text_offset: int = 0,
    ):
        """Read actual prepared Case rows before acceptance, without formally publishing the candidate."""
        if offset < 0 or not 1 <= limit <= runtime.context.settings.page_size or text_offset < 0:
            raise ValueError("Invalid candidate Case page")
        _, verified, kind = validate_candidate(runtime, call_id, path)
        if kind != "task_data":
            raise ValueError("This candidate is not a prepared TaskData collection")
        reader = TaskDataReader(verified)
        text = json.dumps(
            [case.model_dump(mode="json") for case in reader.cases[offset : offset + limit]],
            ensure_ascii=False,
        )
        end = text_offset + runtime.context.settings.preview_chars
        return {
            "text": text[text_offset:end],
            "case_count": len(reader.cases),
            "next_text_offset": end if end < len(text) else None,
            "next_offset": offset + limit if offset + limit < len(reader.cases) else None,
        }

    @tool
    def submit_data_artifacts(
        call_id: str, paths: list[str], runtime: ToolRuntime[DataProcessingContext]
    ):
        """Validate and publish selected data candidates. The Host, not Coding, assigns formal references."""
        checked = [validate_candidate(runtime, call_id, path) for path in paths]
        if not checked or len(paths) != len(set(paths)):
            raise ValueError("Select unique candidate Manifest paths")
        refs = []
        for value, verified, kind in checked:
            if kind == "dataset_profile":
                refs.append(publish_value(runtime, kind, value))
            else:
                target = (
                    runtime.context.directory
                    / "artifacts"
                    / runtime.context.invocation_id
                    / "task_data"
                    / file_digest(verified)
                )
                path = snapshot_task_data(verified, target)
                refs.append(
                    register_file(
                        runtime.context.directory,
                        runtime.context.request,
                        path.relative_to(runtime.context.directory).as_posix(),
                        kind=kind,
                    )
                )
        return tool_result(
            runtime, {"artifacts": [ref.model_dump(mode="json") for ref in refs]}, artifacts=refs
        )

    return [
        inspect_artifact,
        describe_data_contracts,
        inspect_sources,
        invoke_coding,
        inspect_coding,
        inspect_data_candidate,
        inspect_candidate_cases,
        validate_data_candidate,
        submit_data_artifacts,
    ]
