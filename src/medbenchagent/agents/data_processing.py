"""Data Processing tools own candidate acceptance; OpenHands owns code execution."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool, ToolRuntime

from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact, register_file
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.data.dataset import DatasetSource, file_digest
from medbenchagent.data.task_data import (
    SourceManifest,
    DatasetProfile,
    ProfileReader,
    TaskDataManifest,
    TaskDataReader,
    TaskCase,
    PreparedAsset,
    seal_profile,
    seal_task_data,
    snapshot_task_data,
    write_once,
)
from medbenchagent.generation.backends import backend_catalog
from medbenchagent.planning.contracts import TaskPlan
from medbenchagent.runtime.native import chat_model, loop_middleware
from .handoff import (
    WorkState,
    WorkContext,
    WorkResponse,
    WorkInstructions,
    binding,
    get_ref,
    inspect_artifact,
    invoke_work,
    publish_value,
    tool_result,
)


class DataProcessingState(WorkState):
    coding_sessions: list[str]
    coding_calls: dict[str, dict]


@dataclass(frozen=True)
class DataProcessingContext(WorkContext):
    coding: object
    goal: str


def data_contracts():
    return {
        schema.__name__: schema.model_json_schema()
        for schema in (SourceManifest, DatasetProfile, TaskDataManifest, TaskCase, PreparedAsset)
    }


def _allowed_sources(context, call):
    roots = [context.directory / "coding" / call["request"]["workstream_id"] / "workspace"]
    for ref in context.request.inputs:
        if ref.kind not in {"source_manifest", "dataset_profile", "dataset", "task_data"}:
            continue
        path = resolve_artifact(context.directory, ref)
        value = read_json_artifact(context.directory, ref)
        roots.extend(
            (path.parent / source["root"]).resolve() for source in value.get("sources", [])
        )
    return roots


def _candidate(runtime, call_id, path):
    call = runtime.state.get("coding_calls", {})[call_id]
    receipt = next((row for row in call["files"] if row["path"] == path), None)
    if receipt is None:
        raise ValueError("Select a file returned by this Coding call")
    actual = Path(receipt["read_path"]).resolve()
    if file_digest(actual) != receipt["digest"]:
        raise ValueError("Coding candidate changed after execution; make a new Coding call")
    return call, receipt, actual


def validate_candidate(runtime, call_id, path):
    call, receipt, actual = _candidate(runtime, call_id, path)
    schema = {"dataset_profile": DatasetProfile, "task_data": TaskDataManifest}.get(receipt["kind"])
    if schema is None:
        raise ValueError(
            "Submit a dataset_profile or task_data Manifest; supporting files remain referenced members"
        )
    value = schema.model_validate_json(actual.read_bytes())
    allowed = _allowed_sources(runtime.context, call)
    sources = []
    for source in value.sources:
        root = (actual.parent / source.root).resolve()
        if not any(root.is_relative_to(parent.resolve()) for parent in allowed):
            raise ValueError(
                "Candidate source root is outside delegated sources and this Coding workspace"
            )
        sources.append(DatasetSource(source_id=source.source_id, root=str(root)))
    value = value.model_copy(update={"sources": sources})
    if isinstance(value, DatasetProfile):
        sealed = seal_profile(value, actual.parent)
    else:
        if value.backend_id not in {row["backend_id"] for row in backend_catalog()}:
            raise ValueError("Unknown TaskData backend")
        plans = [
            TaskPlan.model_validate(read_json_artifact(runtime.context.directory, ref))
            for ref in runtime.context.request.inputs
            if ref.kind == "task_plan"
        ]
        if plans and not any(
            plan.dataset == value.dataset
            and set(value.scenario_ids) <= plan.scenarios.keys()
            and all(
                value.backend_id in plan.scenarios[key].backend_ids for key in value.scenario_ids
            )
            and set(value.requirement_ids) <= plan.requirements.keys()
            for plan in plans
        ):
            raise ValueError(
                "Candidate backend, Scenarios or requirements do not match a delegated TaskPlan"
            )
        if value.profile is not None:
            profile_ref = get_ref(runtime, value.profile.artifact_id, "dataset_profile")
            if binding(profile_ref) != value.profile:
                raise ValueError("TaskData binds a stale Profile")
            ProfileReader(resolve_artifact(runtime.context.directory, profile_ref))
        sealed = seal_task_data(value, actual.parent)
    content = (sealed.model_dump_json(indent=2) + "\n").encode("utf-8")
    verified = actual.parent / ("validated-" + hashlib.sha256(content).hexdigest() + ".json")
    write_once(verified, content)
    (ProfileReader if isinstance(sealed, DatasetProfile) else TaskDataReader)(verified)
    return sealed, verified, receipt["kind"]


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
            files = sorted(root.iterdir())
            selected = files[offset : offset + runtime.context.settings.page_size]
            rows.append(
                {
                    "source_id": entry.source_id,
                    "root": entry.root,
                    "coverage": "top-level only",
                    "entries": [
                        {"name": path.name, "directory": path.is_dir()} for path in selected
                    ],
                    "next_offset": offset + len(selected)
                    if offset + len(selected) < len(files)
                    else None,
                }
            )
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
            agent_id="coding",
            workstream_id=workstream_id,
            objective=objective,
            inputs=inputs,
            constraints=[
                *context.request.constraints,
                *constraints,
                "Return candidate Manifest paths with kind dataset_profile or task_data. "
                "Supporting files may remain beside the Manifest. Host seals member digests and checks counts. "
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


def build_data_processing_spec(settings, coding, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    agent = create_agent(
        model,
        tools=data_processing_tools(),
        response_format=ToolStrategy(WorkResponse),
        state_schema=DataProcessingState,
        context_schema=DataProcessingContext,
        checkpointer=True,
        name="data_processing_agent",
        middleware=[
            WorkInstructions(DataProcessingState),
            *loop_middleware(
                model,
                settings.loop,
                settings.context_chars,
                write_tools=(
                    "invoke_coding",
                    "validate_data_candidate",
                    "inspect_candidate_cases",
                    "submit_data_artifacts",
                ),
            ),
        ],
    )

    def node(state, config, runtime):
        context = DataProcessingContext(
            runtime.context.directory,
            WorkRequest.model_validate(state["request"]),
            settings,
            instructions,
            "data_processing",
            coding,
            runtime.context.goal,
        )
        return invoke_work(agent, context, config)

    return AgentSpec(
        "Inspect delegated raw sources, produce evidence-grounded Profiles, prepare backend-scoped TaskData, "
        "and validate/submit data. Owns internal OpenHands Coding calls and local repair; requires source_manifest, "
        "dataset_profile or dataset inputs and any applicable TaskPlan/data requirements.",
        node,
    )
