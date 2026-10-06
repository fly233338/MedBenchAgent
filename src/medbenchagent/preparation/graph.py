"""One LangGraph orchestrator for the two preparation sessions."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings

from .context import build_stage_context, resolve_prompt_budget, stage_input_digest
from .contracts import AcceptedHandoff, Diagnostic, Stage, StageRequest, StageResult, ValidationResult, STAGE_SPECS
from .publisher import DatasetPublisher
from .validation import create_accepted_handoff, read_question_set, validate_handoff, validate_prepare_candidate
from .worker import OpenHandsWorker


class PreparationGraphState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    run_directory: str
    preparation_root: str
    bundle_root: str
    sources: list[dict[str, Any]]
    source_roots: dict[str, str]
    source_manifests: dict[str, str]
    source_snapshot_digest: str
    user_request: str
    user_decisions: list[dict[str, Any]]
    stage: str
    status: str
    understand_attempts: int
    prepare_attempts: int
    mapping_revision_cycles: int
    attempt_id: str
    attempt_workspace: str
    attempt_input_digest: str
    stage_result: dict[str, Any] | None
    validation_feedback: dict[str, Any]
    validation: dict[str, Any] | None
    previous_attempt_root: str | None
    pending_questions: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    accepted_handoff: dict[str, Any] | None
    accepted_handoff_ref: str | None
    publication: dict[str, Any] | None
    preparation_attempt_limits: dict[str, int]
    preparation_stage_budgets: dict[str, Any]
    execution_runtime: dict[str, Any]
    data_runtime: dict[str, Any]
    construction_events_path: str
    model: str
    base_url: str
    reasoning_effort: str
    llm_timeout_seconds: float
    llm_max_retries: int


def _event(state, event_type, **values):
    if path := state.get("construction_events_path"):
        record_event(path, event_type, run_id=state.get("run_id"),
                     dataset_id=state["dataset_id"], agent="DatasetProcessingAgent",
                     stage=state.get("stage"), **values)


def _handoff(state) -> AcceptedHandoff | None:
    value = state.get("accepted_handoff")
    return AcceptedHandoff.model_validate(value, strict=False) if value else None


def _limit(state, name):
    return state.get("preparation_attempt_limits", {"understand": 2, "prepare": 2, "mapping_revision_cycles": 1})[name]


def _fresh_attempt(state, **updates):
    return {"attempt_id": "", "attempt_workspace": "", "attempt_input_digest": "",
            "stage_result": None, "validation": None, "pending_questions": [], **updates}


def _retry(state, stage, issues):
    allowed = state.get(f"{stage.value}_attempts", 0) < _limit(state, stage.value)
    return _fresh_attempt(
        state, status="running" if allowed else "blocked", stage=stage.value,
        previous_attempt_root=state.get("attempt_workspace"),
        diagnostics=issues, validation_feedback={"issues": issues},
    )


def _revise_mapping(state, issues):
    cycle = state.get("mapping_revision_cycles", 0)
    if (cycle >= _limit(state, "mapping_revision_cycles")
            or state.get("understand_attempts", 0) >= _limit(state, "understand")
            or state.get("prepare_attempts", 0) >= _limit(state, "prepare")):
        return {"status": "blocked", "diagnostics": issues, "validation_feedback": {"issues": issues}}
    return _fresh_attempt(
        state, status="running", stage=Stage.UNDERSTAND.value,
        mapping_revision_cycles=cycle + 1, accepted_handoff=None, accepted_handoff_ref=None,
        previous_attempt_root=state.get("attempt_workspace"),
        diagnostics=issues, validation_feedback={"issues": issues},
    )


def validate_understand_submission(outputs, state):
    return validate_handoff(outputs, state=state, replay=True)


def validate_candidate_submission(outputs, state):
    handoff = _handoff(state)
    return validate_prepare_candidate(
        Path(outputs["cases.jsonl"]).resolve().parent,
        state=state,
        population_path=handoff.population.path if handoff else None,
    )


def publish_preparation(state, result, validation):
    return DatasetPublisher(state["bundle_root"]).publish(
        Path(state["attempt_workspace"]) / "outputs", state, validation,
    )


def resume_or_start(state):
    current = dict(state)
    if current.get("status") == "complete" and current.get("publication"):
        return current
    if current.get("status") == "needs_input":
        answered = {item["issue_id"] for item in current.get("user_decisions", [])}
        if any(item["issue_id"] not in answered for item in current.get("pending_questions", [])):
            return current
        current.update(_fresh_attempt(
            current, previous_attempt_root=current.get("attempt_workspace"), status="running",
        ))
    if current.get("accepted_handoff_ref") and not current.get("accepted_handoff"):
        current["accepted_handoff"] = json.loads(Path(current["accepted_handoff_ref"]).read_text(encoding="utf-8"))
    handoff = _handoff(current)
    if handoff and handoff.source_snapshot_digest != current["source_snapshot_digest"]:
        raise ValueError("Preparation source snapshot changed; start a new run")
    current.setdefault("stage", Stage.PREPARE.value if handoff else Stage.UNDERSTAND.value)
    current.setdefault("understand_attempts", 0)
    current.setdefault("prepare_attempts", 0)
    current.setdefault("mapping_revision_cycles", 0)
    current.setdefault("validation_feedback", {})
    current.setdefault("user_decisions", [])
    current["status"] = "running"
    return current


def _run_stage(state, stage, worker):
    count = state.get(f"{stage.value}_attempts", 0) + 1
    if count > _limit(state, stage.value):
        return {"status": "blocked", "diagnostics": [{
            "stage": stage.value, "code": "stage_attempt_budget",
            "message": f"{stage.value} attempt budget exhausted",
        }], "stage_result": None}
    attempt_id = f"{stage.value}-{count}"
    workspace = Path(state["preparation_root"]) / "attempts" / stage.value / attempt_id
    for name in ("inputs", "work", "outputs", "state", "logs", "conversations"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    local = {**state, "stage": stage.value, "attempt": count, "attempt_id": attempt_id}
    input_digest = stage_input_digest(stage, local)
    request_path, result_path = workspace / "state/request.json", workspace / "state/result.json"
    previous = json.loads(request_path.read_text(encoding="utf-8")) if request_path.is_file() else None
    same_attempt = bool(
        state.get("status") == "running" and previous
        and previous.get("stage") == stage.value
        and previous.get("attempt_id") == attempt_id
        and previous.get("input_digest") == input_digest
    )
    if previous and not same_attempt:
        raise ValueError("Attempt inputs changed; start a new preparation run")
    if same_attempt and result_path.is_file():
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.input_digest != input_digest:
            raise ValueError("Saved attempt result has a different input digest")
    else:
        packet, _, tokens, execution_key = build_stage_context(stage, local, workspace)
        settings = get_settings().llm
        handoff = _handoff(state) if stage is Stage.PREPARE else None
        artifacts = {} if handoff is None else {
            "source_report.json": handoff.source_report, "mapping.json": handoff.mapping,
            "image_population.jsonl": handoff.population, "inspect_source.py": handoff.inspection_script,
        }
        spec = STAGE_SPECS[stage.value]
        request = StageRequest(
            stage=stage, run_id=state["run_id"], dataset_id=state["dataset_id"],
            attempt=count, attempt_id=attempt_id,
            conversation_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{state['run_id']}:{state['dataset_id']}:{attempt_id}").hex,
            execution_key=execution_key, input_digest=packet.identity["input_digest"],
            source_snapshot_digest=state["source_snapshot_digest"], workspace=str(workspace),
            prompt_path=str(workspace / "inputs/prompt.md"), context_path=str(workspace / "inputs/context.json"),
            source_roots=state["source_roots"], source_manifests=state["source_manifests"],
            input_artifacts=artifacts, required_outputs=spec.required_outputs,
            optional_outputs=spec.optional_outputs, prompt_token_count=tokens,
            budget=resolve_prompt_budget(state, stage),
            construction_events_path=state.get("construction_events_path", ""),
            model=state.get("model") or settings.model or "",
            base_url=state.get("base_url") or settings.base_url or "",
            reasoning_effort=state.get("reasoning_effort") or settings.reasoning_effort or "medium",
            llm_timeout_seconds=state.get("llm_timeout_seconds", settings.timeout_seconds),
            llm_max_retries=state.get("llm_max_retries", settings.max_retries),
            runtime_backend=state.get("execution_runtime", {}).get("backend", "local"),
            resume_existing=same_attempt and any((workspace / "conversations").iterdir()),
            validation_feedback=state.get("validation_feedback", {}),
        )
        atomic_write_json(request_path, request.model_dump(mode="json"))
        _event(local, "stage_started", attempt=count, conversation_id=request.conversation_id, status="running")
        try:
            result = worker.run(request)
        except Exception as exc:
            result = StageResult(
                stage=stage, attempt=count, attempt_id=attempt_id,
                conversation_id=request.conversation_id, execution_key=execution_key,
                input_digest=input_digest, status="failed", diagnostic=str(exc),
                diagnostics=[Diagnostic(stage=stage, code="worker_execution_failed",
                                        category="environment", blocking=True, message=str(exc))],
            )
        atomic_write_json(result_path, result.model_dump(mode="json"))
    return {
        "stage": stage.value, "status": "running", f"{stage.value}_attempts": count,
        "attempt_id": attempt_id, "attempt_workspace": str(workspace),
        "attempt_input_digest": input_digest, "stage_result": result.model_dump(mode="json"),
    }


def _worker_questions(result):
    path = result.outputs.get("semantic_questions.json")
    if not path:
        return [{"issue_id": ref, "stage": result.stage.value,
                 "question": "Preparation needs the missing semantic decision.",
                 "impact": "The stage cannot complete without this decision.",
                 "blocking": True, "resolved": False}
                for ref in result.question_refs]
    questions = read_question_set(path, stage=result.stage)
    selected = [item for item in questions if item["issue_id"] in result.question_refs
                and item.get("blocking") and not item.get("resolved")]
    if not selected:
        return [{"issue_id": ref, "stage": result.stage.value,
                 "question": "Preparation needs the missing semantic decision.",
                 "impact": "The stage cannot complete without this decision.",
                 "blocking": True, "resolved": False}
                for ref in result.question_refs]
    return selected


def _failure_issues(result):
    return [item.model_dump(mode="json") for item in result.diagnostics] or [{
        "stage": result.stage.value, "code": "stage_failed",
        "message": result.diagnostic or result.termination_reason or "Coding session failed",
    }]


def compile_preparation_graph(*, worker=None, checkpointer=None):
    runner = worker or OpenHandsWorker()

    def validate_stage(state, stage):
        if not state.get("stage_result"):
            return {"status": "blocked"}
        result = StageResult.model_validate(state["stage_result"], strict=False)
        if stage is Stage.PREPARE and result.control == "mapping_revision_required":
            return _revise_mapping(state, _failure_issues(result))
        if result.status == "needs_input":
            try:
                questions = _worker_questions(result)
            except ValueError as exc:
                return _retry(state, stage, [{"code": "question_invalid", "message": str(exc)}])
            return {"status": "needs_input", "pending_questions": questions}
        if result.status != "submitted":
            if result.status == "cancelled":
                return {"status": "blocked", "diagnostics": _failure_issues(result)}
            return _retry(state, stage, _failure_issues(result))
        if stage is Stage.UNDERSTAND:
            validation = validate_understand_submission(result.outputs, state)
        else:
            handoff = _handoff(state)
            validation = validate_candidate_submission(
                {"cases.jsonl": str(Path(state["attempt_workspace"]) / "outputs/cases.jsonl")}, state,
            )
        _event(state, "stage_validated", status=validation.status)
        issues = [item.model_dump(mode="json") for item in validation.issues]
        if stage is Stage.PREPARE and validation.control == "mapping_revision_required":
            return _revise_mapping(state, issues)
        if validation.status == "needs_input":
            return {"status": "needs_input",
                    "pending_questions": [item.model_dump(mode="json") for item in validation.questions]}
        if validation.status != "accepted":
            return _retry(state, stage, issues)
        if stage is Stage.UNDERSTAND:
            handoff = create_accepted_handoff(state, result.outputs, validation)
            return _fresh_attempt(
                state, stage=Stage.PREPARE.value, status="running",
                accepted_handoff=handoff.model_dump(mode="json"),
                accepted_handoff_ref=str(Path(state["preparation_root"]) / "handoff" / f"{handoff.digest}.json"),
                validation_feedback={}, diagnostics=[], previous_attempt_root=None,
            )
        return {"status": "validated", "validation": validation.model_dump(mode="json")}

    def publish(state):
        result = StageResult.model_validate(state["stage_result"], strict=False)
        validation = ValidationResult.model_validate(state["validation"], strict=False)
        _event(state, "publish_started", status="running")
        publication = publish_preparation(state, result, validation)
        _event(state, "publish_finished", status="complete", revision=publication.get("revision", ""))
        return {"stage": "publish", "status": "complete", "publication": publication}

    def route(state):
        if state.get("status") in {"needs_input", "blocked", "complete"}:
            return END
        if state.get("status") == "validated":
            return "publish"
        return state.get("stage", "understand")

    graph = StateGraph(PreparationGraphState)
    graph.add_node("resume", resume_or_start)
    graph.add_node("understand", lambda state: _run_stage(state, Stage.UNDERSTAND, runner))
    graph.add_node("prepare", lambda state: _run_stage(state, Stage.PREPARE, runner))
    graph.add_node("validate_understand", lambda state: validate_stage(state, Stage.UNDERSTAND))
    graph.add_node("validate_prepare", lambda state: validate_stage(state, Stage.PREPARE))
    graph.add_node("publish", publish)
    graph.add_edge(START, "resume")
    graph.add_conditional_edges("resume", route)
    graph.add_edge("understand", "validate_understand")
    graph.add_edge("prepare", "validate_prepare")
    graph.add_conditional_edges("validate_understand", route)
    graph.add_conditional_edges("validate_prepare", route)
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)
