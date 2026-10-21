"""One LangGraph orchestrator for the three preparation sessions."""

from __future__ import annotations

import json
import uuid
import shutil
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings

from .context import build_stage_context, resolve_prompt_budget, stage_input_digest, verify_previous_reference
from .contracts import (
    CONTROL_MAPPING_REVISION_REQUIRED,
    AcceptedHandoff,
    ArtifactRef, digest_file, digest_payload,
    Diagnostic,
    Stage,
    StageRequest,
    StageResult,
    ValidationResult,
    STAGE_OUTPUTS,
)
from .publisher import DatasetPublisher
from .validation import validate_stage, accept_mapping, package_dataset, _load_inputs, read_cases
from .images import mapped_images
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
    prepare_text_attempts: int
    prepare_image_attempts: int
    understand_failures: int
    prepare_text_failures: int
    prepare_image_failures: int
    repair_progress: dict[str, Any]
    stage_usage: dict[str, Any]
    text_root: str
    text_artifacts: dict[str, Any]
    package_root: str
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
    revision_basis_handoff: dict[str, Any] | None
    implementation_repair_root: str | None
    publication: dict[str, Any] | None
    preparation_attempt_limits: dict[str, int]
    preparation_stage_budgets: dict[str, Any]
    execution_runtime: dict[str, Any]
    construction_events_path: str
    model: str
    base_url: str
    reasoning_effort: str
    llm_timeout_seconds: float
    llm_max_retries: int


def _event(state, event_type, **values):
    if path := state.get("construction_events_path"):
        record_event(path, event_type, run_id=state.get("run_id"),
                     dataset_id=state["dataset_id"], agent="DataProcessingAgent",
                     stage=state.get("stage"), **values)


def _handoff(state) -> AcceptedHandoff | None:
    value = state.get("accepted_handoff")
    return AcceptedHandoff.model_validate(value, strict=False) if value else None


def _limit(state, name):
    return state.get("preparation_attempt_limits", {"understand": 2, "prepare_text": 2, "prepare_image": 2, "mapping_revision_cycles": 1})[name]


def _fresh_attempt(state, **updates):
    return {"attempt_id": "", "attempt_workspace": "", "attempt_input_digest": "",
            "stage_result": None, "validation": None, "pending_questions": [],
            "revision_basis_handoff": state.get("revision_basis_handoff"),
            "implementation_repair_root": state.get("implementation_repair_root"),
            **updates}


def _candidate_digest(state):
    root = Path(state["attempt_workspace"]) / "work" if state.get("attempt_workspace") else None
    if root is None or not root.exists():
        return digest_payload({})
    files = [path for path in (root / "outputs").rglob("*") if path.is_file()]
    if (root / "accounting.json").is_file():
        files.append(root / "accounting.json")
    return digest_payload({path.relative_to(root).as_posix(): digest_file(path) for path in sorted(files)})


def _retry(state, stage, issues, validation=None):
    """Only host-verified improvement resets the consecutive no-progress budget."""
    history = dict(state.get("repair_progress", {}))
    key = f"{stage.value}:{state.get('mapping_revision_cycles', 0)}"
    prior = history.get(key)
    candidate = _candidate_digest(state)
    checks = set(validation.passed_checks if validation else [])
    signatures = {digest_payload({key: item.get(key) for key in (
        "code", "category", "field", "rule_ids", "issue_ids", "source_refs")}) for item in issues}
    advanced = bool(prior and candidate != prior["candidate_digest"] and (
        checks > set(prior["passed_checks"]) or
        (checks >= set(prior["passed_checks"]) and signatures < set(prior["issues"]))
    ))
    failures = 0 if prior is None or advanced else state.get(f"{stage.value}_failures", 0) + 1
    if prior is None or advanced:
        history[key] = {"candidate_digest": candidate, "passed_checks": sorted(checks),
                        "issues": sorted(signatures)}
    reference_changed = any(item.get("code") == "previous_reference_changed" for item in issues)
    allowed = (not reference_changed
               and failures < _limit(state, stage.value)
               and not any(item.get("category") == "environment" for item in issues))
    return _fresh_attempt(
        state, status="running" if allowed else "blocked", stage=stage.value,
        **{f"{stage.value}_failures": failures},
        repair_progress=history,
        previous_attempt_root=state.get("attempt_workspace"),
        diagnostics=issues, validation_feedback={"issues": issues, "candidate_digest": candidate,
            "passed_checks": sorted(checks), "advanced": advanced, "no_progress_failures": failures},
    )


def _revise_mapping(state, issues):
    cycle = state.get("mapping_revision_cycles", 0)
    previous_prepare_root = state.get("attempt_workspace")
    previous_handoff = state.get("accepted_handoff")
    if (cycle >= _limit(state, "mapping_revision_cycles")
            or state.get("understand_failures", 0) >= _limit(state, "understand")
            or state.get("prepare_text_failures", 0) >= _limit(state, "prepare_text")):
        return {"status": "blocked", "diagnostics": issues, "validation_feedback": {"issues": issues}}
    return _fresh_attempt(
        state, status="running", stage=Stage.UNDERSTAND.value,
        mapping_revision_cycles=cycle + 1, accepted_handoff=None,
        text_root="", text_artifacts={}, package_root="",
        revision_basis_handoff=previous_handoff,
        implementation_repair_root=previous_prepare_root,
        previous_attempt_root=previous_prepare_root,
        diagnostics=issues, validation_feedback={"issues": issues},
    )


def validate_understand_outputs(outputs, state):
    return validate_stage(Path(outputs["mapping.json"]).parent, Stage.UNDERSTAND, state)


def validate_candidate_outputs(outputs, state):
    return validate_stage(Path(outputs["root"]), Stage(state["stage"]), state)


def publish_preparation(state, result, validation):
    from medbenchagent.data.bundle_catalog import verify_bundle
    publication = DatasetPublisher(state["bundle_root"]).publish(state["package_root"], state, validation)
    verified = verify_bundle(publication["revision_path"])
    access, card = verified["ready"]["access"], verified["card"]
    return {**publication, "cases_path": access["cases_path"], "access_manifest": access,
            "card_path": access["dataset_card_path"], "dataset_entry": {
                "dataset_id": state["dataset_id"], "name": card["name"], "description": card["description"],
                "available": True, **{key: access[key] for key in ("cases_path", "image_manifest_path", "profile_path", "dataset_card_path")}}}


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
    handoff = _handoff(current)
    if handoff and handoff.source_snapshot_digest != current["source_snapshot_digest"]:
        raise ValueError("Preparation source snapshot changed; start a new run")
    current.setdefault("stage", Stage.PREPARE_TEXT.value if handoff else Stage.UNDERSTAND.value)
    current.setdefault("understand_attempts", 0)
    current.setdefault("prepare_text_attempts", 0)
    current.setdefault("prepare_image_attempts", 0)
    current.setdefault("mapping_revision_cycles", 0)
    current.setdefault("validation_feedback", {})
    current.setdefault("user_decisions", [])
    current["status"] = "running"
    return current


def _run_stage(state, stage, worker):
    if state.get(f"{stage.value}_failures", 0) >= _limit(state, stage.value):
        return {"status": "blocked", "stage_result": None}
    usage = dict(state.get("stage_usage", {}))
    spent = usage.get(stage.value, {"iterations": 0, "seconds": 0.0})
    budget = resolve_prompt_budget(state, stage)
    if spent["iterations"] >= budget.max_iterations or budget.timeout_seconds - spent["seconds"] < 1:
        return {"status": "blocked", "stage_result": None, "diagnostics": [{
            "code": "stage_budget_exhausted", "stage": stage.value,
            "message": "Cumulative preparation stage execution budget exhausted", "category": "environment"}]}
    remaining = budget.model_copy(update={"max_iterations": budget.max_iterations - spent["iterations"],
                                         "timeout_seconds": int(budget.timeout_seconds - spent["seconds"])})
    count = state.get(f"{stage.value}_attempts", 0) + 1
    attempt_id = f"{stage.value}-{count}"
    epoch = state.get("mapping_revision_cycles", 0)
    session_id = f"{stage.value}-{epoch}"
    root = Path(state["preparation_root"])
    workspace = root / "sessions" / session_id
    snapshot = root / "attempts" / stage.value / attempt_id
    for name in ("work/inputs", "work/outputs", "work/logs", "work/conversations"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    reference = state.get("previous_attempt_root")
    if stage is Stage.UNDERSTAND and state.get("revision_basis_handoff"):
        reference = state.get("implementation_repair_root") or reference
    local = {**state, "stage": stage.value, "attempt": count, "attempt_id": attempt_id,
             "previous_attempt_root": reference}
    input_digest = stage_input_digest(stage, local)
    request_path, result_path = snapshot / "request.json", snapshot / "result.json"
    previous = json.loads(request_path.read_text(encoding="utf-8")) if request_path.is_file() else None
    if previous and previous["input_digest"] != input_digest:
        raise ValueError("Attempt inputs changed; start a new preparation run")
    if previous and result_path.is_file():
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.input_digest != input_digest:
            raise ValueError("Saved attempt result has a different input digest")
    else:
        packet, _, tokens, execution_key = build_stage_context(stage, local, workspace)
        settings = get_settings().llm
        request = StageRequest(
            stage=stage, run_id=state["run_id"], dataset_id=state["dataset_id"],
            attempt=count, attempt_id=attempt_id,
            conversation_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{state['run_id']}:{state['dataset_id']}:{root}:{session_id}").hex,
            execution_key=execution_key, input_digest=input_digest,
            source_snapshot_digest=state["source_snapshot_digest"], workspace=str(workspace),
            prompt_path=str(workspace / "prompt.md"), context_path=str(workspace / "request.json"), context=packet,
            source_roots=state["source_roots"], source_manifests=state["source_manifests"],
            required_outputs=list(STAGE_OUTPUTS[stage]),
            optional_outputs=["metadata.json"] if stage is Stage.PREPARE_TEXT else [],
            prompt_token_count=tokens, budget=remaining,
            construction_events_path=state.get("construction_events_path", ""),
            model=state.get("model") or settings.model or "",
            base_url=state.get("base_url") or settings.base_url or "",
            reasoning_effort=state.get("reasoning_effort") or settings.reasoning_effort or "medium",
            llm_timeout_seconds=state.get("llm_timeout_seconds", settings.timeout_seconds),
            llm_max_retries=state.get("llm_max_retries", settings.max_retries),
            runtime_backend=state.get("execution_runtime", {}).get("backend", "local"),
            previous_attempt_root=reference, worker_runtime=state.get("execution_runtime", {}).get("coding_worker", {}),
            resume_existing=any((workspace / "work/conversations").iterdir()),
            validation_feedback=state.get("validation_feedback", {}),
        )
        if not previous:
            (workspace / "work/control.json").unlink(missing_ok=True)
            (workspace / "result.json").unlink(missing_ok=True)
        atomic_write_json(request_path, request.model_dump(mode="json"))
        atomic_write_json(workspace / "request.json", request.model_dump(mode="json"))
        _event(local, "stage_started", attempt=count, conversation_id=request.conversation_id, status="running")
        started = time.monotonic()
        try:
            result = worker.run(request)
            verify_previous_reference(workspace)
        except Exception as exc:
            result = StageResult(
                stage=stage, attempt=count, attempt_id=attempt_id,
                conversation_id=request.conversation_id, execution_key=execution_key,
                input_digest=input_digest, status="failed", diagnostic=str(exc),
                diagnostics=[Diagnostic(stage=stage, code="worker_execution_failed",
                                        category="environment", blocking=True, message=str(exc))],
            )
        # Immutable candidate snapshots are independent of the continuing SDK session.
        shutil.copytree(workspace / "work", snapshot / "work", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("conversations", "logs"))
        shutil.copy2(workspace / "prompt.md", snapshot / "prompt.md")
        outputs = {name: str(snapshot / "work/outputs" / name) for name in result.outputs}
        result = result.model_copy(update={"outputs": outputs, "input_digest": input_digest,
            "usage": {**result.usage, "host_elapsed_seconds": time.monotonic() - started}})
        atomic_write_json(result_path, result.model_dump(mode="json"))
    usage[stage.value] = {"iterations": spent["iterations"] + max(1, int(result.progress.get("iterations", 0))),
                          "seconds": spent["seconds"] + result.usage.get("host_elapsed_seconds", 0.0)}
    return {"stage": stage.value, "status": "running", f"{stage.value}_attempts": count,
            "stage_usage": usage,
            "attempt_id": attempt_id, "attempt_workspace": str(snapshot),
            "attempt_input_digest": input_digest, "stage_result": result.model_dump(mode="json")}


def _worker_questions(result):
    questions = [item.model_dump(mode="json") for item in result.questions if item.blocking and not item.resolved]
    if not questions:
        raise ValueError("needs_input requires an unresolved blocking question")
    return questions


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
        if stage is not Stage.UNDERSTAND and result.control == CONTROL_MAPPING_REVISION_REQUIRED:
            return _revise_mapping(state, _failure_issues(result))
        if result.status == "needs_input":
            try:
                questions = _worker_questions(result)
            except ValueError as exc:
                return _retry(state, stage, [{"code": "question_invalid", "message": str(exc)}])
            return {"status": "needs_input", "pending_questions": questions}
        if result.status != "submitted":
            if result.status in {"cancelled", "blocked"}:
                return {"status": "blocked", "diagnostics": _failure_issues(result)}
            return _retry(state, stage, _failure_issues(result))
        if stage is Stage.UNDERSTAND:
            validation = validate_understand_outputs(result.outputs, state)
        else:
            handoff = _handoff(state)
            validation = validate_candidate_outputs(
                {"root": str(Path(state["attempt_workspace"]) / "work" / "outputs")}, state,
            )
        _event(state, "stage_validated", status=validation.status)
        issues = [item.model_dump(mode="json") for item in validation.issues]
        if validation.status == "needs_input":
            return {"status": "needs_input",
                    "pending_questions": [item.model_dump(mode="json") for item in validation.questions]}
        if validation.status != "accepted":
            return _retry(state, stage, issues, validation)
        accepted = {f"{stage.value}_failures": 0}
        if stage is Stage.UNDERSTAND:
            handoff = accept_mapping(Path(result.outputs["mapping.json"]).parent, state)
            return _fresh_attempt(
                state, stage=Stage.PREPARE_TEXT.value, status="running",
                accepted_handoff=handoff.model_dump(mode="json"),
                validation_feedback={}, diagnostics=[],
                previous_attempt_root=state.get("implementation_repair_root"),
                revision_basis_handoff=None, implementation_repair_root=None, **accepted,
            )
        if stage is Stage.PREPARE_TEXT:
            root = Path(state["attempt_workspace"]) / "work" / "outputs"
            names = ["cases.jsonl", "metadata.json"]
            artifacts = {name: ArtifactRef(name=name, path=str(root / name), digest=digest_file(root / name)).model_dump(mode="json")
                         for name in names if (root / name).is_file()}
            mapping, plan = _load_inputs(state)
            mapped = mapped_images(mapping, plan, state)
            cases = read_cases(root / "cases.jsonl")
            if not any(mapped[e.value][0].action == "convert"
                       for case in cases for e in case.evidence if e.type == "asset"):
                return {"status": "validated", "text_root": str(root), "text_artifacts": artifacts, **accepted}
            return _fresh_attempt(state, stage=Stage.PREPARE_IMAGE.value, status="running",
                                  text_root=str(root), text_artifacts=artifacts, previous_attempt_root=None, **accepted)
        return {"status": "validated", "validation": validation.model_dump(mode="json"), **accepted}

    def package(state):
        try:
            root, digest = package_dataset(state)
        except (OSError, ValueError, KeyError) as exc:
            return _retry(state, Stage(state["stage"]), [{"code": "final_consistency", "message": str(exc)}])
        validation = ValidationResult(status="accepted", stage=Stage.PREPARE_IMAGE, candidate_digest=digest)
        return {"status": "packaged", "package_root": str(root), "validation": validation.model_dump(mode="json")}

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
            return "package"
        if state.get("status") == "packaged":
            return "publish"
        return state.get("stage", "understand")

    graph = StateGraph(PreparationGraphState)
    graph.add_node("resume", resume_or_start)
    graph.add_node("understand", lambda state: _run_stage(state, Stage.UNDERSTAND, runner))
    graph.add_node("prepare_text", lambda state: _run_stage(state, Stage.PREPARE_TEXT, runner))
    graph.add_node("validate_understand", lambda state: validate_stage(state, Stage.UNDERSTAND))
    graph.add_node("validate_prepare_text", lambda state: validate_stage(state, Stage.PREPARE_TEXT))
    graph.add_node("prepare_image", lambda state: _run_stage(state, Stage.PREPARE_IMAGE, runner))
    graph.add_node("validate_prepare_image", lambda state: validate_stage(state, Stage.PREPARE_IMAGE))
    graph.add_node("package", package)
    graph.add_node("publish", publish)
    graph.add_edge(START, "resume")
    graph.add_conditional_edges("resume", route)
    graph.add_edge("understand", "validate_understand")
    graph.add_edge("prepare_text", "validate_prepare_text")
    graph.add_conditional_edges("validate_understand", route)
    graph.add_conditional_edges("validate_prepare_text", route)
    graph.add_edge("prepare_image", "validate_prepare_image")
    graph.add_conditional_edges("validate_prepare_image", route)
    graph.add_conditional_edges("package", route)
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)
