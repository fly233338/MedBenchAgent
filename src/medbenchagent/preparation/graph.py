"""LangGraph preparation subgraph and deterministic host routing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TypedDict

from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings

from .context import ContextBudgetError, build_stage_context
from .contracts import (
    AcceptedHandoff, ArtifactRef, Diagnostic, ResumeState,
    Stage, StageRequest, StageResult, STAGE_SPECS, digest_file, digest_payload,
)
from .layout import WorkspaceLayout, validate_identifier
from .worker import OpenHandsWorker


class PreparationGraphState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    dataset_scope: str
    run_directory: str
    preparation_root: str
    stage: str
    status: str
    attempt: int
    attempt_mode: str
    sources: list[dict[str, Any]]
    source_roots: dict[str, str]
    source_manifests: dict[str, str]
    artifacts: dict[str, ArtifactRef | dict[str, Any]]
    accepted_artifacts: dict[str, ArtifactRef | dict[str, Any]]
    user_request: str
    user_decisions: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    validation_diagnostics: list[dict[str, Any]]
    pending_questions: list[dict[str, Any]]
    stage_result: dict[str, Any]
    publication: dict[str, Any]
    preparation_workspace: str
    preparation_candidate: dict[str, Any]
    attempt_id: str
    session_seq: int
    parent_attempt_id: str | None
    execution_key: str
    resume_state_ref: str | None
    accepted_handoff: dict[str, Any]
    accepted_handoff_ref: str
    validation: dict[str, Any]
    candidate_artifacts: dict[str, ArtifactRef | dict[str, Any]]
    budgets: dict[str, Any]
    publish_transaction: dict[str, Any]
    current_milestone: str


def preparation_root(state: dict[str, Any]) -> Path:
    dataset_id = str(state.get("dataset_id") or state.get("dataset_scope") or "dataset")
    root = Path(state.get("preparation_root") or (Path(state["run_directory"]) / "preparation" / dataset_id))
    for name in ("inputs", "artifacts", "validation"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def attempt_directory(state: dict[str, Any], stage: Stage, attempt: int) -> Path:
    # The checkpoint chooses the attempt identity.  Directory existence is a
    # recovery signal, never a reason to invent a new attempt number.
    attempt_id = str(state.get("attempt_id") or attempt)
    validate_identifier(attempt_id, field="attempt_id")
    layout = WorkspaceLayout.create(
        preparation_root(state).parent.parent,
        dataset_id=str(state.get("dataset_id") or state.get("dataset_scope") or "dataset"),
        stage=stage,
        attempt_id=attempt_id,
        session_seq=int(state.get("session_seq", 0)),
        source_roots=_source_roots(state),
        backend=str((state.get("execution_runtime") or {}).get("backend", "local")),
    )
    return layout.attempt_root


def _source_roots(state: dict[str, Any]) -> dict[str, str]:
    roots = state.get("source_roots") or {}
    if roots:
        return {str(key): str(value) for key, value in roots.items()}
    return {str(item["id"]): str(item["root"]) for item in state.get("sources", []) if item.get("id") and item.get("root")}


def _event(state: dict[str, Any], event_type: str, *, stage: Stage | str | None = None, attempt: int | None = None, **values: Any) -> None:
    path = state.get("construction_events_path")
    if not path:
        return
    from medbenchagent.runtime.construction import record_event

    stage_value = stage.value if isinstance(stage, Stage) else stage or state.get("stage", "")
    values = dict(values)
    conversation_id = values.pop("conversation_id", None) or state.get("conversation_id")
    record_event(path, event_type,
                 run_id=state.get("run_id"), dataset_id=state.get("dataset_id") or state.get("dataset_scope"),
                 stage=stage_value, attempt=attempt if attempt is not None else state.get("attempt"),
                 session_seq=state.get("session_seq"), milestone=state.get("current_milestone"),
                 conversation_id=conversation_id, **values)


def _candidate_artifacts(result: StageResult, stage: Stage) -> dict[str, ArtifactRef]:
    values = {}
    for name, path in result.outputs.items():
        candidate = Path(path)
        if not candidate.is_file():
            continue
        values[name] = ArtifactRef(name=name, path=str(candidate.resolve()), digest=digest_file(candidate), producer="openhands", stage=stage.value, status="candidate")
    return values


def _accepted_artifacts(result: StageResult, stage: Stage, *, handoff_id: str = "") -> dict[str, ArtifactRef]:
    """Create accepted bindings only after the host validator succeeds."""
    values = _candidate_artifacts(result, stage)
    return {
        name: artifact.model_copy(update={
            "producer": "host",
            "status": "accepted",
            "purpose": f"Accepted {stage.value} handoff {handoff_id}" if handoff_id else f"Accepted {stage.value} artifact",
        })
        for name, artifact in values.items()
    }


def _worker_llm_config(state: dict[str, Any]) -> dict[str, Any]:
    """Resolve the run-frozen Coding Worker configuration at the request boundary.

    Older checkpoints do not contain these fields because they were omitted
    from the specialist context.  The active settings context is the fallback
    for those runs; new runs copy the same values into the child state.
    """
    settings = get_settings().llm
    return {
        "model": str(state.get("model") or settings.model or "").strip(),
        "base_url": str(state.get("base_url") or settings.base_url or "").strip(),
        "reasoning_effort": str(state.get("reasoning_effort") or settings.reasoning_effort or "medium"),
        "llm_timeout_seconds": (state.get("llm_timeout_seconds")
                                if state.get("llm_timeout_seconds") is not None
                                else settings.timeout_seconds),
        "llm_max_retries": (state.get("llm_max_retries")
                            if state.get("llm_max_retries") is not None
                            else settings.max_retries),
    }


def _request(state: dict[str, Any], stage: Stage, attempt: int, attempt_path: Path, packet, token_count: int, execution_key: str) -> StageRequest:
    roots = _source_roots(state)
    manifests = state.get("source_manifests") or {}
    llm = _worker_llm_config(state)
    attempt_id = str(state.get("attempt_id") or attempt_path.name)
    session_seq = int(state.get("session_seq", 0))
    conversation_id = __import__("uuid").uuid5(
        __import__("uuid").NAMESPACE_URL,
        f"{state.get('run_id','run')}:{state.get('dataset_id',state.get('dataset_scope','dataset'))}:{stage.value}:{attempt_id}:{session_seq}",
    ).hex
    return StageRequest(
        stage=stage,
        run_id=str(state.get("run_id") or "run"),
        dataset_id=str(state.get("dataset_id") or state.get("dataset_scope") or "dataset"),
        attempt=attempt,
        conversation_id=conversation_id,
        execution_key=execution_key,
        workspace=str(attempt_path),
        prompt_path=str(attempt_path / "inputs" / "prompt.md"),
        context_path=str(attempt_path / "inputs" / "context.json"),
        source_roots=roots,
        source_manifests={str(key): str(value) for key, value in manifests.items()},
        input_artifacts={name: ArtifactRef.model_validate(value) if not isinstance(value, ArtifactRef) else value for name, value in state.get("accepted_artifacts", {}).items()},
        input_digest=str(packet.identity["input_digest"]),
        source_snapshot_digest=str(packet.identity.get("source_snapshot_digest", "")),
        prompt_token_count=token_count,
        required_outputs=list(STAGE_SPECS[stage.value].required_outputs),
        optional_outputs=list(STAGE_SPECS[stage.value].optional_outputs),
        budget=__import__("medbenchagent.preparation.context", fromlist=["_budget"])._budget(state, stage),
        construction_events_path=str(state.get("construction_events_path") or ""),
        model=llm["model"],
        base_url=llm["base_url"],
        reasoning_effort=llm["reasoning_effort"],
        llm_timeout_seconds=llm["llm_timeout_seconds"],
        llm_max_retries=llm["llm_max_retries"],
        resume_existing=bool(state.get("resume_existing", False)),
        attempt_id=attempt_id,
        session_seq=session_seq,
        parent_attempt_id=state.get("parent_attempt_id"),
        resume_state_ref=state.get("resume_state_ref"),
        contract_digest=digest_payload(STAGE_SPECS[stage.value].model_dump(mode="json")),
        milestone=str(state.get("current_milestone") or ("inventory" if stage is Stage.UNDERSTAND else "implementation")),
        runtime_backend=str((state.get("execution_runtime") or {}).get("backend", "")),
    )


def _run_stage(state: dict[str, Any], stage: Stage, worker: OpenHandsWorker) -> dict[str, Any]:
    # A continuation reuses the checkpoint-selected attempt/conversation;
    # repairs create a fresh attempt.  The old "directory exists => +1"
    # behaviour made recovery duplicate work and changed input identity.
    resume_existing = bool(state.get("resume_existing"))
    attempt = int(state.get("attempt", 0)) if resume_existing and state.get("attempt") else int(state.get("attempt", 0)) + 1
    attempt_path = attempt_directory(state, stage, attempt)
    try:
        attempt = int(attempt_path.name)
    except ValueError:
        attempt = int(state.get("attempt", 0) or 1)
    local = dict(state)
    local["attempt_number"] = attempt
    local["preparation_attempt"] = attempt
    local["attempt_id"] = attempt_path.name
    local["current_milestone"] = "inventory" if stage is Stage.UNDERSTAND else "implementation"
    try:
        packet, _prompt, token_count, execution_key = build_stage_context(stage, local, attempt_path)
    except ContextBudgetError as exc:
        import uuid
        conversation_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{state.get('run_id','run')}:{state.get('dataset_id',state.get('dataset_scope','dataset'))}:{stage.value}:{attempt_path.name}:{state.get('session_seq', 0)}").hex
        result = StageResult(stage=stage, run_id=str(state.get("run_id") or ""), dataset_id=str(state.get("dataset_id") or state.get("dataset_scope") or ""), attempt=attempt, conversation_id=conversation_id, execution_key="context-unavailable",
                             status="failed", diagnostic=str(exc), termination_reason="required_context_too_large",
                             diagnostics=[Diagnostic(stage=stage, code="required_context_too_large", category="contract", blocking=True, message=str(exc))])
        atomic_write_json(attempt_path / "result.json", result.model_dump(mode="json"))
        return {"stage": stage.value, "attempt": attempt, "stage_result": result.model_dump(mode="json"),
                "preparation_workspace": str(attempt_path), "preparation_status": "failed"}
    request = _request(state, stage, attempt, attempt_path, packet, token_count, execution_key)
    atomic_write_json(attempt_path / "request.json", request.model_dump(mode="json"))
    _event(state, "stage_started", stage=stage, attempt=attempt, conversation_id=request.conversation_id,
           execution_key=execution_key, context_path=str(attempt_path / "inputs" / "context.json"),
           prompt_path=str(attempt_path / "inputs" / "prompt.md"), token_count=token_count)
    try:
        result = worker.run(request)
    except Exception as exc:
        code = "required_context_too_large" if isinstance(exc, ContextBudgetError) else "worker_execution_failed"
        reason = code
        result = StageResult(stage=stage, run_id=request.run_id, dataset_id=request.dataset_id, attempt=attempt, conversation_id=request.conversation_id, execution_key=execution_key,
                             status="failed", diagnostic=str(exc), termination_reason=reason, input_digest=request.input_digest,
                             diagnostics=[Diagnostic(stage=stage, code=code, category="contract" if code == "required_context_too_large" else "environment", blocking=True, message=str(exc))])
    atomic_write_json(attempt_path / "result.json", result.model_dump(mode="json"))
    _event(state, "stage_submitted" if result.status == "submitted" else "stage_paused" if result.status == "needs_input" else "stage_failed",
           stage=stage, attempt=attempt, conversation_id=request.conversation_id, status=result.status,
           termination_reason=result.termination_reason, outputs=list(result.outputs), diagnostics=result.diagnostic)
    updates: dict[str, Any] = {
        "stage": stage.value,
        "attempt": attempt,
        "stage_result": result.model_dump(mode="json"),
        "preparation_workspace": str(attempt_path),
        "preparation_status": result.status,
        "attempt_id": request.attempt_id,
        "session_seq": request.session_seq,
        "execution_key": request.execution_key,
    }
    if result.status == "submitted":
        produced = _candidate_artifacts(result, stage)
        updates["candidate_artifacts"] = produced
        updates["artifacts"] = produced
    if result.question_refs:
        updates["pending_questions"] = result.question_refs
    # A worker declaration is a recoverable diagnostic projection.  Host
    # validation and checkpoint state remain authoritative for completion.
    try:
        resume = ResumeState(
            stage=stage, milestone="submission", attempt_id=request.attempt_id,
            session_seq=request.session_seq, input_digest=request.input_digest,
            accepted_handoff_ref=state.get("accepted_handoff_ref"),
            active_issue_ids=[str(item.get("issue_id")) for item in state.get("diagnostics", []) if isinstance(item, dict) and item.get("issue_id")],
            next_action="validate submission" if result.status == "submitted" else result.termination_reason,
        )
        atomic_write_json(attempt_path / "state" / "resume.json", resume.model_dump(mode="json"))
        updates["resume_state_ref"] = str(attempt_path / "state" / "resume.json")
    except Exception:
        pass
    return updates


def run_preparation_graph(state: dict[str, Any], *, worker: OpenHandsWorker | None = None, max_attempts: int = 3) -> dict[str, Any]:
    """Run understand then prepare, stopping at a host-owned handoff.

    This synchronous facade is used by the existing DatasetProcessingAgent and
    is also convenient for recovery tests.  The compiled LangGraph below uses
    the same node functions and routing decisions.
    """
    current = dict(state)
    current = resume_or_start(current)
    current.setdefault("status", "running")
    runner = worker or OpenHandsWorker()
    repair_attempts = int(current.get("repair_attempts", max_attempts))
    stage_attempts = {Stage.UNDERSTAND.value: 0, Stage.PREPARE.value: 0}
    for _ in range(max_attempts + repair_attempts + 2):
        stage = Stage(current["stage"])
        stage_attempts[stage.value] += 1
        if stage_attempts[stage.value] > max_attempts and current.get("attempt_mode") not in {"repair", "clarification"}:
            current.update(status="blocked", diagnostics=[{"code": "stage_attempt_budget", "stage": stage.value, "message": "Stage attempt budget exhausted", "blocking": True}])
            return current
        updates = _run_stage(current, stage, runner)
        current.update(updates)
        result = StageResult.model_validate(current["stage_result"], strict=False)
        if result.status != "submitted":
            current["status"] = result.status
            return current
        if any(item.code == "mapping_revision_required" for item in result.diagnostics):
            current.update(status="running", stage=Stage.UNDERSTAND.value, attempt=0, attempt_mode="repair",
                           previous_outputs_root=str(Path(current.get("preparation_workspace", "")) / "outputs"),
                           diagnostics=[item.model_dump(mode="json") for item in result.diagnostics])
            continue
        if stage is Stage.UNDERSTAND:
            handoff = validate_understand_submission(result.outputs, current)
            _event(current, "stage_validated", stage=Stage.UNDERSTAND, attempt=current.get("attempt"), status=handoff["status"], diagnostics=handoff.get("issues", []))
            if handoff["status"] == "needs_input":
                current.update(status="needs_input", pending_questions=handoff.get("issues", []), stage=Stage.UNDERSTAND.value)
                return current
            if handoff["status"] != "accepted":
                current.update(status="blocked", diagnostics=handoff.get("issues", []), stage=Stage.UNDERSTAND.value)
                return current
            accepted = _accepted_artifacts(result, Stage.UNDERSTAND)
            accepted_handoff = create_accepted_handoff(current, result, handoff)
            current.update(
                accepted_artifacts=accepted,
                artifacts=accepted,
                accepted_handoff=accepted_handoff.model_dump(mode="json"),
                accepted_handoff_ref=str(preparation_root(current) / "handoff" / "understand" / f"{accepted_handoff.handoff_id}.json"),
                stage=Stage.PREPARE.value, attempt=0, attempt_id="", session_seq=0, attempt_mode="initial",
            )
            continue
        if stage is Stage.PREPARE:
            _materialize_accepted_handoff(current, result.outputs)
            validation = validate_candidate_submission(result.outputs, current)
            _event(current, "stage_validated", stage=Stage.PREPARE, attempt=current.get("attempt"), status=validation["status"],
                   diagnostics=validation.get("issues", []))
            if validation["status"] == "needs_input":
                current.update(status="needs_input", stage=Stage.PREPARE.value, pending_questions=validation.get("issues", []))
                return current
            if validation["status"] != "accepted":
                issues = validation.get("issues", [])
                if any("mapping_revision_required" in item.get("code", "") for item in issues if isinstance(item, dict)):
                    current.update(status="running", stage=Stage.UNDERSTAND.value, attempt=0, attempt_mode="repair",
                                   previous_outputs_root=str(Path(current.get("preparation_workspace", "")) / "outputs"),
                                   validation_diagnostics=issues, diagnostics=issues)
                    continue
                current.update(status="blocked", stage=Stage.PREPARE.value, validation_diagnostics=issues, diagnostics=issues)
                return current
            publication = publish_preparation(current, result.outputs, validation)
            _event(current, "stage_completed", stage="publish", attempt=current.get("attempt"), status="complete",
                   publication_path=publication.get("path"))
            current.update(status="complete", stage="publish", publication=publication,
                           preparation_candidate={"candidate_path": str(Path(result.outputs["cases.jsonl"]).parent), "status": "submitted"})
            return current
    current["status"] = "blocked"
    current["diagnostics"] = [{"code": "attempt_budget", "stage": current.get("stage", "understand"), "message": "Preparation attempt budget exhausted", "blocking": True}]
    return current


def _read_questions(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return [{"issue_id": "invalid-semantic-questions", "question": "semantic_questions.json is invalid", "blocking": True}]
    if isinstance(payload, dict):
        return payload.get("questions", []) if isinstance(payload.get("questions", []), list) else []
    return payload if isinstance(payload, list) else []


def validate_understand_submission(outputs: dict[str, str], state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the structural handoff before ``prepare`` receives it."""
    if state and state.get("canonical_validation"):
        from .validation import validate_handoff
        result = validate_handoff(outputs, state=state, replay=True)
        return {
            "status": result.status,
            "issues": [item.model_dump(mode="json") for item in result.issues],
            "questions": [item.model_dump(mode="json") for item in result.questions],
            "validation_result": result.model_dump(mode="json"),
        }
    required = [name for name in STAGE_SPECS[Stage.UNDERSTAND.value].required_outputs if name != "submission.json"]
    issues = []
    for name in required:
        path = Path(outputs.get(name, ""))
        if not path.is_file():
            issues.append({"code": "missing_output", "stage": "understand", "message": f"Missing output: {name}", "path": name, "blocking": True})
    if issues:
        return {"status": "rejected", "issues": issues}
    try:
        from .contracts import PreparationMapping, SourceReport
        report_payload = json.loads(Path(outputs["source_report.json"]).read_text(encoding="utf-8"))
        SourceReport.model_validate(report_payload)
        if not report_payload.get("sources") or not isinstance(report_payload.get("population"), dict):
            raise ValueError("source_report must identify sources and population")
        for index, check in enumerate(report_payload.get("checks", []), 1):
            if not check.get("id") or "result" not in check or not check.get("reproduction"):
                raise ValueError(f"source_report.checks[{index}] must include id, result and reproduction")
        mapping_payload = json.loads(Path(outputs["mapping.json"]).read_text(encoding="utf-8"))
        PreparationMapping.model_validate(mapping_payload)
        for index, rule in enumerate(mapping_payload.get("fields", []), 1):
            if not rule.get("rule_id"):
                raise ValueError(f"mapping.fields[{index}] is missing rule_id")
            if not rule.get("evidence_refs"):
                raise ValueError(f"mapping field {rule['rule_id']} has no evidence_refs")
            if not rule.get("reason"):
                raise ValueError(f"mapping field {rule['rule_id']} has no reason")
        questions = _read_questions(outputs["semantic_questions.json"])
        from .contracts import SemanticQuestion
        for item in questions:
            SemanticQuestion.model_validate(item, strict=False)
        population_ids = set()
        with Path(outputs["image_population.jsonl"]).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                identity = item.get("source_unit_id") or item.get("unit_id") or item.get("image_id") or item.get("asset_id") or item.get("id")
                if not identity:
                    raise ValueError(f"image_population record {line_number} has no stable identity")
                population_ids.add(str(identity))
        if not population_ids:
            raise ValueError("image_population.jsonl must contain at least one source unit")
        compile(Path(outputs["inspect_source.py"]).read_text(encoding="utf-8"), str(outputs["inspect_source.py"]), "exec")
    except Exception as exc:
        return {"status": "rejected", "issues": [{"code": "understand_contract_invalid", "stage": "understand", "message": str(exc), "blocking": True}]}
    blocking = [item for item in questions if item.get("blocking") is True and item.get("resolved") is not True]
    if blocking:
        return {"status": "needs_input", "issues": blocking, "questions": questions}
    return {"status": "accepted", "issues": [], "questions": questions}


def create_accepted_handoff(state: dict[str, Any], result: StageResult, validation: dict[str, Any]) -> AcceptedHandoff:
    """Persist a host-owned immutable semantic handoff after validation."""
    if result.stage is not Stage.UNDERSTAND or result.status != "submitted":
        raise ValueError("Only a submitted understand result can create an AcceptedHandoff")
    required = ("source_report.json", "image_population.jsonl", "mapping.json", "inspect_source.py")
    missing = [name for name in required if not Path(result.outputs.get(name, "")).is_file()]
    if missing:
        raise ValueError(f"Cannot create AcceptedHandoff; missing {missing}")
    dataset_id = str(state.get("dataset_id") or state.get("dataset_scope") or "dataset")
    artifact_refs = {
        name: artifact.model_copy(update={"producer": "host", "status": "accepted"})
        for name, artifact in _candidate_artifacts(result, Stage.UNDERSTAND).items()
        if name in {"source_report.json", "image_population.jsonl", "mapping.json", "semantic_questions.json", "inspect_source.py"}
    }
    def tree_digest(path: str) -> str:
        root = Path(path)
        if root.is_file():
            return digest_file(root)
        values = {
            item.relative_to(root).as_posix(): digest_file(item)
            for item in sorted(root.rglob("*")) if item.is_file()
        }
        return digest_payload(values)

    source_payload = {
        "sources": {key: tree_digest(value) for key, value in _source_roots(state).items()},
        "manifests": {key: tree_digest(value) for key, value in (state.get("source_manifests") or {}).items()},
    }
    decisions = state.get("user_decisions") or state.get("decisions") or []
    handoff_id = digest_payload({
        "dataset_id": dataset_id,
        "source_snapshot": source_payload,
        "mapping": artifact_refs["mapping.json"].digest,
        "population": artifact_refs["image_population.jsonl"].digest,
        "decisions": decisions,
    })[:32]
    handoff = AcceptedHandoff(
        handoff_id=handoff_id,
        dataset_id=dataset_id,
        source_snapshot_digest=digest_payload(source_payload),
        mapping_digest=artifact_refs["mapping.json"].digest,
        population_digest=artifact_refs["image_population.jsonl"].digest,
        source_report_digest=artifact_refs["source_report.json"].digest,
        decisions_digest=digest_payload(decisions),
        inspection_script_digest=artifact_refs["inspect_source.py"].digest,
        inspection_dependency_digests={},
        contract_digest=digest_payload(STAGE_SPECS[Stage.UNDERSTAND.value].model_dump(mode="json")),
        validation_receipt_ref=str(validation.get("receipt_ref") or (preparation_root(state) / "validation" / f"handoff-{handoff_id}.json")),
        artifact_refs=artifact_refs,
        revision=handoff_id,
    )
    receipt_path = Path(handoff.validation_receipt_ref)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(receipt_path, {
        "status": "accepted", "stage": "understand", "handoff_id": handoff.handoff_id,
        "issues": validation.get("issues", []), "artifact_digests": {key: value.digest for key, value in artifact_refs.items()},
    })
    handoff_path = preparation_root(state) / "handoff" / "understand" / f"{handoff_id}.json"
    atomic_write_json(handoff_path, handoff.model_dump(mode="json"))
    return handoff


def resume_or_start(state: dict[str, Any]) -> dict[str, Any]:
    """Choose the next stage from durable facts, never from a UI projection."""
    current = dict(state)
    current.setdefault("status", "running")
    resume_ref = current.get("resume_state_ref")
    if resume_ref and Path(str(resume_ref)).is_file():
        try:
            resume = ResumeState.model_validate_json(Path(str(resume_ref)).read_text(encoding="utf-8"))
            current.setdefault("stage", resume.stage.value)
            current.setdefault("attempt_id", resume.attempt_id)
            current.setdefault("session_seq", resume.session_seq)
            current.setdefault("attempt", int(resume.attempt_id) if str(resume.attempt_id).isdigit() else 1)
            current["resume_existing"] = resume.last_command.status == "running" or bool(current.get("attempt_id"))
        except (OSError, ValueError):
            current.pop("resume_state_ref", None)
    if current.get("publication") and current.get("status") == "complete":
        return current
    accepted = current.get("accepted_handoff")
    accepted_ref = current.get("accepted_handoff_ref")
    if accepted_ref and not accepted:
        path = Path(str(accepted_ref))
        if path.is_file():
            try:
                accepted = AcceptedHandoff.model_validate_json(path.read_text(encoding="utf-8"))
                current["accepted_handoff"] = accepted.model_dump(mode="json")
            except ValueError:
                accepted = None
    if accepted:
        accepted_model = AcceptedHandoff.model_validate(accepted)
        for name, artifact in accepted_model.artifact_refs.items():
            path = Path(artifact.path)
            if not path.is_file() or digest_file(path) != artifact.digest or artifact.status != "accepted":
                accepted = None
                break
    if accepted:
        current.setdefault("stage", Stage.PREPARE.value)
        if current.get("stage") not in {Stage.PREPARE.value, "publish"}:
            current["stage"] = Stage.PREPARE.value
    else:
        current["stage"] = Stage.UNDERSTAND.value
        current.pop("accepted_handoff", None)
        current.pop("accepted_handoff_ref", None)
    current.setdefault("attempt", 0)
    current.setdefault("attempt_mode", "initial")
    current.setdefault("session_seq", 0)
    return current


def _materialize_accepted_handoff(state: dict[str, Any], outputs: dict[str, str]) -> Path:
    """Expose accepted upstream files without replacing prepare-owned files."""
    candidate_root = Path(outputs["cases.jsonl"]).resolve().parent
    handoff = state.get("accepted_handoff") or {}
    source_artifacts = handoff.get("artifact_refs") if isinstance(handoff, dict) else None
    source_artifacts = source_artifacts or state.get("accepted_artifacts") or {}
    for name, value in source_artifacts.items():
        artifact = value if isinstance(value, ArtifactRef) else ArtifactRef.model_validate(value)
        if name not in {"source_report.json", "image_population.jsonl", "mapping.json", "inspect_source.py"}:
            continue
        source = Path(artifact.path)
        target = candidate_root / name
        if source.is_file() and source.resolve() != target.resolve() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return candidate_root


def validate_candidate_submission(outputs: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    """Perform the cheap host gate before the outer finalizer runs."""
    if state.get("canonical_validation"):
        from .validation import validate_prepare_candidate
        result = validate_prepare_candidate(Path(outputs["cases.jsonl"]).parent, state=state,
                                             population_path=state.get("preparation_population_path"))
        return {"status": result.status, "issues": [item.model_dump(mode="json") for item in result.issues],
                "validation_result": result.model_dump(mode="json")}
    required = STAGE_SPECS[Stage.PREPARE.value].required_outputs
    missing = [name for name in required if name != "submission.json" and name not in outputs]
    issues = [{"code": "missing_output", "stage": "prepare", "message": f"Missing submitted output: {name}", "path": name, "blocking": True} for name in missing]
    if issues:
        return {"status": "rejected", "issues": issues}
    manifest = Path(outputs.get("output_manifest.json", ""))
    if not manifest.is_file():
        return {"status": "rejected", "issues": [{"code": "manifest_missing", "stage": "prepare", "message": "output_manifest.json is missing", "blocking": True}]}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("files", []), list):
            raise ValueError("files must be a list")
        registered = {
            (entry.get("path") if isinstance(entry, dict) else entry)
            for entry in payload["files"]
        }
        missing_registered = [name for name in required if name not in {"submission.json", "output_manifest.json"} and name not in registered]
        if missing_registered:
            raise ValueError(f"output_manifest does not register required outputs: {missing_registered}")
        execution_report = json.loads(Path(outputs["execution_report.json"]).read_text(encoding="utf-8"))
        required_report_sections = {"tests", "full_execution", "population_accounting", "mapping_rule_coverage", "limitations"}
        if not required_report_sections <= set(execution_report):
            raise ValueError(f"execution_report is missing sections: {sorted(required_report_sections - set(execution_report))}")
        accepted_mapping = (state.get("accepted_artifacts") or {}).get("mapping.json")
        mapping_digest = accepted_mapping.digest if isinstance(accepted_mapping, ArtifactRef) else (accepted_mapping or {}).get("digest", "") if isinstance(accepted_mapping, dict) else ""
        if mapping_digest and payload.get("mapping_digest") != mapping_digest:
            raise ValueError("output_manifest mapping_digest does not match the accepted mapping")
        accepted_population = (state.get("accepted_artifacts") or {}).get("image_population.jsonl")
        population_digest = accepted_population.digest if isinstance(accepted_population, ArtifactRef) else (accepted_population or {}).get("digest", "") if isinstance(accepted_population, dict) else ""
        if population_digest and payload.get("population_digest") != population_digest:
            raise ValueError("output_manifest population_digest does not match the accepted population")
        question_path = Path(outputs.get("semantic_questions.json", ""))
        if question_path.is_file():
            from .contracts import SemanticQuestion
            questions = _read_questions(str(question_path))
            for item in questions:
                SemanticQuestion.model_validate(item, strict=False)
            blocking = [item for item in questions if item.get("blocking") and not item.get("resolved")]
            if blocking:
                return {"status": "needs_input", "issues": blocking, "manifest": payload}
    except Exception as exc:
        return {"status": "rejected", "issues": [{"code": "manifest_invalid", "stage": "prepare", "message": str(exc), "blocking": True}]}
    return {"status": "accepted", "issues": [], "manifest": payload}


validate_prepare_submission = validate_candidate_submission


def publish_preparation(state: dict[str, Any], outputs: dict[str, str], validation: dict[str, Any]) -> dict[str, Any]:
    """Write a content-bound host publication receipt for a validated candidate."""
    if state.get("strict_publication"):
        from .publisher import DatasetPublisher
        return DatasetPublisher(state["bundle_root"]).publish(Path(outputs["cases.jsonl"]).resolve().parent, state)
    root = preparation_root(state)
    candidate_root = Path(outputs["cases.jsonl"]).resolve().parent
    all_files = {
        path.relative_to(candidate_root).as_posix(): str(path.resolve())
        for path in candidate_root.rglob("*") if path.is_file()
    }
    receipt = {
        "status": "published",
        "dataset_id": str(state.get("dataset_id") or state.get("dataset_scope") or "dataset"),
        "files": {name: {"path": path, "digest": digest_file(path)} for name, path in all_files.items() if Path(path).is_file()},
        "manifest": validation.get("manifest", {}),
    }
    path = root / "publication.json"
    receipt["path"] = str(path)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("files") == receipt["files"]:
            return existing
    atomic_write_json(path, receipt)
    return receipt


def compile_preparation_graph(*, worker: OpenHandsWorker | None = None, checkpointer=None):
    """Compile the single production graph used by runtime and tests."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    runner = worker or OpenHandsWorker()

    def resume(state):
        return resume_or_start(dict(state))

    def understand(state):
        return _run_stage(dict(state), Stage.UNDERSTAND, runner)

    def check_understand(state):
        result = StageResult.model_validate(state["stage_result"], strict=False)
        if result.status != "submitted":
            return {"status": result.status}
        handoff = validate_understand_submission(result.outputs, state)
        if handoff["status"] == "needs_input":
            return {"status": "needs_input", "pending_questions": handoff.get("issues", []), "stage": Stage.UNDERSTAND.value}
        if handoff["status"] != "accepted":
            return {"status": "blocked", "diagnostics": handoff.get("issues", [])}
        accepted = _accepted_artifacts(result, Stage.UNDERSTAND)
        handoff = create_accepted_handoff(state, result, {**handoff, "receipt_ref": ""})
        return {"status": "running", "stage": Stage.PREPARE.value, "attempt": 0,
                "attempt_id": "", "accepted_artifacts": accepted, "artifacts": accepted,
                "accepted_handoff": handoff.model_dump(mode="json"),
                "accepted_handoff_ref": str(preparation_root(state) / "handoff" / "understand" / f"{handoff.handoff_id}.json")}

    def prepare(state):
        return _run_stage(dict(state), Stage.PREPARE, runner)

    def check_prepare(state):
        result = StageResult.model_validate(state["stage_result"], strict=False)
        if result.status != "submitted":
            return {"status": result.status}
        if any(item.code == "mapping_revision_required" for item in result.diagnostics):
            return {"status": "mapping_revision_required", "stage": Stage.UNDERSTAND.value, "attempt": 0,
                    "diagnostics": [item.model_dump(mode="json") for item in result.diagnostics], "attempt_mode": "repair"}
        _materialize_accepted_handoff(state, result.outputs)
        validation = validate_candidate_submission(result.outputs, state)
        if validation["status"] == "needs_input":
            return {"status": "needs_input", "pending_questions": validation.get("issues", []), "stage": Stage.PREPARE.value}
        if validation["status"] != "accepted":
            issues = validation["issues"]
            if any("mapping_revision_required" in item.get("code", "") for item in issues if isinstance(item, dict)):
                return {"status": "mapping_revision_required", "stage": Stage.UNDERSTAND.value, "attempt": 0,
                        "validation_diagnostics": issues, "diagnostics": issues, "attempt_mode": "repair"}
            return {"status": "blocked", "validation_diagnostics": issues, "diagnostics": issues}
        return {"status": "validated", "validation": validation}

    def clarify(state):
        answer = interrupt({"stage": state.get("stage", "understand"), "questions": state.get("pending_questions", [])})
        if isinstance(answer, str):
            issue = (state.get("pending_questions") or [{"issue_id": "clarification"}])[0]
            answer = {"issue_id": issue.get("issue_id", "clarification"), "decision_id": issue.get("issue_id", "clarification"), "answer": answer}
        return {"user_decisions": [*state.get("user_decisions", []), answer], "attempt_mode": "clarification", "status": "running"}

    def publish(state):
        result = StageResult.model_validate(state["stage_result"], strict=False)
        validation = state.get("validation") or validate_candidate_submission(result.outputs, state)
        return {"status": "complete", "stage": "publish", "publication": publish_preparation(state, result.outputs, validation)}

    graph = StateGraph(PreparationGraphState)
    for name, fn in (("resume_or_start", resume), ("understand", understand), ("check_understand", check_understand), ("prepare", prepare), ("check_prepare", check_prepare), ("clarify", clarify), ("publish", publish)):
        graph.add_node(name, fn)
    graph.add_edge(START, "resume_or_start")
    graph.add_conditional_edges("resume_or_start", lambda state: "publish" if state.get("status") == "complete" else "prepare" if state.get("stage") == Stage.PREPARE.value else "understand", {"understand": "understand", "prepare": "prepare", "publish": "publish"})
    graph.add_edge("understand", "check_understand")
    graph.add_conditional_edges("check_understand", lambda state: "clarify" if state.get("status") == "needs_input" else "prepare" if state.get("status") == "running" else END, {"clarify": "clarify", "prepare": "prepare", END: END})
    graph.add_conditional_edges("clarify", lambda state: "understand" if state.get("stage", "understand") == "understand" else "prepare", {"understand": "understand", "prepare": "prepare"})
    graph.add_edge("prepare", "check_prepare")
    graph.add_conditional_edges("check_prepare", lambda state: "publish" if state.get("status") == "validated" else "clarify" if state.get("status") == "needs_input" else "understand" if state.get("status") == "mapping_revision_required" else END,
                               {"publish": "publish", "clarify": "clarify", "understand": "understand", END: END})
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer)
