"""Host-owned preparation state machine and bounded stage context builder."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source, verify_source
from medbenchagent.runtime.dataset_runtime import DatasetAgentResult, dataset_runtime
from medbenchagent.runtime.preparation_state import (
    ArtifactRef,
    MemoryState,
    MemoryRecord,
    PreparationPhase,
    PreparationState,
    PreparationStatus,
    StageBudget,
    StageContext,
    TerminationReason,
    load_memory,
    load_preparation_state,
    preparation_paths,
    save_memory,
    save_preparation_state,
    save_progress,
)
from medbenchagent.runtime.preparation_task import STAGE_OUTPUTS, build_data_preparation_task
from medbenchagent.runtime.run_artifacts import atomic_write_json


STAGES = (PreparationPhase.INVENTORY, PreparationPhase.SEMANTIC,
          PreparationPhase.ADAPTER, PreparationPhase.CANDIDATE)


def _digest_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def _stage_budget(context: dict[str, Any], stage: PreparationPhase) -> StageBudget:
    defaults = {
        PreparationPhase.INVENTORY: (20, 900),
        PreparationPhase.SEMANTIC: (16, 900),
        PreparationPhase.ADAPTER: (32, 1800),
        PreparationPhase.CANDIDATE: (24, 1800),
        PreparationPhase.REPAIR: (24, 1800),
    }
    iterations, timeout = defaults.get(stage, (24, 1800))
    configured = context.get("preparation_stage_budgets", {}).get(stage.value, {})
    return StageBudget(
        max_iterations=int(configured.get("max_iterations", iterations)),
        timeout_seconds=int(configured.get("timeout_seconds", timeout)),
        no_progress_limit=int(configured.get("no_progress_limit", 8)),
        tool_error_limit=int(configured.get("tool_error_limit", 3)),
    )


def _source_digest(context: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    manifests = context.get("source_manifests", {})
    for source_id in sorted(manifests):
        path = Path(manifests[source_id])
        digest.update(source_id.encode("utf-8"))
        digest.update(_digest_tree(path).encode("utf-8"))
    return digest.hexdigest()


def _write_context(stage_dir: Path, state: PreparationState, memory: MemoryState,
                   context: dict[str, Any], stage: PreparationPhase, feedback: str) -> StageContext:
    input_dir = stage_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, ArtifactRef] = {}
    for name, artifact in state.artifacts.items():
        source = Path(artifact.path)
        safe_name = name.replace("/", "__")
        destination = input_dir / safe_name
        if source.is_file():
            shutil.copy2(source, destination)
        inputs[name] = artifact.model_copy(update={"path": f"inputs/{safe_name}"})
    budget = _stage_budget(context, stage)
    objective = {
        PreparationPhase.INVENTORY: "Inventory all configured sources and their complete population.",
        PreparationPhase.SEMANTIC: "Produce source-grounded semantic mappings and explicit questions.",
        PreparationPhase.ADAPTER: "Implement and test the deterministic adapter from frozen mappings.",
        PreparationPhase.CANDIDATE: "Execute the adapter over the complete population and emit core candidate files.",
        PreparationPhase.REPAIR: "Repair the candidate using only the bounded validator diagnostics.",
    }[stage]
    validator_feedback = []
    if feedback:
        validator_feedback.append({"message": feedback[:6000]})
    packet = StageContext(
        run_id=state.run_id, dataset_id=state.dataset_id, phase=stage,
        attempt=state.attempt, objective=objective, inputs=inputs,
        memory=memory, evidence=[], validator_feedback=validator_feedback,
        allowed_outputs=list(STAGE_OUTPUTS.get(stage.value, STAGE_OUTPUTS["candidate"])),
        budget=budget, workspace=str(stage_dir),
    )
    atomic_write_json(stage_dir / "context.json", packet.model_dump(mode="json"))
    shutil.copy2(stage_dir / "context.json", stage_dir / "inputs" / "context.json")
    return packet


def _record_outputs(state: PreparationState, stage: PreparationPhase,
                    result: DatasetAgentResult, stage_dir: Path) -> PreparationState:
    artifacts = dict(state.artifacts)
    for name, path in result.outputs.items():
        output = Path(path)
        if not output.is_file():
            continue
        artifacts[name] = ArtifactRef(name=name, path=str(output.resolve()),
                                      digest=_digest_tree(output), producer="worker",
                                      stage=stage.value, status="candidate")
    return state.model_copy(update={"artifacts": artifacts,
                                    "progress": state.progress.model_copy(update=result.progress or {}),
                                    "partial_outputs": sorted(result.outputs),
                                    "missing_outputs": []})


def _record_artifact_memory(memory: MemoryState, state: PreparationState) -> MemoryState:
    records = list(memory.verified_facts)
    known = {item.id for item in records}
    for name, artifact in state.artifacts.items():
        record_id = f"artifact:{artifact.stage}:{name}"
        if record_id in known:
            continue
        records.append(MemoryRecord(
            id=record_id,
            value={"name": name, "path": artifact.path, "digest": artifact.digest},
            producer="host", scope=state.dataset_id, status="verified",
            artifact_refs=[name],
        ))
    return memory.model_copy(update={"verified_facts": records})


def _copy_candidate(stage_dir: Path, workspace: Path, outputs: list[str]) -> Path:
    target = workspace / "outputs"
    target.mkdir(parents=True, exist_ok=True)
    for name in outputs:
        source = stage_dir / "outputs" / name
        if not source.is_file():
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return target


def _bootstrap(context: dict[str, Any], workspace: Path) -> PreparationState:
    sources = [Source.model_validate(item) for item in context.get("sources", [])]
    if not sources:
        raise ValueError("Dataset preparation requires at least one source")
    for source in sources:
        manifest = Path(context["bundle_root"]) / "source_manifests" / source.id
        inventory_source(source, manifest)
        verify_source(manifest)
    state = PreparationState(run_id=context["run_id"], dataset_id=context["dataset_scope"],
                             source_digest=_source_digest(context))
    state.logs = {
        "stdout": str(workspace / "logs" / "stdout.log"),
        "stderr": str(workspace / "logs" / "stderr.log"),
        "events": context.get("construction_events_path", ""),
    }
    save_preparation_state(workspace, state)
    save_memory(workspace, MemoryState())
    save_progress(workspace, state.progress)
    return state


def run_preparation_orchestrator(context: dict[str, Any]) -> dict[str, Any]:
    """Run bounded preparation stages and return a business-facing outcome."""
    context = dict(context)
    workspace = Path(context["run_directory"]) / "workers" / context["dataset_scope"]
    workspace.mkdir(parents=True, exist_ok=True)
    for directory in ("state", "logs", "conversations", "stages", "outputs"):
        (workspace / directory).mkdir(parents=True, exist_ok=True)

    sources = [Source.model_validate(item) for item in context.get("sources", [])]
    if sources and not context.get("source_manifests"):
        manifests = {item.id: str(Path(context["bundle_root"]) / "source_manifests" / item.id)
                     for item in sources}
        primary = sources[0]
        context.update(source_manifests=manifests, source_manifest=manifests[primary.id],
                       source_root=str(Path(primary.root).resolve()),
                       source_roots={item.id: item.root for item in sources},
                       active_source=primary.model_dump())

    state = load_preparation_state(workspace)
    if state is None:
        state = _bootstrap(context, workspace)
    memory = load_memory(workspace)
    feedback = str(context.get("feedback") or context.get("preparation_feedback") or "")
    if state.status == PreparationStatus.COMPLETE:
        return {"status": "complete", "summary": "Preparation already completed",
                "updates": {"preparation_workspace": str(workspace),
                            "preparation_candidate": {"candidate_path": str(workspace / "outputs"),
                                                       "status": "submitted"},
                            "preparation_status": "complete"}}

    stages = list(STAGES)
    if feedback or state.phase in {PreparationPhase.REPAIR, PreparationPhase.VALIDATION}:
        stages = [PreparationPhase.REPAIR]

    for stage in stages:
        attempt = state.attempt + 1 if stage == state.phase else 1
        state = state.next_session(stage, attempt)
        stage_dir = workspace / "stages" / stage.value / str(state.attempt)
        stage_dir.mkdir(parents=True, exist_ok=True)
        packet = _write_context(stage_dir, state, memory, context, stage, feedback)
        context.update({"preparation_context_path": str(stage_dir / "context.json"),
                        "preparation_attempt": state.attempt,
                        "stage_max_iterations": packet.budget.max_iterations,
                        "stage_timeout_seconds": packet.budget.timeout_seconds,
                        "no_progress_limit": packet.budget.no_progress_limit,
                        "tool_error_limit": packet.budget.tool_error_limit,
                        "preparation_input_digest": _digest_tree(stage_dir / "context.json")})
        task = build_data_preparation_task(context, stage_dir, feedback=feedback,
                                           stage=stage.value, attempt=state.attempt)
        atomic_write_json(stage_dir / "task.json", task.model_dump(mode="json"))
        runtime = dataset_runtime(context)
        try:
            result = runtime.run(task)
        except Exception as exc:
            state = state.model_copy(update={"status": PreparationStatus.FAILED,
                                             "termination_reason": TerminationReason.WORKER_ERROR,
                                             "logs": {**state.logs, "workspace": str(stage_dir)}})
            save_preparation_state(workspace, state)
            return {"status": "failed", "summary": str(exc),
                    "updates": {"preparation_workspace": str(workspace),
                                "preparation_status": "failed",
                                "preparation_termination_reason": "worker_error"}}
        atomic_write_json(stage_dir / "result.json", result.model_dump(mode="json"))
        if result.status != "complete":
            reason = result.termination_reason or (
                "worker_error" if result.status == "failed" else "output_invalid")
            state = state.model_copy(update={
                "status": PreparationStatus(result.status),
                "termination_reason": reason,
                "progress": state.progress.model_copy(update=result.progress or {}),
            })
            save_preparation_state(workspace, state)
            if result.status == "needs_input":
                atomic_write_json(preparation_paths(workspace)["issues"], {
                    "phase": stage.value, "attempt": state.attempt,
                    "question": result.question, "diagnostic": result.diagnostic,
                })
            return {"status": result.status, "summary": result.diagnostic or result.question,
                    "updates": {"preparation_workspace": str(workspace),
                                "preparation_conversation_id": result.conversation_id,
                                "preparation_status": result.status,
                                "preparation_question": result.question,
                                "preparation_termination_reason": reason}}
        state = _record_outputs(state, stage, result, stage_dir)
        memory = _record_artifact_memory(memory, state)
        save_memory(workspace, memory)
        save_preparation_state(workspace, state)
        save_progress(workspace, state.progress)
        atomic_write_json(preparation_paths(workspace)["artifacts"],
                          {key: value.model_dump(mode="json") for key, value in state.artifacts.items()})
        if stage in {PreparationPhase.CANDIDATE, PreparationPhase.REPAIR}:
            candidate = _copy_candidate(stage_dir, workspace, list(result.outputs))
            state = state.model_copy(update={"phase": PreparationPhase.VALIDATION,
                                             "status": PreparationStatus.RUNNING})
            save_preparation_state(workspace, state)
            return {"status": "complete", "summary": "Preparation candidate submitted for host validation",
                    "updates": {"preparation_workspace": str(workspace),
                                "preparation_conversation_id": result.conversation_id,
                                "preparation_status": "complete",
                                "preparation_candidate": {"candidate_path": str(candidate),
                                                           "source_ids": [item["id"] for item in context.get("sources", [])],
                                                           "status": "submitted"}}}
        feedback = ""

    return {"status": "blocked", "summary": "Preparation state machine stopped without a candidate",
            "updates": {"preparation_workspace": str(workspace), "preparation_status": "blocked"}}


def mark_preparation_complete(workspace: str | Path) -> None:
    if not str(workspace):
        return
    state = load_preparation_state(workspace)
    if state is None:
        return
    state = state.model_copy(update={"phase": PreparationPhase.VALIDATION,
                                    "status": PreparationStatus.COMPLETE,
                                    "termination_reason": TerminationReason.COMPLETED})
    save_preparation_state(workspace, state)


def mark_preparation_blocked(workspace: str | Path, reason: str = "validation_failed") -> None:
    if not str(workspace):
        return
    state = load_preparation_state(workspace)
    if state is None:
        return
    state = state.model_copy(update={"phase": PreparationPhase.VALIDATION,
                                    "status": PreparationStatus.BLOCKED,
                                    "termination_reason": reason})
    save_preparation_state(workspace, state)

