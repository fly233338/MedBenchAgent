"""Deterministic stage context construction.

This module never calls an LLM.  It selects files from the frozen run state,
projects small structured summaries, maps source paths to worker-visible paths,
and renders the exact prompt saved for an attempt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest
from medbenchagent.data.models import AssetEntry, CaseRecord
from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import (
    ATTEMPT_MODES,
    CONTRACT_RULES,
    STAGE_SPECS,
    ArtifactRef,
    AttemptContext,
    ContextPacket,
    Diagnostic,
    InputBinding,
    AttemptMode,
    SourceBinding,
    Stage,
    StageBudget,
    StageOutputs,
    UserDecision,
    Submission,
    QuestionSet,
    OutputManifest,
    SourceReport,
    PreparationMapping,
    ExecutionReport,
    contract_file_names,
    digest_file,
    digest_payload,
)


class ContextBudgetError(ValueError):
    """The required part of a stage context cannot fit the configured budget."""

    code = "required_context_too_large"


@dataclass(frozen=True)
class MaterializedInputs:
    root: Path
    sources: list[SourceBinding]
    bindings: list[InputBinding]
    artifacts: dict[str, ArtifactRef]
    values: dict[str, Any]


def _sha256_tree(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        return digest_file(target)
    if target.is_dir():
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            digest.update(child.relative_to(target).as_posix().encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def _safe_relative(path: str | Path) -> str:
    value = Path(path).as_posix().lstrip("/")
    if not value or value == "." or ".." in Path(value).parts:
        raise ValueError(f"Worker input path must be relative: {path}")
    return value


def _copy_file(source: str | Path, destination: Path) -> str:
    source_path = Path(source)
    if not source_path.is_file():
        raise ValueError(f"Required preparation input is missing: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return digest_file(source_path)


def _copy_tree(source: str | Path, destination: Path) -> str:
    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(f"Required preparation input directory is missing: {source_path}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_path.rglob("*")):
        relative = child.relative_to(source_path)
        target = destination / relative
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
    return _sha256_tree(source_path)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return digest_file(path)


def _write_json(path: Path, value: Any) -> str:
    atomic_write_json(path, value)
    return digest_file(path)


def _source_items(graph_state: dict[str, Any]) -> list[dict[str, Any]]:
    values = graph_state.get("sources") or []
    result = []
    for item in values:
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json")
        if not isinstance(item, dict) or not item.get("id") or not item.get("root"):
            raise ValueError("Every preparation source requires an id and root")
        result.append(dict(item))
    if not result:
        raise ValueError("Dataset preparation requires at least one source")
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("Preparation source IDs must be unique")
    return result


def _artifact_items(graph_state: dict[str, Any], *, accepted_only: bool = False) -> dict[str, ArtifactRef]:
    # Accepted inputs have one authoritative container.  A plain ``artifacts``
    # mapping is always a candidate projection and must never be promoted by
    # a truthiness fallback.
    values = graph_state.get("accepted_artifacts") if accepted_only else graph_state.get("artifacts")
    values = values or {}
    if isinstance(values, list):
        values = {str(item.get("name")): item for item in values if isinstance(item, dict) and item.get("name")}
    result: dict[str, ArtifactRef] = {}
    for name, value in values.items():
        if isinstance(value, ArtifactRef):
            artifact = value
        elif isinstance(value, dict):
            payload = {"name": name, **value}
            payload.setdefault("producer", "host" if accepted_only else "worker")
            payload.setdefault("stage", "understand")
            if not payload.get("status"):
                payload["status"] = "candidate"
            if not payload.get("digest") and payload.get("path") and Path(payload["path"]).is_file():
                payload["digest"] = digest_file(payload["path"])
            artifact = ArtifactRef.model_validate(payload)
        else:
            path = Path(str(value))
            if not path.is_file():
                continue
            artifact = ArtifactRef(name=name, path=str(path), digest=digest_file(path), producer="worker",
                                   stage="understand", status="candidate")
        if accepted_only and artifact.status != "accepted":
            continue
        result[name] = artifact
    return result


def _source_manifest(graph_state: dict[str, Any], source_id: str) -> Path | None:
    manifests = graph_state.get("source_manifests") or {}
    value = manifests.get(source_id) if isinstance(manifests, dict) else None
    if value:
        return Path(value)
    bundle = graph_state.get("bundle_root")
    if bundle:
        candidate = Path(bundle) / "source_manifests" / source_id
        if candidate.exists():
            return candidate
    return None


def _environment(graph_state: dict[str, Any]) -> dict[str, Any]:
    environment = graph_state.get("environment") or graph_state.get("data_runtime") or graph_state.get("execution_runtime") or {}
    if hasattr(environment, "model_dump"):
        environment = environment.model_dump(mode="json")
    return dict(environment) if isinstance(environment, dict) else {"value": str(environment)}


def _decisions(graph_state: dict[str, Any]) -> list[UserDecision]:
    raw = graph_state.get("user_decisions") or graph_state.get("decisions") or []
    if isinstance(raw, dict):
        raw = [
            (dict(value, issue_id=str(key), decision_id=str(value.get("decision_id") or key)) if isinstance(value, dict)
             else {"issue_id": str(key), "decision_id": str(key), "answer_reference": str(value)})
            for key, value in raw.items()
        ]
    result = []
    issues = graph_state.get("issues") or {}
    for value in raw:
        if isinstance(value, UserDecision):
            result.append(value)
        elif isinstance(value, dict):
            try:
                result.append(UserDecision.model_validate(value))
            except Exception:
                # Clarification state stores an evidence reference.  Resolve
                # the complete answer text here; previews and reference IDs
                # are navigation metadata and cannot enter worker context.
                answer = value.get("answer")
                issue = issues.get(str(value.get("issue_id"))) if isinstance(issues, dict) else None
                answer_reference = value.get("answer_reference") or (issue or {}).get("answer_reference")
                if answer is None and answer_reference and issue:
                    try:
                        from medbenchagent.runtime.clarifications import answer_payload
                        payload = answer_payload(graph_state, issue)
                        answer = payload.get("text") or "\n".join(item.get("answer", "") for item in payload.get("exchanges", []))
                    except Exception:
                        answer = None
                if value.get("issue_id") and isinstance(answer, str) and answer.strip():
                    result.append(UserDecision(
                        decision_id=str(value.get("decision_id") or value["issue_id"]),
                        issue_id=str(value["issue_id"]),
                        source_ids=[str(item) for item in value.get("source_ids", [])],
                        fields=[str(item) for item in value.get("fields", [])],
                        answer=answer,
                        structured_value=value.get("structured_value"),
                        evidence_refs=[str(item) for item in value.get("evidence_refs", [])],
                    ))
    return result


def _attempt(graph_state: dict[str, Any]) -> AttemptContext:
    raw = graph_state.get("attempt_context") or graph_state.get("attempt") or {}
    if isinstance(raw, AttemptContext):
        return raw
    if isinstance(raw, int):
        raw = {}
    raw = dict(raw) if isinstance(raw, dict) else {}
    mode = raw.get("mode") or graph_state.get("attempt_mode") or "initial"
    if mode not in ATTEMPT_MODES:
        raise ValueError(f"Unknown preparation attempt mode: {mode}")
    raw["mode"] = AttemptMode(mode)
    return AttemptContext.model_validate(raw)


def build_attempt_context(graph_state: dict[str, Any]) -> AttemptContext:
    return _attempt(graph_state)


def select_applicable_decisions(graph_state: dict[str, Any], stage: Stage | str) -> list[UserDecision]:
    """Select decisions already scoped to the current source/field set."""
    Stage(stage)
    decisions = _decisions(graph_state)
    source_ids = {str(item.get("id")) for item in graph_state.get("sources", []) if isinstance(item, dict)}
    fields = {str(item) for item in graph_state.get("active_fields", [])}
    issue_ids = {str(item) for item in graph_state.get("target_issue_ids", [])}
    result = []
    for item in decisions:
        if item.source_ids and not source_ids.intersection(item.source_ids):
            continue
        if item.fields and fields and not fields.intersection(item.fields):
            continue
        if issue_ids and item.issue_id not in issue_ids and (item.source_ids or item.fields):
            # Dataset-scoped decisions remain applicable even without an issue
            # selector. A selector only narrows source/field issues.
            continue
        result.append(item)
    return result


def _budget(graph_state: dict[str, Any], stage: Stage) -> StageBudget:
    configured = graph_state.get("preparation_stage_budgets") or {}
    if hasattr(configured, "model_dump"):
        configured = configured.model_dump(mode="json")
    values = dict(configured.get(stage.value, {})) if isinstance(configured, dict) else {}
    return StageBudget(
        max_iterations=int(values.get("max_iterations", graph_state.get("stage_max_iterations", 40))),
        timeout_seconds=int(values.get("timeout_seconds", graph_state.get("stage_timeout_seconds", 1800))),
        no_progress_limit=int(values.get("no_progress_limit", graph_state.get("no_progress_limit", 8))),
        tool_error_limit=int(values.get("tool_error_limit", graph_state.get("tool_error_limit", 3))),
        repair_attempts=int(values.get("repair_attempts", graph_state.get("repair_attempts", 2))),
        transport_retries=int(values.get("transport_retries", graph_state.get("transport_retries", 2))),
        semantic_revision_cycles=int(values.get("semantic_revision_cycles", graph_state.get("semantic_revision_cycles", 2))),
        total_seconds=int(values.get("total_seconds", graph_state.get("preparation_total_seconds", 14_400))),
        total_iterations=int(values.get("total_iterations", graph_state.get("preparation_total_iterations", 300))),
        total_cost=values.get("total_cost", graph_state.get("preparation_total_cost")),
        initial_message_tokens=int(values.get("initial_message_tokens", graph_state.get("initial_message_tokens", 12_000))),
        model_context_tokens=values.get("model_context_tokens", graph_state.get("model_context_tokens")),
        output_reserve_tokens=int(values.get("output_reserve_tokens", graph_state.get("output_reserve_tokens", 4_096))),
        conversation_reserve_tokens=int(values.get("conversation_reserve_tokens", graph_state.get("conversation_reserve_tokens", 8_192))),
        safety_margin_tokens=int(values.get("safety_margin_tokens", graph_state.get("safety_margin_tokens", 1_024))),
    )


def resolve_prompt_budget(graph_state: dict[str, Any], stage: Stage | str = Stage.UNDERSTAND) -> StageBudget:
    """Resolve the configured initial-message and worker progress budget."""
    return _budget(graph_state, Stage(stage))


def _contract_payload(name: str) -> dict[str, Any]:
    if name == "submission.schema.json":
        return Submission.model_json_schema()
    if name == "question.schema.json":
        return QuestionSet.model_json_schema()
    if name == "output_manifest.schema.json":
        return OutputManifest.model_json_schema()
    if name == "source_report.schema.json":
        return SourceReport.model_json_schema()
    if name == "mapping.schema.json":
        return PreparationMapping.model_json_schema()
    if name == "execution_report.schema.json":
        return ExecutionReport.model_json_schema()
    if name == "case.schema.json":
        return CaseRecord.model_json_schema()
    if name == "asset.schema.json":
        return AssetEntry.model_json_schema()
    if name == "presentation.schema.json":
        return PresentationRecord.model_json_schema()
    if name == "split.schema.json":
        return SplitManifest.model_json_schema()
    if name == "sampling.schema.json":
        return SamplingManifest.model_json_schema()
    if name == "rules.md":
        return {"text": CONTRACT_RULES}
    raise ValueError(f"Unknown preparation contract file: {name}")


def resolve_stage_inputs(stage: Stage | str, graph_state: dict[str, Any]) -> dict[str, Any]:
    """Resolve only the inputs declared by the selected stage."""
    current = Stage(stage.stage if hasattr(stage, "stage") else stage)
    spec = STAGE_SPECS[current.value]
    sources = _source_items(graph_state)
    # Only host-accepted artifacts enter the accepted input directory.  Drafts
    # and failed candidates are copied through previous_outputs instead.
    artifacts = _artifact_items(graph_state, accepted_only=True)
    manifests = dict(graph_state.get("source_manifests") or {})
    for source in sources:
        if source["id"] not in manifests:
            candidate = _source_manifest(graph_state, source["id"])
            if candidate is not None:
                manifests[source["id"]] = str(candidate)
    return {
        "stage": current,
        "spec": spec,
        "sources": sources,
        "artifacts": artifacts,
        "decisions": select_applicable_decisions(graph_state, current),
        "environment": _environment(graph_state),
        "user_request": str(graph_state.get("user_request") or graph_state.get("user_requirement") or ""),
        "validation": graph_state.get("validation_report") or graph_state.get("validation_path") or graph_state.get("preparation_validation_path"),
        "previous_outputs": graph_state.get("previous_outputs_root") or graph_state.get("previous_attempt_root"),
        "previous_work": graph_state.get("previous_work_root") or graph_state.get("previous_root"),
        "target_issue_ids": list(graph_state.get("target_issue_ids") or []),
        "completed_work": list(graph_state.get("completed_work") or []),
        "remaining_work": list(graph_state.get("remaining_work") or []),
        "invalidated_work": list(graph_state.get("invalidated_work") or []),
        "source_manifests": manifests,
        "attempt": _attempt(graph_state),
        "budget": _budget(graph_state, current),
    }


def validate_input_identities(inputs: dict[str, Any]) -> None:
    """Reject changed accepted files before constructing a worker prompt."""
    for name, artifact in inputs.get("artifacts", {}).items():
        path = Path(artifact.path)
        if not path.is_file():
            raise ValueError(f"Accepted artifact is missing: {name}")
        actual = digest_file(path)
        if artifact.digest and actual != artifact.digest:
            raise ValueError(f"Accepted artifact digest changed: {name}")
    for source_id, value in (inputs.get("source_manifests") or {}).items():
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Source manifest is missing: {source_id}")


def materialize_worker_inputs(inputs: dict[str, Any], destination: str | Path) -> MaterializedInputs:
    """Create a read-only input snapshot for one attempt."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    sources: list[SourceBinding] = []
    bindings: list[InputBinding] = []
    values = dict(inputs)

    _write_json(root / "user_request.json", {"request": inputs.get("user_request", "")})
    bindings.append(InputBinding(path="inputs/user_request.json", digest=digest_file(root / "user_request.json"), role="user_request", reason="Original preparation request."))
    _write_json(root / "environment.json", inputs.get("environment", {}))
    bindings.append(InputBinding(path="inputs/environment.json", digest=digest_file(root / "environment.json"), role="environment", reason="Frozen execution environment."))
    decisions = [item.model_dump(mode="json") if isinstance(item, UserDecision) else item for item in inputs.get("decisions", [])]
    _write_json(root / "decisions.json", {"decisions": decisions})
    bindings.append(InputBinding(path="inputs/decisions.json", digest=digest_file(root / "decisions.json"), role="decisions", reason="Resolved decisions in scope for this stage."))

    source_index = []
    source_manifests = inputs.get("source_manifests") or {}
    backend = str((inputs.get("environment") or {}).get("backend", "local"))
    for source in inputs["sources"]:
        source_id = str(source["id"])
        manifest = Path(source_manifests.get(source_id, "")) if source_manifests.get(source_id) else None
        manifest_destination = root / "sources" / source_id
        manifest_digest = ""
        if manifest and manifest.exists():
            manifest_digest = _copy_tree(manifest, manifest_destination)
            for child in sorted(manifest_destination.rglob("*")):
                if child.is_file():
                    bindings.append(InputBinding(path=f"inputs/sources/{source_id}/{child.relative_to(manifest_destination).as_posix()}", digest=digest_file(child), role="source_manifest", read_policy="stream_during_execution", reason=f"Frozen manifest for source {source_id}."))
        worker_manifest = f"inputs/sources/{source_id}"
        worker_root = f"/workspace/sources/{source_id}" if backend == "docker" else str(Path(source["root"]).resolve())
        binding = SourceBinding(source_id=source_id, root=worker_root, manifest=worker_manifest, digest=manifest_digest or digest_payload({key: value for key, value in source.items() if key != "root"}))
        sources.append(binding)
        source_index.append({"source_id": source_id, "root": binding.root, "manifest": worker_manifest, "digest": binding.digest, "identity": {key: value for key, value in source.items() if key != "root"}})
    _write_json(root / "source_index.json", {"sources": source_index})
    bindings.append(InputBinding(path="inputs/source_index.json", digest=digest_file(root / "source_index.json"), role="source_index", reason="All configured sources and worker-visible paths."))

    contracts_root = root / "contracts"
    contracts_root.mkdir(parents=True, exist_ok=True)
    for name in contract_file_names(inputs["stage"]):
        target = contracts_root / name
        if name == "rules.md":
            digest = _write_text(target, CONTRACT_RULES)
        else:
            digest = _write_json(target, _contract_payload(name))
        bindings.append(InputBinding(path=f"inputs/contracts/{name}", digest=digest, role="contracts", reason="Host-maintained output contract."))

    for name, artifact in inputs.get("artifacts", {}).items():
        source = Path(artifact.path)
        target = root / "accepted" / name
        digest = _copy_file(source, target)
        accepted = artifact.model_copy(update={"path": f"inputs/accepted/{name}", "digest": digest, "status": "accepted"})
        values.setdefault("accepted_artifacts", {})[name] = accepted
        bindings.append(InputBinding(path=f"inputs/accepted/{name}", digest=digest, role="accepted", reason=f"Accepted upstream artifact {name}."))

    previous = inputs.get("previous_outputs")
    previous_work = inputs.get("previous_work")
    if previous or previous_work:
        previous_root = root / "previous"
        previous_root.mkdir(parents=True, exist_ok=True)
        copied = False
        if previous:
            previous_path = Path(previous)
            if previous_path.is_dir():
                _copy_tree(previous_path, previous_root / "outputs")
                for child in previous_path.iterdir():
                    target = previous_root / child.name
                    if child.is_file():
                        shutil.copy2(child, target)
                copied = True
        if previous_work:
            previous_work_path = Path(previous_work)
            if previous_work_path.is_dir():
                _copy_tree(previous_work_path, previous_root / "work")
                copied = True
        if copied:
            digest = _sha256_tree(previous_root)
            bindings.append(InputBinding(path="inputs/previous", digest=digest, role="previous", read_policy="on_failure", required=False, reason="Selected work and outputs from the previous attempt."))

    validation = inputs.get("validation")
    if validation:
        if isinstance(validation, (dict, list)):
            digest = _write_json(root / "validation.json", validation)
        else:
            digest = _copy_file(validation, root / "validation.json")
        bindings.append(InputBinding(path="inputs/validation.json", digest=digest, role="validation", read_policy="before_work", reason="Structured issues assigned to this attempt."))

    config_name = "inspection_config.json" if inputs["stage"] is Stage.UNDERSTAND else "preparation_config.json"
    config_payload = {
        "sources": [item.model_dump(mode="json") for item in sources],
        "source_manifests": {item.source_id: item.manifest for item in sources},
        "accepted_artifacts": {
            name: (artifact.model_copy(update={"path": f"inputs/accepted/{name}"}).model_dump(mode="json") if isinstance(artifact, ArtifactRef) else {**artifact, "path": f"inputs/accepted/{name}"})
            for name, artifact in values.get("accepted_artifacts", {}).items()
        },
        "input_digest": digest_payload({name: artifact.digest for name, artifact in values.get("accepted_artifacts", {}).items()}),
        "options": {"mode": "inventory" if inputs["stage"] is Stage.UNDERSTAND else "sample_or_full"},
    }
    config_digest = _write_json(root / config_name, config_payload)
    bindings.append(InputBinding(path=f"inputs/{config_name}", digest=config_digest, role="execution", read_policy="before_work", reason="Host-generated fixed script input configuration."))

    return MaterializedInputs(root=root, sources=sources, bindings=bindings, artifacts=values.get("accepted_artifacts", inputs.get("artifacts", {})), values=values)


def _json_summary(path: Path, *, max_items: int = 8) -> Any:
    full_ref = f"inputs/accepted/{path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"full_ref": full_ref, "digest": digest_file(path)}
    def bounded(item: Any) -> Any:
        if not isinstance(item, list):
            return item
        return {
            "count": len(item),
            "sample": item[:max_items],
            "truncated": len(item) > max_items,
            "full_ref": full_ref,
        }
    if isinstance(value, dict):
        return {key: bounded(item) for key, item in value.items()}
    return bounded(value)


def _project_facts(stage: Stage, materialized: MaterializedInputs) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "environment": materialized.values.get("environment", {}),
        "known_limitations": [],
    }
    accepted = materialized.root / "accepted"
    for name in ("source_report.json", "mapping.json", "execution_report.json"):
        candidate = accepted / name
        if candidate.is_file():
            summary = _json_summary(candidate)
            if name == "source_report.json":
                facts["source_report_summary"] = summary
                if isinstance(summary, dict):
                    facts["population_summary"] = summary.get("population", {})
            elif name == "mapping.json":
                if isinstance(summary, dict):
                    facts["grouping_rules"] = summary.get("grouping", {})
                    facts["join_rules"] = summary.get("joins", [])
                    facts["mapping_summary"] = {key: summary.get(key) for key in ("population_policy", "case_identity", "grouping", "resolved_decision_refs", "limitations") if key in summary}
            else:
                facts["execution_summary"] = summary
    population = accepted / "image_population.jsonl"
    if population.is_file():
        with population.open(encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        facts["population_summary"] = {**dict(facts.get("population_summary", {})), "unit_count": count, "path": "inputs/accepted/image_population.jsonl", "digest": digest_file(population)}
    return facts


def _project_files(stage: Stage, materialized: MaterializedInputs, attempt: AttemptContext) -> list[InputBinding]:
    files = list(materialized.bindings)
    required_roles = {"source_index", "environment", "decisions", "contracts"}
    if stage is Stage.PREPARE:
        required_roles.update({"accepted"})
    result = []
    for binding in files:
        if binding.role in required_roles or binding.role in {"previous", "validation"} or binding.required:
            local = materialized.root.parent / Path(binding.path).relative_to("inputs") if binding.path.startswith("inputs/") else None
            size = local.stat().st_size if local and local.is_file() else 0
            result.append(binding.model_copy(update={"size_bytes": size}))
    return result


def project_context(stage: Stage | str, inputs: dict[str, Any], materialized: MaterializedInputs, graph_state: dict[str, Any]) -> ContextPacket:
    current = Stage(stage)
    spec = STAGE_SPECS[current.value]
    attempt = inputs["attempt"]
    attempt_number = int(graph_state.get("preparation_attempt") or graph_state.get("attempt_number") or 1)
    attempt_id = str(graph_state.get("attempt_id") or attempt_number)
    session_seq = int(graph_state.get("session_seq", 0))
    source_snapshot_digest = digest_payload({
        "sources": [item.model_dump(mode="json") for item in materialized.sources],
        "manifests": inputs.get("source_manifests", {}),
    })
    input_payload = {
        "stage": current.value,
        "sources": [item.model_dump(mode="json") for item in materialized.sources],
        "artifacts": {name: item.digest for name, item in inputs.get("artifacts", {}).items()},
        "decisions": [item.digest for item in inputs.get("decisions", [])],
        "validation": inputs.get("validation") or "",
        "user_request": inputs.get("user_request", ""),
    }
    input_digest = digest_payload(input_payload)
    identity = {
        "run_id": str(graph_state.get("run_id") or "run"),
        "dataset_id": str(graph_state.get("dataset_id") or graph_state.get("dataset_scope") or "dataset"),
        "stage": current.value,
        "attempt": attempt_number,
        "attempt_id": attempt_id,
        "session_seq": session_seq,
        "input_digest": input_digest,
        "source_snapshot_digest": source_snapshot_digest,
        "accepted_handoff_ref": graph_state.get("accepted_handoff_ref"),
        "execution_environment_digest": digest_payload(inputs.get("environment", {})),
    }
    decisions = [item.model_dump(mode="json") for item in inputs.get("decisions", [])]
    active = decisions
    bindings = {}
    for name, artifact in inputs.get("artifacts", {}).items():
        bindings[name] = {
            "path": f"inputs/accepted/{name}", "digest": artifact.digest,
            "stage": artifact.stage, "status": artifact.status,
            "producer": artifact.producer, "acceptance_ref": graph_state.get("accepted_handoff_ref"),
            "depends_on": list(artifact.depends_on),
        }
    attempt_payload = attempt.model_dump(mode="json")
    attempt_payload.update({
        "mode": AttemptMode(str(attempt_payload.get("mode", "initial"))),
        "attempt_id": attempt_id,
        "session_seq": session_seq,
        "input_digest": input_digest,
        "execution_key": str(graph_state.get("execution_key") or ""),
        "attempt_dir": str(graph_state.get("preparation_workspace") or ""),
        "resume_state_ref": graph_state.get("resume_state_ref"),
        "previous_root": inputs.get("previous_outputs"),
        "completed_steps": list(inputs.get("completed_work") or []),
        "remaining_steps": list(inputs.get("remaining_work") or []),
        "invalidated_steps": list(inputs.get("invalidated_work") or []),
    })
    packet = ContextPacket(
        identity=identity,
        task={
            "objective": spec.objective,
            "user_request": inputs.get("user_request", ""),
            "success_conditions": spec.success_conditions,
        },
        sources=materialized.sources,
        bindings=bindings,
        decisions={"active": active, "full_path": "inputs/decisions.json"},
        facts=_project_facts(current, materialized),
        files=_project_files(current, materialized, attempt),
        outputs=StageOutputs(root="outputs", required=spec.required_outputs, optional=spec.optional_outputs, contracts_root="inputs/contracts", files=spec.files),
        attempt=AttemptContext.model_validate(attempt_payload),
        diagnostics=_diagnostics(graph_state),
        execution={
            "cwd": "outputs",
            "python": "python",
            "commands": STAGE_SPECS[current.value].commands,
            "source_bindings": [item.model_dump(mode="json") for item in materialized.sources],
            "input_config": "inputs/inspection_config.json" if current is Stage.UNDERSTAND else "inputs/preparation_config.json",
            "output_dir": "outputs",
        },
        budgets=inputs["budget"].model_dump(mode="json"),
        accepted_handoff_ref=graph_state.get("accepted_handoff_ref"),
    )
    return packet


def _diagnostics(graph_state: dict[str, Any]) -> list[Diagnostic]:
    raw = graph_state.get("diagnostics") or graph_state.get("validation_diagnostics") or []
    if isinstance(raw, dict):
        raw = raw.get("issues", [])
    result = []
    for item in raw:
        if isinstance(item, Diagnostic):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(Diagnostic.model_validate(item))
            except Exception:
                result.append(Diagnostic(stage=Stage.PREPARE, code=str(item.get("code") or item.get("kind") or "validation"), message=str(item.get("message") or item.get("error") or item), blocking=True))
    return result


_ATTEMPT_PROMPTS = {
    "initial": """Attempt mode: initial

Start the assigned stage from the current frozen inputs. Use accepted
upstream artifacts only when the host has supplied their acceptance binding.
Do not search previous conversations for missing context. Establish the
stage checklist, perform the first unfinished step, and record progress
and evidence in the declared working files.""",
    "repair": """Attempt mode: repair

Read the validator report, target issue IDs, ResumeState, and previous-work
index before editing. If a required repair input is absent, report a contract
blocker instead of reconstructing the failure from a short summary.

Treat previous outputs as repair candidates. Verify their input and code
identities, preserve unaffected work, and repair the responsible files.
Rerun affected checks and record an execution or validation reference for
each addressed issue. Keep unresolved issues explicit.

In prepare, do not change accepted semantics to make validation pass.
Request mapping_revision_required with evidence when semantic changes are
needed. In understand, revise the affected draft rules and resubmit the
handoff for host acceptance before prepare can continue.""",
    "clarification": """Attempt mode: clarification

Read the applicable decisions, including the complete answer text, issue
references, scope, and frozen input identity. An answer reference is not
the answer. If the text cannot be read, report the missing input.

Apply each decision only to its declared scope. Revisit the invalidated work
listed by the host and preserve independently valid work. Do not repeat an
answered question. If a decision remains ambiguous or conflicts with the
contract, state the remaining conflict with evidence.

Continue the stage assigned by the host. A prepare-stage answer that changes
accepted semantics must return through understand and handoff acceptance.""",
    "continue": """Attempt mode: continue

Continue the same semantic attempt in a fresh conversation. Read ResumeState,
the accepted input bindings, relevant working files, execution references,
and the remaining and invalidated steps.

Verify reusable work against its input and code identities. Check whether
the last command is complete, failed, cancelled, or still running. Do not
launch a duplicate command or repeat successful work solely because this
conversation is new.

Resume the first unfinished or invalidated step. Treat narrative summaries
as navigation aids. File contents, execution records, and host validation
determine what is complete. Update progress before the next handoff.""",
}


def read_prompt(name: str) -> str:
    path = Path(__file__).parents[1] / "prompts" / name
    if not path.is_file():
        raise FileNotFoundError(f"Preparation prompt is missing: {path}")
    return path.read_text(encoding="utf-8")


def render_attempt_mode(mode: str | AttemptContext) -> str:
    value = mode.mode.value if isinstance(mode, AttemptContext) else str(mode)
    if value not in _ATTEMPT_PROMPTS:
        raise ValueError(f"Unknown preparation attempt mode: {value}")
    return _ATTEMPT_PROMPTS[value]


def render_prompt(stage: Stage | str, packet: ContextPacket) -> str:
    current = Stage(stage)
    stage_prompt_name = {
        Stage.UNDERSTAND: "data_preparation_understand.md",
        Stage.PREPARE: "data_preparation_prepare.md",
    }[current]
    common = read_prompt("data_preparation_agent.md")
    stage_prompt = read_prompt(stage_prompt_name)
    spec = STAGE_SPECS[current.value]
    contract_text = json.dumps({
        "stage": spec.stage.value,
        "required_outputs": spec.required_outputs,
        "optional_outputs": spec.optional_outputs,
        "files": spec.files,
        "commands": spec.commands,
        "completion_by_outcome": spec.completion_by_outcome,
        "contracts_root": packet.outputs.contracts_root,
    }, ensure_ascii=False, indent=2)
    return "\n\n".join([
        common,
        stage_prompt,
        render_attempt_mode(packet.attempt),
        "StageSpec file and command contract:\n```json\n" + contract_text + "\n```",
        "Current host context:",
        "```json\n" + packet.model_dump_json(indent=2) + "\n```",
        "The same context packet is available at ./inputs/context.json. Follow its file-reading requirements and output contracts.",
        """Submission protocol

Write outputs/submission.json using the supplied schema and the current
stage's outcome-specific requirements.

- submitted: all required stage deliverables exist, the declared checks
  actually ran and passed, and no unresolved blocking issue remains.
- needs_input: a genuine missing user fact blocks correct processing;
  include valid question IDs and preserve the useful partial work.
- blocked: a technical or contract failure prevents completion; include
  structured diagnostics and the condition required to resume.

For a prepare-stage semantic conflict, use blocked and include
mapping_revision_required with its required evidence and rule references.
Do not fabricate success files to satisfy the submitted file list when
requesting input or reporting a blocker.

List only files that exist within the output contract. Register dependencies,
reference actual checks, and identify remaining or partial work accurately.
Never claim host acceptance or publication in the submission.

After writing the submission and confirming that no command is still
running, call finish. The host will validate the submission together with
the process result, execution evidence, and independent artifact checks.""",
    ])


def _token_count(text: str) -> int:
    # A conservative deterministic estimate.  If a tokenizer is configured
    # by a caller it can replace this function; context correctness must not
    # depend on an optional tokenizer package.
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def fit_context_budget(packet: ContextPacket, *, budget: StageBudget, renderer=None) -> tuple[ContextPacket, str, int]:
    """Fit non-critical summaries while preserving a complete valid packet."""
    current = packet
    render = renderer or (lambda value: render_prompt(packet.identity["stage"], value))
    prompt = render(current)
    token_count = _token_count(prompt)
    configured = budget.initial_message_tokens
    if budget.model_context_tokens is not None:
        configured = min(configured, budget.model_context_tokens - budget.output_reserve_tokens - budget.conversation_reserve_tokens - budget.safety_margin_tokens)
    if configured < 256:
        raise ContextBudgetError("required_context_too_large: configured model budget leaves no room for required context")
    if token_count <= configured:
        return current, prompt, token_count

    # Shrink only optional facts and diagnostic detail; required identities,
    # rules, decisions, file bindings and outputs remain complete.
    facts = dict(current.facts)
    for key in ("source_report_summary", "execution_summary"):
        if key in facts:
            facts[key] = {"summary": "See the complete file under inputs/accepted.", "path": f"inputs/accepted/{'source_report.json' if key == 'source_report_summary' else 'execution_report.json'}"}
    if isinstance(facts.get("join_rules"), list):
        facts["join_rules"] = facts["join_rules"][:4]
    reduced = current.model_copy(update={"facts": facts})
    prompt = render(reduced)
    token_count = _token_count(prompt)
    if token_count <= configured:
        return reduced, prompt, token_count

    if reduced.diagnostics:
        diagnostics = []
        for item in reduced.diagnostics:
            diagnostics.append(item.model_copy(update={"message": item.message[:240]}))
        reduced = reduced.model_copy(update={"diagnostics": diagnostics})
        prompt = render(reduced)
        token_count = _token_count(prompt)
        if token_count <= configured:
            return reduced, prompt, token_count
    raise ContextBudgetError("required_context_too_large: required rules, decisions, bindings and output contract exceed the stage budget")


def _execution_key(stage: Stage, packet: ContextPacket, graph_state: dict[str, Any]) -> str:
    payload = {
        "stage": stage.value,
        "input_digest": packet.identity["input_digest"],
        "source_snapshot_digest": packet.identity.get("source_snapshot_digest", ""),
        "accepted_handoff_ref": packet.identity.get("accepted_handoff_ref", ""),
        "decisions": packet.decisions,
        "contract_digest": digest_payload(packet.outputs.model_dump(mode="json")),
        "environment": packet.identity.get("execution_environment_digest", ""),
        "validator_issues": graph_state.get("target_issue_ids", []),
        "schema_digests": graph_state.get("schema_digests", {}),
        "prompt_digest": digest_payload(render_prompt(stage, packet)),
        "code_revision": graph_state.get("code_revision", ""),
        "worker_dependency_digest": graph_state.get("worker_dependency_digest", ""),
        "model": graph_state.get("model", ""),
        "base_url": graph_state.get("base_url", ""),
        "reasoning_effort": graph_state.get("reasoning_effort", ""),
    }
    return digest_payload(payload)


def build_stage_context(stage: Stage | str, graph_state: dict[str, Any], attempt_dir: str | Path) -> tuple[ContextPacket, str, int, str]:
    """Build and persist one attempt's input snapshot and exact task prompt."""
    current = Stage(stage)
    inputs = resolve_stage_inputs(current, graph_state)
    validate_input_identities(inputs)
    attempt_path = Path(attempt_dir)
    input_root = attempt_path / "inputs"
    materialized = materialize_worker_inputs(inputs, input_root)
    packet = project_context(current, inputs, materialized, graph_state)
    packet, prompt, token_count = fit_context_budget(packet, budget=inputs["budget"])
    write_context_snapshot(attempt_path, packet)
    write_prompt_snapshot(attempt_path, prompt)
    execution_key = _execution_key(current, packet, graph_state)
    return packet, prompt, token_count, execution_key


def write_context_snapshot(attempt_dir: str | Path, packet: ContextPacket) -> Path:
    path = Path(attempt_dir) / "inputs" / "context.json"
    _write_json(path, packet.model_dump(mode="json"))
    return path


def write_prompt_snapshot(attempt_dir: str | Path, prompt: str) -> Path:
    path = Path(attempt_dir) / "inputs" / "prompt.md"
    _write_text(path, prompt)
    return path
