"""Deterministic context construction for one preparation stage."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest
from medbenchagent.data.models import AssetEntry, CaseRecord
from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import (
    CONTRACT_RULES,
    STAGE_SPECS,
    AcceptedHandoff,
    ArtifactRef,
    ContextPacket,
    ExecutionReport,
    InputBinding,
    PreparationMapping,
    QuestionSet,
    SourceBinding,
    SourceReport,
    Stage,
    StageBudget,
    StageOutputs,
    UserDecision,
    digest_file,
    digest_payload,
    contract_file_names,
)


class ContextBudgetError(ValueError):
    code = "required_context_too_large"


@dataclass(frozen=True)
class MaterializedInputs:
    root: Path
    sources: list[SourceBinding]
    bindings: list[InputBinding]
    artifacts: dict[str, ArtifactRef]
    handoff: AcceptedHandoff | None
    values: dict[str, Any]


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
        target = destination / child.relative_to(source_path)
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
    return digest_payload({
        item.relative_to(source_path).as_posix(): digest_file(item)
        for item in sorted(source_path.rglob("*")) if item.is_file()
    })


def _write_text(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    return digest_file(path)


def _write_json(path: Path, value: Any) -> str:
    atomic_write_json(path, value)
    return digest_file(path)


def _source_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in state.get("sources") or []:
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


def _source_manifest(state: dict[str, Any], source_id: str) -> Path | None:
    manifests = state.get("source_manifests") or {}
    value = manifests.get(source_id) if isinstance(manifests, dict) else None
    if value:
        return Path(value)
    bundle = state.get("bundle_root")
    if bundle:
        candidate = Path(bundle) / "source_manifests" / source_id
        if candidate.exists():
            return candidate
    return None


def _environment(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("environment") or state.get("data_runtime") or state.get("execution_runtime") or {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {"value": str(value)}


def _decisions(state: dict[str, Any]) -> list[UserDecision]:
    return [item if isinstance(item, UserDecision) else UserDecision.model_validate(item)
            for item in state.get("user_decisions", [])]


def _budget(state: dict[str, Any], stage: Stage) -> StageBudget:
    configured = state.get("preparation_stage_budgets") or {}
    values = configured.get(stage.value, {}) if isinstance(configured, dict) else {}
    return StageBudget(
        max_iterations=int(values.get("max_iterations", state.get("stage_max_iterations", 40))),
        timeout_seconds=int(values.get("timeout_seconds", state.get("stage_timeout_seconds", 1800))),
        initial_message_tokens=int(values.get("initial_message_tokens", state.get("initial_message_tokens", 12_000))),
        model_context_tokens=values.get("model_context_tokens", state.get("model_context_tokens")),
        output_reserve_tokens=int(values.get("output_reserve_tokens", state.get("output_reserve_tokens", 4_096))),
        conversation_reserve_tokens=int(values.get("conversation_reserve_tokens", state.get("conversation_reserve_tokens", 8_192))),
        safety_margin_tokens=int(values.get("safety_margin_tokens", state.get("safety_margin_tokens", 1_024))),
    )


def resolve_prompt_budget(state: dict[str, Any], stage: Stage | str = Stage.UNDERSTAND) -> StageBudget:
    return _budget(state, Stage(stage))


def _contract_payload(name: str) -> dict[str, Any]:
    from .contracts import OutputManifest, PopulationRecord, Submission, ImageConversionPlan, ImageRequest, PopulationOutcome

    mapping = {
        "image_conversion_plan.schema.json": ImageConversionPlan.model_json_schema(),
        "image_request.schema.json": ImageRequest.model_json_schema(),
        "population_outcome.schema.json": PopulationOutcome.model_json_schema(),
        "submission.schema.json": Submission.model_json_schema(),
        "question.schema.json": QuestionSet.model_json_schema(),
        "output_manifest.schema.json": OutputManifest.model_json_schema(),
        "source_report.schema.json": SourceReport.model_json_schema(),
        "population.schema.json": PopulationRecord.model_json_schema(),
        "mapping.schema.json": PreparationMapping.model_json_schema(),
        "execution_report.schema.json": ExecutionReport.model_json_schema(),
        "case.schema.json": CaseRecord.model_json_schema(),
        "asset.schema.json": AssetEntry.model_json_schema(),
        "presentation.schema.json": PresentationRecord.model_json_schema(),
        "split.schema.json": SplitManifest.model_json_schema(),
        "sampling.schema.json": SamplingManifest.model_json_schema(),
    }
    if name == "rules.md":
        return {"text": CONTRACT_RULES}
    if name not in mapping:
        raise ValueError(f"Unknown preparation contract file: {name}")
    return mapping[name]


def _handoff(state: dict[str, Any]) -> AcceptedHandoff | None:
    value = state.get("accepted_handoff")
    if not value:
        return None
    return value if isinstance(value, AcceptedHandoff) else AcceptedHandoff.model_validate(value)


def resolve_stage_inputs(stage: Stage | str, state: dict[str, Any]) -> dict[str, Any]:
    current = Stage(stage)
    spec = STAGE_SPECS[current.value]
    sources = _source_items(state)
    manifests = dict(state.get("source_manifests") or {})
    for source in sources:
        if source["id"] not in manifests:
            candidate = _source_manifest(state, source["id"])
            if candidate is not None:
                manifests[source["id"]] = str(candidate)
    handoff = _handoff(state)
    artifacts: dict[str, ArtifactRef] = {}
    if current is not Stage.UNDERSTAND:
        if handoff is None:
            raise ValueError("Prepare requires an accepted handoff")
        artifacts = {
            "source_report.json": handoff.source_report,
            "mapping.json": handoff.mapping,
            "image_population.jsonl": handoff.population,
            "inspect_source.py": handoff.inspection_script,
            "image_conversion_plan.json": handoff.image_conversion_plan,
        }
    if current is Stage.PREPARE_IMAGE:
        for name, value in state.get("text_artifacts", {}).items():
            artifacts[name] = ArtifactRef.model_validate(value, strict=False)
    return {
        "stage": current,
        "spec": spec,
        "sources": sources,
        "artifacts": artifacts,
        "handoff": handoff,
        "revision_basis_handoff": (
            state.get("revision_basis_handoff") if current is Stage.UNDERSTAND else None
        ),
        "decisions": _decisions(state),
        "environment": _environment(state),
        "user_request": str(state.get("user_request") or state.get("user_requirement") or ""),
        "validation_feedback": dict(state.get("validation_feedback") or {}),
        "previous_attempt_root": state.get("previous_attempt_root"),
        "preparation_root": state.get("preparation_root"),
        "source_manifests": manifests,
        "budget": _budget(state, current),
    }


def validate_input_identities(inputs: dict[str, Any]) -> None:
    for name, artifact in inputs.get("artifacts", {}).items():
        path = Path(artifact.path)
        if not path.is_file():
            raise ValueError(f"Accepted artifact is missing: {name}")
        if digest_file(path) != artifact.digest:
            raise ValueError(f"Accepted artifact digest changed: {name}")
    for source_id, value in (inputs.get("source_manifests") or {}).items():
        if not Path(value).exists():
            raise ValueError(f"Source manifest is missing: {source_id}")


def _reference_snapshot(root: Path, scopes: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    entries: dict[str, Any] = {}
    for scope in scopes:
        candidate = root / scope
        if not candidate.is_dir():
            continue
        for path in sorted(candidate.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            entries[path.relative_to(root).as_posix()] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return digest_payload(entries), entries


def _validate_reference_root(inputs: dict[str, Any], value: str | Path) -> Path:
    root = Path(value).resolve()
    if not root.is_dir():
        raise ValueError(f"Previous preparation attempt is missing: {root}")
    preparation_root = inputs.get("preparation_root")
    if preparation_root:
        allowed = Path(preparation_root).resolve()
        if not root.is_relative_to(allowed):
            raise ValueError(f"Previous preparation attempt is outside the preparation root: {root}")
    return root


def verify_previous_reference(attempt_dir: str | Path) -> None:
    reference_path = Path(attempt_dir) / "inputs" / "previous_reference.json"
    if not reference_path.is_file():
        return
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    root = Path(reference["host_root"])
    _, current = _reference_snapshot(root, ("work", "outputs", "inputs/accepted"))
    if current != reference.get("entries", {}):
        raise ValueError("Previous preparation reference changed during the coding stage")


def materialize_worker_inputs(inputs: dict[str, Any], destination: str | Path) -> MaterializedInputs:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    bindings: list[InputBinding] = []
    values = dict(inputs)
    _write_json(root / "user_request.json", {"request": inputs.get("user_request", "")})
    bindings.append(InputBinding(path="inputs/user_request.json", digest=digest_file(root / "user_request.json"), role="user_request", reason="Original preparation request."))
    _write_json(root / "environment.json", inputs.get("environment", {}))
    bindings.append(InputBinding(path="inputs/environment.json", digest=digest_file(root / "environment.json"), role="environment", reason="Frozen execution environment."))
    decisions = [item.model_dump(mode="json") for item in inputs.get("decisions", [])]
    _write_json(root / "decisions.json", {"decisions": decisions})
    bindings.append(InputBinding(path="inputs/decisions.json", digest=digest_file(root / "decisions.json"), role="decisions", reason="Resolved user decisions."))

    sources: list[SourceBinding] = []
    source_index = []
    backend = str((inputs.get("environment") or {}).get("backend", "local"))
    for source in inputs["sources"]:
        source_id = str(source["id"])
        manifest = Path(inputs["source_manifests"].get(source_id, ""))
        manifest_destination = root / "sources" / source_id
        manifest_digest = _copy_tree(manifest, manifest_destination) if manifest.is_dir() else ""
        for child in sorted(manifest_destination.rglob("*")):
            if child.is_file():
                bindings.append(InputBinding(
                    path=f"inputs/sources/{source_id}/{child.relative_to(manifest_destination).as_posix()}",
                    digest=digest_file(child), role="source_manifest",
                    read_policy="stream_during_execution",
                    reason=f"Frozen manifest for source {source_id}.",
                ))
        worker_root = f"/workspace/sources/{source_id}" if backend == "docker" else str(Path(source["root"]).resolve())
        binding = SourceBinding(
            source_id=source_id, root=worker_root,
            manifest=f"inputs/sources/{source_id}", digest=manifest_digest or digest_payload(source),
        )
        sources.append(binding)
        source_index.append({
            "source_id": source_id, "root": binding.root,
            "manifest": binding.manifest, "digest": binding.digest,
            "identity": {key: value for key, value in source.items() if key != "root"},
        })
    _write_json(root / "source_index.json", {"sources": source_index})
    bindings.append(InputBinding(path="inputs/source_index.json", digest=digest_file(root / "source_index.json"), role="source_index", reason="Frozen source bindings."))

    contracts_root = root / "contracts"
    contracts_root.mkdir(parents=True, exist_ok=True)
    for name in contract_file_names(inputs["stage"]):
        target = contracts_root / name
        digest = _write_text(target, CONTRACT_RULES) if name == "rules.md" else _write_json(target, _contract_payload(name))
        bindings.append(InputBinding(path=f"inputs/contracts/{name}", digest=digest, role="contracts", reason="Host-maintained output contract."))

    artifacts = dict(inputs.get("artifacts") or {})
    if artifacts:
        accepted_root = root / "accepted"
        for name, artifact in artifacts.items():
            target = accepted_root / name
            digest = _copy_file(artifact.path, target)
            bindings.append(InputBinding(path=f"inputs/accepted/{name}", digest=digest, role="accepted", reason=f"Accepted handoff artifact {name}."))
        values["accepted_artifacts"] = artifacts

    previous = inputs.get("previous_attempt_root")
    previous_root = None
    if previous:
        previous_root = _validate_reference_root(inputs, previous)
        reference_digest, reference_entries = _reference_snapshot(
            previous_root, ("work", "outputs", "inputs/accepted"),
        )
        worker_root = "/workspace/previous" if backend == "docker" else str(previous_root)
        reference = {
            "host_root": str(previous_root),
            "worker_root": worker_root,
            "digest": reference_digest,
            "entries": reference_entries,
        }
        _write_json(root / "previous_reference.json", reference)
        values["previous_reference"] = reference
        bindings.append(InputBinding(
            path="previous", worker_path=worker_root, host_path=str(previous_root),
            digest=reference_digest, role="previous", read_policy="on_failure", required=False,
            reason="Previous stage work and outputs are exposed through a read-only reference.",
        ))

    revision_handoff_value = inputs.get("revision_basis_handoff")
    if revision_handoff_value and previous_root is not None:
        revision_handoff = (
            revision_handoff_value
            if isinstance(revision_handoff_value, AcceptedHandoff)
            else AcceptedHandoff.model_validate(revision_handoff_value, strict=False)
        )
        revision_basis = {}
        for field in ("source_report", "mapping", "population", "inspection_script", "image_conversion_plan"):
            artifact = getattr(revision_handoff, field)
            candidate = previous_root / "inputs" / "accepted" / artifact.name
            if not candidate.is_file():
                raise ValueError(f"Revision basis artifact is missing: {artifact.name}")
            worker_path = (
                f"/workspace/previous/inputs/accepted/{artifact.name}"
                if backend == "docker" else str(candidate)
            )
            revision_basis[field] = {
                "name": artifact.name, "path": worker_path, "digest": artifact.digest,
            }
            bindings.append(InputBinding(
                path=f"previous/inputs/accepted/{artifact.name}",
                worker_path=worker_path, host_path=str(candidate), digest=artifact.digest,
                role="revision_basis", read_policy="before_work", required=False,
                reason="Previous accepted semantic artifact used as a revision basis.",
            ))
        values["revision_basis"] = {
            "handoff_digest": revision_handoff.digest,
            "artifacts": revision_basis,
        }

    feedback = inputs.get("validation_feedback") or {}
    if feedback:
        digest = _write_json(root / "validation.json", feedback)
        bindings.append(InputBinding(path="inputs/validation.json", digest=digest, role="validation", reason="Host validation feedback."))

    config_name = "inspection_config.json" if inputs["stage"] is Stage.UNDERSTAND else "preparation_config.json"
    config_payload = {
        "sources": [item.model_dump(mode="json") for item in sources],
        "source_manifests": {item.source_id: item.manifest for item in sources},
        "accepted_artifacts": {
            name: {"path": f"inputs/accepted/{name}", "digest": artifact.digest, "stage": artifact.stage, "status": "accepted"}
            for name, artifact in artifacts.items()
        },
        "input_digest": digest_payload({name: artifact.digest for name, artifact in artifacts.items()}),
        "options": {"mode": "inventory" if inputs["stage"] is Stage.UNDERSTAND else "sample_or_full"},
    }
    digest = _write_json(root / config_name, config_payload)
    bindings.append(InputBinding(path=f"inputs/{config_name}", digest=digest, role="execution", reason="Host-generated fixed script input configuration."))
    handoff = inputs.get("handoff")
    if handoff is not None:
        updates = {}
        for field in ("source_report", "mapping", "population", "inspection_script", "image_conversion_plan"):
            artifact = getattr(handoff, field)
            updates[field] = artifact.model_copy(update={"path": f"inputs/accepted/{artifact.name}"})
        receipt_path = root / "handoff_validation.json"
        if Path(handoff.validation_receipt_ref).is_file():
            _copy_file(handoff.validation_receipt_ref, receipt_path)
            updates["validation_receipt_ref"] = "inputs/handoff_validation.json"
        handoff = handoff.model_copy(update=updates)
    return MaterializedInputs(root=root, sources=sources, bindings=bindings, artifacts=artifacts,
                              handoff=handoff, values=values)


def _json_summary(path: Path, *, max_items: int = 8) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": f"inputs/accepted/{path.name}", "digest": digest_file(path)}
    if isinstance(value, list):
        return {"count": len(value), "sample": value[:max_items], "truncated": len(value) > max_items, "path": f"inputs/accepted/{path.name}"}
    if isinstance(value, dict):
        return {key: (item[:max_items] if isinstance(item, list) else item) for key, item in value.items()}
    return value


def _project_facts(stage: Stage, materialized: MaterializedInputs) -> dict[str, Any]:
    facts: dict[str, Any] = {"environment": materialized.values.get("environment", {}), "known_limitations": []}
    accepted = materialized.root / "accepted"
    for name in ("source_report.json", "mapping.json", "execution_report.json"):
        candidate = accepted / name
        if candidate.is_file():
            summary = _json_summary(candidate)
            facts[f"{name[:-5]}_summary"] = summary
            if name == "source_report.json" and isinstance(summary, dict):
                facts["population_summary"] = summary.get("population", {})
    population = accepted / "image_population.jsonl"
    if population.is_file():
        count = sum(1 for line in population.open(encoding="utf-8") if line.strip())
        facts["population_summary"] = {**dict(facts.get("population_summary", {})), "unit_count": count, "path": "inputs/accepted/image_population.jsonl", "digest": digest_file(population)}
    if revision_basis := materialized.values.get("revision_basis"):
        facts["revision_basis"] = revision_basis
    if previous_reference := materialized.values.get("previous_reference"):
        facts["previous_reference"] = {
            "worker_root": previous_reference["worker_root"],
            "digest": previous_reference["digest"],
            "scopes": sorted({item.split("/", 1)[0] for item in previous_reference["entries"]}),
        }
    return facts


def _project_files(stage: Stage, materialized: MaterializedInputs) -> list[InputBinding]:
    required_roles = {"user_request", "source_index", "environment", "decisions", "contracts", "execution"}
    if stage is not Stage.UNDERSTAND:
        required_roles.add("accepted")
    required_roles.add("revision_basis")
    return [
        binding.model_copy(update={"size_bytes": (
            binding.size_bytes
            or (Path(binding.host_path).stat().st_size if binding.host_path and Path(binding.host_path).is_file() else 0)
            or ((materialized.root.parent / Path(binding.path).relative_to("inputs")).stat().st_size
                if binding.path.startswith("inputs/")
                and (materialized.root.parent / Path(binding.path).relative_to("inputs")).is_file()
                else 0)
        )})
        for binding in materialized.bindings
        if binding.required or binding.role in required_roles or binding.role in {"previous", "validation"}
    ]


def stage_input_digest(stage: Stage | str, state: dict[str, Any]) -> str:
    current = Stage(stage)
    handoff = _handoff(state) if current is not Stage.UNDERSTAND else None
    revision_basis = state.get("revision_basis_handoff")
    if revision_basis:
        revision_basis = (
            revision_basis.digest if isinstance(revision_basis, AcceptedHandoff)
            else AcceptedHandoff.model_validate(revision_basis, strict=False).digest
        )
    return digest_payload({
        "stage": current.value,
        "source_snapshot_digest": state.get("source_snapshot_digest") or digest_payload(state.get("source_manifests", {})),
        "handoff": handoff.digest if handoff else None,
        "text_artifacts": state.get("text_artifacts", {}) if current is Stage.PREPARE_IMAGE else {},
        "revision_basis_handoff": revision_basis,
        "previous_attempt_root": state.get("previous_attempt_root"),
        "implementation_repair_root": state.get("implementation_repair_root"),
        "decisions": [item.digest for item in _decisions(state)],
        "validation_feedback": state.get("validation_feedback") or {},
        "user_request": state.get("user_request") or state.get("user_requirement") or "",
    })


def project_context(stage: Stage | str, inputs: dict[str, Any], materialized: MaterializedInputs, state: dict[str, Any]) -> ContextPacket:
    current = Stage(stage)
    spec = STAGE_SPECS[current.value]
    attempt = int(state.get("attempt") or state.get("stage_attempt") or 1)
    source_snapshot_digest = str(state.get("source_snapshot_digest") or digest_payload({
        "sources": [item.model_dump(mode="json") for item in materialized.sources],
        "manifests": inputs.get("source_manifests", {}),
    }))
    input_digest = stage_input_digest(current, state)
    identity = {
        "run_id": str(state.get("run_id") or "run"),
        "dataset_id": str(state.get("dataset_id") or state.get("dataset_scope") or "dataset"),
        "stage": current.value,
        "attempt": attempt,
        "attempt_id": str(state.get("attempt_id") or f"{current.value}-{attempt}"),
        "input_digest": input_digest,
        "source_snapshot_digest": source_snapshot_digest,
    }
    return ContextPacket(
        identity=identity,
        task={"objective": spec.objective, "user_request": inputs.get("user_request", ""), "success_conditions": spec.success_conditions},
        sources=materialized.sources,
        accepted_handoff=materialized.handoff,
        decisions=inputs.get("decisions", []),
        facts=_project_facts(current, materialized),
        files=_project_files(current, materialized),
        outputs=StageOutputs(root="outputs", required=spec.required_outputs, optional=spec.optional_outputs, contracts_root="inputs/contracts", files=spec.files),
        validation_feedback=inputs.get("validation_feedback") or {},
        execution={
            "cwd": "outputs", "python": "python", "commands": spec.commands,
            "source_bindings": [item.model_dump(mode="json") for item in materialized.sources],
            "input_config": "inputs/inspection_config.json" if current is Stage.UNDERSTAND else "inputs/preparation_config.json",
            "output_dir": "outputs",
            "previous_reference": (
                {
                    "worker_root": materialized.values["previous_reference"]["worker_root"],
                    "digest": materialized.values["previous_reference"]["digest"],
                }
                if materialized.values.get("previous_reference") else {}
            ),
            "revision_basis": materialized.values.get("revision_basis", {}),
        },
        budgets=inputs["budget"].model_dump(mode="json"),
    )


def render_attempt_mode(packet: ContextPacket) -> str:
    if packet.facts.get("revision_basis"):
        return ("This is a semantic mapping revision. Treat facts.revision_basis as the old semantic "
                "baseline and validation_feedback as the conflict evidence; produce a new mapping.")
    if packet.validation_feedback:
        return "This is a repair attempt. Read the host validation feedback before changing files."
    if int(packet.identity.get("attempt", 1)) > 1:
        return "This is a retry attempt. Preserve valid work and address the current failure."
    return "This is the initial attempt for the assigned stage."


def render_prompt(stage: Stage | str, packet: ContextPacket) -> str:
    current = Stage(stage)
    names = {Stage.UNDERSTAND: "data_preparation_understand.md", Stage.PREPARE_TEXT: "data_preparation_prepare_text.md", Stage.PREPARE_IMAGE: "data_preparation_prepare_image.md"}
    common = read_prompt("data_preparation_agent.md")
    stage_prompt = read_prompt(names[current])
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
        render_attempt_mode(packet),
        "Stage contract:\n```json\n" + contract_text + "\n```",
        "Current host context:\n```json\n" + packet.model_dump_json(indent=2) + "\n```",
        "The same context packet is available at ./inputs/context.json. Follow its file-reading requirements and output contracts.",
        """Submission protocol

Write outputs/submission.json using the supplied schema.

- submitted: all required stage deliverables exist and the declared checks actually ran.
- needs_input: a genuine missing user fact blocks correct processing; include question IDs.
- blocked: a technical or contract failure prevents completion; include structured diagnostics.
- In prepare, use diagnostic code mapping_revision_required only when the accepted semantics are wrong.
  The semantic conflict classes are semantic_conflict, population_definition_conflict,
  grouping_semantics_conflict, and field_semantics_conflict. Output, schema, manifest,
  dependency, and script failures stay implementation diagnostics and must be repaired in prepare.
- A previous reference is read-only repair material. Do not modify files under its worker path.

Never claim host acceptance or publication. After writing the submission and confirming that no command is running, call finish.""",
    ])


def _token_count(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def fit_context_budget(packet: ContextPacket, *, budget: StageBudget, renderer=None) -> tuple[ContextPacket, str, int]:
    render = renderer or (lambda value: render_prompt(packet.identity["stage"], value))
    prompt = render(packet)
    token_count = _token_count(prompt)
    configured = budget.initial_message_tokens
    if budget.model_context_tokens is not None:
        configured = min(configured, budget.model_context_tokens - budget.output_reserve_tokens - budget.conversation_reserve_tokens - budget.safety_margin_tokens)
    if configured < 256:
        raise ContextBudgetError("required_context_too_large: configured model budget leaves no room for required context")
    if token_count <= configured:
        return packet, prompt, token_count
    facts = dict(packet.facts)
    for key in ("source_report_summary", "execution_report_summary"):
        if key in facts:
            facts[key] = {"path": f"inputs/accepted/{key.replace('_summary', '.json')}"}
    reduced = packet.model_copy(update={"facts": facts})
    prompt = render(reduced)
    token_count = _token_count(prompt)
    if token_count <= configured:
        return reduced, prompt, token_count
    if reduced.validation_feedback:
        reduced = reduced.model_copy(update={"validation_feedback": {"summary": "Read inputs/validation.json for the complete report."}})
        prompt = render(reduced)
        token_count = _token_count(prompt)
        if token_count <= configured:
            return reduced, prompt, token_count
    raise ContextBudgetError("required_context_too_large: required stage context exceeds the stage budget")


def _execution_key(stage: Stage, packet: ContextPacket, state: dict[str, Any]) -> str:
    return digest_payload({
        "stage": stage.value,
        "input_digest": packet.identity["input_digest"],
        "source_snapshot_digest": packet.identity["source_snapshot_digest"],
        "handoff": packet.accepted_handoff.digest if packet.accepted_handoff else "",
        "decisions": [item.digest for item in packet.decisions],
        "contract": packet.outputs.model_dump(mode="json"),
        "prompt": digest_payload(render_prompt(stage, packet)),
        "model": state.get("model", ""),
        "base_url": state.get("base_url", ""),
        "reasoning_effort": state.get("reasoning_effort", ""),
    })


def build_stage_context(stage: Stage | str, state: dict[str, Any], attempt_dir: str | Path) -> tuple[ContextPacket, str, int, str]:
    current = Stage(stage)
    inputs = resolve_stage_inputs(current, state)
    validate_input_identities(inputs)
    attempt_path = Path(attempt_dir)
    materialized = materialize_worker_inputs(inputs, attempt_path / "inputs")
    packet = project_context(current, inputs, materialized, {**state, "attempt": state.get("attempt") or state.get("stage_attempt") or 1})
    packet, prompt, token_count = fit_context_budget(packet, budget=inputs["budget"])
    write_context_snapshot(attempt_path, packet)
    write_prompt_snapshot(attempt_path, prompt)
    return packet, prompt, token_count, _execution_key(current, packet, state)


def write_context_snapshot(attempt_dir: str | Path, packet: ContextPacket) -> Path:
    path = Path(attempt_dir) / "inputs" / "context.json"
    _write_json(path, packet.model_dump(mode="json"))
    return path


def write_prompt_snapshot(attempt_dir: str | Path, prompt: str) -> Path:
    path = Path(attempt_dir) / "inputs" / "prompt.md"
    _write_text(path, prompt)
    return path


def read_prompt(name: str) -> str:
    path = Path(__file__).parents[1] / "prompts" / name
    if not path.is_file():
        raise FileNotFoundError(f"Preparation prompt is missing: {path}")
    return path.read_text(encoding="utf-8")
