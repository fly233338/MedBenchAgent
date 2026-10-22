"""Explicitly scripted example agents; no model calls or medical interpretation."""

import base64
import hashlib
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from medbenchagent.data.dataset import DatasetManifest, DatasetRef, FileDatasetReader
from medbenchagent.data.models import (
    AssetEntry, CaseRecord, CaseRelation, EntityRef, Evidence, EvidenceRef, SourceRef,
)
from medbenchagent.planning.contracts import DataRequirement, SemanticTaskSpec

from .artifacts import read_json_artifact, resolve_artifact, write_artifact, write_json_artifact
from .contracts import ArtifactRef, AskUser, Delegate, Finish, WorkNeed, WorkRequest, WorkResult
from .graph import AgentInput, AgentOutput, AgentSpec, CompilationContext, SupervisorView


def _latest(references, kind):
    return next((ref for ref in reversed(references) if ref.kind == kind), None)


def _reader(directory, references):
    dataset = _latest(references, "dataset")
    if dataset is None:
        raise ValueError("The demo request needs a Dataset Manifest")
    return FileDatasetReader(resolve_artifact(directory, dataset))


def _dataset_ref(reader):
    manifest = reader.get_manifest()
    return DatasetRef(dataset_id=manifest.dataset_id, revision=manifest.revision)


def _overview(reader):
    cases = reader.list_cases().items
    return {
        "dataset": _dataset_ref(reader).model_dump(),
        "case_ids": [case.case_id for case in cases],
        "fields": sorted({evidence.name for case in cases for evidence in case.evidence}),
    }


def create_synthetic_inputs(
    directory: Path, compilation_id: str, *, initial_evidence: bool = False
) -> list[ArtifactRef]:
    request = WorkRequest(
        compilation_id=compilation_id, request_id=f"{compilation_id}-input",
        agent_id="fixture", objective="Create a tiny synthetic Case dataset",
    )
    # Two distinct Asset identities can have identical content.
    pixels = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aRZk"
        "AAAAASUVORK5CYII="
    )
    write_json_artifact(
        directory, request, "dataset/source.json", kind="source",
        value={"case-a": 2, "case-b": 5},
    )
    cases, assets = [], []
    for suffix, value in (("a", 2), ("b", 5)):
        case_id = f"case-{suffix}"
        source = SourceRef(source_id="raw", path="source.json", record=case_id)
        write_artifact(
            directory, request, f"dataset/image-{suffix}.png", kind="image", content=pixels,
        )
        assets.append(AssetEntry(
            asset_id=f"asset-{suffix}", kind="image", format="png", source_id="raw",
            source_path=f"image-{suffix}.png", file_digest=hashlib.sha256(pixels).hexdigest(),
            width=1, height=1,
        ))
        cases.append(CaseRecord(
            case_id=case_id, patient_id="synthetic-patient",
            evidence=[
                Evidence(
                    id="image", name="view", type="asset", value=f"asset-{suffix}",
                    view_id=f"view-{suffix}", study_id="synthetic-study",
                    input_visibility="model_visible",
                ),
                Evidence(
                    id="value", name="value", type="value", value=value,
                    entity=EntityRef(kind="sample", id=case_id),
                    input_visibility="model_visible", source_refs=[source],
                ),
            ],
            relations=[CaseRelation(
                subject=EvidenceRef(case_id=case_id, evidence_id="value"), predicate="describes",
                object=EvidenceRef(case_id=case_id, evidence_id="image"), source_refs=[source],
            )],
        ))
    for name, rows in (("cases.jsonl", cases), ("assets.jsonl", assets)):
        write_artifact(
            directory, request, f"dataset/{name}", kind="records",
            content=("\n".join(row.model_dump_json() for row in rows) + "\n").encode("utf-8"),
        )
    write_json_artifact(
        directory, request, "dataset/mapping.json", kind="mapping",
        value={"case_unit": "synthetic sample", "fields": {"value": "synthetic input number"}},
    )
    manifest = DatasetManifest(
        dataset_id="synthetic", revision="fixture", cases_path="cases.jsonl",
        assets_path="assets.jsonl", sources=[{"source_id": "raw", "root": "."}],
        mapping_path="mapping.json",
    )
    dataset = write_json_artifact(
        directory, request, "dataset/manifest.json", kind="dataset", value=manifest.model_dump(),
    )
    inputs = [dataset]
    if initial_evidence:
        inputs.append(write_json_artifact(
            directory, request, "overview.json", kind="dataset_overview",
            value=_overview(_reader(directory, inputs)),
        ))
    return inputs


def coding_node(state: AgentInput, runtime: Runtime[CompilationContext]) -> AgentOutput:
    request = WorkRequest.model_validate(state["request"])
    directory = runtime.context.directory
    reader = _reader(directory, request.inputs)
    needed = _latest(request.inputs, "data_requirement")
    if needed is None:
        output = write_json_artifact(
            directory, request, "overview.json", kind="dataset_overview", value=_overview(reader),
        )
        summary = "Inspected the synthetic dataset"
    else:
        requirement = DataRequirement.model_validate(read_json_artifact(directory, needed))
        if requirement.dataset != _dataset_ref(reader):
            raise ValueError("Data requirement belongs to another Dataset snapshot")
        output = write_json_artifact(
            directory, request, "derived.json", kind="derived_view",
            value={
                "dataset": requirement.dataset.model_dump(),
                "requirement_id": requirement.requirement_id,
                "rows": [
                    {"case_id": case_id,
                     "source_evidence_ids": ["value"],
                     "doubled_value": reader.get_evidence(case_id, "value").value * 2}
                    for case_id in requirement.case_ids
                ],
            },
        )
        summary = "Produced the requested Case-linked derived values"
    result = WorkResult(
        request_id=request.request_id, outcome="completed", summary=summary, artifacts=[output],
    )
    return {"result": result.model_dump(mode="json")}


class PlannerState(AgentInput, AgentOutput, total=False):
    planning_pass: int
    pending_requirement_id: str | None


def build_scripted_planner():
    def plan(state: PlannerState, runtime: Runtime[CompilationContext]):
        request = WorkRequest.model_validate(state["request"])
        directory = runtime.context.directory
        reader = _reader(directory, request.inputs)
        case_ids = [case.case_id for case in reader.list_cases().items]
        evidence = [EvidenceRef(case_id=case_id, evidence_id="value") for case_id in case_ids]
        planning_pass = state.get("planning_pass", 0) + 1
        derived = _latest(request.inputs, "derived_view")
        artifacts, needs = [], []
        if derived is None:
            requirement = DataRequirement(
                requirement_id="double-value", dataset=_dataset_ref(reader), case_ids=case_ids,
                purpose="Provide a reference answer for each synthetic Case",
                representation="JSON rows with case_id, doubled_value and source_evidence_ids",
                evidence_refs=evidence,
                acceptance_criteria=["Preserve Case IDs", "Double the original value exactly"],
            )
            requirement_ref = write_json_artifact(
                directory, request, "requirement.json", kind="data_requirement",
                value=requirement.model_dump(mode="json"),
            )
            artifacts.append(requirement_ref)
            needs.append(WorkNeed(
                need_id=requirement.requirement_id, kind="work",
                summary=requirement.purpose, detail_ref=requirement_ref,
            ))
            pending = requirement.requirement_id
        else:
            data = read_json_artifact(directory, derived)
            if data["dataset"] != _dataset_ref(reader).model_dump():
                raise ValueError("Derived view belongs to another Dataset snapshot")
            if data["requirement_id"] != state.get("pending_requirement_id"):
                raise ValueError("Derived view does not answer the pending requirement")
            expected = {key: reader.get_evidence(key, "value").value * 2 for key in case_ids}
            rows = data["rows"]
            if (len(rows) != len(expected)
                    or {row["case_id"]: row["doubled_value"] for row in rows} != expected
                    or any(row["source_evidence_ids"] != ["value"] for row in rows)):
                raise ValueError("Derived values do not satisfy the data requirement")
            pending = None
        task = SemanticTaskSpec(
            task_id="synthetic-double", dataset=_dataset_ref(reader),
            scenario_id="synthetic-values", task_family="open_vqa",
            objective="Read a synthetic Case value and report twice that value",
            input_requirements=["The explicitly selected synthetic value"],
            reference_requirements=["Case-linked doubled_value from the derived view"],
            output_paradigm="number", evidence_refs=evidence,
            feasibility="needs_data" if pending else "feasible",
            requirement_ids=[pending] if pending else [],
        )
        artifacts.append(write_json_artifact(
            directory, request, "plan.json", kind="task_plan",
            value=task.model_dump(mode="json"),
        ))
        result = WorkResult(
            request_id=request.request_id, outcome="needs_input" if pending else "completed",
            summary="Need a derived representation" if pending else "Task now has reference data",
            artifacts=artifacts, needs=needs,
        )
        return {
            "planning_pass": planning_pass, "pending_requirement_id": pending,
            "result": result.model_dump(mode="json"),
        }

    builder = StateGraph(
        PlannerState, input_schema=AgentInput, output_schema=AgentOutput, context_schema=CompilationContext,
    )
    builder.add_node("plan", plan)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", END)
    return builder.compile(checkpointer=True)


def demo_agents() -> dict[str, AgentSpec]:
    return {
        "coding": AgentSpec("Inspect or transform the synthetic data", coding_node),
        "task_planner": AgentSpec("Design and continue a synthetic task", build_scripted_planner()),
    }


def scripted_supervisor(directory: Path, *, pause_for_input: bool = False):
    """A response-driven test policy, explicitly not a model's autonomous reasoning."""
    def decide(view: SupervisorView):
        references = list(view.artifacts)
        if view.last_result and view.last_result.outcome in {"failed", "blocked"}:
            return Finish(summary=f"Scripted run stopped: {view.last_result.summary}")
        plan = _latest(references, "task_plan")
        if plan and read_json_artifact(directory, plan)["feasibility"] == "feasible":
            return Finish(summary="Scripted collaboration complete; no benchmark was generated")
        if _latest(references, "dataset_overview") is None:
            return Delegate(
                agent_id="coding", objective="Inspect the available synthetic data",
                inputs=references,
            )
        if pause_for_input and _latest(references, "derived_view") and not view.answers:
            return AskUser(question="Continue planning with the new synthetic evidence?")
        if view.last_result and any(need.kind == "work" for need in view.last_result.needs):
            return Delegate(
                agent_id="coding", objective=view.last_result.needs[0].summary,
                inputs=references,
            )
        return Delegate(
            agent_id="task_planner", objective="Design or revise the evidence-grounded task",
            inputs=references,
        )
    return decide
