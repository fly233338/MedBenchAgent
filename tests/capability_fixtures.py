"""Small real files and direct tool calls for the new capability boundaries; no model service."""

import json
from dataclasses import replace
from types import SimpleNamespace
import inspect

from langgraph.types import Command

from medbenchagent.agents.data_processing import DataProcessingContext, data_processing_tools
from medbenchagent.agents.generation import generation_tools
from medbenchagent.agents.handoff import WorkContext, WorkInstructions, binding
from medbenchagent.agents.semantic_planner import (
    PlannerContext,
    PlannerContextMiddleware,
    planner_tools,
)
from medbenchagent.compilation.artifacts import (
    read_json_artifact,
    write_json_artifact,
    resolve_artifact,
)
from medbenchagent.compilation.coding import OpenHandsCodingAgent
from medbenchagent.compilation.coding_worker import CodingCandidate
from medbenchagent.compilation.contracts import ArtifactRef, WorkRequest
from medbenchagent.data.dataset import DatasetRef, DatasetSource, file_digest
from medbenchagent.data.models import SourceRef
from medbenchagent.data.task_data import (
    DatasetProfile,
    ProfileObservation,
    ProfileReader,
    SourceManifest,
    TaskCase,
    TaskField,
    TaskDataManifest,
    TaskDataReader,
    PreparedAsset,
)
from medbenchagent.generation.execution import CandidateReader
from medbenchagent.prompts import load_prompt
from medbenchagent.settings import (
    load_coding_settings,
    load_planner_settings,
    load_work_agent_settings,
)
from test_compilation_coding import FakeRunner


class ToolCalls:
    def __init__(self, tools, context, known, *, state=None, middleware=None):
        self.tools = {tool.name: tool for tool in tools}
        self.context, self.known = context, known
        self.state = state if state is not None else {}
        self.number = 0
        self.state.update(
            (middleware or WorkInstructions()).before_agent(
                self.state, SimpleNamespace(context=context)
            )
            or {}
        )

    def call(self, name, **arguments):
        self.number += 1
        function = self.tools[name].func
        if "runtime" in inspect.signature(function).parameters:
            arguments["runtime"] = SimpleNamespace(
                context=self.context, state=self.state, tool_call_id=f"tool-{self.number}"
            )
        if (
            "base_revision" in inspect.signature(function).parameters
            and "base_revision" not in arguments
        ):
            path = self.context.directory / self.state["draft_path"]
            arguments["base_revision"] = json.loads(path.read_text())["revision"]
        value = function(**arguments)
        if isinstance(value, Command):
            self.state.update(value.update)
            self.known.update(
                {
                    row["artifact_id"]: ArtifactRef.model_validate(row)
                    for row in self.state.get("submitted", [])
                }
            )
            return json.loads(value.update["messages"][-1].content)
        return value


def build_capabilities(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notes.txt").write_text(
        "Synthetic case with two inputs, label yes, report and region annotations.",
        encoding="utf-8",
    )
    (raw / "left.png").write_bytes(b"synthetic-image-bytes")
    (raw / "right.volume").write_bytes(b"synthetic-volume-bytes")
    dataset = DatasetRef(dataset_id="synthetic-capabilities", revision="source")
    source = SourceManifest(
        dataset=dataset, sources=[DatasetSource(source_id="raw", root=str(raw))]
    )
    source_file = tmp_path / "sources.json"
    source_file.write_text(source.model_dump_json(), encoding="utf-8")
    source_ref = write_json_artifact(
        directory,
        WorkRequest(request_id="input", compilation_id="cap", agent_id="input", objective="Source"),
        "sources.json",
        kind="source_manifest",
        value=source.model_dump(mode="json"),
    )
    known = {source_ref.artifact_id: source_ref}
    citation = SourceRef(source_id="raw", path="notes.txt", record="case-1")
    profile = DatasetProfile(
        dataset=dataset,
        sources=source.sources,
        description="Synthetic fixture",
        coverage="sampled",
        inspected_scope="one synthetic source row",
        fields=[{"name": "views", "count": 2}],
        observations=[
            ProfileObservation(
                observation_id="source-fields",
                description="Two inputs and prepared targets",
                source_refs=[citation],
            )
        ],
    )
    mutable = {}

    def produce(payload):
        from pathlib import Path

        output = Path(payload["output_directory"])
        if "profile" not in mutable:
            (output / "profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
            files = [{"path": "profile.json", "kind": "dataset_profile"}]
        else:
            files = []
            region = {
                "asset_id": "left",
                "box": [0.1, 0.2, 0.8, 0.9],
                "coordinate_system": "normalized_xyxy",
            }
            fields = [
                TaskField(
                    name="views",
                    type="asset",
                    value=["left", "right"],
                    input_visibility="model_visible",
                    source_refs=[citation],
                ),
                TaskField(
                    name="context",
                    type="text",
                    value="Two synthetic views",
                    input_visibility="model_visible",
                    source_refs=[citation],
                ),
                *[
                    TaskField(
                        name=name,
                        type=kind,
                        value=value,
                        input_visibility="reference_only",
                        source_refs=[citation],
                    )
                    for name, kind, value in (
                        ("label", "value", "yes"),
                        ("text", "text", "synthetic answer"),
                        ("report", "text", "synthetic report"),
                        ("region", "value", region),
                        ("detections", "value", [{**region, "label": "finding"}]),
                    )
                ],
            ]
            assets = [
                PreparedAsset(
                    asset_id=name,
                    source_id="raw",
                    source_path=filename,
                    file_digest=file_digest(raw / filename),
                    kind=kind,
                    format=fmt,
                )
                for name, filename, kind, fmt in (
                    ("left", "left.png", "image", "png"),
                    ("right", "right.volume", "volume", "synthetic"),
                )
            ]
            for backend in ("vqa", "report_generation", "visual_grounding", "detection"):
                destination = output / backend
                destination.mkdir()
                scopes = mutable["scenarios"] if backend == "vqa" else mutable["scenarios"][:1]
                case = TaskCase(
                    case_id="case-1", scenario_ids=scopes, fields=fields, source_refs=[citation]
                )
                (destination / "cases.jsonl").write_text(
                    case.model_dump_json() + "\n", encoding="utf-8"
                )
                (destination / "assets.jsonl").write_text(
                    "".join(asset.model_dump_json() + "\n" for asset in assets), encoding="utf-8"
                )
                manifest = TaskDataManifest(
                    dataset=dataset,
                    sources=source.sources,
                    backend_id=backend,
                    scenario_ids=scopes,
                    requirement_ids=[f"prepare-{backend}"],
                    profile=binding(mutable["profile"]),
                )
                (destination / "manifest.json").write_text(
                    manifest.model_dump_json(), encoding="utf-8"
                )
                files.append({"path": f"{backend}/manifest.json", "kind": "task_data"})
        return CodingCandidate(
            request_id=payload["request"]["request_id"],
            outcome="completed",
            summary="Prepared candidates",
            files=files,
        )

    runner = FakeRunner(produce)
    coding_settings = load_coding_settings()
    coding_settings = replace(coding_settings, llm=replace(coding_settings.llm, model="offline"))
    coding = OpenHandsCodingAgent(
        coding_settings, instructions=load_prompt("coding.md"), runner=runner
    )

    def request(identifier, agent, inputs=None):
        return WorkRequest(
            compilation_id="cap",
            request_id=identifier,
            agent_id=agent,
            objective="Exercise capabilities",
            inputs=list(known.values()) if inputs is None else inputs,
        )

    dp_settings = load_work_agent_settings("data_processing")
    dp = ToolCalls(
        data_processing_tools(),
        DataProcessingContext(
            directory,
            request("profile", "data_processing"),
            dp_settings,
            load_prompt("data_processing.md"),
            "data_processing",
            coding,
            "Prepare data",
        ),
        known,
    )
    result = dp.call(
        "invoke_coding",
        objective="Inspect sources",
        input_artifact_ids=[source_ref.artifact_id],
        constraints=[],
    )
    assert not dp.state["submitted"]
    dp.call("validate_data_candidate", call_id=result["call_id"], path="profile.json")
    profile_result = dp.call(
        "submit_data_artifacts", call_id=result["call_id"], paths=["profile.json"]
    )
    profile_ref = ArtifactRef.model_validate(profile_result["artifacts"][0])
    mutable["profile"] = profile_ref
    reader = ProfileReader(resolve_artifact(directory, profile_ref))
    planner_settings = load_planner_settings()
    planning_context = PlannerContext(
        directory,
        request("scenarios", "task_planner"),
        reader,
        profile_ref,
        planner_settings,
        load_prompt("planner.md"),
    )
    planner = ToolCalls(
        planner_tools(), planning_context, known, middleware=PlannerContextMiddleware()
    )
    proof = [{"observation_id": "source-fields"}]
    from medbenchagent.planning.contracts import (
        AxisAssessment,
        ProfileEvidenceRef,
        DataRequirement,
        SemanticTaskSpec,
        TaskPlan,
    )

    evidence = [ProfileEvidenceRef.model_validate(value) for value in proof]
    for axis, values in (
        ("input_structure", ["multi-input"]),
        ("operation", ["recognition", "description"]),
        ("clinical_target", ["synthetic"]),
    ):
        planner.call(
            "record_axis_assessment",
            axis=axis,
            assessments=[
                AxisAssessment(
                    value=value,
                    supported=True,
                    reason="Synthetic source observation",
                    evidence_refs=evidence,
                )
                for value in values
            ],
        )
    planner.call("build_candidate_scenarios")
    scenarios = [
        row["id"] for row in planner.call("inspect_task_plan", section="scenarios")["items"]
    ]
    mutable["scenarios"] = scenarios
    for index, scenario in enumerate(scenarios):
        planner.call(
            "record_scenario_decision",
            scenario_id=scenario,
            keep=True,
            reason="Supported by source observation",
            evidence_refs=evidence,
        )
        planner.call(
            "set_scenario_backends",
            scenario_id=scenario,
            backend_ids=["vqa", "report_generation", "visual_grounding", "detection"]
            if index == 0
            else ["vqa"],
        )
    for backend in ("vqa", "report_generation", "visual_grounding", "detection"):
        planner.call(
            "propose_data_requirement",
            requirement=DataRequirement(
                requirement_id=f"prepare-{backend}",
                dataset=dataset,
                purpose="Prepare task Cases",
                representation="Backend Cases",
                evidence_refs=evidence,
                acceptance_criteria=["Preserve source facts"],
                affected_scenario_ids=scenarios if backend == "vqa" else scenarios[:1],
                backend_id=backend,
            ),
        )
    scene_result = planner.call("publish_task_plan")
    scenario_ref = next(
        known[row["artifact_id"]] for row in scene_result["artifacts"] if row["kind"] == "task_plan"
    )
    assert not read_json_artifact(directory, scenario_ref)["tasks"]
    dp = ToolCalls(
        data_processing_tools(),
        replace(dp.context, request=request("prepare", "data_processing")),
        known,
        state=dp.state,
    )
    result = dp.call(
        "invoke_coding",
        objective="Prepare backend Cases",
        input_artifact_ids=list(known),
        constraints=[],
        workstream_id=dp.state["coding_sessions"][0],
    )
    published = dp.call(
        "submit_data_artifacts",
        call_id=result["call_id"],
        paths=[row["path"] for row in result["files"]],
    )
    task_refs = {}
    task_readers = {}
    for raw_ref in published["artifacts"]:
        ref = ArtifactRef.model_validate(raw_ref)
        task_reader = TaskDataReader(resolve_artifact(directory, ref))
        task_refs[task_reader.manifest.backend_id] = ref
        task_readers[ref.artifact_id] = task_reader
    planner = ToolCalls(
        planner_tools(),
        replace(
            planning_context, request=request("leaves", "task_planner"), task_data=task_readers
        ),
        known,
        state=planner.state,
        middleware=PlannerContextMiddleware(),
    )
    for backend, ref in task_refs.items():
        planner.call(
            "accept_data_requirement",
            requirement_id=f"prepare-{backend}",
            artifact_id=ref.artifact_id,
            reason="Inspected Cases satisfy the prepared-field requirement",
            evidence_refs=evidence,
        )
    definitions = (
        ("closed_vqa", "vqa", "label", "label"),
        ("open_vqa", "vqa", "text", "text"),
        ("report_generation", "report_generation", "report", "text"),
        ("visual_grounding", "visual_grounding", "region", "bbox"),
        ("detection", "detection", "detections", "detections"),
    )
    for family, backend, reference, output in definitions:
        task = SemanticTaskSpec(
            task_id=family,
            dataset=dataset,
            scenario_id=scenarios[0],
            task_family=family,
            objective="Answer from prepared evidence",
            input_requirements=["Ordered input views"],
            reference_requirements=[reference],
            output_paradigm=output,
            evidence_refs=evidence,
            feasibility="feasible",
            requirement_ids=[f"prepare-{backend}"],
            capability_category="Localization"
            if family in {"visual_grounding", "detection"}
            else "Generation"
            if family == "report_generation"
            else "Recognition",
            task_data=binding(task_refs[backend]),
            assembly={
                "backend_id": backend,
                "input_fields": ["views"],
                "question_templates": ["{context}: {task_objective}. {options}"],
                "question_fields": {"context": "context"},
                "reference_field": reference,
                "output_spec": output,
                "options": ["yes", "no"] if family == "closed_vqa" else None,
            },
        )
        planner.call("save_semantic_task", task=task)
    leaf_result = planner.call("publish_task_plan")
    leaf_ref = next(
        known[row["artifact_id"]] for row in leaf_result["artifacts"] if row["kind"] == "task_plan"
    )
    goal = write_json_artifact(
        directory,
        request("target", "input", []),
        "target.json",
        kind="benchmark_request",
        value={"sample_count": 5},
    )
    known[goal.artifact_id] = goal
    generation = ToolCalls(
        generation_tools(),
        WorkContext(
            directory,
            request("generate", "generation"),
            load_work_agent_settings("generation"),
            load_prompt("generation.md"),
            "generation",
        ),
        known,
    )
    specs = []
    for family, backend, _, _ in definitions:
        specs.append(
            generation.call(
                "prepare_task_execution",
                task_plan_id=leaf_ref.artifact_id,
                task_id=family,
                task_data_id=task_refs[backend].artifact_id,
                sample_count=1,
                seed=7,
            )["artifact_id"]
        )
    allocation = generation.call("allocate_generation_budget", spec_ids=specs, requested_count=5)[
        "artifact_id"
    ]
    batches = [
        generation.call(
            "generate_task_batch", spec_id=identifier, allocation_id=allocation, offset=0, limit=1
        )["artifact_id"]
        for identifier in specs
    ]
    candidate_ref = known[
        generation.call(
            "finalize_generation", spec_ids=specs, batch_ids=batches, allocation_id=allocation
        )["artifact_id"]
    ]
    return SimpleNamespace(
        directory=directory,
        known=known,
        candidate_ref=candidate_ref,
        candidate=CandidateReader(resolve_artifact(directory, candidate_ref)),
        generation=generation,
        coding=coding,
        source_file=source_file,
        profile=reader,
        task_refs=task_refs,
        plan=TaskPlan.model_validate(read_json_artifact(directory, leaf_ref)),
    )
