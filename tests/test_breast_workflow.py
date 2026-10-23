"""BrEaST-shaped offline collaboration through real native agents and host checks."""
import json
from pathlib import Path
import runpy

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from breast_fixtures import create_breast_source
from coding_fixtures import FakeRunner
from native_agent_fixtures import ScriptedChatModel, call, current_view
from medbenchagent.agents._shared.handoff import binding
from medbenchagent.agents.data_processing.coding.adapter import OpenHandsCodingAgent
from medbenchagent.agents.data_processing.coding.worker import CodingCandidate
from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact
from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.contracts import ArtifactRef, WorkRequest
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.compilation.run import create_run, resume_run
from medbenchagent.data.dataset import DatasetRef, DatasetSource, FileDatasetReader
from medbenchagent.data.task_data import DatasetProfile, SourceManifest, TaskCase, TaskDataManifest, PreparedAsset
from medbenchagent.data.tabular import inspect_table
from medbenchagent.evaluation.loader import load_benchmark_package
from medbenchagent.generation.execution import CandidateReader
from medbenchagent.runtime.graph_runner import initial_state
from medbenchagent.settings import CodingSettings


class BreastScript:
    stages = [
        ("profile", "data_processing", "profile"),
        ("discover", "task_planner", "discover_scenarios"),
        ("prepare-classification", "data_processing", "prepare_task_data"),
        ("prepare-spatial", "data_processing", "prepare_task_data"),
        ("design", "task_planner", "design_leaves"),
        ("generate", "generation", "generate"),
        ("reject", "review", "review_candidate"),
        ("repair", "task_planner", "design_leaves"),
        ("regenerate", "generation", "generate"),
        ("approve", "review", "review_candidate"),
        ("deliver", "delivery", "publish"),
    ]

    def __init__(self, directory, raw, workbook):
        self.directory, self.raw, self.workbook = directory, raw, workbook
        self.known, self.latest, self.positions, self.receipts = {}, {}, {}, {}
        self.scenarios, self.coding_calls, self.completed = {}, [], []
        self.stream = None
        self.dataset = DatasetRef(dataset_id="BrEaST", revision="fixture")
        self.proof = [{"observation_id": "workbook"}]

    def observe(self, result):
        for ref in result.artifacts:
            self.known[ref.artifact_id] = ref
            self.latest[ref.kind] = ref

    def ref(self, kind):
        return self.latest[kind].artifact_id

    def requirement(self, key):
        return next(ref for ref in reversed(list(self.known.values())) if ref.kind == "data_requirement"
                    and read_json_artifact(self.directory, ref)["requirement_id"] == key)

    def produce(self, payload):
        objective = payload["request"]["objective"]
        self.coding_calls.append(objective)
        output = Path(payload["output_directory"])
        if objective == "prepare-spatial":
            return CodingCandidate(request_id=payload["request"]["request_id"], outcome="blocked",
                summary="Multi-image mask association is unknown", issues=[{"code": "mask_association_unknown",
                    "message": "Case case-001 has two images and no explicit mask-to-image mapping"}])
        if objective == "profile":
            prepare = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_breast_example.py"))["prepare"]
            self.canonical = prepare(self.workbook, self.raw, output / "canonical")
            reader = FileDatasetReader(self.canonical)
            self.cases = reader.list_cases().items
            self.sources = [DatasetSource(source_id=source.source_id,
                                         root=str(self.canonical.parent / source.root)) for source in reader.get_manifest().sources]
            stats = inspect_table(self.workbook, id_columns=["CaseID"])
            assert stats["rows_scanned"] == 4
            profile = DatasetProfile(dataset=self.dataset, sources=self.sources, description="Four source lesion records",
                coverage="complete", inspected_scope="Four workbook rows and named PNG assets",
                fields=stats["columns"], samples=self.cases,
                observations=[{"observation_id": "workbook", "description": "Three recorded labels; one unknown label; multiview mask association unknown",
                               "source_refs": [{"source_id": "metadata", "path": "source.xlsx"}]}],
                unknowns=["case-001 mask association", "case-004 pathology"])
            filename, kind, value = "profile.json", "dataset_profile", profile
        else:
            reader = FileDatasetReader(self.canonical)
            cases, assets = [], {}
            for source_case in self.cases:
                label = next(e for e in source_case.evidence if e.name == "pathology")
                if label.value is None:
                    continue
                images = [e for e in source_case.evidence if e.input_visibility == "model_visible"]
                for image in images:
                    asset = reader.get_asset(image.value)
                    assets[asset.asset_id] = PreparedAsset(**{key: getattr(asset, key) for key in
                        ("asset_id", "source_id", "source_path", "file_digest", "kind", "format", "metadata")})
                cases.append(TaskCase(case_id=source_case.case_id, scenario_ids=[self.scenarios["recognition"]],
                    source_refs=label.source_refs, fields=[
                        {"name": "views", "type": "asset", "value": [e.value for e in images], "input_visibility": "model_visible",
                         "source_refs": [ref.model_dump() for e in images for ref in e.source_refs]},
                        {"name": "label", "type": "value", "value": label.value, "input_visibility": "reference_only",
                         "source_refs": [ref.model_dump() for ref in label.source_refs], "ref": ["views"]}]))
            for name, rows in (("cases.jsonl", cases), ("assets.jsonl", assets.values())):
                (output / name).write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
            value = TaskDataManifest(dataset=self.dataset, sources=self.sources, backend_id="vqa",
                scenario_ids=[self.scenarios["recognition"]], requirements={"classification": binding(self.requirement("classification"))},
                profile=binding(self.latest["dataset_profile"]), case_granularity="source lesion record",
                preparation_parameters={"missing_label_case_ids": ["case-004"]})
            filename, kind = "task-data.json", "task_data"
        (output / filename).write_text(value.model_dump_json(), encoding="utf-8")
        return CodingCandidate(request_id=payload["request"]["request_id"], outcome="completed", summary="Measured source data",
                               files=[{"path": filename, "kind": kind}])

    def leaf(self, repair=False):
        if repair:
            leaf = read_json_artifact(self.directory, self.latest["task_plan"])["tasks"]["classify"]
            leaf["revision"] += 1
            leaf["change_reason"] = "Address review wording"
            leaf["assembly"]["question_templates"] = ["Using these views, select the recorded lesion category: {options}"]
            return leaf
        return {"task_id": "classify", "dataset": self.dataset.model_dump(), "scenario_id": self.scenarios["recognition"],
                "task_family": "closed_vqa", "objective": "Classify the lesion", "input_requirements": ["views"],
                "reference_requirements": ["recorded pathology"], "output_paradigm": "label", "evidence_refs": self.proof,
                "feasibility": "feasible", "requirement_ids": ["classification"], "capability_category": "Recognition",
                "task_data": binding(self.latest["task_data"]).model_dump(),
                "assembly": {"backend_id": "vqa", "input_fields": ["views"], "reference_field": "label",
                             "question_templates": ["Which label? {options}"], "options": ["benign", "malignant"], "output_spec": "label"}}

    def actions(self, stage, request):
        result = self.receipts.setdefault(stage, {})
        if stage in {"profile", "prepare-classification", "prepare-spatial"}:
            actions = [("invoke_coding", lambda: {"objective": stage, "input_artifact_ids": [r["artifact_id"] for r in request["inputs"]],
                                                  "constraints": [], **({"workstream_id": self.stream} if self.stream else {})})]
            if stage != "prepare-spatial":
                actions.append(("submit_data_artifacts", lambda: {"call_id": result["invoke_coding"]["call_id"],
                                                                  "paths": [result["invoke_coding"]["files"][0]["path"]]}))
            return actions
        if stage == "discover":
            actions = []
            for axis, values in (("input_structure", ["multiview"]), ("operation", ["recognition", "localization"]), ("clinical_target", ["breast-lesion"])):
                actions.append(("record_axis_assessment", lambda axis=axis, values=values: {"axis": axis,
                    "assessments": [{"value": value, "supported": True, "reason": "Source observations", "evidence_refs": self.proof} for value in values]}))
            actions += [("build_candidate_scenarios", lambda: {}), ("inspect_task_plan", lambda: {"section": "scenarios", "limit": 2})]
            for operation, backend, key in (("recognition", "vqa", "classification"), ("localization", "visual_grounding", "spatial")):
                actions += [
                    ("record_scenario_decision", lambda operation=operation: {"scenario_id": self.scenarios[operation], "keep": True,
                         "reason": "Useful task; missing representation remains a technical requirement", "evidence_refs": self.proof}),
                    ("set_scenario_backends", lambda operation=operation, backend=backend: {"scenario_id": self.scenarios[operation], "backend_ids": [backend]}),
                    ("propose_data_requirement", lambda operation=operation, backend=backend, key=key: {"requirement": {
                        "requirement_id": key, "dataset": self.dataset.model_dump(), "purpose": key, "representation": "TaskData", "case_granularity": "source lesion record",
                        "case_ids": ["case-001"] if key == "spatial" else [],
                        "evidence_refs": self.proof, "acceptance_criteria": ["Only source-backed answers"],
                        "affected_scenario_ids": [self.scenarios[operation]], "backend_id": backend,
                        "required_fields": [{"name": "views", "type": "asset", "input_visibility": "model_visible", "purpose": "Images"},
                                            {"name": "label" if key == "classification" else "region", "type": "value", "input_visibility": "reference_only", "purpose": "Reference"}]}}),
                ]
            return actions + [("publish_task_plan", lambda: {})]
        if stage in {"design", "repair"}:
            actions = [] if stage == "repair" else [("accept_data_requirement", lambda: {
                "requirement_id": "classification", "artifact_id": self.ref("task_data"), "reason": "Three source-backed labels; unknown row omitted explicitly", "evidence_refs": self.proof})]
            return actions + [("save_semantic_task", lambda: {"task": self.leaf(stage == "repair")}), ("publish_task_plan", lambda: {})]
        if stage in {"generate", "regenerate"}:
            return [
                ("prepare_task_execution", lambda: {"task_plan_id": self.ref("task_plan"), "task_id": "classify", "task_data_id": self.ref("task_data"), "sample_count": 4, "seed": 7}),
                ("allocate_generation_budget", lambda: {"spec_ids": [result["prepare_task_execution"]["artifact_id"]], "requested_count": 4}),
                ("generate_task_batch", lambda: {"spec_id": result["prepare_task_execution"]["artifact_id"], "allocation_id": result["allocate_generation_budget"]["artifact_id"], "offset": 0, "limit": 4}),
                ("finalize_generation", lambda: {"spec_ids": [result["prepare_task_execution"]["artifact_id"]], "batch_ids": [result["generate_task_batch"]["artifact_id"]], "allocation_id": result["allocate_generation_budget"]["artifact_id"]}),
            ]
        if stage in {"reject", "approve"}:
            return [("run_candidate_audit", lambda: {"candidate_id": self.ref("generation_candidate")}),
                    ("finalize_review", lambda: {"target_id": self.ref("generation_candidate"), "audit_id": result["run_candidate_audit"]["artifact_id"],
                     "findings": [] if stage == "approve" else [{"object_id": "classify", "category": "template", "blocking": True,
                         "summary": "Clarify which images and source label the question refers to", "evidence_artifact_ids": [self.ref("task_plan")], "suggested_owner": "task_planner"}],
                     "checked_task_ids": ["classify"], "checked_scenario_ids": [self.scenarios["recognition"]],
                     "checked_item_ids": [item.item_id for item in CandidateReader(resolve_artifact(self.directory, self.latest["generation_candidate"])).items],
                     "unchecked_scope": [], "approved": stage == "approve"})]
        raise AssertionError(stage)

    def respond(self, messages, index):
        view = current_view(messages)
        if "request" not in view:
            assert not view.get("feedback"), view.get("feedback")
            count = view["catalog_totals"]["work"]
            if count == 3 and not view["user_answers"]:
                return call("SupervisorResponse", {"decision": {"action": "ask_user", "question": "Resume the offline checkpoint?"}}, index)
            if count == len(self.stages):
                return call("SupervisorResponse", {"decision": {"action": "finish", "summary": "Three of four requested items; spatial association unresolved",
                    "goal_result": "partial", "evidence_artifact_ids": [self.ref("delivery_manifest")]}}, index)
            stage, agent, operation = self.stages[count]
            if count == 0:
                target, kind, ids = self.ref("source_manifest"), "artifact", []
            elif count == 1:
                target, kind, ids = self.ref("dataset_profile"), "artifact", []
            elif count in {2, 3, 4, 5, 7, 8}:
                target = self.ref("task_plan")
                kind, ids = ("requirement", ["classification" if count == 2 else "spatial"]) if count in {2, 3} else (
                    ("scenario", [self.scenarios["recognition"]]) if count == 4 else ("task", ["classify"]))
            else:
                target, kind, ids = self.ref("generation_candidate"), "artifact", []
            unique_kinds = {"task_plan", "generation_candidate", "audit_report", "review_report"}
            selected = [ref.artifact_id for ref in self.known.values() if ref.kind not in unique_kinds or ref == self.latest[ref.kind]]
            return call("SupervisorResponse", {"decision": {"action": "delegate", "agent_id": agent, "operation": operation,
                "objective": stage, "scope": [{"artifact_id": target, "kind": kind, "object_ids": ids}],
                "input_artifact_ids": selected, "constraints": ["Requested four; preserve missing labels and unknown mask associations"]}}, index)
        request = view["request"]
        stage, identifier = request["objective"], request["request_id"]
        position = self.positions.get(identifier, 0)
        actions = self.actions(stage, request)
        if position:
            name = actions[position - 1][0]
            previous = next(m for m in reversed(messages) if m.type == "tool" and m.tool_call_id == f"step-{identifier}-{position}")
            assert previous.status != "error", (stage, name, previous.content)
            result = json.loads(previous.content)
            self.receipts[stage][name] = result
            if name == "invoke_coding":
                self.stream = result["request"]["workstream_id"]
            if name == "inspect_task_plan":
                self.scenarios = {row["value"]["operation"]: row["id"] for row in result["items"]}
        if position == len(actions):
            response = {"outcome": "blocked" if stage == "prepare-spatial" else "completed", "summary": stage,
                        "artifact_ids": [ref["artifact_id"] for ref in view["submitted"]]}
            if stage == "prepare-spatial":
                response["issues"] = self.receipts[stage]["invoke_coding"]["issues"]
            return call("PlannerResponse" if request["agent_id"] == "task_planner" else "WorkResponse", response, index)
        name, arguments = actions[position]
        args = arguments()
        if request["agent_id"] == "task_planner" and not name.startswith("inspect"):
            args["base_revision"] = view["plan"]["revision"]
        self.positions[identifier] = position + 1
        response = call(name, args, index)
        response.tool_calls[0]["id"] = f"step-{identifier}-{position + 1}"
        return response


def test_breast_partial_delivery_with_blocked_sibling_repair_and_resume(tmp_path, monkeypatch):
    import medbenchagent.runtime.native as native
    import medbenchagent.runtime.model_services as services
    import medbenchagent.compilation.policy as policy
    import medbenchagent.agents.data_processing.coding.adapter as coding
    import medbenchagent.agents.data_processing.coding.skills as skills
    from medbenchagent.agents._shared import prompting

    raw, directory = tmp_path / "raw", tmp_path / "run"
    script = BreastScript(directory, raw, create_breast_source(raw))
    source = tmp_path / "source.json"
    source.write_text(SourceManifest(dataset=script.dataset, sources=[DatasetSource(source_id="raw", root=str(raw))]).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: "offline")
    options = create_run(directory, compilation_id="breast", goal="Four traceable questions", source_manifest=source, sample_count=4)
    for snapshot in [options.supervisor_settings, options.planner_settings, *options.professional_settings.values()]:
        limits = next(value for key, value in snapshot.items() if key != "llm")
        limits["context_chars"] = 2000000
        limits["loop"].update(context_tokens=500000, summary_trigger_tokens=300000, summary_keep_tokens=10000)
    (directory / "run.json").write_text(options.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(native, "chat_model", lambda *_: ScriptedChatModel(responder=script.respond))
    runner = FakeRunner(script.produce)
    monkeypatch.setattr(coding, "OpenHandsCodingAgent", lambda settings, **kw: OpenHandsCodingAgent(settings, **kw, runner=runner))
    original_check = policy.check_result
    def checked(root, request, result):
        result = original_check(root, request, result)
        assert result.outcome == ("blocked" if request.objective == "prepare-spatial" else "completed"), result
        script.observe(result)
        script.completed.append((request, result))
        return result
    monkeypatch.setattr(policy, "check_result", checked)
    initial = initial_state(directory, options)
    for row in initial["artifacts"]:
        ref = ArtifactRef.model_validate(row)
        script.known[ref.artifact_id] = script.latest[ref.kind] = ref
    config = {"configurable": {"thread_id": "breast"}, "recursion_limit": 100}
    with assemble_compilation(options, directory) as (agents, context), SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(agents, checkpointer=saver)
        state = graph.invoke(initial, config, context=context)
        assert state["status"] == "waiting" and len(state["history"]) == 3
        ready = context.readiness(tuple(ArtifactRef.model_validate(row) for row in state["artifacts"]))
        assert any(row["operation"] == "design_leaves" and row["scope"][0]["object_ids"] == [script.scenarios["recognition"]] for row in ready["available_operations"])
        assert any(row["operation"] == "design_leaves" and row["scope"][0]["object_ids"] == [script.scenarios["localization"]] for row in ready["blocked_reasons"])
    data_ref = script.latest["task_data"]
    data_mtime = resolve_artifact(directory, data_ref).stat().st_mtime_ns
    monkeypatch.setattr(prompting, "load_prompt", lambda *_: pytest.fail("Resume loaded live Prompt"))
    monkeypatch.setattr(skills, "files", lambda *_: pytest.fail("Resume loaded live Skill"))
    restored = resume_run(directory, "breast")
    replay = OpenHandsCodingAgent(CodingSettings.from_snapshot(restored.coding_settings),
        instructions=restored.prompts["coding"].text, skills=restored.coding_skills,
        runner=FakeRunner(lambda _: pytest.fail("Completed Coding request executed again")))
    result, _ = replay.execute_candidate(directory, WorkRequest.model_validate(runner.payloads[-1]["request"]), options.goal)
    assert result.outcome == "completed"
    with assemble_compilation(restored, directory) as (agents, context), SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(agents, checkpointer=saver)
        state = graph.invoke(Command(resume="Continue"), config, context=context)
    assert state["goal_result"] == "partial" and len(state["history"]) == 11
    for record, (stage, agent, operation) in zip(state["history"], script.stages, strict=True):
        assert (record["request"]["objective"], record["request"]["agent_id"], record["request"]["operation"]) == (stage, agent, operation)
        assert record["request"]["scope"] and record["request"]["inputs"]
    assert state["history"][3]["result"]["issues"][0]["code"] == "mask_association_unknown"
    package = load_benchmark_package(resolve_artifact(directory, script.latest["delivery_manifest"]).parent)
    assert len(package["items"]) == 3
    assert package["manifest"]["counts"]["requested"] == 4 and package["manifest"]["counts"]["shortfall"] == 1
    assert len(next(item for item in package["items"] if item["case_id"] == "case-001")["inputs"]) == 2
    assert all("recorded lesion category" in row["question"] for row in package["items"])
    plan = read_json_artifact(directory, script.latest["task_plan"])
    assert "spatial" in plan["requirements"] and "spatial" not in plan["acceptances"]
    assert plan["scenarios"][script.scenarios["localization"]]["decision"] == "keep"
    assert script.latest["task_data"] == data_ref and resolve_artifact(directory, data_ref).stat().st_mtime_ns == data_mtime
    assert script.coding_calls == ["profile", "prepare-classification", "prepare-spatial"]
    repair = next(request for request, _ in script.completed if request.objective == "repair")
    assert repair.scope[0].kind == "task" and repair.scope[0].object_ids == ["classify"]
    assert all(request.inputs and request.scope for request, _ in script.completed)
