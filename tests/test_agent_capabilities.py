"""Two small offline collaboration checks using real tools, native agents and files."""
import json
import shutil
from types import SimpleNamespace

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from medbenchagent.agents.benchmark_review import review_tools
from medbenchagent.agents.delivery import delivery_spec
from medbenchagent.agents.handoff import WorkContext
from medbenchagent.compilation.artifacts import resolve_artifact
from medbenchagent.compilation.contracts import WorkRequest, WorkScope, GoalSpec, ArtifactRef
from medbenchagent.compilation.run import create_run
from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.evaluation.loader import load_benchmark_package
from medbenchagent.prompts import freeze_composition
from medbenchagent.settings import load_work_agent_settings
from capability_fixtures import ToolCalls, build_capabilities
from coding_fixtures import FakeRunner
from native_agent_fixtures import ScriptedChatModel, call, current_view


def test_profile_taskdata_leaf_generation_capabilities(tmp_path, monkeypatch):
    bundle = build_capabilities(tmp_path)
    request = WorkRequest(compilation_id="cap", request_id="cap-000006", agent_id="review",
        operation="review_candidate", objective="Review generated content",
        scope=[WorkScope(artifact_id=bundle.candidate_ref.artifact_id)], inputs=list(bundle.known.values()))
    review = ToolCalls(review_tools(), WorkContext(bundle.directory, request,
        load_work_agent_settings("review"), {key: freeze_composition(key).text for key in ("review_plan", "review_candidate")}, "review"), bundle.known)
    audit = review.call("run_candidate_audit", candidate_id=bundle.candidate_ref.artifact_id)
    review.call("finalize_review", target_id=bundle.candidate_ref.artifact_id, audit_id=audit["artifact_id"],
        findings=[], checked_task_ids=list(bundle.plan.tasks),
        checked_scenario_ids=list({i.scenario_id for i in bundle.candidate.items}),
        checked_item_ids=[i.item_id for i in bundle.candidate.items], unchecked_scope=[], approved=True)
    stages = [*bundle.stages, review]
    delivery = WorkRequest(compilation_id="cap", request_id="cap-000007", agent_id="delivery", operation="publish",
        objective="Publish reviewed content", scope=[WorkScope(artifact_id=bundle.candidate_ref.artifact_id)],
        inputs=list(bundle.known.values()))
    remapping, positions = {}, {}

    def translate(value):
        if isinstance(value, str):
            return remapping.get(value, value)
        if isinstance(value, list):
            return [translate(v) for v in value]
        if isinstance(value, dict):
            return {key: translate(v) for key, v in value.items()}
        return value

    def learn(expected, actual):
        if isinstance(expected, str) and isinstance(actual, str) and expected != actual:
            remapping[expected] = actual
        elif isinstance(expected, dict) and isinstance(actual, dict):
            for key in expected.keys() & actual.keys():
                learn(expected[key], actual[key])
        elif isinstance(expected, list) and isinstance(actual, list):
            for a, b in zip(expected, actual):
                learn(a, b)

    def respond(messages, index):
        view = current_view(messages)
        if "request" not in view:
            assert not view.get("feedback"), view.get("feedback")
            count = view["catalog_totals"]["work"]
            if count == 5 and not view["user_answers"]:
                return call("SupervisorResponse", {"decision": {"action": "ask_user", "question": "Continue the offline review?"}}, index)
            if count == 7:
                refs = [r["artifact_id"] for r in view["available_artifacts"] if r["kind"] == "delivery_manifest"]
                return call("SupervisorResponse", {"decision": {"action": "finish", "summary": "Delivered five items",
                    "goal_result": "succeeded", "evidence_artifact_ids": refs}}, index)
            req = stages[count].context.request if count < 6 else delivery
            decision = {"action": "delegate", "agent_id": req.agent_id, "operation": req.operation,
                "objective": req.objective, "scope": [s.model_dump() for s in req.scope],
                "input_artifact_ids": [r.artifact_id for r in req.inputs]}
            return call("SupervisorResponse", {"decision": translate(decision)}, index)
        req = view["request"]
        stage = next(s for s in stages if s.context.request.request_id == req["request_id"])
        position = positions.get(req["request_id"], 0)
        if position:
            previous = stage.trace[position - 1]
            result = next(m for m in reversed(messages) if m.type == "tool" and m.tool_call_id == previous["id"])
            assert result.status != "error", result.content
            learn(previous["result"], json.loads(result.content))
        if position == len(stage.trace):
            schema = "PlannerResponse" if req["agent_id"] == "task_planner" else "WorkResponse"
            return call(schema, {"outcome": "completed", "summary": "Operation complete",
                "artifact_ids": [r["artifact_id"] for r in view["submitted"]]}, index)
        step = stage.trace[position]
        positions[req["request_id"]] = position + 1
        response = call(step["name"], translate(step["args"]), index)
        response.tool_calls[0]["id"] = step["id"]
        return response

    import medbenchagent.runtime.native as native
    import medbenchagent.runtime.model_services as services
    import medbenchagent.agents.coding as coding
    original = coding.OpenHandsCodingAgent
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: "offline")
    monkeypatch.setattr(native, "chat_model", lambda *_: ScriptedChatModel(responder=respond))
    monkeypatch.setattr(coding, "OpenHandsCodingAgent", lambda settings, **kw: original(settings, **kw, runner=FakeRunner(bundle.produce)))
    directory = tmp_path / "native"
    options = create_run(directory, compilation_id="cap", goal="Deliver five items", source_manifest=bundle.source_file, sample_count=5)
    # Keep the scripted fixture focused on collaboration, not forced summarization.
    snapshots = [options.supervisor_settings, options.planner_settings, *options.professional_settings.values()]
    for snapshot in snapshots:
        section = next(v for k, v in snapshot.items() if k != "llm")
        section["context_chars"] = 2000000
        section["loop"].update(context_tokens=500000, summary_trigger_tokens=300000, summary_keep_tokens=10000)
    shutil.copytree(bundle.directory / "artifacts" / "cap-input", directory / "artifacts" / "cap-input")
    config = {"configurable": {"thread_id": "cap"}, "recursion_limit": 100}
    with assemble_compilation(options, directory) as (agents, context), SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(agents, checkpointer=saver)
        state = graph.invoke({"compilation_id": "cap", "goal": options.goal,
            "artifacts": [bundle.source_ref.model_dump(), bundle.target_ref.model_dump()]}, config, context=context)
        assert state["status"] == "waiting", state
        assert state["history"][1]["result"]["outcome"] == "completed"
        assert state["history"][1]["result"]["needs"]
        assert "messages" not in state
    with assemble_compilation(options, directory) as (agents, context), SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(agents, checkpointer=saver)
        state = graph.invoke(Command(resume="Continue"), config, context=context)
        assert state["goal_result"] == "succeeded", state
        output = next(ArtifactRef.model_validate(r) for r in state["artifacts"] if r["kind"] == "delivery_manifest")
        assert len(load_benchmark_package(resolve_artifact(directory, output).parent)["items"]) == 5
        assert len(state["history"]) == 7


def test_review_and_delivery_bind_exact_candidate_contents(tmp_path):
    bundle = build_capabilities(tmp_path)
    request = WorkRequest(
        compilation_id="cap",
        request_id="review",
        agent_id="review", operation="review_candidate", scope=[WorkScope(artifact_id=bundle.candidate_ref.artifact_id)],
        objective="Inspect candidate",
        inputs=list(bundle.known.values()),
    )
    review = ToolCalls(
        review_tools(),
        WorkContext(
            bundle.directory,
            request,
            load_work_agent_settings("review"),
            {key: freeze_composition(key).text for key in ("review_plan", "review_candidate")},
            "review",
        ),
        bundle.known,
    )
    audit = review.call("run_candidate_audit", candidate_id=bundle.candidate_ref.artifact_id)
    assert audit["passed"] and audit["checked_count"] == 5
    report = review.call(
        "finalize_review",
        target_id=bundle.candidate_ref.artifact_id,
        audit_id=audit["artifact_id"],
        findings=[],
        checked_task_ids=list(bundle.plan.tasks),
        checked_scenario_ids=list({item.scenario_id for item in bundle.candidate.items}),
        checked_item_ids=[item.item_id for item in bundle.candidate.items],
        unchecked_scope=[],
        approved=True,
    )
    request = WorkRequest(
        compilation_id="cap",
        request_id="deliver",
        agent_id="delivery", operation="publish", scope=[WorkScope(artifact_id=bundle.candidate_ref.artifact_id)],
        objective="Publish the reviewed candidate",
        inputs=list(bundle.known.values()),
    )
    result = delivery_spec().node(
        {"request": request.model_dump()},
        SimpleNamespace(context=SimpleNamespace(directory=bundle.directory, goal_spec=GoalSpec())),
    )["result"]
    assert result["outcome"] == "completed"
    from medbenchagent.compilation.contracts import ArtifactRef

    package = resolve_artifact(
        bundle.directory, ArtifactRef.model_validate(result["artifacts"][0])
    ).parent
    loaded = load_benchmark_package(package)
    assert len(loaded["items"]) == 5
    from medbenchagent.generation.publication import export_benchmark
    exported = tmp_path / "exported"
    export_benchmark(package, exported)
    assert (exported / "review_report.json").read_bytes() == (package / "review_report.json").read_bytes()
    assert len(load_benchmark_package(exported)["items"]) == 5
    from pathlib import Path

    assert all(
        Path(entry["value"]["path"]).is_relative_to(package)
        for item in loaded["items"]
        for entry in item["inputs"]
    )
    bundle.generation.call(
        "quarantine_generated_items",
        batch_ids=[p.artifact_id for p in bundle.candidate.manifest.batches],
        item_ids=[bundle.candidate.items[0].item_id],
        reason="Needs semantic repair",
    )
    allocation = next(ref.artifact_id for ref in bundle.known.values() if ref.kind == "allocation")
    changed = bundle.generation.call(
        "finalize_generation",
        spec_ids=[p.artifact_id for p in bundle.candidate.manifest.specs],
        batch_ids=[p.artifact_id for p in bundle.candidate.manifest.batches],
        allocation_id=allocation,
    )["artifact_id"]
    stale_inputs = [
        ref
        for key, ref in bundle.known.items()
        if ref.kind != "generation_candidate" or key == changed
    ]
    request = request.model_copy(update={"request_id": "reject-stale", "inputs": stale_inputs})
    rejected = delivery_spec().node(
        {"request": request.model_dump()},
        SimpleNamespace(context=SimpleNamespace(directory=bundle.directory, goal_spec=GoalSpec())),
    )["result"]
    assert rejected["outcome"] == "failed" and not rejected["artifacts"]
    assert report["approved"]

    # A semantic rejection routes only the Leaf back for repair. Prepared data is reused.
    from dataclasses import replace
    from medbenchagent.agents.semantic_planner import planner_tools, PlannerContextMiddleware
    from medbenchagent.agents.generation import generation_tools
    from medbenchagent.generation.contracts import ReviewFinding
    plan_ref = next(r for r in bundle.generation.context.request.inputs if r.kind == "task_plan")
    review.call("finalize_review", target_id=bundle.candidate_ref.artifact_id,
        audit_id=audit["artifact_id"], findings=[ReviewFinding(object_id="closed_vqa", category="template",
            blocking=True, summary="Clarify the question", evidence_artifact_ids=[plan_ref.artifact_id],
            suggested_owner="task_planner")], checked_task_ids=list(bundle.plan.tasks),
        checked_scenario_ids=list({i.scenario_id for i in bundle.candidate.items}),
        checked_item_ids=[i.item_id for i in bundle.candidate.items], unchecked_scope=[], approved=False)
    original = bundle.stages[3]
    repair_request = original.context.request.model_copy(update={"request_id": "repair-leaf",
        "scope": [WorkScope(artifact_id=plan_ref.artifact_id, kind="task", object_ids=["closed_vqa"])],
        "inputs": [r for r in bundle.known.values() if r.kind != "task_plan" or r == plan_ref]})
    planner = ToolCalls(planner_tools(), replace(original.context, request=repair_request), bundle.known,
        middleware=PlannerContextMiddleware())
    leaf = bundle.plan.tasks["closed_vqa"].model_copy(deep=True)
    leaf.revision += 1
    leaf.change_reason = "Address the review's wording issue"
    leaf.assembly.question_templates = ["Revised wording: {context}. {options}"]
    planner.call("save_semantic_task", task=leaf)
    published = planner.call("publish_task_plan")
    revised = next(bundle.known[r["artifact_id"]] for r in published["artifacts"] if r["kind"] == "task_plan")
    assert all(resolve_artifact(bundle.directory, ref).is_file() for ref in bundle.task_refs.values())
    inputs = [r for r in bundle.known.values() if r.kind != "task_plan" or r == revised]
    generation = ToolCalls(generation_tools(), replace(bundle.generation.context,
        request=bundle.generation.context.request.model_copy(update={"request_id": "regenerate", "inputs": inputs,
            "scope": [WorkScope(artifact_id=revised.artifact_id)]})), bundle.known)
    specs = [generation.call("prepare_task_execution", task_plan_id=revised.artifact_id, task_id=task.task_id,
        task_data_id=task.task_data.artifact_id, sample_count=1, seed=7)["artifact_id"] for task in bundle.plan.tasks.values()]
    allocation = generation.call("allocate_generation_budget", spec_ids=specs, requested_count=5)["artifact_id"]
    batches = [generation.call("generate_task_batch", spec_id=spec, allocation_id=allocation, offset=0, limit=1)["artifact_id"] for spec in specs]
    repaired = generation.call("finalize_generation", spec_ids=specs, batch_ids=batches, allocation_id=allocation)["artifact_id"]
    repaired_ref = bundle.known[repaired]
    from medbenchagent.generation.execution import CandidateReader
    repaired_items = CandidateReader(resolve_artifact(bundle.directory, repaired_ref)).items
    review = ToolCalls(review_tools(), replace(review.context, request=review.context.request.model_copy(update={
        "request_id": "rereview", "inputs": list(bundle.known.values()),
        "scope": [WorkScope(artifact_id=repaired)]})), bundle.known)
    fresh_audit = review.call("run_candidate_audit", candidate_id=repaired)
    assert fresh_audit["passed"]
    fresh_review = review.call("finalize_review", target_id=repaired, audit_id=fresh_audit["artifact_id"], findings=[],
        checked_task_ids=list(bundle.plan.tasks), checked_scenario_ids=list({i.scenario_id for i in repaired_items}),
        checked_item_ids=[i.item_id for i in repaired_items], unchecked_scope=[], approved=True)
    selected = [r for r in bundle.known.values() if r.kind not in {"generation_candidate", "audit_report", "review_report"}
                or r.artifact_id in {repaired, fresh_audit["artifact_id"], fresh_review["artifact_id"]}]
    final = request.model_copy(update={"request_id": "repaired-delivery", "inputs": selected,
        "scope": [WorkScope(artifact_id=repaired)]})
    result = delivery_spec().node({"request": final.model_dump()}, SimpleNamespace(context=SimpleNamespace(
        directory=bundle.directory, goal_spec=GoalSpec())))["result"]
    assert result["outcome"] == "completed"
    assert len(load_benchmark_package(resolve_artifact(bundle.directory, ArtifactRef.model_validate(result["artifacts"][0])).parent)["items"]) == 5
