"""Two offline capability checks; no autonomous workflow or live model validation."""

from types import SimpleNamespace
import pytest

from medbenchagent.agents.benchmark_review import review_tools
from medbenchagent.agents.delivery import delivery_spec
from medbenchagent.agents.handoff import WorkContext
from medbenchagent.compilation.artifacts import resolve_artifact
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.run import create_run
from medbenchagent.compilation.assembly import assemble_compilation
from medbenchagent.evaluation.loader import load_benchmark_package
from medbenchagent.prompts import load_prompt
from medbenchagent.settings import load_work_agent_settings
from capability_fixtures import ToolCalls, build_capabilities
from native_agent_fixtures import ScriptedChatModel


def test_profile_taskdata_leaf_generation_capabilities(tmp_path, monkeypatch):
    bundle = build_capabilities(tmp_path)
    assert bundle.profile.profile.samples == []
    assert len(bundle.task_refs) == 4 and len(bundle.candidate.items) == 5
    assert bundle.candidate.manifest.counts["shortfall"] == 0
    assert all(
        [entry.value["asset_id"] for entry in item.inputs] == ["left", "right"]
        for item in bundle.candidate.items
    )
    assert {item.output_spec for item in bundle.candidate.items} == {
        "label",
        "text",
        "bbox",
        "detections",
    }
    from medbenchagent.generation.backends import validate_leaf
    from medbenchagent.data.task_data import TaskDataReader

    leaf = bundle.plan.tasks["closed_vqa"]
    leaked = leaf.model_copy(
        update={"assembly": leaf.assembly.model_copy(update={"reference_field": "context"})}
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_leaf(
            leaked, TaskDataReader(resolve_artifact(bundle.directory, bundle.task_refs["vqa"]))
        )
    import medbenchagent.runtime.model_services as services
    import medbenchagent.runtime.native as native

    monkeypatch.setattr(services, "resolve_model_service", lambda *_args: "offline")
    from native_agent_fixtures import current_view, call

    def respond(messages, index):
        view = current_view(messages)
        if "request" in view:
            if messages[-1].type == "human":
                return call("list_generation_backends", {}, index)
            return call(
                "WorkResponse", {"outcome": "blocked", "summary": "Bind ready Leaf inputs"}, index
            )
        action = (
            {"action": "finish", "summary": "Capability inspected"}
            if view["history"]
            else {
                "action": "delegate",
                "agent_id": "generation",
                "objective": "Inspect supported backends",
            }
        )
        return call("SupervisorResponse", {"decision": action}, index)

    monkeypatch.setattr(native, "chat_model", lambda *_args: ScriptedChatModel(responder=respond))
    options = create_run(
        tmp_path / "assembly",
        compilation_id="assembled",
        goal="Expose capabilities",
        supervisor_mode="llm",
        planner_mode="llm",
        coding_mode="openhands",
        data_processing_mode="llm",
        generation_mode="llm",
        review_mode="llm",
        delivery_enabled=True,
        source_manifest=bundle.source_file,
        sample_count=5,
    )
    with assemble_compilation(options, tmp_path / "assembly") as (agents, context):
        assert set(agents) == {
            "data_processing",
            "task_planner",
            "generation",
            "review",
            "delivery",
        }
        assert context.decide.instructions == options.prompts["supervisor"].text
        from langgraph.checkpoint.sqlite import SqliteSaver
        from medbenchagent.compilation.graph import build_compilation_graph

        with SqliteSaver.from_conn_string(
            str(tmp_path / "assembly" / "checkpoints.sqlite")
        ) as saver:
            graph = build_compilation_graph(agents, checkpointer=saver)
            result = graph.invoke(
                {"compilation_id": "assembled", "goal": options.goal},
                {"configurable": {"thread_id": "assembled"}},
                context=context,
            )
            assert result["history"][0]["result"]["outcome"] == "blocked"
    assert set(options.prompts) == {
        "supervisor",
        "planner",
        "coding",
        "data_processing",
        "generation",
        "review",
    }


def test_review_and_delivery_bind_exact_candidate_contents(tmp_path):
    bundle = build_capabilities(tmp_path)
    request = WorkRequest(
        compilation_id="cap",
        request_id="review",
        agent_id="review",
        objective="Inspect candidate",
        inputs=list(bundle.known.values()),
    )
    review = ToolCalls(
        review_tools(),
        WorkContext(
            bundle.directory,
            request,
            load_work_agent_settings("review"),
            load_prompt("review.md"),
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
        agent_id="delivery",
        objective="Publish the reviewed candidate",
        inputs=list(bundle.known.values()),
    )
    result = delivery_spec().node(
        {"request": request.model_dump()},
        SimpleNamespace(context=SimpleNamespace(directory=bundle.directory)),
    )["result"]
    assert result["outcome"] == "completed"
    from medbenchagent.compilation.contracts import ArtifactRef

    package = resolve_artifact(
        bundle.directory, ArtifactRef.model_validate(result["artifacts"][0])
    ).parent
    loaded = load_benchmark_package(package)
    assert len(loaded["items"]) == 5
    from medbenchagent.generation.delivery import export_benchmark
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
        SimpleNamespace(context=SimpleNamespace(directory=bundle.directory)),
    )["result"]
    assert rejected["outcome"] == "failed" and not rejected["artifacts"]
    assert report["approved"]
