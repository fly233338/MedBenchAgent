"""Real multi-dataset construction with deterministic model doubles and checkpoint recovery."""

import json
from dataclasses import replace
from pathlib import Path
import pytest
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from medbenchagent import settings
from medbenchagent.workflow.graph import compile_benchmark_graph
from medbenchagent.agents.finalization import restore_delivery
from test_benchmark_info import grounded
from test_generation import _task, _grounding
from test_dataset_cards import prepared_dataset


@pytest.mark.parametrize("dataset_count,ask_global,revise,template_repair", [(1, False, False, False), (2, False, False, False), (2, True, False, False), (2, False, True, False), (1, False, False, True)])
def test_complete_build(tmp_path, monkeypatch, harness_model, dataset_count, ask_global, revise, template_repair):
    template = tmp_path / "core.json"
    template.write_text(json.dumps({"concepts": {"status": {"origin": "core", "category": "clinical", "description": "Status", "require": "Explicit annotation"}}}), encoding="utf-8")
    configured = settings.get_settings()
    configured = replace(configured, paths=replace(configured.paths, blueprint_template=template,
                         dataset_catalog=tmp_path / "catalog.json", materials_root=tmp_path / "materials",
                         runs_root=tmp_path / "runs", prepared_datasets_root=tmp_path / "prepared"))
    monkeypatch.setattr(settings, "_SETTINGS", configured)
    identifiers = ["a", "b"][:dataset_count]
    for identifier in identifiers:
        prepared_dataset(tmp_path, identifier, monkeypatch)
    info = {"clinical_scope": "Breast disease", "diseases": [], "modalities": ["Mammography"], "language": "en",
            "sample_scope": [{"id": "modality", "kind": "modality", "description": "Only Mammography", "quote": "Mammography", "modalities": ["Mammography"]}],
            "evaluation_focus": "Classify status", "sample_count": dataset_count * 2 + 2}
    changed = {**info, "sample_count": 1, "language": "zh"}
    briefs = iter([info, changed])
    monkeypatch.setattr("medbenchagent.planning.benchmark_info.complete_json", lambda *a: grounded(next(briefs) if revise else info, "Mammography"))
    monkeypatch.setattr("medbenchagent.planning.dataset_selection.complete_json", lambda *a: {"dataset_ids": identifiers})
    cycle = [("specify", {}), ("select_datasets", {})] + [(tool, {"dataset_id": d}) for d in identifiers for tool in ("plan", "prepare_templates")]
    prefix = [("specify", {}), ("select_datasets", {}), ("plan", {"dataset_id": "a"}), ("ask_user", {"question": "Confirm goal"})] if ask_global else cycle + [("ask_user", {"question": "Change target?"})] if revise else []
    if ask_global:
        cycle = cycle[2:]
    if template_repair:
        cycle += [("plan", {"dataset_id": "a"}), ("prepare_templates", {"dataset_id": "a"})]
    supervisor = iter(prefix + cycle + [("allocate", {})] + [(tool, {"dataset_id": d}) for d in (identifiers[:1] if revise else identifiers) for tool in ("generate",)] + [("finalize", {}), ("finish", {})])
    finalization = iter([("inspect_dataset", {"dataset_id": d}) for d in (identifiers[:1] if revise else identifiers)] +
                        [("assemble_candidates", {}), ("balance_options", {}), ("validate_delivery", {}), ("publish_delivery", {}), ("finish", {})])
    leaf = _task("letter", ["image.annotation.status"])["protocol"]
    basic = [("save_leaf", {"leaf_id": "same.leaf", "concept_id": "status", "description": "Status", "protocol": leaf, "mapping_grounding": _grounding(leaf)}),
             ("complete_concept", {"concept_id": "status", "status": "complete", "reason": "Direct annotation"}),
             ("coverage_gap", {}), ("finish_discovery", {}), ("review_tree", {}), ("finish", {})]
    repaired = [("save_leaf", {**basic[0][1], "protocol": {**leaf, "question_intent": "Classify the explicitly annotated status"}})] + basic[1:]
    basic = [("resolve_sample_scope", {"resolutions": [{"requirement_id": "modality", "status": "implemented",
             "reason": "Exact modality", "modality_bindings": [{"modality": "Mammography", "values": ["Mammography"]}], "conditions": [{"requirement_id": "modality", "field": "image.modality",
             "operator": "in", "values": ["Mammography"]}]}]})] + basic
    planning_actions = iter(([("ask_user", {"question": "Confirm goal", "scope": "global"})] if ask_global else []) + basic * (dataset_count * (2 if revise else 1)) + (repaired if template_repair else []))
    allocated = [False]
    def model(instructions, prompt):
        if instructions.startswith("Finalize the complete"):
            name, arguments = next(finalization)
        elif instructions.startswith("Build a two-level"):
            assert json.loads(prompt)["state"]["delegation"]["objective"] == "Follow the requested benchmark scope"
            name, arguments = next(planning_actions)
        elif instructions.startswith("Distribute"):
            assert json.loads(prompt)["state"]["delegation"]["objective"] == "Follow the requested benchmark scope"
            if not allocated[0]:
                groups = json.loads(prompt)["state"]["groups"]
                name, arguments = "submit", {"concepts": {k: {"weight": 1.0, "reason": "Requested ability"} for k in groups}}
                allocated[0] = True
            else:
                name, arguments = "finish", {}
        else:
            pending = json.loads(prompt)["state"].get("global_issue")
            if pending and pending["status"] == "answered":
                return {"tool": "resolve_clarification", "arguments": {"issue_id": pending["issue_id"],
                        "kind": "requirement_change" if revise else "semantic"}, "reason": "Classify received answer"}
            name, arguments = next(supervisor)
            if name == "ask_user" and pending:
                arguments = {**arguments, "issue_id": pending["issue_id"]}
        if name in {"plan", "allocate"}:
            arguments = {**arguments, "delegation": {"revision_notes": "Diagnostic note"}}
        return {"tool": name, "arguments": arguments, "reason": "Required action"}
    harness_model(model)
    monkeypatch.setattr("medbenchagent.agents.task_planning.complete_json", lambda *a: {"issues": []})
    template_reviews = []
    def templates(system, prompt, **kwargs):
        language = json.loads(prompt)["specification"]["language"]
        assert language in ({"zh", "en"} if revise else {"en"})
        if system.startswith("Review"):
            template_reviews.append(1)
            return {"verdict": "revise_plan" if template_repair and len(template_reviews) == 1 else "approved", "reason": "Clarify task intent"}
        return {"templates": ["图像显示哪种状态？", "请选择图中的状态。", "判断图像对应的状态。"] if language == "zh" else ["What status is shown?", "Which status applies?", "Identify the status."]}
    monkeypatch.setattr("medbenchagent.generation.templates.complete_json", templates)
    graph = compile_benchmark_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "integration"}, "recursion_limit": 300}
    output = graph.invoke({"user_requirement": "Mammography: use datasets " + ", ".join(identifiers), "entry_intent": "build_benchmark"}, config)
    if ask_global or revise:
        assert output["__interrupt__"]
        output = graph.invoke(Command(resume="Use one Chinese question" if revise else "Confirmed goal"), config)
    assert output["status"] == "complete"
    assert "context" not in output and "result" not in output
    assert output["item_count"] == (1 if revise else 2 * dataset_count)
    assert output["total_shortfall"] == output["planning_shortfall"] == (0 if revise else 2)
    manifest = json.loads(Path(output["manifest_path"]).read_text(encoding="utf-8"))
    assert set(manifest["datasets"]) == set(identifiers)
    assert manifest["schema_version"] == "benchmark-manifest.v4"
    rows = [json.loads(line) for line in Path(output["audit_artifacts"]["audited"]).read_text(encoding="utf-8").splitlines()]
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["task_id"] == row["dataset"] + "::same.leaf" for row in rows)
    assert all((Path(output["delivery_path"]) / row["image_path"]).is_file() for row in rows)
    for path in output["dataset_task_plans"].values():
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "allocation" not in plan and "sample_count" not in plan["tasks"]["same.leaf"]
        assert plan["tasks"]["same.leaf"]["eligible_count"] == 2
        if template_repair:
            assert plan["reworks_used"] == 1
    before = Path(output["audit_artifacts"]["audited"]).read_bytes()
    private = graph.get_state(config).values["context"]
    if ask_global:
        assert private["global_issue"] is None
        child = json.loads(Path(private["child_states"]["a"]["planning"]).read_text(encoding="utf-8"))["context"]
        assert child["global_issue"] is None
        from medbenchagent.runtime.inspection import evidence
        assert [evidence(child, ref)["value"]["text"] for ref in child["decisions"].values()] == ["Confirmed goal"]
    restore_delivery(private)
    assert Path(output["audit_artifacts"]["audited"]).read_bytes() == before
    # A resumed run uses its saved Blueprint, not an edited external template.
    from medbenchagent.runtime.dependencies import verify_resume
    template.write_text('{"concepts":{}}', encoding="utf-8")
    verify_resume(private)
