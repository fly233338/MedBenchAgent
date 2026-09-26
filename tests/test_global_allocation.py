import copy
import pytest
from medbenchagent.agents.allocation import allocate, allocation_groups


def plan(leaves, origin="core"):
    return {"concepts": {"ability": {"description": "Ability", "origin": origin}},
            "tasks": {k: {"concept_id": "ability", "eligible_count": n} for k, n in leaves.items()}}


def intent(plans):
    return {"concepts": {k: {"weight": 1.0, "reason": "Requested"} for k in allocation_groups(plans)}}


def test_dataset_budget_does_not_grow_with_leaf_count():
    plans = {"a": plan({"one": 100}), "b": plan({"one": 100, "two": 100, "three": 100})}
    result = allocate(plans, intent(plans), 12)
    assert result["counts"] == {"a": {"one": 6}, "b": {"one": 2, "two": 2, "three": 2}}


def test_capacity_redistribution_and_real_shortfall():
    plans = {"a": plan({"same": 1}), "b": plan({"same": 5}), "empty": {"concepts": {}, "tasks": {}}}
    result = allocate(plans, intent(plans), 9)
    assert result["allocated_count"] == 6 and result["planning_shortfall"] == 3
    assert result["counts"]["empty"] == {}


def test_extensions_are_not_merged_by_name():
    plans = {"a": plan({"same": 3}, "extension"), "b": plan({"same": 3}, "extension")}
    assert len(allocation_groups(plans)) == 2


def test_zero_weight_child_does_not_inflate_parent_capacity():
    plans = {"a": plan({"one": 10}), "b": plan({"one": 1})}
    proposal = intent(plans)
    key = next(iter(proposal["concepts"]))
    proposal["datasets"] = {key: {"a": {"weight": 0.0, "reason": "Excluded"}, "b": {"weight": 1.0, "reason": "Selected"}}}
    saved = copy.deepcopy(plans)
    result = allocate(plans, proposal, 5)
    assert result["counts"] == {"a": {"one": 0}, "b": {"one": 1}}
    assert result["planning_shortfall"] == 4 and saved == plans


def test_missing_weight_is_rejected():
    plans = {"a": plan({"one": 1})}
    with pytest.raises(ValueError, match="exactly"):
        allocate(plans, {"concepts": {}}, 1)


def allocation_context(tmp_path):
    import json
    from medbenchagent.runtime.dependencies import publish
    c = {"run_id": "allocation", "run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op"),
         "construction_events_path": str(tmp_path / "events"), "selected_dataset_ids": ["a", "b"],
         "benchmark_info": {"sample_count": 6}, "user_requirement": "Build", "dataset_task_plans": {}}
    publish(c, "Requirement", {"allocation_preferences": []}, ())
    for kind in ("DatasetSelection", "AllocationConfig", "AllocationDelegation"):
        publish(c, kind, {}, ())
    for dataset in ("a", "b"):
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps(plan({"one": 10})), encoding="utf-8")
        c["dataset_task_plans"][dataset] = str(path)
        publish(c, f"dataset/{dataset}/Plan", {"path": str(path)}, ())
        publish(c, f"dataset/{dataset}/Templates", {}, (f"dataset/{dataset}/Plan",))
    return c


def test_no_preferences_allocate_without_any_model_call(tmp_path, monkeypatch, harness_model):
    import json
    from pathlib import Path
    from medbenchagent.agents.allocation import compile_allocation_agent
    c = allocation_context(tmp_path)
    harness_model(lambda *a: pytest.fail("Default allocation must not call LLM"))
    output = compile_allocation_agent().invoke({"context": c})
    assert output["result"]["status"] == "complete"
    data = json.loads(Path(output["context"]["allocation_path"]).read_text(encoding="utf-8"))
    assert data["counts"] == {"a": {"one": 3}, "b": {"one": 3}}


def test_unequal_weights_require_current_requirement_preferences(tmp_path):
    from medbenchagent.agents.allocation import submit, _plans
    from medbenchagent.runtime.dependencies import publish
    c = allocation_context(tmp_path)
    proposal = intent(_plans(c))
    key = next(iter(proposal["concepts"]))
    proposal["datasets"] = {key: {"a": {"weight": 2.0, "reason": "Prefer A"}, "b": {"weight": 1.0, "reason": "Default"}}}
    with pytest.raises(ValueError, match="preference references"):
        submit(c, **proposal)
    publish(c, "Requirement", {"allocation_preferences": [{"id": "prefer-a", "description": "Prefer A", "quote": "Prefer A"}]}, ())
    proposal["datasets"][key]["a"]["preference_references"] = ["prefer-a"]
    assert submit(c, **proposal)["status"] == "success"
    publish(c, "Requirement", {"allocation_preferences": []}, ())
    with pytest.raises(ValueError, match="preference references"):
        submit(c, **proposal)
