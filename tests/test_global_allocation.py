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
