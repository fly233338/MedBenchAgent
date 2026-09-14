import pytest
from medbenchagent.planning.sample_counts import allocate_leaf_counts


def test_clipping_redistributes_without_final_shortfall():
    result = allocate_leaf_counts({"a": 9, "b": 1}, {"a": 1, "b": 20}, 10)
    assert result["counts"] == {"a": 1, "b": 9}
    assert result["planning_shortfall"] == 0


def test_capacity_exhaustion_reports_only_final_total_shortfall():
    result = allocate_leaf_counts({"a": 9, "b": 1}, {"a": 1, "b": 2}, 10)
    assert result["counts"] == {"a": 1, "b": 2}
    assert result["allocated_count"] == 3
    assert result["planning_shortfall"] == 7


def test_stable_ties_and_zero_weight():
    assert allocate_leaf_counts({"b": 1, "c": 0, "a": 1}, {"a": 5, "b": 5, "c": 5}, 3)["counts"] == {"a": 2, "b": 1, "c": 0}


@pytest.mark.parametrize("weights", [{"a": float("nan")}, {"a": -1}, {"a": 0}, {"a": True}])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        allocate_leaf_counts(weights, {"a": 2}, 3)
