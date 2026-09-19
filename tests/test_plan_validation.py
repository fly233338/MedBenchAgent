"""Persisted tree shape and readiness errors use paths that preserve opaque IDs."""

import copy

import pytest

from medbenchagent.planning.tree import validate_draft, validate_plan
from test_generation import _task


@pytest.fixture
def plan():
    leaf = _task("letter", ["image.annotation.status"], sample_count=2)
    leaf.pop("question_templates")
    leaf["eligible_count"] = leaf.pop("sample_count")
    return {
        "benchmark_info": {"sample_count": 2, "language": "en"},
        "dataset": {"dataset_id": "test"},
        "concepts": {
            "concept": {
                "description": "Status",
                "category": "test",
                "origin": "core",
                "status": "complete",
                "reason": "Direct annotation",
            }
        },
        "tasks": {"concept.child.leaf": leaf},
        "proposals": {},
        "revision": 3,
        "discovery_complete": True,
        "review": {"revision": 3, "issues": []},
        "reworks_used": 1,
        "correction_open": False,
        "retired_ids": [],
        "coverage": {"used_fields": ["image.annotation.status"], "unused_fields": []},

    }


def test_dotted_identity_is_not_a_third_level(plan):
    assert validate_draft(plan) == validate_plan(plan) == []


@pytest.mark.parametrize(
    "path,value",
    [
        (["tasks", "concept.child.leaf", "children"], ["nested"]),
        (["tasks", "concept.child.leaf", "question_templates"], ["Old template"]),
        (["concepts", "concept", "protocol"], {}),
        (["unknown_field"], True),
        (["tasks", "concept.child.leaf", "concept_id"], "concept.child.leaf"),
        (["tasks", "concept.child.leaf", "category"], "other"),
        (["tasks", "concept.child.leaf", "protocol", "metric"], "unknown"),
    ],
)
def test_invalid_structure_reports_exact_field_path(plan, path, value):
    target = plan
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    issues = validate_plan(plan)
    assert issues
    # Protocol's own cross-field validator reports the Protocol object.
    assert any(issue["path"] == path or issue["path"] == path[:-1] for issue in issues)
    assert all(isinstance(issue["message"], str) for issue in issues)


def test_identity_collision_and_retirement(plan):
    plan["concepts"]["concept.child.leaf"] = copy.deepcopy(plan["concepts"]["concept"])
    assert any("disjoint" in issue["message"] for issue in validate_draft(plan))
    del plan["concepts"]["concept.child.leaf"]
    plan["retired_ids"] = ["concept.child.leaf"]
    assert any("Retired" in issue["message"] for issue in validate_draft(plan))


@pytest.mark.parametrize(
    "value", [None, [], {}, {"concepts": {}, "tasks": {}}, {"review": {"revision": 1}}]
)
def test_malformed_plan_returns_issues_without_exceptions(value):
    issues = validate_plan(value)
    assert issues
    assert all(set(issue) == {"path", "message"} for issue in issues)


def test_incomplete_draft_is_structurally_valid_but_not_ready(plan):
    plan["concepts"]["concept"]["status"] = "pending"
    plan["discovery_complete"] = False
    plan["review"] = {}
    plan["tasks"]["concept.child.leaf"].pop("eligible_count")
    assert not validate_draft(plan)
    paths = [issue["path"] for issue in validate_plan(plan)]
    assert ["concepts", "concept", "status"] in paths
    assert ["review"] in paths and ["tasks", "concept.child.leaf", "eligible_count"] in paths


def test_incomplete_review_and_allocation_are_reported(plan):
    plan["review"].pop("issues")
    assert any(issue["path"] == ["review"] for issue in validate_plan(plan))
    plan["review"]["issues"] = []
    plan["tasks"]["concept.child.leaf"]["eligible_count"] = "two"
    assert any(issue["path"] == ["tasks", "concept.child.leaf", "eligible_count"] for issue in validate_plan(plan))
