"""Protocol facts, opaque identities and bounded incremental planning."""

import copy
import json
from pathlib import Path

import pytest

from medbenchagent.agents import task_planning as planning
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.planning.tree import validate_plan
from test_generation import _task, _record


def context(tmp_path, monkeypatch):
    rows = [_record(0, {"status": "benign"}), _record(1, {"status": "malignant"})]
    annotation = tmp_path / "annotation.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    c = {
        "run_id": "test",
        "run_directory": str(tmp_path),
        "operation_directory": str(tmp_path / "op"),
        "annotation_path": str(annotation),
        "selected_dataset_id": "test",
        "benchmark_info": {"sample_count": 4, "language": "en"},
        "construction_events_path": str(tmp_path / "events.jsonl"),
        "limits": {"max_reworks": 1},
    }
    for kind in ("Requirement", "DataConfig", "Dataset", "Materials", "PlanningConfig"):
        publish(c, kind, {"value": 1}, ())
    monkeypatch.setattr(
        planning,
        "_base",
        lambda _: {
            "benchmark_info": c["benchmark_info"],
            "dataset": {"dataset_id": "test"},
            "concepts": {"core": {"description": "Core", "origin": "core", "category": "basic"}},
        },
    )
    monkeypatch.setattr(
        planning,
        "_profile",
        lambda _: {
            "total_images": 2,
            "structures": [],
            "fields": {"image.annotation.status": {"non_empty_images": 2}},
        },
    )
    return c


def call(c, function, **kwargs):
    index = len(list(Path(c["run_directory"]).glob("op-*")))
    c["operation_directory"] = str(Path(c["run_directory"]) / f"op-{index}")
    output = function(c, **kwargs)
    c.update(output.get("updates", {}))
    return output


def save(c, leaf_id="opaque.identity", concept_id="core"):
    return call(
        c,
        planning.save_leaf,
        leaf_id=leaf_id,
        concept_id=concept_id,
        description="Classify status",
        protocol=_task("letter", ["image.annotation.status"])["protocol"],
        evidence=[],
    )


def complete(c, concept_id="core"):
    return call(
        c,
        planning.complete_concept,
        concept_id=concept_id,
        status="complete",
        reason="Explicit annotations",
    )


def ready(c, monkeypatch):
    save(c)
    complete(c)
    call(c, planning.coverage_gap)
    call(c, planning.finish_discovery)
    monkeypatch.setattr(planning, "complete_json", lambda *a: {"issues": []})
    call(c, planning.review_tree)
    call(c, planning.allocate, weights={"opaque.identity": {"weight": 1.0, "reason": "Coverage"}})


def test_plan_allows_final_capacity_shortfall_and_invalidates_on_revision(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    tree = planning._tree(c)
    assert tree["allocation"]["planning_shortfall"] == 2
    assert not validate_plan(tree)
    assert call(c, planning.finish)["status"] == "complete"
    require(c, "Plan")
    save(c)
    with pytest.raises(ValueError, match="stale"):
        require(c, "Plan")
    assert "finish" not in planning.available_tools(c)


def test_review_budget_persists_after_redelegation_and_cannot_publish(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    save(c)  # first correction round
    complete(c)
    issues = [{"action": "revise", "node_ids": ["opaque.identity"], "reason": "Still overlaps"}]
    monkeypatch.setattr(planning, "complete_json", lambda *a: {"issues": issues})
    assert call(c, planning.review_tree)["status"] == "blocked"
    resumed = copy.deepcopy(c)
    assert planning._tree(resumed)["reworks_used"] == 1
    assert save(resumed)["status"] == "blocked"
    with pytest.raises(ValueError, match="review"):
        planning.finish(resumed)


def test_rag_available_before_any_leaf_and_exact_field_values(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    assert planning._tree(c)["tasks"] == {}
    assert "search_materials" in planning.available_tools(c)
    chunks = [{"pdf": "book.pdf", "page": 1, "chunk_id": "c1", "text": "Terminology"}]
    monkeypatch.setattr(
        planning,
        "_rag",
        lambda _: type("Rag", (), {"retrieve": lambda self, query, limit: chunks})(),
    )
    output = call(c, planning.search_materials, query="Explain concept", limit=1)
    assert json.loads(output["summary"]) == chunks
    assert c["retrieved_evidence"] == [{"pdf": "book.pdf", "page": 1, "chunk_id": "c1"}]
    values = json.loads(planning.read_field_values(c, field="image.annotation.status")["summary"])
    assert {row["value"] for row in values["values"]} == {"benign", "malignant"}


def test_extension_gate_acceptance_rejection_and_reparent_identity(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    proposal = dict(
        concept_id="extension",
        description="Different ability",
        category="basic",
        fields=["image.annotation.status"],
        clinical_rationale="Clinically relevant",
        requirement_relevance="Target ability",
        cannot_fit_existing="Different capability",
        proposed_leaf_types=["letter_classification"],
        evidence=[],
    )
    with pytest.raises(ValueError, match="coverage"):
        planning.propose_extension(c, **proposal)
    save(c)
    complete(c)
    call(c, planning.coverage_gap)
    monkeypatch.setattr(
        planning, "complete_json", lambda *a: {"accepted": False, "reason": "Can fit core"}
    )
    call(c, planning.propose_extension, **proposal)
    assert "extension" not in planning._tree(c)["concepts"]
    monkeypatch.setattr(
        planning, "complete_json", lambda *a: {"accepted": True, "reason": "Distinct capability"}
    )
    call(c, planning.propose_extension, **proposal)
    assert planning._tree(c)["concepts"]["extension"]["origin"] == "extension"
    accepted = json.loads(planning.read_proposal(c, node_id="extension")["summary"])
    assert accepted["cannot_fit_existing"] == proposal["cannot_fit_existing"]
    assert accepted["decision"]["accepted"] is True
    assert "read_proposal" in planning.available_tools(c)
    save(c, concept_id="extension")
    assert set(planning._tree(c)["tasks"]) == {"opaque.identity"}
    assert planning._tree(c)["tasks"]["opaque.identity"]["concept_id"] == "extension"
    call(
        c,
        planning.complete_concept,
        concept_id="core",
        status="unsupported",
        reason="Moved to extension",
    )
    call(c, planning.propose_extension, **{**proposal, "category": "new-category"})
    assert planning._tree(c)["tasks"]["opaque.identity"]["category"] == "new-category"
    assert (
        json.loads(planning.read_proposal(c, node_id="extension")["summary"])["category"]
        == "new-category"
    )
    with pytest.raises(ValueError, match="concept"):
        save(c, leaf_id="third-level", concept_id="opaque.identity")


def test_unknown_answer_values_and_fabricated_citations_rejected(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    protocol = _task("letter", ["image.annotation.status"])["protocol"]
    protocol["answer_contract"]["value_mapping"][0]["raw_value"] = "invented"
    with pytest.raises(ValueError, match="absent"):
        planning.save_leaf(
            c,
            leaf_id="leaf",
            concept_id="core",
            description="Intent",
            protocol=protocol,
            evidence=[],
        )
    with pytest.raises(ValueError, match="retrieved"):
        planning.save_leaf(
            c,
            leaf_id="leaf",
            concept_id="core",
            description="Intent",
            protocol=_task("letter", ["image.annotation.status"])["protocol"],
            evidence=[{"pdf": "fake", "page": 1, "chunk_id": "fake"}],
        )


def test_upstream_refresh_does_not_reset_spent_budget(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    save(c)
    publish(c, "Dataset", {"value": 2}, ())
    assert planning._tree(c)["reworks_used"] == 1
    assert set(planning._tree(c)["tasks"]) == {"opaque.identity"}
    assert planning._tree(c)["concepts"]["core"]["status"] == "pending"


def test_plan_validator_rejects_leaf_count_drift(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    tree = planning._tree(c)
    tree["tasks"]["opaque.identity"]["sample_count"] += 1
    assert any(issue["message"] == "Invalid leaf allocation" for issue in validate_plan(tree))


def test_budget_exhausted_review_is_not_retried_with_new_identity(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    save(c)
    complete(c)
    monkeypatch.setattr(
        planning,
        "complete_json",
        lambda *a: {
            "issues": [{"action": "revise", "node_ids": ["opaque.identity"], "reason": "Conflict"}]
        },
    )
    call(c, planning.review_tree)
    old = planning._tree(c)
    assert save(c, leaf_id="new.identity")["status"] == "blocked"
    assert planning._tree(c) == old


def test_cached_exhausted_review_preserves_blocked_result_and_reference(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    save(c)
    complete(c)
    issues = [{"action": "revise", "node_ids": ["opaque.identity"], "reason": "Conflict"}]
    calls = []

    def review(*args):
        calls.append(1)
        return {"issues": issues}

    monkeypatch.setattr(planning, "complete_json", review)
    first = call(c, planning.review_tree)
    again = planning.review_tree(copy.deepcopy(c))
    assert first["status"] == again["status"] == "blocked"
    assert first["unresolved_questions"] == again["unresolved_questions"] == issues
    assert first["artifacts"] == again["artifacts"] == {"draft": c["draft_path"]}
    assert calls == [1]
    assert "finish" not in planning.available_tools(c)


def test_cached_review_then_one_correction_round_and_successful_recheck(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    save(c)
    complete(c)
    call(c, planning.coverage_gap)
    call(c, planning.finish_discovery)
    responses = iter(
        [
            {
                "issues": [
                    {"action": "revise", "node_ids": ["opaque.identity"], "reason": "Clarify"}
                ]
            },
            {"issues": []},
        ]
    )
    calls = []

    def review(*args):
        calls.append(1)
        return next(responses)

    monkeypatch.setattr(planning, "complete_json", review)
    first = call(c, planning.review_tree)
    cached = planning.review_tree(c)
    assert first["status"] == cached["status"] == "success"
    assert cached["unresolved_questions"] == first["unresolved_questions"]
    assert calls == [1]
    save(c)
    complete(c)
    assert planning._tree(c)["reworks_used"] == 1
    assert call(c, planning.review_tree)["status"] == "success"
    assert calls == [1, 1]
    call(c, planning.allocate, weights={"opaque.identity": {"weight": 1.0, "reason": "Coverage"}})
    assert call(c, planning.finish)["status"] == "complete"


def test_invalid_persisted_draft_does_not_crash_tool_gate(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    tree = planning._tree(c)
    tree["tasks"]["opaque.identity"]["protocol"]["metric"] = "unknown"
    Path(c["draft_path"]).write_text(json.dumps(tree), encoding="utf-8")
    assert "finish" not in planning.available_tools(c)
    assert "opaque.identity" in planning.describe(c)["validation_error"]


def test_upstream_changes_refresh_metadata_preserve_identity_and_revalidate(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    save(c)
    old = planning._tree(c)
    c["benchmark_info"] = {"sample_count": 1, "language": "zh"}
    c["selected_dataset_id"] = "new-dataset"
    publish(c, "Requirement", {"brief": c["benchmark_info"]}, ())
    publish(c, "Dataset", {"value": 2}, ())
    refreshed = planning._tree(c)
    assert refreshed["benchmark_info"] == c["benchmark_info"]
    assert refreshed["dataset"] == {"dataset_id": "new-dataset"}
    assert refreshed["tasks"] == old["tasks"]
    assert refreshed["reworks_used"] == 1
    assert refreshed["review"] == {} and not refreshed["discovery_complete"]
    assert "allocation" not in refreshed and "coverage" not in refreshed
    assert "finish" not in planning.available_tools(c)
    complete(c)  # revalidates retained Protocol against current data
    call(c, planning.coverage_gap)
    call(c, planning.finish_discovery)
    call(c, planning.review_tree)
    call(
        c, planning.allocate, weights={"opaque.identity": {"weight": 1.0, "reason": "Updated goal"}}
    )
    output = call(c, planning.finish)
    plan = json.loads(Path(output["updates"]["working_blueprint_path"]).read_text(encoding="utf-8"))
    assert plan["benchmark_info"] == c["benchmark_info"]
    assert plan["dataset"]["dataset_id"] == "new-dataset"
    assert plan["allocation"]["requested_count"] == plan["allocation"]["allocated_count"] == 1
    assert plan["reworks_used"] == 1


def test_readiness_gate_never_rescans_annotations(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    monkeypatch.setattr(
        planning, "load_annotation_jsonl", lambda *a: pytest.fail("Readiness must not scan data")
    )
    assert "finish" in planning.available_tools(c)
    assert not validate_plan(planning._tree(c))
