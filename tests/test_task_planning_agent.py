"""Protocol facts, opaque identities and bounded incremental planning."""

import copy
import json
from pathlib import Path

import pytest

from medbenchagent.agents import task_planning as planning
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.inspection import evidence, evidence_page
from medbenchagent.planning.tree import validate_plan
from test_generation import _task, _record, _grounding


def context(tmp_path, monkeypatch):
    rows = [_record(0, {"status": "benign"}), _record(1, {"status": "malignant"})]
    annotation = tmp_path / "annotation.jsonl"
    annotation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    c = {
        "run_id": "test",
        "run_directory": str(tmp_path),
        "operation_directory": str(tmp_path / "op"),
        "annotation_path": str(annotation),
        "selected_dataset_id": "test", "dataset_scope": "test",
        "benchmark_info": {"sample_count": 4, "language": "en"},
        "construction_events_path": str(tmp_path / "events.jsonl"),
        "limits": {"max_reworks": 1},
    }
    for kind in ("Requirement", "DataConfig", "Dataset", "DatasetCard", "Materials", "PlanningConfig", "PlanningDelegation"):
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
    from medbenchagent.data.dataset import prepare_annotation_dataset
    _, c["dataset_access_manifest"] = prepare_annotation_dataset(annotation, tmp_path / "cache")
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
        mapping_grounding=_grounding(_task("letter", ["image.annotation.status"])["protocol"]),
        evidence=[],
    )


def test_scope_coverage_is_required_and_injected(tmp_path, monkeypatch):
    from medbenchagent.planning.scope import validate_scope_plan
    c = context(tmp_path, monkeypatch)
    c["sample_scope"] = [{"id": "population", "kind": "population", "description": "Only benign cases", "quote": "benign"}]
    with pytest.raises(ValueError, match="Resolve every"):
        save(c)
    condition = {"requirement_id": "population", "field": "image.annotation.status", "operator": "in", "values": ["benign"]}
    call(c, planning.resolve_sample_scope, resolutions=[{"requirement_id": "population", "status": "implemented",
         "reason": "Explicit status", "conditions": [condition]}])
    save(c)
    tree = planning._tree(c)
    assert tree["tasks"]["opaque.identity"]["eligible_count"] == 1
    validate_scope_plan(tree, c["sample_scope"])
    tree["tasks"]["opaque.identity"]["protocol"]["sample_conditions"] = []
    with pytest.raises(ValueError, match="differs"):
        validate_scope_plan(tree, c["sample_scope"])


def test_document_discovery_pages_existing_inventory(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    manifest = tmp_path / "source_manifests" / "source"
    manifest.mkdir(parents=True)
    source = tmp_path / "raw"
    source.mkdir()
    (source / "codes.md").write_text("0 means benign", encoding="utf-8")
    (manifest / "source.json").write_text(json.dumps({"root": str(source)}), encoding="utf-8")
    (manifest / "files.jsonl").write_text('\n'.join(json.dumps({"path": p}) for p in ["codes.md", "manual.pdf", "image.png"]), encoding="utf-8")
    from medbenchagent.data.documentation import freeze_document
    freeze_document(tmp_path, "source", source, "codes.md")
    (source / "second.md").write_text("Another source reference", encoding="utf-8")
    freeze_document(tmp_path, "source", source, "second.md")
    first = json.loads(planning.list_dataset_documents(c, limit=1)["summary"])
    assert first["total"] == 2 and first["next_offset"] == 1
    assert first["documents"][0]["source_id"] == "source"
    assert "0 means benign" in planning.read_dataset_document(c, path=first["documents"][0]["path"])["summary"]
    second = json.loads(planning.list_dataset_documents(c, offset=1)["summary"])
    assert second["documents"][0]["readable"] is True
    with pytest.raises(ValueError, match="not frozen"):
        planning.read_dataset_document(c, path="source/manual.pdf")


def test_mapping_requires_complete_and_dataset_specific_grounding(tmp_path, monkeypatch):
    from medbenchagent.planning.tree import validate_mapping_grounding
    c = context(tmp_path, monkeypatch)
    protocol = _task("letter", ["image.annotation.status"])["protocol"]
    args = {"leaf_id": "leaf", "concept_id": "core", "description": "Status", "protocol": protocol, "evidence": []}
    with pytest.raises(ValueError, match="cover every"):
        planning.save_leaf(c, **args)
    grounding = _grounding(protocol)
    assert planning.save_leaf(c, **args, mapping_grounding=grounding)["status"] == "success"
    bad = copy.deepcopy(grounding)
    bad[0]["field"] = "image.annotation.other"
    with pytest.raises(ValueError, match="answer field"):
        validate_mapping_grounding(protocol, bad, [])
    coded = copy.deepcopy(protocol)
    coded["answer_contract"]["value_mapping"][0]["raw_value"] = 0
    coded_grounding = _grounding(coded)
    with pytest.raises(ValueError, match="Direct mapping"):
        validate_mapping_grounding(coded, coded_grounding, [])
    coded_grounding[0].update(mode="interpreted", evidence_references=["codebook"])
    basis = [{"field": "image.annotation.status", "reference": "codebook", "source": "material", "dataset_specific": False}]
    with pytest.raises(ValueError, match="dataset-specific"):
        validate_mapping_grounding(coded, coded_grounding, basis)
    basis[0]["dataset_specific"] = True
    assert validate_mapping_grounding(coded, coded_grounding, basis)
    # A once-reviewed Plan cannot publish if mapping coverage disappears.
    ready(c, monkeypatch)
    tree = planning._tree(c)
    tree["tasks"]["opaque.identity"]["mapping_grounding"] = []
    assert any(issue["path"][-1] == "mapping_grounding" for issue in validate_plan(tree))


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


def test_plan_allows_final_capacity_shortfall_and_invalidates_on_revision(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    tree = planning._tree(c)
    assert tree["tasks"]["opaque.identity"]["eligible_count"] == 2
    assert "allocation" not in tree
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
    c.update(rag_capability={"available": True, "reasons": []}, selected_material_files=["book.pdf"])
    assert planning._tree(c)["tasks"] == {}
    assert "search_materials" in planning.available_tools(c)
    chunks = [{"pdf": "book.pdf", "page": 1, "chunk_id": "c1", "text": "Terminology"}]
    monkeypatch.setattr(
        planning,
        "_rag",
        lambda _: type("Rag", (), {"retrieve": lambda self, query, limit: chunks})(),
    )
    output = call(c, planning.search_materials, query="Explain concept", limit=1)
    returned = json.loads(output["summary"])
    assert returned[0]["chunk_id"] == "c1"
    assert evidence(c, returned[0]["reference"])["value"] == chunks[0]
    assert "retrieved_evidence" not in c and "retrieved_passages" not in c
    values = json.loads(planning.read_field_values(c, field="image.annotation.status")["summary"])
    assert {row["value"] for row in values["inspection"]["values"]} == {"benign", "malignant"}


def test_remote_rag_outage_propagates_for_checkpoint_resume(tmp_path, monkeypatch):
    from medbenchagent.rag.remote import RagUnavailable

    c = context(tmp_path, monkeypatch)

    def unavailable(_query, _limit):
        raise RagUnavailable("Reranker API request failed")

    monkeypatch.setattr(planning, "_rag", lambda _: type("Rag", (), {"retrieve":
                        lambda self, query, limit: unavailable(query, limit)})())
    with pytest.raises(RagUnavailable, match="Reranker API"):
        planning.search_materials(c, query="query", limit=1)


def test_optional_materials_initialize_without_rag_and_keep_existing_selection(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    c["artifact_versions"].pop("dataset/test/Materials")
    monkeypatch.setattr("medbenchagent.runtime.preflight.rag_capability", lambda _: {"available": False, "reasons": ["No models"]})
    initialized = planning.initialize_planning(c)
    require(initialized, "Materials")
    assert initialized["selected_material_files"] == []
    assert "save_leaf" in planning.available_tools(initialized)
    assert "search_materials" not in planning.available_tools(initialized)
    assert "list_materials" not in planning.available_tools(initialized)
    first = initialized["artifact_versions"]["dataset/test/Materials"]
    initialized["selected_material_files"] = ["existing.pdf"]
    planning.initialize_planning(initialized)
    assert initialized["selected_material_files"] == ["existing.pdf"]
    assert initialized["artifact_versions"]["dataset/test/Materials"] == first


def test_available_rag_requires_material_selection(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    c["artifact_versions"].pop("dataset/test/Materials")
    monkeypatch.setattr("medbenchagent.runtime.preflight.rag_capability", lambda _: {"available": True, "reasons": []})
    planning.initialize_planning(c)
    assert "dataset/test/Materials" not in c["artifact_versions"]
    assert "select_materials" in planning.available_tools(c)
    assert "save_leaf" not in planning.available_tools(c)


def test_search_default_passes_configured_limit(tmp_path, monkeypatch):
    from medbenchagent.runtime.tool_contracts import SearchArguments
    c = context(tmp_path, monkeypatch)
    limits = []
    monkeypatch.setattr(planning, "_rag", lambda _: type("Rag", (), {"retrieve": lambda self, query, limit: limits.append(limit) or []})())
    planning.search_materials(c, **SearchArguments(query="Medical evidence").model_dump())
    planning.search_materials(c, query="Medical evidence", limit=2)
    assert limits == [None, 2]  # None lets LocalPdfRag use settings.rag.result_limit.


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
        inspection_references=[json.loads(call(c, planning.read_field_values, field="image.annotation.status")["summary"])["inspection_reference"]],
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
    with pytest.raises(ValueError, match="retrieved|Unknown evidence"):
        planning.save_leaf(
            c,
            leaf_id="leaf",
            concept_id="core",
            description="Intent",
            protocol=_task("letter", ["image.annotation.status"])["protocol"],
            evidence=[{"reference": "fake", "pdf": "fake", "page": 1, "chunk_id": "fake"}],
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
    tree["tasks"]["opaque.identity"]["eligible_count"] = -1
    assert any(issue["path"] == ["tasks", "opaque.identity", "eligible_count"] for issue in validate_plan(tree))


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
    output = call(c, planning.finish)
    plan = json.loads(Path(output["updates"]["working_blueprint_path"]).read_text(encoding="utf-8"))
    assert plan["benchmark_info"] == c["benchmark_info"]
    assert plan["dataset"]["dataset_id"] == "new-dataset"
    assert plan["tasks"]["opaque.identity"]["eligible_count"] == 2
    assert "allocation" not in plan
    assert plan["reworks_used"] == 1


def test_readiness_gate_never_rescans_annotations(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    monkeypatch.setattr(
        planning, "iter_records", lambda *a: pytest.fail("Readiness must not scan data")
    )
    assert "finish" in planning.available_tools(c)
    assert not validate_plan(planning._tree(c))


def test_unsupported_dataset_publishes_empty_plan(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    call(c, planning.read_field_values, field="image.annotation.status")
    call(c, planning.complete_concept, concept_id="core", status="unsupported", reason="No relevant annotation",
         inspection_references=[r["reference"] for r in evidence_page(c)["entries"]])
    call(c, planning.coverage_gap)
    call(c, planning.finish_discovery)
    monkeypatch.setattr(planning, "complete_json", lambda *a: {"issues": []})
    call(c, planning.review_tree)
    assert call(c, planning.finish)["status"] == "complete"
    assert planning._tree(c)["tasks"] == {}


def test_unsupported_rejects_missing_forged_stale_and_foreign_inspections(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    arguments = dict(concept_id="core", status="unsupported", reason="No suitable labels")
    for references in ([], ["invented"]):
        with pytest.raises(ValueError, match="[Ii]nspection|[Ee]vidence"):
            planning.complete_concept(c, **arguments, inspection_references=references)
    call(c, planning.read_field_values, field="image.annotation.status")
    references = [r["reference"] for r in evidence_page(c)["entries"]]
    foreign = copy.deepcopy(c)
    foreign["selected_dataset_id"] = "other"
    foreign["dataset_scope"] = "other"
    with pytest.raises(ValueError, match="another dataset"):
        planning.complete_concept(foreign, **arguments, inspection_references=references)
    publish(c, "Dataset", {"revision": 2}, ())
    with pytest.raises(ValueError, match="stale"):
        planning.complete_concept(c, **arguments, inspection_references=references)


def test_review_receives_mapping_and_verified_semantic_basis(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    protocol = _task("letter", ["image.annotation.status"])["protocol"]
    from medbenchagent.runtime.clarifications import new_issue, record_answer, resolve_semantic_answer
    issue = new_issue("TaskPlanningAgent", c, "labels", "What do the status labels mean?")
    issue = resolve_semantic_answer(c, record_answer(c, issue, "benign means Benign; malignant means Malignant"))
    c["issues"] = {issue["issue_id"]: issue}
    reference = issue["answer_reference"]
    basis = [{"field": "image.annotation.status", "source": "user", "reference": reference,
              "quote": "benign means Benign; malignant means Malignant", "interpretation": "Preserve annotated pathology status"}]
    call(c, planning.save_leaf, leaf_id="leaf", concept_id="core", description="Status",
         protocol=protocol, evidence=[], semantic_basis=basis, mapping_grounding=_grounding(protocol))
    complete(c)
    call(c, planning.coverage_gap)
    call(c, planning.finish_discovery)
    captured = {}

    def review(system, prompt, *contracts):
        captured.update(json.loads(prompt))
        assert "Opaque codes" in system
        return {"issues": []}

    monkeypatch.setattr(planning, "complete_json", review)
    call(c, planning.review_tree)
    leaf = captured["leaves"]["leaf"]
    assert leaf["protocol"]["answer_contract"]["value_mapping"] == protocol["answer_contract"]["value_mapping"]
    assert leaf["semantic_basis"] == [{**b, "dataset_specific": False} for b in basis]
    assert leaf["eligible_count"] == 2
    assert leaf["field_values"]["image.annotation.status"]["complete"]
    basis[0]["quote"] = "fabricated source text"
    with pytest.raises(ValueError, match="exact quotation"):
        planning.save_leaf(c, leaf_id="leaf", concept_id="core", description="Status",
                           protocol=protocol, evidence=[], semantic_basis=basis, mapping_grounding=_grounding(protocol))


def test_zero_capacity_and_changed_delegation_block_publication(tmp_path, monkeypatch):
    c = context(tmp_path, monkeypatch)
    ready(c, monkeypatch)
    call(c, planning.finish)
    publish(c, "PlanningDelegation", {"objective": "Revise priorities"}, ())
    with pytest.raises(ValueError, match="stale"):
        require(c, "Plan")
    assert planning._tree(c)["review"] == {}
    monkeypatch.setattr(planning, "validate_dataset_protocol", lambda *args: 0)
    with pytest.raises(ValueError, match="positive eligible_count"):
        save(c)
