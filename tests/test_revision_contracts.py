import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from compilation_fixtures import compile_fixture
from test_compilation import cases, approve
from test_global_construction import _plan
from test_preparation_graph import Worker, state
from medbenchagent.agents import task_planning
from medbenchagent.data.sampling import representative_samples
from medbenchagent.generation.delivery import publish_benchmark
from medbenchagent.planning.context import bounded_evidence, EvidenceContextBudgetError
from medbenchagent.planning.bir import BIRStore
from medbenchagent.preparation.graph import compile_preparation_graph
from medbenchagent.preparation.contracts import Stage
from medbenchagent.runtime.openhands_worker import send_host_turn
from medbenchagent.workflow.graph import specify, available_stages


def test_required_queried_field_survives_large_catalog_and_omissions_are_visible():
    catalog = [{"name": f"field-{index}"} for index in range(80)]
    packet = bounded_evidence({"evidence_catalog": catalog, "required_evidence_names": ["field-79"]})
    assert packet["evidence_catalog"][0]["name"] == "field-79"
    assert len(packet["evidence_catalog"]) == 30
    assert len(packet["omissions"]["evidence_names"]) == 50
    with pytest.raises(EvidenceContextBudgetError, match="Required planning evidence"):
        bounded_evidence({"annotation_semantics": {"meaning": "x" * 65000}})


def test_sampling_covers_rare_label_missing_value_and_numeric_boundary():
    records = [{"case_id": str(i), "evidence": [
        {"name": "label", "type": "value", "value": "common", "ref": []},
        {"name": "size", "type": "value", "value": i, "ref": []}]} for i in range(50)]
    records.extend([
        {"case_id": "rare", "evidence": [{"name": "label", "type": "value", "value": "rare", "ref": ["image"]}]},
        {"case_id": "missing", "evidence": [{"name": "label", "type": "value", "value": None, "ref": []}]},
    ])
    selected = {row["case_id"] for row in representative_samples(records, 6)}
    assert {"rare", "missing", "0", "49"} <= selected


def test_preparation_repair_reuses_session_and_preserves_candidate_snapshot(tmp_path):
    worker = Worker(fail_image=True)
    outcome = compile_preparation_graph(worker=worker).invoke(state(tmp_path))
    assert outcome["status"] == "complete"
    images = [request for request in worker.requests if request.stage is Stage.PREPARE_IMAGE]
    assert len(images) == 2
    assert images[0].conversation_id == images[1].conversation_id
    assert images[0].workspace == images[1].workspace
    snapshots = Path(outcome["preparation_root"]) / "attempts/prepare_image"
    first = next((snapshots / "prepare_image-1/work/outputs/images").glob("*.png"))
    second = next((snapshots / "prepare_image-2/work/outputs/images").glob("*.png"))
    assert first.read_bytes() == b"bad"
    assert second.read_bytes() != b"bad"


def test_clarification_does_not_consume_preparation_failure_budget(tmp_path):
    worker = Worker(question=True)
    initial = state(tmp_path)
    initial["preparation_attempt_limits"] = {"understand": 1, "prepare_text": 1, "prepare_image": 1, "mapping_revision_cycles": 0}
    graph = compile_preparation_graph(worker=worker)
    waiting = graph.invoke(initial)
    waiting["user_decisions"] = [{"issue_id": "meaning", "answer": "finding"}]
    outcome = graph.invoke(waiting)
    assert outcome["status"] == "complete"
    assert worker.requests[0].conversation_id == worker.requests[1].conversation_id
    assert outcome.get("understand_failures", 0) == 0


def test_sdk_host_feedback_is_deduplicated_using_persisted_event_history():
    events, messages = [], []
    def send(text):
        messages.append(text)
        events.append(SimpleNamespace(model_dump=lambda **kwargs: {"message": text}))
    conversation = SimpleNamespace(state=SimpleNamespace(events=events), send_message=send)
    assert send_host_turn(conversation, "round-1", "Fix mapping")
    assert not send_host_turn(conversation, "round-1", "Fix mapping")
    assert send_host_turn(conversation, "round-2", "User clarification")
    assert len(messages) == 2


def test_optional_clinical_goal_and_language_defaults(tmp_path, monkeypatch):
    import medbenchagent.workflow.graph as workflow
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    monkeypatch.setattr(workflow, "preflight_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(workflow, "extract_benchmark_info", lambda *args, **kwargs: {
        "clinical_scope": None, "language": None, "sample_count": None,
        "evidence": [], "allocation_preferences": [], "sample_scope": [], "dataset_mentions": ["d"]})
    context = {"bir_path": str(store.root), "run_id": "test", "user_requirement": "Compile d",
               "operation_directory": str(tmp_path / "runtime")}
    updates = specify(context)["updates"]
    info = store.load("benchmark_info.json")
    assert not info["missing_fields"]
    assert info["clinical_scope"] is None
    assert info["language"] == "en"
    assert "global_issue" not in updates


@pytest.mark.parametrize("explicit,expected", [(None, 2), (1, 1)])
def test_intake_resolves_image_limit_from_frozen_settings_or_user(tmp_path, monkeypatch, explicit, expected):
    import medbenchagent.workflow.graph as workflow
    from medbenchagent.runtime.run_artifacts import initialize_run
    from medbenchagent.settings import run_settings
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    run = initialize_run(tmp_path / "runs", bir_path=store.root, graph_overrides={"max_total_tasks_per_image": 2})
    monkeypatch.setattr(workflow, "preflight_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(workflow, "extract_benchmark_info", lambda *args, **kwargs:
                        {"max_total_tasks_per_image": explicit, "sample_scope": [], "allocation_preferences": []})
    context = {**run, "user_requirement": "Compile d", "operation_directory": str(tmp_path / "operation")}
    with run_settings(run["effective_config_path"]):
        specify(context)
    assert store.load("benchmark_info.json")["max_total_tasks_per_image"] == expected
    assert "max_total_tasks_per_image" not in context and "configured_total_tasks_per_image" not in context


def test_supervisor_has_review_and_rework_choices_and_publication_respects_rejection(tmp_path, monkeypatch):
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    context.update(brief_ready=True, user_requirement="Compile d", brief_requirement="Compile d")
    tools = available_stages(context)
    assert {"review_plan", "revise_plan", "allocate", "audit", "inspect_compilation"} <= tools
    assert "publish" not in tools
    approve(context, monkeypatch)
    assert "publish" in available_stages(context)
    store = BIRStore(context["bir_path"])
    saved = store.load("review.json")
    saved.update(status="rejected", findings=[{"id": "issue", "blocking": True}])
    store.update("review.json", saved, run_id="test", actor="test")
    with pytest.raises(ValueError, match="blocking semantic review"):
        publish_benchmark(context)


def test_explicit_value_pagination_preserves_full_domain(tmp_path):
    plan = _plan("d", {"leaf": 61})
    vocabulary = [f"label-{index:02}" for index in range(61)]
    plan.leaf_tasks["leaf"] = plan.leaf_tasks["leaf"].model_copy(update={"options": vocabulary})
    records = []
    for index, label in enumerate(vocabulary):
        row = cases()[0]
        row["case_id"] = str(index)
        row["evidence"][1]["value"] = label
        records.append(row)
    context = compile_fixture(tmp_path, plan, records)
    first = task_planning.read_field_values(context, name="label", offset=0, limit=50)["data"]
    second = task_planning.read_field_values(context, name="label", offset=first["next_offset"], limit=50)["data"]
    assert [row["value"] for row in [*first["values"], *second["values"]]] == vocabulary
    assert second["next_offset"] is None
    assert json.loads(BIRStore(context["bir_path"])._path("task_plan.json").read_text())["revision"] == 0


def test_scoped_rework_carries_review_evidence_and_enforces_budget(tmp_path):
    from medbenchagent.workflow.supervision import revise_plan
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    context["limits"] = {"max_reworks": 1}
    store = BIRStore(context["bir_path"])
    finding = {"id": "wording", "scenario_id": "scenario-d", "owner": "planning", "blocking": True,
               "summary": "Question assumes a positive finding", "evidence_refs": ["evidence_catalog:label"]}
    store.update("review.json", {"status": "rejected", "findings": [finding]}, run_id="test", actor="test")
    revise_plan(context, scenario_ids=["scenario-d"], reason="Correct the wording", issue_ids=["wording"])
    request = store.load("task_plan.json")["review"]["revision_request"]
    assert request["findings"] == [finding]
    with pytest.raises(ValueError, match="rework budget"):
        revise_plan(context, scenario_ids=["scenario-d"], reason="Try again")
