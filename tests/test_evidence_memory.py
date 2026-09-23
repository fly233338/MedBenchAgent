import json

import pytest

from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.inspection import inspected, inspection, evidence_page, read_evidence, save_evidence
from medbenchagent.runtime.contexts import save_context, load_context
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.clarifications import new_issue, record_answer, resolve_semantic_answer


def context(tmp_path, role="TaskPlanningAgent"):
    c = {"run_id": "memory", "run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op"),
         "construction_events_path": str(tmp_path / "events.jsonl"), "dataset_scope": "dataset", "agent_role": role}
    publish(c, "Dataset", {"revision": 1}, ())
    return c


def test_evidence_is_recoverable_without_payload_in_state_or_index(tmp_path):
    c = context(tmp_path)
    large = "medical facts " * 20000
    output = inspected(c, "read_dataset_document", {"path": "source/book.md"}, {"text": large}, ("Dataset",))
    c.update(output["updates"])
    first = json.loads(output["summary"].split(', "inspection":')[0] + "}")["inspection_reference"]
    for i in range(8):
        c.update(inspected(c, "read_fields", {"field": str(i)}, {"values": [i]}, ("Dataset",))["updates"])
    assert len(json.dumps(c)) < 3000
    assert len(json.dumps(evidence_page(c))) < 3000
    state = tmp_path / "private.json"
    save_context(state, c, "planning", "dataset")
    restored = load_context(state, "planning", "dataset")
    assert inspection(restored, first)["value"]["text"] == large
    page = json.loads(read_evidence(restored, reference=first, pointer=["text"])["summary"])
    assert len(page["text"]) == 4000 and page["next_offset"] == 4000
    with pytest.raises(ValueError, match="role/dataset"):
        load_context(state, "preparation", "dataset")
    publish(restored, "Dataset", {"revision": 2}, ())
    assert evidence_page(restored)["entries"] == []
    with pytest.raises(ValueError, match="stale"):
        inspection(restored, first)


def test_evidence_cannot_cross_roles_or_sources_and_pages_structures(tmp_path):
    c = context(tmp_path, "DataProcessingAgent")
    publish(c, "source/a/Source", {}, ())
    ref = save_evidence(c, "table", {}, {"rows": list(range(100))}, ("source/a/Source",), source_ids=["a"])
    page = json.loads(read_evidence(c, reference=ref, pointer=["rows"], offset=20, limit=3)["summary"])
    assert page["entries"] == [20, 21, 22] and page["next_offset"] == 23
    with pytest.raises(ValueError, match="another source"):
        inspection(c, ref, source_id="b")
    with pytest.raises(ValueError, match="another dataset or agent"):
        read_evidence({**c, "agent_role": "TaskPlanningAgent"}, reference=ref)


def test_successful_queries_do_not_consume_rework_budget(tmp_path, monkeypatch):
    c = context(tmp_path)
    c["limits"] = {"max_reworks": 0, "max_calls": 12}
    calls = []
    def query(_):
        calls.append(1)
        return result("success", "Read facts")
    actions = iter(["query"] * 7 + ["finish"])
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", lambda *args: {
        "tool": next(actions), "arguments": {}, "reason": "Recall facts"})
    graph = compile_agent("TaskPlanningAgent", "Read facts", {"query": query}, lambda _: {},
                          lambda _: result("complete", "Done"), read_tools=frozenset({"query"}))
    output = graph.invoke({"context": c}, {"recursion_limit": 30})
    assert output["result"]["status"] == "complete" and len(calls) == 7


def test_source_version_creates_new_issue_and_multisource_is_explicit(tmp_path):
    from medbenchagent.data.contracts import PreparationEvidence
    from medbenchagent.data.preparation_evidence import semantic_evidence
    c = context(tmp_path, "DataProcessingAgent")
    c.update(source_scope="b", sources=[{"id": "a"}, {"id": "b"}])
    for source in ("a", "b"):
        publish(c, f"source/{source}/Source", {"revision": 1}, ())
    issue = new_issue("DataProcessingAgent", c, "labels", "What do labels mean?")
    assert issue == new_issue("DataProcessingAgent", c, "labels", "What do labels mean?")
    resolved = resolve_semantic_answer(c, record_answer(c, issue, "A is benign"))
    c["issues"] = {resolved["issue_id"]: resolved}
    evidence = [PreparationEvidence(source="user", reference=resolved["answer_reference"], quote="A is benign")]
    semantic_evidence(c, evidence, "b")
    with pytest.raises(ValueError, match="covering this source"):
        semantic_evidence(c, evidence, "a")
    multi = new_issue("DataProcessingAgent", c, "labels", "Same labels in a and b?", subject_scope="multi_source", source_ids=["a", "b"])
    multi = resolve_semantic_answer(c, record_answer(c, multi, "A is benign in both sources"))
    c["issues"][multi["issue_id"]] = multi
    cited = [PreparationEvidence(source="user", reference=multi["answer_reference"], quote="A is benign in both sources")]
    semantic_evidence(c, cited, "a")
    semantic_evidence(c, cited, "b")
    publish(c, "source/b/Source", {"revision": 2}, ())
    assert new_issue("DataProcessingAgent", c, "labels", "What do labels mean?")["issue_id"] != issue["issue_id"]
    with pytest.raises(ValueError, match="stale"):
        semantic_evidence(c, cited, "b")


def test_display_clarification_interrupt_resume_and_parameter_binding(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from medbenchagent.agents.data_processing import save_image_plan
    c = context(tmp_path, "DataProcessingAgent")
    c.update(source_scope="a", sources=[{"id": "a"}])
    publish(c, "Source", {}, ())
    inventory = tmp_path / "images.json"
    inventory.write_text(json.dumps({"im": {"copy": False, "preserve_supported": False, "window": None, "invert": False}}), encoding="utf-8")
    c["image_inventory_path"] = str(inventory)
    publish(c, "ImageInventory", {"path": str(inventory)}, ("Source",))
    actions = iter(["ask_user", "finish"])
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", lambda *args: {
        "tool": (tool := next(actions)), "arguments": {"question": "Which window for im?", "image_ids": ["im"]} if tool == "ask_user" else {},
        "reason": "Missing display policy"})
    policy = {"encoding": "window", "low": 0.0, "high": 80.0, "invert": True}
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", lambda *args: {"policies": {"im": policy}})
    graph = compile_agent("DataProcessingAgent", "Prepare", {}, lambda _: {}, lambda _: result("complete", "Done"), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "display"}}
    initial = graph.invoke({"context": c}, config)
    assert initial["__interrupt__"]
    resumed = graph.invoke(Command(resume="Use 0 to 80 and invert"), config)["context"]
    issue = resumed["clarification"]
    overrides = {"im": {**policy, "decision_reference": issue["issue_id"]}}
    assert save_image_plan(resumed, overrides=overrides)["status"] == "success"
    overrides["im"]["high"] = 90.0
    with pytest.raises(ValueError, match="parameters differ"):
        save_image_plan(resumed, overrides=overrides)
    publish(resumed, "ImageInventory", {"revision": 2}, ("Source",))
    with pytest.raises(ValueError, match="stale"):
        save_image_plan(resumed, overrides=overrides)
