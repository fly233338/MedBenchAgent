import json
from pathlib import Path
from types import SimpleNamespace

from medbenchagent.construction import (
    construction_report,
    event_context,
    record_event,
    record_human_interaction,
    record_llm_call,
    record_revision_applied,
)


def test_construction_events_and_report_v2_preserve_human_input(tmp_path: Path) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    record_event(
        events_path,
        "llm_call",
        run_id="run",
        dataset_id="vindr-mammo",
        task_id="view",
        agent="SpecificationAgent",
        operation="generation",
        status="success",
        latency_ms=12,
        input_tokens=0,
        output_tokens=0,
        token_usage_available=False,
        call_id="call-1",
    )
    record_event(events_path, "run_started", run_id="run")
    record_event(events_path, "generation_summary", task_counts={"view": 1}, shortfalls={"view": 0})
    record_event(events_path, "audit_summary", audited=1, rejected=0)
    record_human_interaction(
        events_path,
        run_id="run",
        dataset_id="vindr-mammo",
        agent="TaskPlanningAgent",
        stage="materials",
        user_input="选择两份指南",
        parse_status="success",
        action="confirm",
        selected_objects=2,
    )
    record_event(events_path, "run_completed", run_id="run")

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 6
    assert events[4]["user_input"] == "选择两份指南"
    report = construction_report(events_path)
    assert report["schema_version"] == "construction-report.v2"
    assert report["llm"]["calls"] == 1
    assert report["llm"]["per_audited_item"] == {"calls": 1, "tokens": 0}
    assert report["items"] == {"target": 1, "raw": 1, "audited": 1, "rejected": 0, "shortfall": 0}
    assert report["agents"]["SpecificationAgent"]["calls"] == 1
    assert report["human_stages"]["materials"] == {
        "confirmed": 1,
        "revision_rounds": 0,
        "modified_objects": 0,
        "selected_objects": 2,
    }
    assert "generation_review_override" not in report["human_stages"]
    assert "audit_override" not in report["human_stages"]


def test_three_human_turns_reduce_to_two_applied_revisions_and_one_confirmation(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    first = record_human_interaction(
        events_path,
        run_id="run",
        dataset_id=None,
        agent="SpecificationAgent",
        stage="benchmark_info",
        user_input="把语言改成中文",
        parse_status="success",
        action="revise",
    )
    record_revision_applied(
        events_path,
        run_id="run",
        dataset_id=None,
        agent="SpecificationAgent",
        stage="benchmark_info",
        interaction_id=first,
        modified_object_ids=["language"],
    )
    second = record_human_interaction(
        events_path,
        run_id="run",
        dataset_id=None,
        agent="SpecificationAgent",
        stage="benchmark_info",
        user_input="题量改成 20",
        parse_status="success",
        action="revise",
    )
    record_revision_applied(
        events_path,
        run_id="run",
        dataset_id=None,
        agent="SpecificationAgent",
        stage="benchmark_info",
        interaction_id=second,
        modified_object_ids=["sample_count"],
    )
    record_human_interaction(
        events_path,
        run_id="run",
        dataset_id=None,
        agent="SpecificationAgent",
        stage="benchmark_info",
        user_input="确认",
        parse_status="success",
        action="confirm",
    )

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert sum(event["event_type"] == "human_interaction" for event in events) == 3
    assert sum(event["event_type"] == "revision_applied" for event in events) == 2
    stage = construction_report(events_path)["human_stages"]["benchmark_info"]
    assert stage == {"confirmed": 1, "revision_rounds": 2, "modified_objects": 2}


def test_failed_human_input_is_retained_without_affecting_report(tmp_path: Path) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    record_human_interaction(
        events_path,
        run_id="run",
        dataset_id=None,
        agent="TaskPlanningAgent",
        stage="task_plan",
        user_input="无法解析的内容",
        parse_status="failed",
        error_type="ValueError",
    )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["user_input"] == "无法解析的内容"
    assert event["parse_status"] == "failed"
    assert construction_report(events_path)["human_stages"]["task_plan"] == {
        "confirmed": 0,
        "revision_rounds": 0,
        "modified_objects": 0,
    }


def test_llm_events_include_contextual_model_and_component(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    with event_context(
        construction_events_path=events_path,
        run_id="run",
        agent="GenerationAgent",
        operation="generate",
        component="letter",
    ):
        record_llm_call(
            status="success",
            started=0,
            response=SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3), id="response-1"
            ),
        )

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["agent"] == "GenerationAgent"
    assert event["operation"] == "generate"
    assert event["component"] == "letter"
    assert event["model"] == "deepseek-v4-flash"
