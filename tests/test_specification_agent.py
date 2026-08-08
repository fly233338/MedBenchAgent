import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import medbenchagent.agents.specification as specification_agent
from medbenchagent.agents import (
    compile_specification_agent,
)
from medbenchagent.construction import construction_report
from medbenchagent.run_artifacts import initialize_run


def test_specification_agent_preserves_interrupt_payload_and_resume(
    tmp_path: Path, monkeypatch
) -> None:
    run = initialize_run(tmp_path / "runs")
    benchmark_info = {"disease": "breast cancer", "sample_count": 1}
    monkeypatch.setattr(
        specification_agent,
        "extract_benchmark_info",
        lambda *_args: benchmark_info,
    )
    monkeypatch.setattr(specification_agent, "missing_fields", lambda _info: [])
    monkeypatch.setattr(
        specification_agent,
        "_benchmark_review_action",
        lambda *_args: {"action": "confirm"},
    )
    graph = compile_specification_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "specification-resume"}}

    interrupted = graph.invoke(
        {
            "user_requirement": "Build a benchmark.",
            "run_id": run["run_id"],
            "construction_events_path": run["construction_events_path"],
            "working_blueprint_path": run["working_blueprint_path"],
        },
        config,
    )

    assert interrupted["__interrupt__"][0].value == {
        "kind": "benchmark_info_confirmation",
        "benchmark_info": benchmark_info,
        "question": "Confirm this BenchmarkInfo, or describe the correction.",
    }
    resumed = graph.invoke(Command(resume="confirm"), config)
    assert resumed == {
        "user_requirement": "Build a benchmark.",
        "benchmark_info": benchmark_info,
        "working_blueprint_path": run["working_blueprint_path"],
    }
    blueprint_path = Path(resumed["working_blueprint_path"])
    assert blueprint_path.is_file()
    assert (
        json.loads(blueprint_path.read_text(encoding="utf-8"))["benchmark_info"] == benchmark_info
    )
    human_events = [
        event
        for event in map(
            json.loads,
            Path(run["construction_events_path"]).read_text(encoding="utf-8").splitlines(),
        )
        if event["event_type"] == "human_interaction"
    ]
    assert [(event["action"], event["user_input"]) for event in human_events] == [
        ("submit", "Build a benchmark."),
        ("confirm", "confirm"),
    ]


def test_specification_agent_records_two_applied_revisions_before_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    run = initialize_run(tmp_path / "runs")
    infos = iter(
        [
            {"language": "en", "sample_count": 10},
            {"language": "zh", "sample_count": 10},
            {"language": "zh", "sample_count": 20},
        ]
    )
    actions = iter(
        [
            {"action": "revise", "feedback": "use Chinese"},
            {"action": "revise", "feedback": "make 20 questions"},
            {"action": "confirm"},
        ]
    )
    monkeypatch.setattr(specification_agent, "extract_benchmark_info", lambda *_args: next(infos))
    monkeypatch.setattr(specification_agent, "missing_fields", lambda _info: [])
    monkeypatch.setattr(
        specification_agent, "_benchmark_review_action", lambda *_args: next(actions)
    )
    graph = compile_specification_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "specification-revisions"}}
    graph.invoke(
        {
            "user_requirement": "Build a benchmark.",
            "run_id": run["run_id"],
            "construction_events_path": run["construction_events_path"],
            "working_blueprint_path": run["working_blueprint_path"],
        },
        config,
    )
    graph.invoke(Command(resume="把语言改成中文"), config)
    graph.invoke(Command(resume="题量改成 20"), config)
    graph.invoke(Command(resume="确认"), config)

    events = [
        json.loads(line)
        for line in Path(run["construction_events_path"]).read_text(encoding="utf-8").splitlines()
    ]
    review_events = [
        event
        for event in events
        if event["event_type"] == "human_interaction" and event.get("action") != "submit"
    ]
    applied = [event for event in events if event["event_type"] == "revision_applied"]
    assert [event["user_input"] for event in review_events] == [
        "把语言改成中文",
        "题量改成 20",
        "确认",
    ]
    assert [event["modified_object_ids"] for event in applied] == [
        ["language"],
        ["sample_count"],
    ]
    assert construction_report(run["construction_events_path"])["human_stages"][
        "benchmark_info"
    ] == {"confirmed": 1, "revision_rounds": 2, "modified_objects": 2}
