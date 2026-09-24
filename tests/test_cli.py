import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from medbenchagent import cli


def _settings(tmp_path: Path) -> SimpleNamespace:
    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text('{"concepts":{"core":{}}}\n', encoding="utf-8")
    materials = tmp_path / "materials"
    materials.mkdir(exist_ok=True)
    catalog = tmp_path / "dataset_catalog.json"
    catalog.write_text('{"datasets":[]}\n', encoding="utf-8")
    return SimpleNamespace(
        llm=SimpleNamespace(model="deepseek-v4-flash", base_url="http://127.0.0.1:8005/v1"),
        generation=SimpleNamespace(
            seed=17,
            max_tasks_per_image=2,
            max_total_tasks_per_image=None,
            question_temperature=0.4,
        ),
        rag=SimpleNamespace(embedding_base_url="http://127.0.0.1:8001/v1",
                            embedding_model="Qwen3-Embedding",
                            reranker_base_url="http://127.0.0.1:8002/v1",
                            reranker_model="Qwen3-Reranker"),
        default_config_path=tmp_path / "default.toml",
        local_config_path=None,
        environment_overrides=(),
        paths=SimpleNamespace(
            blueprint_template=blueprint,
            materials_root=materials,
            dataset_catalog=catalog,
            runs_root=tmp_path / "runs",
        ),
    )


def test_preflight_aggregates_missing_local_resources(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.paths.blueprint_template.unlink()
    settings.rag.embedding_model = ""
    settings.rag.reranker_base_url = "http://missing-host/v1"
    annotation = tmp_path / "dataset" / "annotation.jsonl"
    settings.paths.dataset_catalog.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": "demo",
                        "name": "Demo",
                        "description": "Demo dataset",
                        "annotation_path": str(annotation),
                        "available": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    errors = "\n".join(cli._preflight_errors(settings))

    assert "Blueprint template does not exist" in errors
    assert "Catalog entry must contain exactly" in errors
    assert "Embedding model" not in errors
    assert "Reranker model" not in errors
    assert "PDF" not in errors
    assert "No dataset is ready for a complete run" not in errors
    assert "annotation.jsonl" not in errors
    assert "image_manifest.jsonl" not in errors


def test_preflight_accepts_complete_local_resources(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    category = settings.paths.materials_root / "Guideline"
    category.mkdir()
    (category / "guide.pdf").write_bytes(b"%PDF-local")
    assert cli._preflight_errors(settings) == []


def test_preflight_treats_rag_as_optional_and_blueprint_as_build_only(tmp_path):
    settings = _settings(tmp_path)
    settings.rag.embedding_model = ""
    settings.rag.reranker_model = ""
    assert cli._preflight_errors(settings) == []
    settings.paths.blueprint_template.unlink()
    assert cli._preflight_errors(settings, build=False) == []
    assert cli._preflight_errors(settings, build=True)


def test_build_reports_preflight_before_graph_creation(tmp_path: Path) -> None:
    output: list[str] = []

    settings = _settings(tmp_path)
    settings.paths.blueprint_template.unlink()
    result = cli.run_interactive_build(
        input_fn=lambda _prompt: "requirement",
        output=output.append,
        settings=settings,
    )

    assert result == 2
    assert "Local preflight failed" in "\n".join(output)


def test_build_handles_eof_and_keyboard_interrupt(tmp_path: Path) -> None:
    for error in (EOFError(), KeyboardInterrupt()):
        output: list[str] = []

        def fail(_prompt: str, current: BaseException = error) -> str:
            raise current

        result = cli.run_interactive_build(
            input_fn=fail,
            output=output.append,
            settings=_settings(tmp_path),
        )

        assert result == 130
        assert "Reopen the experiment to resume" in "\n".join(output)


def test_build_failure_reports_preserved_run_directory(tmp_path: Path, monkeypatch) -> None:
    class FailingGraph:
        def stream(self, *_args: object, **_kwargs: object):
            yield (), "updates", {"initialize": {"run_directory": "runs/failed-run"}}
            raise RuntimeError("model unavailable")

        def get_state(self, _config: dict) -> SimpleNamespace:
            raise AssertionError("Failure diagnostics must use the recorded run directory")

    output: list[str] = []
    monkeypatch.setattr(cli, "_preflight_errors", lambda _settings, **kwargs: [])

    result = cli.run_interactive_build(
        input_fn=lambda _prompt: "requirement",
        output=output.append,
        settings=_settings(tmp_path),
        graph=FailingGraph(),
    )

    rendered = "\n".join(output)
    assert result == 1
    assert "RuntimeError: model unavailable" in rendered
    assert "Preserved run artifacts: runs/failed-run" in rendered


def test_recursion_limit_scales_with_dataset_size() -> None:
    assert cli._recursion_limit({}) == 1_000
    assert cli._recursion_limit({"dataset_profile": {"total_images": 10}}) == 1_000
    assert cli._recursion_limit({"dataset_profile": {"total_images": 2_000}}) == 8_100


def test_parser_uses_session_entry_and_keeps_maintenance_commands() -> None:
    parser = cli._parser()
    assert parser.parse_args([]).command is None
    assert parser.parse_args(["register", "--dataset-id", "a", "--bundle", "b"]).command == "register"
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["build"])
    assert error.value.code == 2


def test_clarification_commands_keep_answer_separate_from_slash_commands(tmp_path):
    settings = _settings(tmp_path)
    snapshot = SimpleNamespace(values={}, next=(), tasks=())
    graph = SimpleNamespace(get_state=lambda _: snapshot)
    values = iter(["/status", "/config", "/new another", "English"])
    output = []
    assert cli._read_clarification("Your response: ", lambda _: next(values), output.append,
                                    settings, graph, {}) == "English"
    assert "Run status: running" in output
    assert any(line.startswith("Active config: ") for line in output)
    assert any("use /status, /config or /quit" in line for line in output)
    with pytest.raises(cli.SessionQuit):
        cli._read_clarification("Your response: ", lambda _: "/quit", output.append,
                                settings, graph, {})


def test_tool_result_update_is_streamed():
    output = []
    cli._render_update({"execute": {"observations": [
        {"tool": "search_materials", "status": "success", "summary": "2 passages"},
    ]}}, output.append)
    assert output == ["[tool] search_materials: success — 2 passages"]


def test_stream_contract_handles_actual_langgraph_events_and_interrupts():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt, Command
    from typing import TypedDict

    class State(TypedDict, total=False):
        answer: str

    def ask(state):
        return {"answer": interrupt({"question": "Which scope?", "kind": "clarification"})}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "cli-stream"}}
    messages, questions = [], []
    for value in ({}, Command(resume="All datasets")):
        for event in graph.stream(value, config, stream_mode=["tasks", "updates", "custom"], subgraphs=True):
            namespace, mode, payload = cli._event_parts(event)
            if mode == "tasks":
                cli._render_task(namespace, payload, messages.append)
            elif mode == "updates":
                question = cli._interrupt_payload(payload)
                if question:
                    questions.append(question["question"])
                cli._render_update(payload, messages.append)
    assert questions == ["Which scope?"]
    assert "[start] ask" in messages and "[done] ask" in messages
    assert graph.get_state(config).values["answer"] == "All datasets"


def test_stream_rejects_unknown_shapes_without_scanning_business_payloads():
    from langgraph.types import Interrupt
    for event in ({}, ("updates", {}), ([], "updates", {}), ((7,), "updates", {}),
                  ((), "unknown", {}), ((), "updates", [])):
        with pytest.raises(ValueError, match="Expected stream event"):
            cli._event_parts(event)
    nested = {"node": {"business": {"__interrupt__": (Interrupt({"question": "Not a real interrupt"}),)}}}
    assert cli._interrupt_payload(nested) is None
    with pytest.raises(ValueError, match="nonempty question"):
        cli._interrupt_payload({"__interrupt__": (Interrupt({}),)})
    with pytest.raises(ValueError, match="LangGraph Interrupt"):
        cli._interrupt_payload({"__interrupt__": ({"value": {"question": "guessed"}},)})
