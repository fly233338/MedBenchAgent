import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from medbenchagent import cli


def _settings(tmp_path: Path) -> SimpleNamespace:
    embedding = tmp_path / "embedding"
    reranker = tmp_path / "reranker"
    embedding.mkdir(exist_ok=True)
    reranker.mkdir(exist_ok=True)
    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text("{}\n", encoding="utf-8")
    materials = tmp_path / "materials"
    materials.mkdir(exist_ok=True)
    catalog = tmp_path / "dataset_catalog.json"
    catalog.write_text('{"datasets":[]}\n', encoding="utf-8")
    return SimpleNamespace(
        llm=SimpleNamespace(model="deepseek-v4-flash", base_url="http://127.0.0.1:8005/v1"),
        generation=SimpleNamespace(
            seed=17,
            max_tasks_per_image=2,
            question_temperature=0.4,
        ),
        rag=SimpleNamespace(embedding_model=embedding, reranker_model=reranker),
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
    settings.rag.embedding_model = None
    settings.rag.reranker_model = tmp_path / "missing-reranker"
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
    assert "Embedding model is not configured" in errors
    assert "Reranker model directory does not exist" in errors
    assert "No categorized PDF medical materials were found" in errors
    assert "No dataset is ready for a complete run" not in errors
    assert "annotation.jsonl" not in errors
    assert "image_manifest.jsonl" not in errors


def test_preflight_accepts_complete_local_resources(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    category = settings.paths.materials_root / "Guideline"
    category.mkdir()
    (category / "guide.pdf").write_bytes(b"%PDF-local")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    annotation = dataset / "annotation.jsonl"
    annotation.write_text("{}\n", encoding="utf-8")
    (dataset / "image_manifest.jsonl").write_text("{}\n", encoding="utf-8")
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

    assert cli._preflight_errors(settings) == []


def test_build_reports_preflight_before_graph_creation(tmp_path: Path) -> None:
    output: list[str] = []

    result = cli.run_interactive_build(
        input_fn=lambda _prompt: "requirement",
        output=output.append,
        settings=_settings(tmp_path),
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
        assert "build --resume" in "\n".join(output)


def test_build_failure_reports_preserved_run_directory(tmp_path: Path, monkeypatch) -> None:
    class FailingGraph:
        def stream(self, *_args: object, **_kwargs: object):
            yield (), "updates", {"initialize": {"run_directory": "runs/failed-run"}}
            raise RuntimeError("model unavailable")

        def get_state(self, _config: dict) -> SimpleNamespace:
            raise AssertionError("Failure diagnostics must use the recorded run directory")

    output: list[str] = []
    monkeypatch.setattr(cli, "_preflight_errors", lambda _settings: [])

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


def test_parser_keeps_build_entry_and_removes_balance_command() -> None:
    parser = cli._parser()

    assert parser.parse_args(["build"]).command == "build"
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["balance-answers"])
    assert error.value.code == 2


def test_named_experiment_opens_completed_checkpoint_without_execution(tmp_path, monkeypatch):
    from medbenchagent.runtime.experiments import create_experiment, get_experiment
    settings = _settings(tmp_path)
    entry = create_experiment(settings.paths.runs_root, "pilot")
    snapshot = SimpleNamespace(values={"status": "complete", "run_id": "run", "run_directory": "runs/run",
                                      "context": {"run_intent": {"intent": "build_benchmark"},
                                                  "selected_dataset_ids": ["a"]}}, next=(), tasks=())

    class FinishedGraph:
        def get_state(self, config):
            assert config["configurable"]["thread_id"] == entry["thread_id"]
            return snapshot

        def stream(self, *args, **kwargs):
            raise AssertionError("Completed experiments must not execute")

    messages = []
    assert cli.run_interactive_build(settings=settings, graph=FinishedGraph(), experiment=entry,
                                     input_fn=lambda _: (_ for _ in ()).throw(AssertionError("No new input")),
                                     output=messages.append) == 0
    saved = get_experiment(settings.paths.runs_root, "pilot")
    assert saved["status"] == "complete" and saved["datasets"] == ["a"]
    assert not any("Resume identifier" in message for message in messages)


def test_experiment_registry_and_snapshot_status(tmp_path):
    from medbenchagent.runtime.experiments import create_experiment, list_experiments, snapshot_updates
    root = tmp_path / "runs"
    entry = create_experiment(root, "prepare-one", dataset_id="one")
    assert list_experiments(root)[0]["thread_id"] == entry["thread_id"]
    with pytest.raises(ValueError, match="already exists"):
        create_experiment(root, "prepare-one")
    snapshot = SimpleNamespace(values={}, next=("supervisor",), tasks=(SimpleNamespace(interrupts=("question",)),))
    assert snapshot_updates(snapshot, interrupted=True)["status"] == "waiting_input"
    snapshot.tasks = ()
    assert snapshot_updates(snapshot)["status"] == "resumable"
    assert snapshot_updates(snapshot, interrupted=True)["status"] == "interrupted"
    args = cli._parser().parse_args(["experiment", "new", "pilot", "--dataset", "a", "--dataset", "b"])
    assert args.dataset_ids == ["a", "b"]


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


def test_checkpoint_diagnostic_failure_preserves_original_error_and_status(tmp_path):
    import sqlite3
    from medbenchagent.runtime.experiments import create_experiment, get_experiment
    settings = _settings(tmp_path)
    entry = create_experiment(settings.paths.runs_root, "failed-pilot")

    class FailingGraph:
        started = False

        def get_state(self, config):
            if self.started:
                raise sqlite3.DatabaseError("checkpoint unreadable")
            return SimpleNamespace(values={}, next=(), tasks=())

        def stream(self, *args, **kwargs):
            self.started = True
            yield (), "updates", {"initialize": {"run_directory": "runs/preserved"}}
            raise RuntimeError("original model failure")

    messages = []
    status = cli.run_interactive_build(settings=settings, graph=FailingGraph(), experiment=entry,
                                       input_fn=lambda _: "Prepare dataset", output=messages.append)
    rendered = "\n".join(messages)
    assert status == 1
    assert "RuntimeError: original model failure" in rendered
    assert "DatabaseError: checkpoint unreadable" in rendered
    assert "Preserved run artifacts: runs/preserved" in rendered
    saved = get_experiment(settings.paths.runs_root, entry["name"])
    assert saved["status"] == "interrupted" and saved["run_directory"] == "runs/preserved"
