import json
from pathlib import Path
from types import SimpleNamespace

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


def _interrupt(kind: str) -> dict:
    payloads = {
        "benchmark_info_confirmation": {
            "benchmark_info": {"disease": "breast cancer", "sample_count": 2},
        },
        "dataset_selection_confirmation": {
            "recommendation": {
                "dataset_id": "demo",
                "name": "Demo",
                "description": "Demo dataset",
            },
            "datasets": [{"id": "demo", "name": "Demo", "available": True}],
        },
        "task_plan_review": {
            "dataset_profile": {"total_images": 5, "fields": {"image.annotation.view": {}}},
            "task_plan": {
                "tasks": {
                    "view": {
                        "decision": "retain",
                        "supporting_fields": ["image.annotation.view"],
                        "metric": "accuracy",
                        "reason": "View is available.",
                    }
                }
            },
        },
        "sample_count_confirmation": {"total": 2, "sample_counts": {"view": 2}},
        "materials_choose": {
            "materials": [
                {
                    "category": "Guideline",
                    "category_label": "Guideline",
                    "file": "Guideline/guide.pdf",
                }
            ],
            "error": "",
        },
        "question_design_confirmation": {
            "tasks": {
                "view": {
                    "question_templates": ["Which view is shown?"],
                    "options": ["CC", "MLO"],
                    "generation_logic": "Use the approved view.",
                    "medical_task_description": "Tests view recognition.",
                    "citations": [],
                }
            }
        },
    }
    return {"kind": kind, **payloads[kind], "question": "Confirm?"}


class _FakeGraph:
    kinds = [
        "benchmark_info_confirmation",
        "dataset_selection_confirmation",
        "task_plan_review",
        "sample_count_confirmation",
        "materials_choose",
        "question_design_confirmation",
    ]

    def __init__(self, report_path: Path) -> None:
        self.index = 0
        self.completed = False
        self.inputs: list[object] = []
        self.configs: list[dict] = []
        self.report_path = report_path

    def stream(self, value: object, config: dict, **kwargs: object):
        self.inputs.append(value)
        self.configs.append(dict(config))
        assert kwargs == {
            "stream_mode": ["tasks", "updates", "custom"],
            "subgraphs": True,
        }
        if self.index < len(self.kinds):
            if self.index == 0:
                yield (
                    (),
                    "updates",
                    {
                        "run_initialize": {
                            "run_directory": "runs/run-1",
                            "working_blueprint_path": "runs/run-1/blueprint.json",
                            "effective_config_path": "runs/run-1/effective_config.json",
                        }
                    },
                )
            kind = self.kinds[self.index]
            self.index += 1
            yield (
                ("SpecificationAgent:task",),
                "tasks",
                {"id": "task", "name": "benchmark_extract", "input": {}},
            )
            yield (
                ("SpecificationAgent:task",),
                "updates",
                {"__interrupt__": (SimpleNamespace(value=_interrupt(kind)),)},
            )
            return
        yield (
            ("ClinicalDesignAgent:task",),
            "custom",
            {"kind": "rag_progress", "task_id": "view", "completed": 1, "total": 1, "remaining": 0},
        )
        yield (
            ("GenerationAgent:task",),
            "custom",
            {
                "kind": "generation_progress",
                "stage": "vqa",
                "cursor": 5,
                "completed_counts": {"view": 2},
                "target_counts": {"view": 2},
                "rejections": {},
                "exhausted": False,
            },
        )
        yield (
            (),
            "updates",
            {
                "AuditAgent": {
                    "audit_artifacts": {
                        "audited": "runs/run-1/audited.jsonl",
                        "report": str(self.report_path),
                    }
                }
            },
        )
        yield ((), "updates", {"run_finalize": {"manifest_path": "manifest.json"}})
        self.completed = True

    def get_state(self, _config: dict) -> SimpleNamespace:
        values = {
            "dataset_profile": {"total_images": 5},
            "run_id": "run-1",
            "run_directory": "runs/run-1",
            "item_count": 2,
            "task_counts": {"view": 2},
            "generation_shortfalls": {"view": 0},
            "raw_output_path": "runs/run-1/raw.jsonl",
            "manifest_path": "runs/run-1/benchmark_manifest.json",
            "audit_artifacts": {
                "audited": "runs/run-1/audited.jsonl",
                "flagged": "runs/run-1/flagged.jsonl",
                "rejected": "runs/run-1/rejected.jsonl",
                "report": str(self.report_path),
            },
        }
        return SimpleNamespace(values=values, next=() if self.completed else ("pending",))


def test_build_prompts_for_requirement_and_drives_all_interrupts(
    tmp_path: Path, monkeypatch
) -> None:
    report = tmp_path / "audit_report.json"
    report.write_text(
        json.dumps(
            {
                "hard_audit": {"accepted": 2, "rejected": 0},
                "semantic_audit": {"flagged": 1},
            }
        ),
        encoding="utf-8",
    )
    graph = _FakeGraph(report)
    answers = iter(["", "Build a breast ultrasound benchmark", *(["confirm"] * 6)])
    output: list[str] = []
    monkeypatch.setattr(cli, "_preflight_errors", lambda _settings: [])

    result = cli.run_interactive_build(
        input_fn=lambda _prompt: next(answers),
        output=output.append,
        settings=_settings(tmp_path),
        graph=graph,
    )

    assert result == 0
    assert graph.inputs[0] == {"user_requirement": "Build a breast ultrasound benchmark"}
    assert [getattr(value, "resume", None) for value in graph.inputs[1:]] == ["confirm"] * 6
    rendered = "\n".join(output)
    assert "Input cannot be empty" in rendered
    for question in cli._QUESTION_TEXT.values():
        assert question in rendered
    assert "[start] SpecificationAgent / benchmark_extract" in rendered
    assert "[waiting] Confirm the BenchmarkInfo" in rendered
    assert "metric=accuracy" in rendered
    assert "=== Run initialized ===" in rendered
    assert "Run ID (directory name): run-1" in rendered
    assert rendered.count("Run ID (directory name): run-1") == 1
    assert "Run directory: runs/run-1" in rendered
    assert "[artifact] working_blueprint_path: runs/run-1/blueprint.json" in rendered
    assert "[artifact] audited: runs/run-1/audited.jsonl" in rendered
    assert "[RAG] view: 1/1, remaining=0" in rendered
    assert "[generation] stage=vqa cursor=5" in rendered
    assert "Audit summary: accepted=2, rejected=0, semantic_flags=1" in rendered
    assert "Run directory: runs/run-1" in rendered
    assert "Audited JSONL: runs/run-1/audited.jsonl" in rendered
    assert "seed=17" in rendered
    assert graph.configs[-1]["recursion_limit"] == 1_000


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
    assert "No dataset is ready for a complete run" in errors
    assert "annotation.jsonl" in errors
    assert "image_manifest.jsonl" in errors


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
        assert "cannot resume after the process exits" in "\n".join(output)


def test_build_failure_reports_preserved_run_directory(tmp_path: Path, monkeypatch) -> None:
    class FailingGraph:
        def stream(self, *_args: object, **_kwargs: object):
            raise RuntimeError("model unavailable")
            yield

        def get_state(self, _config: dict) -> SimpleNamespace:
            return SimpleNamespace(values={"run_directory": "runs/failed-run"}, next=("pending",))

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


def test_parser_keeps_build_parameter_free_and_balance_arguments() -> None:
    parser = cli._parser()

    assert parser.parse_args(["build"]).command == "build"
    balance = parser.parse_args(
        ["balance-answers", "--input", "raw.jsonl", "--output", "balanced.jsonl"]
    )
    assert balance.input_path == "raw.jsonl"
    assert balance.output_path == "balanced.jsonl"
    assert balance.seed == 0
