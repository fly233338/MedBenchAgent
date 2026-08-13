import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest
import torch
from PIL import Image

from medbenchagent.evaluation.contracts import (
    ImageEntry,
    InferenceSample,
    ModelOutput,
    TaskSpec,
)
from medbenchagent.evaluation.data import EvaluationDataError, load_benchmark
from medbenchagent.evaluation.eval import (
    _all_task_metric_groups,
    _task_score,
    evaluate_inference,
)
from medbenchagent.evaluation.inference import (
    DistributedContext,
    _mark_rank_complete,
    _parser as inference_parser,
    _validate_args as validate_inference_args,
    _wait_for_rank_completion,
    run_inference,
    shard_indices,
)
from medbenchagent.evaluation.parsing import PredictionParseError, parse_prediction
from medbenchagent.evaluation.scoring import bbox_iou, item_score, macro_f1, option_value


summary_lines = run_path(
    str(Path(__file__).resolve().parents[1] / "results" / "summarize_task_metrics.py")
)["summary_lines"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _letter_item(index: int = 0) -> dict:
    return {
        "id": f"item-{index}",
        "task_id": "diagnosis",
        "dataset": "dataset-one",
        "image_id": "image-one",
        "image_path": "opaque/source.dcm",
        "question": "What is the diagnosis?",
        "options": ["benign", "malignant"],
        "answer": "B",
    }


def _fixture(
    tmp_path: Path,
    *,
    items: list[dict] | None = None,
    tasks: dict | None = None,
    language: str = "en",
) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "images" / "image-one.png"
    image_path.parent.mkdir()
    image = Image.new("L", (20, 10))
    image.putpixel((0, 0), 255)
    image.save(image_path)
    image_manifest = tmp_path / "image_manifest.jsonl"
    _write_jsonl(
        image_manifest,
        [
            {
                "image_id": "image-one",
                "png_path": "images/image-one.png",
                "width": 20,
                "height": 10,
                "sha256": _sha256(image_path),
            }
        ],
    )
    benchmark = tmp_path / "benchmark.dataset-one.audited.jsonl"
    _write_jsonl(benchmark, items or [_letter_item()])
    manifest = tmp_path / "benchmark_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "benchmark-manifest.v1",
            "benchmark_id": "benchmark-test",
            "benchmark_info": {"language": language},
            "dataset": {
                "dataset_id": "dataset-one",
                "image_manifest_sha256": _sha256(image_manifest),
            },
            "tasks": tasks
            or {
                "diagnosis": {
                    "category": "clinical",
                    "response_format": "letter",
                    "metrics": ["accuracy"],
                }
            },
            "artifacts": {
                "audited": {
                    "path": benchmark.name,
                    "sha256": _sha256(benchmark),
                }
            },
        },
    )
    return benchmark, manifest, image_manifest


def _mixed_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    items = [
        _letter_item(0),
        {
            **_letter_item(1),
            "task_id": "detection",
            "options": None,
            "answer": [1, 1, 10, 8],
        },
        {
            **_letter_item(2),
            "task_id": "report_generation",
            "options": None,
            "answer": "No acute finding.",
        },
    ]
    tasks = {
        "diagnosis": {
            "category": "clinical",
            "response_format": "letter",
            "metrics": ["accuracy"],
        },
        "detection": {
            "category": "spatial_localization",
            "response_format": "bbox",
            "metrics": ["miou"],
        },
        "report_generation": {
            "category": "report",
            "response_format": "free_text",
            "metrics": ["bleu"],
        },
    }
    return _fixture(tmp_path, items=items, tasks=tasks)


def _sample(response_format: str, answer: object, options: list[str] | None = None) -> InferenceSample:
    item = {
        "id": "one",
        "task_id": "task",
        "dataset": "dataset",
        "image_id": "image",
        "image_path": "opaque",
        "question": "Question?",
        "options": options,
        "answer": answer,
    }
    metric = {"letter": "accuracy", "free_text": "bleu", "bbox": "miou"}[response_format]
    return InferenceSample(
        index=0,
        item=item,
        task=TaskSpec("task", "category", response_format, metric),
        image=ImageEntry("image", Path("image.png"), "image.png", 200, 100, "a" * 64),
    )


def test_load_benchmark_validates_manifest_and_png(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)

    loaded = load_benchmark(benchmark, manifest, image_manifest)

    assert loaded.benchmark_id == "benchmark-test"
    assert loaded.samples[0].image.png_path == (tmp_path / "images/image-one.png").resolve()
    assert loaded.samples[0].task.metric == "accuracy"


def test_load_benchmark_rejects_unrecorded_benchmark_hash(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    benchmark.write_text(benchmark.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(EvaluationDataError, match="not recorded"):
        load_benchmark(benchmark, manifest, image_manifest)


def test_load_benchmark_rejects_changed_png(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    Image.new("L", (20, 10), 100).save(tmp_path / "images/image-one.png")

    with pytest.raises(EvaluationDataError, match="PNG sha256"):
        load_benchmark(benchmark, manifest, image_manifest)


def test_load_benchmark_rejects_unknown_task(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    item = _letter_item()
    item["task_id"] = "unknown"
    _write_jsonl(benchmark, [item])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"]["audited"]["sha256"] = _sha256(benchmark)
    _write_json(manifest, payload)

    with pytest.raises(EvaluationDataError, match="unknown task_id"):
        load_benchmark(benchmark, manifest, image_manifest)


def test_load_benchmark_rejects_escaping_image_path(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    row = json.loads(image_manifest.read_text(encoding="utf-8"))
    row["png_path"] = "../outside.png"
    _write_jsonl(image_manifest, [row])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["dataset"]["image_manifest_sha256"] = _sha256(image_manifest)
    _write_json(manifest, payload)

    with pytest.raises(EvaluationDataError, match="escapes"):
        load_benchmark(benchmark, manifest, image_manifest)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B", "B"),
        (" b\n", "B"),
        ("A.", "A"),
        (" e.\n", "E"),
        ("A. one", "A"),
        (" b. TWO\n", "B"),
        ("A. E.", "A"),
        ("A. because this finding is characteristic", "A"),
        ("A. malignant", "A"),
        ("(A)", "A"),
        ("(B) DENSITY B", "B"),
        ("(B. Asymmetry)", "B"),
        ("（B）", "B"),
        ("（B） DENSITY B", "B"),
        ("（B．Asymmetry）", "B"),
    ],
)
def test_parse_letter_prediction(raw: str, expected: str) -> None:
    sample = _sample("letter", "B", ["one", "two", "three", "four", "five"])
    assert parse_prediction(ModelOutput(raw), sample) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":"B"}',
        "malignant",
        "The answer is B.",
        "[B]",
    ],
)
def test_parse_letter_prediction_rejects_non_letter_output(raw: str) -> None:
    with pytest.raises(PredictionParseError, match="one valid option letter"):
        parse_prediction(ModelOutput(raw), _sample("letter", "B", ["benign", "malignant"]))


def test_parse_free_text_prediction_preserves_raw_answer() -> None:
    sample = _sample("free_text", "No acute finding.")

    assert parse_prediction(ModelOutput("  No acute finding.\n"), sample) == "No acute finding."
    assert parse_prediction(ModelOutput('{"answer":"No acute finding."}'), sample) == (
        '{"answer":"No acute finding."}'
    )


def test_parse_free_text_prediction_rejects_empty_output() -> None:
    with pytest.raises(PredictionParseError, match="non-empty text"):
        parse_prediction(ModelOutput(" \n"), _sample("free_text", "answer"))


def test_parse_prediction_prefers_officially_extracted_answer() -> None:
    sample = _sample("letter", "B", ["benign", "malignant"])

    assert parse_prediction(ModelOutput("reasoning</think>B", answer_text="B"), sample) == "B"


def test_empty_extracted_answer_does_not_fall_back_to_raw_output() -> None:
    sample = _sample("letter", "B", ["benign", "malignant"])

    with pytest.raises(PredictionParseError, match="one valid option letter"):
        parse_prediction(ModelOutput("B", answer_text=""), sample)


def test_parse_bbox_rescales_qwen_coordinate_space() -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])
    output = ModelOutput('{"bbox":[10,5,50,25]}', coordinate_size=(100, 50))

    assert parse_prediction(output, sample) == [20.0, 10.0, 100.0, 50.0]


def test_parse_bbox_rescales_qwen3_fixed_coordinate_space() -> None:
    sample = _sample("bbox", [138.8, 42.0, 184.0, 78.0])
    output = ModelOutput(
        '```json\n{"bbox":[694,420,920,780]}\n```',
        coordinate_size=(1000, 1000),
    )

    assert parse_prediction(output, sample) == [138.8, 42.0, 184.0, 78.0]


def test_parse_bbox_reports_declared_coordinate_space_for_outside_box() -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])

    with pytest.raises(
        PredictionParseError,
        match=r"\[694\.0, 420\.0, 920\.0, 780\.0\].*640x512",
    ):
        parse_prediction(
            ModelOutput('{"bbox":[694,420,920,780]}', coordinate_size=(640, 512)),
            sample,
        )


def test_parse_bbox_rescales_internvl_normalized_coordinate_space() -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])
    output = ModelOutput(
        "reasoning### The final answer is:{\"bbox\":[100,100,500,500]}",
        coordinate_size=(1000, 1000),
        answer_text='{"bbox":[100,100,500,500]}',
    )

    assert parse_prediction(output, sample) == [20.0, 10.0, 100.0, 50.0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[10,5,50,25]", [20.0, 10.0, 100.0, 50.0]),
        ('```json\n{"bbox":[10,5,50,25]}\n```', [20.0, 10.0, 100.0, 50.0]),
        (
            '```json\n{"bbox_2d":[10,5,50,25],"label":"lesion"}\n```',
            [20.0, 10.0, 100.0, 50.0],
        ),
        ("The final bbox is bbox = (10, 5, 50, 25).", [20.0, 10.0, 100.0, 50.0]),
        (
            'bbox: [10,5,50,25]\nFinal answer: {"bbox":[10,5,50,25]}',
            [20.0, 10.0, 100.0, 50.0],
        ),
    ],
)
def test_parse_bbox_accepts_one_distinct_candidate(raw: str, expected: list[float]) -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])
    assert parse_prediction(ModelOutput(raw, coordinate_size=(100, 50)), sample) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":[10,5,50,25]}',
        '{"bbox":[true,5,50,25]}',
        '{"bbox":[10,5,150,25]}',
        'bbox: [10,5,50,25] then bbox_2d: [20,5,60,25]',
        'bbox_2d: [[10,5,50,25]]',
        'not_bbox: [10,5,50,25]',
    ],
)
def test_parse_bbox_rejects_missing_ambiguous_or_invalid_candidates(raw: str) -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])
    with pytest.raises(PredictionParseError):
        parse_prediction(ModelOutput(raw, coordinate_size=(100, 50)), sample)


def test_metrics_cover_iou_and_semantic_macro_f1() -> None:
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert _task_score("miou", [{"score": 0.25}, {"score": 0.75}], "en") == 0.5
    first = _letter_item()
    second = {**_letter_item(1), "options": ["malignant", "benign"], "answer": "A"}
    assert option_value(first, "B") == option_value(second, "A") == "malignant"
    assert macro_f1(["malignant", "benign"], ["malignant", None]) == pytest.approx(0.5)


class _FakeAdapter:
    def generate(self, samples: list[InferenceSample], *, max_new_tokens: int) -> list[ModelOutput]:
        assert max_new_tokens == 32
        return [
            ModelOutput(
                "private reasoning",
                answer_text=chr(ord("A") + sample.item["options"].index("malignant")),
            )
            for sample in samples
        ]


class _MixedAdapter:
    def generate(self, samples: list[InferenceSample], *, max_new_tokens: int) -> list[ModelOutput]:
        outputs = {
            "letter": ModelOutput("B"),
            "bbox": ModelOutput('{"bbox_2d":[1,1,10,8],"label":"lesion"}'),
            "free_text": ModelOutput("No acute finding."),
        }
        return [outputs[sample.task.response_format] for sample in samples]


def _args(
    benchmark: Path,
    manifest: Path,
    image_manifest: Path,
    output: Path,
    *,
    task_types: tuple[str, ...] = ("letter",),
) -> argparse.Namespace:
    return argparse.Namespace(
        adapter="fake",
        model_path="models/fake",
        benchmark=str(benchmark),
        benchmark_manifest=str(manifest),
        image_manifest=str(image_manifest),
        task_types=list(task_types),
        output=str(output),
        batch_size=2,
        max_new_tokens=32,
        dtype="auto",
        attn_implementation="sdpa",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        bleu_threshold=0.5,
        bbox_iou_threshold=0.5,
        overwrite=False,
    )


def test_inference_parser_requires_unique_task_types() -> None:
    parser = inference_parser()
    required = [
        "--model-path",
        "model",
        "--benchmark",
        "benchmark.jsonl",
        "--benchmark-manifest",
        "manifest.json",
        "--image-manifest",
        "images.jsonl",
        "--output",
        "output.json",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(required)

    args = parser.parse_args([*required, "--task-types", "letter", "bbox", "free_text"])
    validate_inference_args(parser, args)
    assert args.task_types == ["letter", "bbox", "free_text"]

    duplicate_args = parser.parse_args([*required, "--task-types", "bbox", "bbox"])
    with pytest.raises(SystemExit):
        validate_inference_args(parser, duplicate_args)


def test_load_benchmark_filters_formats_and_reindexes_samples(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _mixed_fixture(tmp_path)

    loaded = load_benchmark(
        benchmark,
        manifest,
        image_manifest,
        task_types=("bbox", "letter"),
    )

    assert [sample.index for sample in loaded.samples] == [0, 1]
    assert [sample.task.response_format for sample in loaded.samples] == ["letter", "bbox"]


def test_inference_rejects_requested_format_without_loading_adapter(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    adapter_loaded = False

    def load_adapter(*_args: object, **_kwargs: object) -> _FakeAdapter:
        nonlocal adapter_loaded
        adapter_loaded = True
        return _FakeAdapter()

    with pytest.raises(EvaluationDataError, match="bbox"):
        run_inference(
            _args(
                benchmark,
                manifest,
                image_manifest,
                tmp_path / "inference.json",
                task_types=("letter", "bbox"),
            ),
            DistributedContext(0, 1, 0, torch.device("cpu"), False),
            adapter_loader=load_adapter,
        )

    assert adapter_loaded is False


def test_evaluation_writes_response_format_keyed_reports(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _mixed_fixture(tmp_path)
    inference_path = tmp_path / "inference.json"
    task_types = ("free_text", "letter", "bbox")
    run_inference(
        _args(
            benchmark,
            manifest,
            image_manifest,
            inference_path,
            task_types=task_types,
        ),
        DistributedContext(0, 1, 0, torch.device("cpu"), False),
        adapter_loader=lambda *_args, **_kwargs: _MixedAdapter(),
    )

    report_path = tmp_path / "report.json"
    evaluate_inference(inference_path, report_path)

    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert inference["metadata"]["task_types"] == list(task_types)
    assert [item["coordinate_size"] for item in inference["items"]] == [
        [20, 10],
        [20, 10],
        [20, 10],
    ]
    assert list(report) == list(task_types)
    for task_type in task_types:
        section = report[task_type]
        assert section["schema_version"] == "medbench-evaluation.v1"
        assert section["metadata"]["task_type"] == task_type
        assert section["summary"]["total"] == 1
        assert {task["response_format"] for task in section["tasks"].values()} == {task_type}
        assert len(section["items"]) == 1
    assert report["bbox"]["tasks"]["detection"]["score"] == 1.0
    assert report["bbox"]["tasks"]["detection"]["correct_rate"] == 1.0


def test_evaluation_reparses_letter_raw_prediction_without_model_run(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    inference_path = tmp_path / "inference.json"
    run_inference(
        _args(benchmark, manifest, image_manifest, inference_path),
        DistributedContext(0, 1, 0, torch.device("cpu"), False),
        adapter_loader=lambda *_args, **_kwargs: _FakeAdapter(),
    )

    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    payload["items"][0].update(
        {
            "raw_prediction": "（B）",
            "prediction": None,
            "item_score": 0.0,
            "correct": False,
            "error": {"code": "parse_error", "message": "old extractor"},
        }
    )
    _write_json(inference_path, payload)

    report_path = tmp_path / "reparsed.report.json"
    evaluate_inference(inference_path, report_path, reparse=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["letter"]["metadata"]["reparsed_from_raw_prediction"] is True
    assert report["letter"]["items"][0]["prediction"] == "B"
    assert report["letter"]["items"][0]["correct"] is True


@pytest.mark.parametrize(
    ("adapter", "raw_prediction"),
    [
        ("qwen3_vl_8b_instruct", '{"bbox":[50,100,500,800]}'),
        (
            "qwen3_vl_8b_thinking",
            'reasoning with bbox [900,900,950,950]</think>{"bbox":[50,100,500,800]}',
        ),
    ],
)
def test_evaluation_reparses_legacy_qwen3_bbox_without_model_run(
    tmp_path: Path, adapter: str, raw_prediction: str
) -> None:
    benchmark, manifest, image_manifest = _mixed_fixture(tmp_path)
    inference_path = tmp_path / "inference.json"
    run_inference(
        _args(
            benchmark,
            manifest,
            image_manifest,
            inference_path,
            task_types=("bbox",),
        ),
        DistributedContext(0, 1, 0, torch.device("cpu"), False),
        adapter_loader=lambda *_args, **_kwargs: _MixedAdapter(),
    )
    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    payload["metadata"]["model"]["adapter"] = adapter
    payload["items"][0].pop("coordinate_size")
    payload["items"][0].update(
        {
            "raw_prediction": raw_prediction,
            "prediction": None,
            "item_score": 0.0,
            "correct": False,
            "error": {"code": "parse_error", "message": "legacy coordinate bug"},
        }
    )
    _write_json(inference_path, payload)

    report_path = tmp_path / "reparsed.report.json"
    evaluate_inference(inference_path, report_path, reparse_bbox=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))["bbox"]
    assert report["metadata"]["reparsed_from_raw_prediction"] is True
    assert report["summary"]["errors"] == 0
    assert report["items"][0]["prediction"] == [1.0, 1.0, 10.0, 8.0]
    assert report["items"][0]["item_score"] == 1.0
    assert report["items"][0]["correct"] is True


def test_bbox_reparse_rejects_legacy_dynamic_space_without_recorded_size(
    tmp_path: Path,
) -> None:
    benchmark, manifest, image_manifest = _mixed_fixture(tmp_path)
    inference_path = tmp_path / "inference.json"
    run_inference(
        _args(
            benchmark,
            manifest,
            image_manifest,
            inference_path,
            task_types=("bbox",),
        ),
        DistributedContext(0, 1, 0, torch.device("cpu"), False),
        adapter_loader=lambda *_args, **_kwargs: _MixedAdapter(),
    )
    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    payload["metadata"]["model"]["adapter"] = "qwen2_5_vl_7b_instruct"
    payload["items"][0].pop("coordinate_size")
    _write_json(inference_path, payload)

    with pytest.raises(EvaluationDataError, match="recorded coordinate_size"):
        evaluate_inference(
            inference_path,
            tmp_path / "report.json",
            reparse_bbox=True,
        )


def test_inference_and_evaluation_end_to_end_with_fake_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress: dict[str, object] = {}
    progress_updates: list[int] = []

    class FakeProgress:
        def __init__(self, **kwargs: object) -> None:
            progress.update(kwargs)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def update(self, count: int) -> None:
            progress_updates.append(count)

    monkeypatch.setattr("medbenchagent.evaluation.inference.tqdm", FakeProgress)
    items = [_letter_item(0), _letter_item(1)]
    benchmark, manifest, image_manifest = _fixture(tmp_path, items=items)
    output = tmp_path / "inference.json"
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)

    result = run_inference(
        _args(benchmark, manifest, image_manifest, output),
        context,
        adapter_loader=lambda *_args, **_kwargs: _FakeAdapter(),
    )
    report_path = tmp_path / "report.json"
    evaluate_inference(output, report_path)

    inference = json.loads(output.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    letter_report = report["letter"]
    assert result == str(output)
    assert progress["total"] == 2
    assert progress["desc"] == "rank 0"
    assert progress_updates == [2]
    assert not (tmp_path / "inference.json.parts").exists()
    assert [item["prediction"] for item in inference["items"]] == ["B", "B"]
    assert [item["raw_prediction"] for item in inference["items"]] == [
        "private reasoning",
        "private reasoning",
    ]
    assert inference["metadata"]["task_types"] == ["letter"]
    assert letter_report["metadata"]["task_type"] == "letter"
    assert letter_report["summary"] == {
        "total": 2,
        "correct": 2,
        "incorrect": 0,
        "correct_rate": 1.0,
        "macro_f1": 1.0,
        "errors": 0,
        "error_counts": {},
    }
    assert letter_report["tasks"]["diagnosis"]["score"] == 1.0
    assert letter_report["tasks"]["diagnosis"]["macro_f1"] == 1.0
    assert letter_report["categories"]["clinical"]["macro_f1"] == 1.0
    assert letter_report["items"] == [
        {
            "index": 0,
            "item_id": "item-0",
            "task_id": "diagnosis",
            "prediction": "B",
            "raw_prediction": "private reasoning",
            "item_score": 1.0,
            "correct": True,
            "error": None,
        },
        {
            "index": 1,
            "item_id": "item-1",
            "task_id": "diagnosis",
            "prediction": "B",
            "raw_prediction": "private reasoning",
            "item_score": 1.0,
            "correct": True,
            "error": None,
        },
    ]


def test_evaluation_macro_f1_uses_semantic_options_after_balancing(tmp_path: Path) -> None:
    items = [
        _letter_item(0),
        {**_letter_item(1), "options": ["malignant", "benign"], "answer": "A"},
    ]
    tasks = {
        "diagnosis": {
            "category": "clinical",
            "response_format": "letter",
            "metrics": ["macro_f1"],
        }
    }
    benchmark, manifest, image_manifest = _fixture(tmp_path, items=items, tasks=tasks)
    output = tmp_path / "inference.json"
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)
    run_inference(
        _args(benchmark, manifest, image_manifest, output),
        context,
        adapter_loader=lambda *_args, **_kwargs: _FakeAdapter(),
    )

    report_path = tmp_path / "report.json"
    evaluate_inference(output, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [row["prediction"] for row in json.loads(output.read_text())["items"]] == ["B", "A"]
    assert report["letter"]["tasks"]["diagnosis"]["score"] == 1.0


def test_metric_groups_average_parent_tasks_instead_of_question_types(tmp_path: Path) -> None:
    items = [
        {**_letter_item(0), "task_id": "finding_attribute.shape"},
        {**_letter_item(1), "task_id": "finding_attribute.margin", "answer": "A"},
        {**_letter_item(2), "task_id": "diagnosis"},
    ]
    tasks = {
        "finding_attribute.shape": {
            "category": "clinical",
            "response_format": "letter",
            "metrics": ["accuracy"],
            "parent_task_id": "finding_attribute",
        },
        "finding_attribute.margin": {
            "category": "clinical",
            "response_format": "letter",
            "metrics": ["accuracy"],
            "parent_task_id": "finding_attribute",
        },
        "diagnosis": {
            "category": "clinical",
            "response_format": "letter",
            "metrics": ["accuracy"],
        },
    }
    benchmark, manifest, image_manifest = _fixture(tmp_path, items=items, tasks=tasks)
    output = tmp_path / "inference.json"
    run_inference(
        _args(benchmark, manifest, image_manifest, output),
        DistributedContext(0, 1, 0, torch.device("cpu"), False),
        adapter_loader=lambda *_args, **_kwargs: _FakeAdapter(),
    )
    report_path = tmp_path / "report.json"
    evaluate_inference(output, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["letter"]["metric_groups"] == {
        "accuracy": {"task_count": 2, "task_macro_score": 0.75},
        "f1": {"task_count": 2, "task_macro_score": 0.75},
    }
    assert report["letter"]["tasks"]["finding_attribute.shape"]["parent_task_id"] == (
        "finding_attribute"
    )


def test_metric_groups_do_not_infer_parent_tasks_from_task_names() -> None:
    groups = _all_task_metric_groups(
        {
            "finding_attribute.shape": {
                "parent_task_id": None,
                "correct_rate": 1.0,
                "score": 1.0,
            },
            "finding_attribute.margin": {
                "parent_task_id": None,
                "correct_rate": 0.0,
                "score": 0.0,
            },
            "diagnosis": {
                "parent_task_id": None,
                "correct_rate": 1.0,
                "score": 1.0,
            },
        }
    )

    assert groups == {
        "accuracy": {"task_count": 3, "task_macro_score": pytest.approx(2 / 3)},
        "f1": {"task_count": 3, "task_macro_score": pytest.approx(2 / 3)},
    }


def test_summary_helper_supports_legacy_and_grouped_reports() -> None:
    legacy = {
        "tasks": {
            "diagnosis": {
                "parent_task_id": None,
                "correct_rate": 0.75,
                "score": 0.5,
            }
        }
    }
    grouped = {
        "letter": legacy,
        "bbox": {
            "tasks": {
                "detection": {
                    "parent_task_id": None,
                    "correct_rate": 0.25,
                    "score": 0.4,
                },
                "visual_grounding": {
                    "parent_task_id": None,
                    "correct_rate": 0.5,
                    "score": 0.6,
                },
            }
        },
    }

    assert summary_lines(legacy) == ["diagnosis", "0.7500", "0.5000"]
    assert summary_lines(grouped) == [
        "letter",
        "diagnosis",
        "0.7500",
        "0.5000",
        "",
        "bbox",
        "detection,visual_grounding",
        "0.4000,0.6000",
        "0.2500,0.5000",
    ]


def test_inference_requires_overwrite_for_existing_output(tmp_path: Path) -> None:
    benchmark, manifest, image_manifest = _fixture(tmp_path)
    output = tmp_path / "inference.json"
    output.write_text("existing", encoding="utf-8")
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_inference(
            _args(benchmark, manifest, image_manifest, output),
            context,
            adapter_loader=lambda *_args, **_kwargs: _FakeAdapter(),
        )


def test_eval_threshold_override_recomputes_correctness(tmp_path: Path) -> None:
    item = {
        **_letter_item(),
        "task_id": "report",
        "options": None,
        "answer": "No acute finding.",
    }
    tasks = {
        "report": {
            "category": "report",
            "response_format": "free_text",
            "metrics": ["bleu"],
        }
    }
    benchmark, manifest, image_manifest = _fixture(tmp_path, items=[item], tasks=tasks)
    loaded = load_benchmark(benchmark, manifest, image_manifest)
    sample = loaded.samples[0]
    score = item_score(sample, "No finding.", "en")
    inference = tmp_path / "inference.json"
    _write_json(
        inference,
        {
            "schema_version": "medbench-inference.v1",
            "metadata": {
                "benchmark_id": "benchmark-test",
                "dataset_id": "dataset-one",
                "language": "en",
                "model": {"adapter": "fake"},
                "correctness": {"bleu_threshold": 0.5, "bbox_iou_threshold": 0.5},
                "inputs": {"item_count": 1},
            },
            "items": [
                {
                    "index": 0,
                    "item": item,
                    "task": {"category": "report", "response_format": "free_text", "metric": "bleu"},
                    "image": {"png_path": "images/image-one.png", "width": 20, "height": 10},
                    "prediction": "No finding.",
                    "raw_prediction": '{"answer":"No finding."}',
                    "item_score": score,
                    "correct": score >= 0.5,
                    "error": None,
                }
            ],
        },
    )

    report_path = tmp_path / "report.json"
    evaluate_inference(inference, report_path, bleu_threshold=0.0)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    free_text_report = report["free_text"]
    assert free_text_report["metadata"]["thresholds"]["overridden"] is True
    assert free_text_report["summary"]["correct"] == 1
    assert free_text_report["summary"]["macro_f1"] is None
    assert free_text_report["tasks"]["report"]["macro_f1"] is None
    assert free_text_report["categories"]["report"]["macro_f1"] is None


def test_shard_indices_are_complete_and_non_overlapping() -> None:
    shards = [list(shard_indices(10, rank, 3)) for rank in range(3)]
    assert sorted(index for shard in shards for index in shard) == list(range(10))
    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]


def test_rank_completion_wait_returns_after_all_markers_exist(tmp_path: Path) -> None:
    _mark_rank_complete(tmp_path, 0)
    _mark_rank_complete(tmp_path, 1)

    _wait_for_rank_completion(tmp_path, 2)
