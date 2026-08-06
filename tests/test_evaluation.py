import argparse
import hashlib
import json
from pathlib import Path

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
from medbenchagent.evaluation.eval import evaluate_inference
from medbenchagent.evaluation.inference import (
    DistributedContext,
    run_inference,
    shard_indices,
)
from medbenchagent.evaluation.parsing import PredictionParseError, parse_prediction
from medbenchagent.evaluation.scoring import bbox_iou, item_score, macro_f1, option_value


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
    ],
)
def test_parse_letter_prediction(raw: str, expected: str) -> None:
    sample = _sample("letter", "B", ["one", "two", "three", "four", "five"])
    assert parse_prediction(ModelOutput(raw), sample) == expected


@pytest.mark.parametrize(
    "raw",
    ['{"answer":"B"}', "malignant", "The answer is B.", "(B)", "A. E.", "A. because"],
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


def test_parse_bbox_rescales_internvl_normalized_coordinate_space() -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])
    output = ModelOutput(
        "reasoning### The final answer is:{\"bbox\":[100,100,500,500]}",
        coordinate_size=(1000, 1000),
        answer_text='{"bbox":[100,100,500,500]}',
    )

    assert parse_prediction(output, sample) == [20.0, 10.0, 100.0, 50.0]


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":[10,5,50,25]}',
        "[10,5,50,25]",
        '```json\n{"bbox":[10,5,50,25]}\n```',
        '{"bbox":[10,5,50,25],"label":"lesion"}',
        '{"bbox":[true,5,50,25]}',
        '{"bbox":[10,5,150,25]}',
    ],
)
def test_parse_bbox_rejects_outputs_outside_strict_contract(raw: str) -> None:
    sample = _sample("bbox", [20.0, 10.0, 100.0, 50.0])
    with pytest.raises(PredictionParseError):
        parse_prediction(ModelOutput(raw, coordinate_size=(100, 50)), sample)


def test_metrics_cover_iou_and_semantic_macro_f1() -> None:
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
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


def _args(benchmark: Path, manifest: Path, image_manifest: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        adapter="fake",
        model_path="models/fake",
        benchmark=str(benchmark),
        benchmark_manifest=str(manifest),
        image_manifest=str(image_manifest),
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
    assert report["summary"] == {
        "total": 2,
        "correct": 2,
        "incorrect": 0,
        "correct_rate": 1.0,
        "errors": 0,
        "error_counts": {},
    }
    assert report["tasks"]["diagnosis"]["score"] == 1.0
    assert report["items"] == [
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
    assert report["tasks"]["diagnosis"]["score"] == 1.0


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
    assert report["metadata"]["thresholds"]["overridden"] is True
    assert report["summary"]["correct"] == 1


def test_shard_indices_are_complete_and_non_overlapping() -> None:
    shards = [list(shard_indices(10, rank, 3)) for rank in range(3)]
    assert sorted(index for shard in shards for index in shard) == list(range(10))
    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
