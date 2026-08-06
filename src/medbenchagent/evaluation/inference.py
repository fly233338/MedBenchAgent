"""Single-node torchrun inference for generated Medical VLM benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm.auto import tqdm

from medbenchagent.evaluation.contracts import InferenceSample, LoadedBenchmark
from medbenchagent.evaluation.data import load_benchmark
from medbenchagent.evaluation.io import atomic_write_json
from medbenchagent.evaluation.models import load_model_adapter
from medbenchagent.evaluation.models.base import ModelAdapter
from medbenchagent.evaluation.parsing import PredictionParseError, parse_prediction
from medbenchagent.evaluation.scoring import item_correct, item_score


AdapterLoader = Callable[..., ModelAdapter]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    process_group: bool

    def barrier(self) -> None:
        if self.process_group:
            dist.barrier()


def initialize_distributed() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Medical VLM inference requires CUDA.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK {local_rank} does not identify an available CUDA device.")
    torch.cuda.set_device(local_rank)
    process_group = world_size > 1
    if process_group:
        dist.init_process_group(backend="nccl")
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        process_group=process_group,
    )


def shard_indices(item_count: int, rank: int, world_size: int) -> range:
    return range(rank, item_count, world_size)


def _parts_directory(output: Path) -> Path:
    return output.with_name(output.name + ".parts")


def _prepare_output(output: Path, parts: Path, overwrite: bool, context: DistributedContext) -> None:
    if context.rank == 0:
        if (output.exists() or parts.exists()) and not overwrite:
            raise FileExistsError(
                f"Inference output or temporary parts already exist; use --overwrite: {output}"
            )
        if overwrite:
            if output.is_file():
                output.unlink()
            if parts.is_dir():
                shutil.rmtree(parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        parts.mkdir(parents=False, exist_ok=False)
    context.barrier()


def _result_record(
    sample: InferenceSample,
    *,
    raw_prediction: str,
    prediction: str | list[float] | None,
    score: float,
    correct: bool,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "index": sample.index,
        "item": sample.item,
        "task": {
            "category": sample.task.category,
            "response_format": sample.task.response_format,
            "metric": sample.task.metric,
            "parent_task_id": sample.task.parent_task_id,
        },
        "image": {
            "png_path": sample.image.relative_png_path,
            "width": sample.image.width,
            "height": sample.image.height,
        },
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "item_score": score,
        "correct": correct,
        "error": error,
    }


def _write_rank_results(
    path: Path,
    benchmark: LoadedBenchmark,
    adapter: ModelAdapter,
    *,
    context: DistributedContext,
    batch_size: int,
    max_new_tokens: int,
    bleu_threshold: float,
    bbox_iou_threshold: float,
) -> None:
    indices = list(shard_indices(len(benchmark.samples), context.rank, context.world_size))
    language = benchmark.benchmark_info["language"]
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        with tqdm(
            total=len(indices),
            desc=f"rank {context.rank}",
            unit="item",
            position=context.rank,
            dynamic_ncols=True,
        ) as progress:
            for start in range(0, len(indices), batch_size):
                batch = [benchmark.samples[index] for index in indices[start : start + batch_size]]
                outputs = adapter.generate(batch, max_new_tokens=max_new_tokens)
                if len(outputs) != len(batch):
                    raise RuntimeError("Model adapter returned an unexpected number of outputs.")
                for sample, output in zip(batch, outputs):
                    try:
                        prediction = parse_prediction(output, sample)
                        score = item_score(sample, prediction, language)
                        correct = item_correct(
                            sample,
                            score,
                            bleu_threshold=bleu_threshold,
                            bbox_iou_threshold=bbox_iou_threshold,
                        )
                        error = None
                    except PredictionParseError as exc:
                        prediction, score, correct = None, 0.0, False
                        error = {"code": "parse_error", "message": str(exc)}
                    destination.write(
                        json.dumps(
                            _result_record(
                                sample,
                                raw_prediction=output.raw_text,
                                prediction=prediction,
                                score=score,
                                correct=correct,
                                error=error,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                destination.flush()
                progress.update(len(batch))


def _read_parts(parts: Path, expected_count: int, world_size: int) -> list[dict[str, Any]]:
    results = []
    for rank in range(world_size):
        path = parts / f"rank-{rank}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing inference rank output: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("index"), int):
                raise RuntimeError(f"Invalid inference rank output: {path}")
            results.append(value)
    results.sort(key=lambda result: result["index"])
    if len(results) != expected_count or [result["index"] for result in results] != list(
        range(expected_count)
    ):
        raise RuntimeError("Inference rank outputs are incomplete or contain duplicate indices.")
    return results


def _metadata(args: argparse.Namespace, benchmark: LoadedBenchmark, context: DistributedContext) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_id": benchmark.benchmark_id,
        "dataset_id": benchmark.dataset_id,
        "language": benchmark.benchmark_info["language"],
        "model": {
            "adapter": args.adapter,
            "model_path": args.model_path,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "distributed": {
            "world_size": context.world_size,
            "batch_size_per_rank": args.batch_size,
        },
        "correctness": {
            "bleu_threshold": args.bleu_threshold,
            "bbox_iou_threshold": args.bbox_iou_threshold,
        },
        "inputs": {
            "benchmark_sha256": benchmark.benchmark_sha256,
            "benchmark_manifest_sha256": benchmark.benchmark_manifest_sha256,
            "image_manifest_sha256": benchmark.image_manifest_sha256,
            "item_count": len(benchmark.samples),
        },
    }


def run_inference(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    adapter_loader: AdapterLoader = load_model_adapter,
) -> str | None:
    output = Path(args.output)
    parts = _parts_directory(output)
    _prepare_output(output, parts, args.overwrite, context)
    benchmark = load_benchmark(
        args.benchmark,
        args.benchmark_manifest,
        args.image_manifest,
        rank=context.rank,
        world_size=context.world_size,
    )
    adapter = adapter_loader(
        args.adapter,
        args.model_path,
        device=context.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    _write_rank_results(
        parts / f"rank-{context.rank}.jsonl",
        benchmark,
        adapter,
        context=context,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        bleu_threshold=args.bleu_threshold,
        bbox_iou_threshold=args.bbox_iou_threshold,
    )
    context.barrier()
    if context.rank == 0:
        results = _read_parts(parts, len(benchmark.samples), context.world_size)
        atomic_write_json(
            output,
            {
                "schema_version": "medbench-inference.v1",
                "metadata": _metadata(args, benchmark, context),
                "items": results,
            },
        )
        shutil.rmtree(parts)
    context.barrier()
    return str(output) if context.rank == 0 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="qwen2_5_vl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--bleu-threshold", type=float, default=0.5)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive.")
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        parser.error("--min-pixels and --max-pixels must define a positive ordered range.")
    if not 0 <= args.bleu_threshold <= 1 or not 0 <= args.bbox_iou_threshold <= 1:
        parser.error("Correctness thresholds must be between 0 and 1.")


@record
def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    context = initialize_distributed()
    try:
        result = run_inference(args, context)
        if result is not None:
            print(result)
    finally:
        if context.process_group and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
