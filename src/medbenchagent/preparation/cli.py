"""Worker-side preparation CLI backed by the canonical host models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import Stage
from .assembly import validate_stage_outputs
from .validation import (
    validate_handoff,
    validate_prepare_candidate,
    validate_understand_candidate,
)


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): str(path) for path in root.rglob("*") if path.is_file()}


def _context(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("context must contain an object")
    return value


def _write_report(path: str | Path, result: Any) -> None:
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    atomic_write_json(path, payload)


def validate_handoff_command(args: argparse.Namespace) -> int:
    candidate = Path(args.candidate).resolve()
    context = _context(args.context)
    result = validate_handoff(_files(candidate), state=context, replay=args.replay)
    _write_report(args.report, result)
    return 0 if result.status == "accepted" else 2


def validate_understand_command(args: argparse.Namespace) -> int:
    """Run the worker-visible, side-effect-free Understand validator."""
    candidate = Path(args.candidate).resolve()
    context = _context(args.context)
    source_ids = [item.get("source_id") for item in context.get("sources", []) if isinstance(item, dict)]
    result = validate_understand_candidate(_files(candidate), known_source_ids=source_ids)
    _write_report(args.report, result)
    return 0 if result.status == "accepted" else 2


def validate_candidate_command(args: argparse.Namespace) -> int:
    context = _context(args.context)
    context["source_roots"] = {
        item.get("source_id"): item.get("root")
        for item in context.get("sources", [])
        if isinstance(item, dict) and item.get("source_id") and item.get("root")
    }
    result = validate_prepare_candidate(
        Path(args.candidate), state=context, persist=False,
    )
    _write_report(args.report, result)
    return 0 if result.status == "accepted" else 2


def validate_stage_command(args):
    context = _context(args.context)
    context["source_roots"] = {row["source_id"]: row["root"] for row in context["sources"]}
    if args.stage is Stage.PREPARE_IMAGE:
        accepted = Path(args.context).parent / "accepted"
        context["text_root"] = str(accepted)
        context["text_artifacts"] = {name: {"path": str(accepted / name), "digest": value["digest"]}
            for name, value in context.get("artifacts", {}).items()
            if name in {"cases.jsonl", "provenance.jsonl", "image_requests.jsonl"}}
    result = validate_stage_outputs(args.candidate, args.stage, context)
    _write_report(args.report, result)
    return 0 if result.status == "accepted" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m medbenchagent.preparation.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    handoff = sub.add_parser("validate-handoff", help="validate an understand candidate")
    handoff.add_argument("--context", required=True)
    handoff.add_argument("--candidate", required=True)
    handoff.add_argument("--report", required=True)
    handoff.add_argument("--replay", action="store_true")
    handoff.set_defaults(function=validate_handoff_command)
    understand = sub.add_parser("validate-understand", help="validate worker-visible Understand invariants")
    understand.add_argument("--context", required=True)
    understand.add_argument("--candidate", required=True)
    understand.add_argument("--report", required=True)
    understand.set_defaults(function=validate_understand_command)
    candidate = sub.add_parser("validate-candidate", help="validate a prepare candidate")
    candidate.add_argument("--context", required=True)
    candidate.add_argument("--candidate", required=True)
    candidate.add_argument("--report", required=True)
    candidate.set_defaults(function=validate_candidate_command)
    for stage in (Stage.PREPARE_TEXT, Stage.PREPARE_IMAGE):
        command = sub.add_parser(f"validate-{stage.value.replace('_', '-')}")
        command.add_argument("--context", required=True)
        command.add_argument("--candidate", required=True)
        command.add_argument("--report", required=True)
        command.set_defaults(function=validate_stage_command, stage=stage)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parsed = build_parser().parse_args(argv)
        return int(parsed.function(parsed))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
