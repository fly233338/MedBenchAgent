"""Worker-side preparation CLI backed by the canonical host models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import digest_file
from .validation import (
    project_grouping,
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


def package_command(args: argparse.Namespace) -> int:
    candidate = Path(args.candidate).resolve()
    context = _context(args.context)
    for field in ("source_report", "mapping", "population", "inspection_script"):
        artifact = context.get("accepted_handoff", {}).get(field)
        if artifact:
            target = candidate / artifact["name"]
            if target.exists() and digest_file(target) != artifact["digest"]:
                raise ValueError(f"Candidate changed accepted artifact: {artifact['name']}")
            if not target.exists():
                shutil.copy2(artifact["path"], target)
    mapping_path = candidate / "mapping.json"
    if not mapping_path.is_file():
        raise ValueError("package requires mapping.json")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    atomic_write_json(candidate / "grouping_policy.json", project_grouping(mapping))
    report_path = candidate / "execution_report.json"
    if not report_path.is_file():
        raise ValueError("package requires execution_report.json produced by real executions")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("tests") or not report.get("full_execution"):
        raise ValueError("execution_report must contain actual tests and full_execution records")
    files = []
    for path in sorted(candidate.rglob("*")):
        if not path.is_file() or path.name in {"output_manifest.json", "submission.json"}:
            continue
        files.append({"path": path.relative_to(candidate).as_posix(), "kind": "output", "digest": digest_file(path), "producer": "worker"})
    dependencies = []
    for value in context.get("files", []):
        if isinstance(value, dict) and value.get("path"):
            dependencies.append({"path": value["path"], "role": value.get("role", "input"), "digest": value.get("digest", "")})
    atomic_write_json(candidate / "output_manifest.json", {
        "files": files,
        "dependencies": dependencies,
        "mapping_digest": digest_file(mapping_path),
        "population_digest": digest_file(candidate / "image_population.jsonl") if (candidate / "image_population.jsonl").is_file() else "",
    })
    return 0


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
    package = sub.add_parser("package", help="build deterministic grouping and output manifests")
    package.add_argument("--context", required=True)
    package.add_argument("--candidate", required=True)
    package.set_defaults(function=package_command)
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
