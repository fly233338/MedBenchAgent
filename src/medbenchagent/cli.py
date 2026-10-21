"""Command-line arguments, prompts and terminal rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from medbenchagent.runtime import sessions

if TYPE_CHECKING:
    from medbenchagent.runtime.conversation_service import ConversationService
    from medbenchagent.runtime.graph_runner import RunEvent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medd2b")
    parser.add_argument("--config", type=Path, help="TOML template for newly created experiments")
    commands = parser.add_subparsers(dest="command")
    create = commands.add_parser("create", help="Create an experiment")
    create.add_argument("experiment")
    run = commands.add_parser("run", help="Submit a request to an existing experiment")
    run.add_argument("experiment")
    run.add_argument("request")
    run.add_argument("--dataset", dest="dataset_id")
    batch = commands.add_parser("batch", help="Compile datasets independently")
    batch.add_argument("experiment")
    batch.add_argument("dataset_ids", nargs="+")
    batch.add_argument("--request", default="Compile this dataset into a benchmark")
    resume = commands.add_parser("resume", help="Resume a session run by its ID")
    resume.add_argument("experiment")
    resume.add_argument("run_id")
    register = commands.add_parser("register", help="Register a prepared Dataset Bundle")
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--bundle", required=True)
    export = commands.add_parser("export", help="Export a published portable benchmark package")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    return parser


def _render(event: RunEvent) -> None:
    payload = event.payload
    if event.kind == "initialized":
        print(f"Run {event.run_session_id} initialized: {payload.get('run_directory')}")
    elif event.kind == "progress":
        print(payload.get("text", ""))
    elif event.kind == "artifact":
        print(f"[artifact] {payload.get('name')}: {payload.get('path')}")
    elif event.kind == "view":
        from medbenchagent.runtime.presentation import compilation_text
        print(compilation_text({"view": payload, "status": "running"}))
    elif event.kind == "finished":
        print(f"Run {event.run_session_id}: {payload.get('status')}. {payload.get('summary', '')}")
        if payload.get("run_directory"):
            print(f"Run directory: {payload['run_directory']}")
    elif event.kind == "error":
        print(f"Run {event.run_session_id}: {payload.get('status')}. {payload.get('summary', '')}", file=sys.stderr)
        if payload.get("recovery_hint"):
            print(payload["recovery_hint"], file=sys.stderr)
        if payload.get("run_directory"):
            print(f"Preserved run artifacts: {payload['run_directory']}", file=sys.stderr)


def _consume(service: ConversationService, events: Iterable[RunEvent]) -> int:
    status = 0
    events = iter(events)
    while True:
        event = next(events, None)
        if event is None:
            return status
        if event.kind == "needs_input":
            question = event.payload["question"]
            print(f"\n{question}")
            close = getattr(events, "close", None)
            if close is not None:
                close()
            try:
                answer = input("Your response: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nRun checkpoint preserved; use the resume command later.", file=sys.stderr)
                return 130
            if not answer:
                print("Answer cannot be empty; use the resume command later.", file=sys.stderr)
                return 2
            service.answer(event.run_session_id, answer)
            events = iter(service.resume(event.run_session_id, answer=answer))
            continue
        _render(event)
        if event.kind == "error":
            status = event.payload.get("exit_code", 1)
            if status == 130:
                return status
        elif event.kind == "finished" and event.payload.get("status") in {"blocked", "failed"}:
            status = 1


def run_command(args: argparse.Namespace) -> int:
    if args.command == "export":
        from medbenchagent.generation.delivery import export_benchmark
        print(export_benchmark(args.source, args.destination))
        return 0
    if args.command == "register":
        from medbenchagent.data.dataset_catalog import register_dataset

        print(json.dumps(register_dataset(args.dataset_id, args.bundle), ensure_ascii=False, indent=2))
        return 0
    if args.command == "create":
        sessions.create_experiment(args.experiment, template=args.config)
        print(f"Created experiment {args.experiment}")
        return 0
    from medbenchagent.runtime.conversation_service import ConversationService

    service = ConversationService(args.experiment)
    service.reconcile()
    if args.command == "run":
        submission = service.submit(args.request, dataset_id=args.dataset_id)
        for run_id in submission.run_ids:
            print(f"Queued run: {run_id}")
        return _consume(service, service.drive())
    if args.command == "batch":
        submission = service.submit_batch(args.dataset_ids, args.request)
        for run_id in submission.run_ids:
            print(f"Queued compilation: {run_id}")
        return _consume(service, service.drive())
    if args.command == "resume":
        return _consume(service, service.resume(args.run_id))
    raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    """Keep an already installed console script usable until its launcher is rebuilt."""
    from medbenchagent.entrypoint import main as entrypoint_main

    entrypoint_main()
