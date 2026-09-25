"""Command-line arguments, prompts and terminal rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from medbenchagent.runtime import sessions
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.graph_runner import RunEvent
from medbenchagent.runtime.llm import ModelUnavailable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medbenchagent")
    parser.add_argument("--config", type=Path, help="TOML template for newly created experiments")
    commands = parser.add_subparsers(dest="command")
    create = commands.add_parser("create", help="Create an experiment")
    create.add_argument("experiment")
    run = commands.add_parser("run", help="Submit a request to an existing experiment")
    run.add_argument("experiment")
    run.add_argument("request")
    resume = commands.add_parser("resume", help="Resume a session run by its ID")
    resume.add_argument("experiment")
    resume.add_argument("run_id")
    intake = commands.add_parser("resume-intake", help="Answer pending preparation questions")
    intake.add_argument("experiment")
    register = commands.add_parser("register", help="Register a prepared Dataset Bundle")
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--bundle", required=True)
    refresh = commands.add_parser("refresh-card", help="Refresh dataset metadata")
    refresh.add_argument("--dataset-id", required=True)
    return parser


def _render(event: RunEvent) -> None:
    payload = event.payload
    if event.kind == "initialized":
        print(f"Run {event.run_session_id} initialized: {payload.get('run_directory')}")
    elif event.kind == "progress":
        print(payload.get("text", ""))
    elif event.kind == "artifact":
        print(f"[artifact] {payload.get('name')}: {payload.get('path')}")
    elif event.kind == "finished":
        print(f"Run {event.run_session_id}: {payload.get('status')}. {payload.get('summary', '')}")
        if payload.get("run_directory"):
            print(f"Run directory: {payload['run_directory']}")
    elif event.kind == "error":
        print(f"Run {event.run_session_id}: {payload.get('status')}. {payload.get('summary', '')}")
        if payload.get("run_directory"):
            print(f"Preserved run artifacts: {payload['run_directory']}")


def _consume(service: ConversationService, events: Iterable[RunEvent]) -> int:
    status = 0
    for event in events:
        if event.kind == "needs_input":
            question = event.payload["question"]
            print(f"\n{question}")
            close = getattr(events, "close", None)
            if close is not None:
                close()
            try:
                answer = input("Your response: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nRun checkpoint preserved; use the resume command later.")
                return 130
            if not answer:
                print("Answer cannot be empty; use the resume command later.")
                return 2
            service.answer(event.run_session_id, answer)
            return _consume(service, service.resume(event.run_session_id, answer=answer))
        _render(event)
        if event.kind == "error":
            status = 1
    return status


def _answer_intake(service: ConversationService) -> int:
    while pending := service.pending_intake():
        question = pending["questions"][pending["current"]]
        print(f"Question {pending['current'] + 1} of {len(pending['questions'])}: {question['prompt']}")
        try:
            answer = input("Your response: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAnswers saved; use resume-intake later.")
            return 130
        try:
            result = service.answer_intake(answer)
        except (ValueError, ModelUnavailable) as exc:
            print(f"Invalid answer: {exc}")
            continue
        if result.run_ids:
            for run_id in result.run_ids:
                print(f"Queued run: {run_id}")
    return _consume(service, service.drive())


def run_command(args: argparse.Namespace) -> int:
    if args.command == "register":
        from medbenchagent.data.dataset_catalog import register_dataset

        print(json.dumps(register_dataset(args.dataset_id, args.bundle), ensure_ascii=False, indent=2))
        return 0
    if args.command == "refresh-card":
        from medbenchagent.data.dataset_catalog import refresh_card

        print(json.dumps(refresh_card(args.dataset_id), ensure_ascii=False, indent=2))
        return 0
    if args.command == "create":
        sessions.create_experiment(args.experiment, template=args.config)
        print(f"Created experiment {args.experiment}")
        return 0
    service = ConversationService(args.experiment)
    service.reconcile()
    if args.command == "run":
        submission = service.submit(args.request)
        if submission.reply:
            print(submission.reply)
        if submission.question:
            return _answer_intake(service)
        for run_id in submission.run_ids:
            print(f"Queued run: {run_id}")
        return _consume(service, service.drive())
    if args.command == "resume-intake":
        if service.pending_intake() is None:
            raise ValueError("No preparation question is pending")
        return _answer_intake(service)
    if args.command == "resume":
        return _consume(service, service.resume(args.run_id))
    raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    """Keep an already installed console script usable until its launcher is rebuilt."""
    from medbenchagent.entrypoint import main as entrypoint_main

    entrypoint_main()
