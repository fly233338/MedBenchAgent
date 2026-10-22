"""Run or resume the explicitly scripted Compilation Core example."""

import argparse
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import TypeAdapter

from .contracts import Identifier
from .demo import create_synthetic_inputs, demo_agents, scripted_supervisor
from .graph import CoreContext, build_compilation_graph


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scripted collaboration demo: no model, OpenHands or medical data calls.",
    )
    parser.add_argument("command", choices=("run", "resume"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--compilation-id", required=True)
    parser.add_argument("--initial-evidence", action="store_true")
    parser.add_argument("--pause-for-input", action="store_true")
    parser.add_argument("--answer")
    args = parser.parse_args(argv)
    try:
        identifier = TypeAdapter(Identifier).validate_python(args.compilation_id)
        directory = args.directory.resolve()
        options_path = directory / "demo.json"
        if args.command == "run":
            if args.answer is not None:
                raise ValueError("--answer is only valid for resume")
            if options_path.exists() or (directory / "checkpoints.sqlite").exists():
                raise ValueError("This directory already contains a run; use resume")
            directory.mkdir(parents=True, exist_ok=True)
            options = {
                "compilation_id": identifier, "initial_evidence": args.initial_evidence,
                "pause_for_input": args.pause_for_input,
            }
            with options_path.open("x", encoding="utf-8") as stream:
                json.dump(options, stream)
        else:
            if args.initial_evidence or args.pause_for_input:
                raise ValueError("Resume uses the original demo options")
            options = json.loads(options_path.read_text(encoding="utf-8"))
            if options["compilation_id"] != identifier:
                raise ValueError("The directory belongs to another Compilation")
            if not (directory / "checkpoints.sqlite").is_file():
                raise ValueError("No saved checkpoint exists")
        context = CoreContext(
            directory=directory,
            decide=scripted_supervisor(directory, pause_for_input=options["pause_for_input"]),
        )
        config = {"configurable": {"thread_id": identifier}, "recursion_limit": 60}
        with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
            graph = build_compilation_graph(demo_agents(), checkpointer=saver)
            if args.command == "run":
                inputs = create_synthetic_inputs(
                    directory, identifier, initial_evidence=options["initial_evidence"],
                )
                invocation = {
                    "compilation_id": identifier,
                    "goal": "Demonstrate Case-linked multi-agent handoffs with synthetic values",
                    "artifacts": [ref.model_dump(mode="json") for ref in inputs],
                }
            else:
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    raise ValueError("No state exists for this Compilation")
                pending = any(task.interrupts for task in snapshot.tasks)
                if pending and args.answer is None:
                    raise ValueError("This run is waiting for --answer")
                if not pending and args.answer is not None:
                    raise ValueError("There is no pending question to answer")
                invocation = Command(resume=args.answer) if pending else None
            if args.command == "run" or graph.get_state(config).next:
                graph.invoke(invocation, config, context=context, durability="sync")
            snapshot = graph.get_state(config)
            state = snapshot.values
            print(json.dumps({
                "compilation_id": identifier, "mode": "scripted",
                "status": state.get("status"), "summary": state.get("summary", ""),
                "history": state.get("history", []), "artifacts": state.get("artifacts", []),
                "questions": [item.value for task in snapshot.tasks for item in task.interrupts],
            }, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
