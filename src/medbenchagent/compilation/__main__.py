"""Run or resume independent coordination with scripted professional agents."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .contracts import Contract, Identifier, Text
from .demo import DEFAULT_DEMO_GOAL, create_synthetic_inputs, demo_agents, scripted_supervisor
from .graph import CompilationContext, build_compilation_graph


class RunOptions(Contract):
    compilation_id: Identifier
    goal: Text
    supervisor_mode: Literal["scripted", "llm"]
    initial_evidence: bool
    pause_for_input: bool
    model_settings: dict[str, Any] | None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compilation with a scripted or real Supervisor and scripted professional agents.",
    )
    parser.add_argument("command", choices=("run", "resume"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--compilation-id", required=True)
    parser.add_argument("--supervisor", choices=("scripted", "llm"),
                        help="run default: scripted; resume uses saved mode")
    parser.add_argument("--config", type=Path, help="TOML model configuration for a new llm run")
    parser.add_argument("--goal", help="Goal for a new run; defaults to the synthetic task")
    parser.add_argument("--initial-evidence", action="store_true")
    parser.add_argument("--pause-for-input", action="store_true",
                        help="Scripted Supervisor demonstration only")
    parser.add_argument("--answer")
    args = parser.parse_args(argv)
    model_settings = None
    close_clients = None
    try:
        directory = args.directory.resolve()
        options_path = directory / "run.json"
        if args.command == "run":
            if args.answer is not None:
                raise ValueError("--answer is only valid for resume")
            if options_path.exists() or (directory / "checkpoints.sqlite").exists():
                raise ValueError("This directory already contains a run; use resume")
            mode = args.supervisor or "scripted"
            if mode == "scripted" and args.config is not None:
                raise ValueError("--config requires --supervisor llm")
            if mode == "llm" and args.pause_for_input:
                raise ValueError("--pause-for-input belongs to the scripted Supervisor")
            if mode == "llm":
                from medbenchagent.runtime.model_services import resolve_model_service
                from medbenchagent.settings import get_llm_api_key, load_supervisor_settings

                model_settings = load_supervisor_settings(args.config)
                selected = resolve_model_service(
                    "Chat", model_settings.llm.base_url, model_settings.llm.model, get_llm_api_key(),
                )
                model_settings = replace(
                    model_settings, llm=replace(model_settings.llm, model=selected),
                )
            options = RunOptions(
                compilation_id=args.compilation_id,
                goal=args.goal if args.goal is not None else DEFAULT_DEMO_GOAL,
                supervisor_mode=mode, initial_evidence=args.initial_evidence,
                pause_for_input=args.pause_for_input,
                model_settings=model_settings.snapshot() if model_settings else None,
            )
            directory.mkdir(parents=True, exist_ok=True)
            with options_path.open("x", encoding="utf-8") as stream:
                stream.write(options.model_dump_json(indent=2) + "\n")
        else:
            if (args.initial_evidence or args.pause_for_input or args.supervisor is not None
                    or args.config is not None or args.goal is not None):
                raise ValueError("Resume uses the original mode, model, goal and options")
            options = RunOptions.model_validate_json(options_path.read_text(encoding="utf-8"))
            if options.compilation_id != args.compilation_id:
                raise ValueError("The directory belongs to another Compilation")
            if not (directory / "checkpoints.sqlite").is_file():
                raise ValueError("No saved checkpoint exists")
            if options.supervisor_mode == "llm":
                from medbenchagent.settings import SupervisorSettings, load_project_env

                # Credentials may be rotated; saved model parameters are not overridden by env.
                load_project_env()
                if options.model_settings is None:
                    raise ValueError("The llm run has no saved model settings")
                model_settings = SupervisorSettings.from_snapshot(options.model_settings)
            elif options.model_settings is not None:
                raise ValueError("A scripted run cannot contain model settings")
        if options.supervisor_mode == "llm":
            from medbenchagent.runtime.llm import clear_client_cache
            from .supervisor import ModelSupervisor

            close_clients = clear_client_cache
            if options.pause_for_input:
                raise ValueError("The llm Supervisor decides when user input is needed")
            decide = ModelSupervisor(directory, model_settings)
        else:
            decide = scripted_supervisor(directory, pause_for_input=options.pause_for_input)
        context = CompilationContext(directory=directory, decide=decide)
        identifier = options.compilation_id
        config = {"configurable": {"thread_id": identifier}, "recursion_limit": 60}
        with SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
            graph = build_compilation_graph(demo_agents(), checkpointer=saver)
            if args.command == "run":
                inputs = create_synthetic_inputs(
                    directory, identifier, initial_evidence=options.initial_evidence,
                )
                invocation = {
                    "compilation_id": identifier, "goal": options.goal,
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
                "compilation_id": identifier, "supervisor_mode": options.supervisor_mode,
                "agent_mode": "scripted",
                "model": model_settings.llm.model if model_settings else None,
                "status": state.get("status"), "summary": state.get("summary", ""),
                "history": state.get("history", []), "artifacts": state.get("artifacts", []),
                "questions": [item.value for task in snapshot.tasks for item in task.interrupts],
            }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    finally:
        if close_clients is not None:
            close_clients()


if __name__ == "__main__":
    main()
