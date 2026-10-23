"""Independent CLI: load a run, assemble selected implementations, and invoke LangGraph."""

import argparse
import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .assembly import assemble_compilation
from .contracts import WorkRecord, WorkRequest
from .artifacts import register_file
from medbenchagent.data.dataset import snapshot_dataset
from .demo import DEFAULT_DEMO_GOAL, create_synthetic_inputs
from .graph import build_compilation_graph
from .interfaces import record_summary
from .run import create_run, resume_run


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compilation with native Supervisor/Planner and optional OpenHands Coding.",
    )
    parser.add_argument("command", choices=("run", "resume"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--compilation-id", required=True)
    parser.add_argument("--supervisor", choices=("scripted", "llm"),
                        help="run default: scripted; resume uses saved mode")
    parser.add_argument("--coding", choices=("scripted", "openhands"),
                        help="run default: scripted; resume uses saved mode")
    parser.add_argument("--planner", choices=("scripted", "llm"), help="run default: scripted")
    parser.add_argument("--dataset-manifest", type=Path, help="Canonical Dataset manifest to freeze for a new run")
    parser.add_argument("--config", type=Path, help="TOML configuration for a new real Supervisor or Coding run")
    parser.add_argument("--goal", help="Goal for a new run; defaults to the synthetic task")
    parser.add_argument("--initial-evidence", action="store_true")
    parser.add_argument("--pause-for-input", action="store_true", help="Scripted Supervisor demonstration only")
    parser.add_argument("--answer")
    args = parser.parse_args(argv)
    try:
        directory = args.directory.resolve()
        if args.command == "run":
            if args.answer is not None:
                raise ValueError("--answer is only valid for resume")
            options = create_run(
                directory, compilation_id=args.compilation_id,
                goal=args.goal if args.goal is not None else (
                    "Discover and publish an evidence-grounded semantic Task Plan for the supplied Dataset."
                    if args.dataset_manifest is not None or args.planner == "llm" else DEFAULT_DEMO_GOAL),
                supervisor_mode=args.supervisor or "scripted", coding_mode=args.coding or "scripted",
                planner_mode=args.planner or "scripted", dataset_manifest=args.dataset_manifest,
                config_path=args.config, initial_evidence=args.initial_evidence,
                pause_for_input=args.pause_for_input,
            )
        else:
            if (args.initial_evidence or args.pause_for_input or args.supervisor is not None
                    or args.coding is not None or args.config is not None or args.goal is not None):
                raise ValueError("Resume uses the original mode, model, goal and options")
            if args.planner is not None or args.dataset_manifest is not None:
                raise ValueError("Resume uses the original Planner and Dataset")
            options = resume_run(directory, args.compilation_id)
        identifier = options.compilation_id
        config = {"configurable": {"thread_id": identifier}, "recursion_limit": 60}
        with assemble_compilation(options, directory) as (agents, context), \
                SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
            graph = build_compilation_graph(agents, checkpointer=saver)
            if args.command == "run":
                # Synthetic input creation belongs to this demo entrypoint, never Assembly.
                if options.dataset_manifest is not None:
                    request = WorkRequest(compilation_id=identifier, request_id=f"{identifier}-input",
                                          agent_id="input", objective="Freeze the input Dataset")
                    path = snapshot_dataset(Path(options.dataset_manifest),
                                            directory / "artifacts" / request.request_id / "dataset")
                    inputs = [register_file(directory, request, path.relative_to(directory).as_posix(), kind="dataset")]
                else:
                    inputs = create_synthetic_inputs(directory, identifier, initial_evidence=options.initial_evidence)
                invocation = {"compilation_id": identifier, "goal": options.goal,
                              "artifacts": [ref.model_dump(mode="json") for ref in inputs]}
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
                "coding_mode": options.coding_mode, "planner_mode": options.planner_mode,
                "supervisor_model": options.supervisor_settings["llm"]["model"] if options.supervisor_settings else None,
                "coding_model": options.coding_settings["llm"]["model"] if options.coding_settings else None,
                "planner_model": options.planner_settings["llm"]["model"] if options.planner_settings else None,
                "status": state.get("status"), "summary": state.get("summary", ""),
                "history": [record_summary(WorkRecord.model_validate(row)) for row in state.get("history", [])],
                "artifacts": state.get("artifacts", []), "workstreams": state.get("workstreams", []),
                "questions": [item.value for task in snapshot.tasks for item in task.interrupts],
            }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
