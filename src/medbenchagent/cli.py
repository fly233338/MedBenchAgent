"""One CLI for user-facing Runs and the canonical Compilation core."""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from medbenchagent.runtime.intake import InputDraft, SelectedFile, select_dataset
from medbenchagent.runtime.graph_runner import execute_compilation, inspect_compilation
from medbenchagent.runtime import sessions
from medbenchagent.settings import load_paths


def _parser():
    parser = argparse.ArgumentParser(prog="medd2b")
    sub = parser.add_subparsers(dest="command")
    create = sub.add_parser("create", help="Create one Run and editable config")
    create.add_argument("name")
    create.add_argument("--config", type=Path)
    run = sub.add_parser("run", help="Start a Compilation from explicit real inputs")
    run.add_argument("name", nargs="?")
    run.add_argument("--directory", type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--prepare-only", action="store_true", help="Freeze a run for Studio without executing agents")
    run.add_argument("--goal")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--source-manifest", type=Path)
    source.add_argument("--dataset-manifest", type=Path)
    source.add_argument("--dataset", help="Registered dataset ID")
    source.add_argument("--source-dir", type=Path, action="append")
    run.add_argument("--key-file", type=Path, action="append", default=[])
    for command in (run,):
        _goal_arguments(command)
    batch = sub.add_parser("batch", help="Create an independent Run for each registered Dataset")
    batch.add_argument("dataset_ids", nargs="+")
    batch.add_argument("--goal", default="Compile and deliver a traceable benchmark.")
    _goal_arguments(batch)
    resume = sub.add_parser("resume")
    resume.add_argument("--directory", type=Path, required=True)
    resume.add_argument("--answer")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--directory", type=Path, required=True)
    register = sub.add_parser("register")
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--manifest", type=Path, required=True)
    register.add_argument("--kind", choices=("source", "dataset"), default="dataset")
    export = sub.add_parser("export")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    demo = sub.add_parser("demo", help="Create a tiny explicit source fixture; does not call models")
    demo.add_argument("--directory", type=Path, required=True)
    sub.add_parser("tui")
    return parser


def _goal_arguments(parser):
    parser.add_argument("--goal-kind", choices=("profile", "plan", "benchmark"))
    parser.add_argument("--plan-level", choices=("scenarios", "executable_leaves"))
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--no-limited-delivery", action="store_true", default=None)


def _consume(events):
    status = 0
    for event in events:
        print(json.dumps({"run_id": event.run_session_id, "event": event.kind, **event.payload}, ensure_ascii=False))
        if event.kind == "finished" and event.payload.get("goal_result") in {"blocked", "failed"}:
            status = 1
    return status


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {None, "tui"}:
            from medbenchagent.tui.app import MBAApp
            if not sys.stdout.isatty() or __import__("os").getenv("TERM") == "dumb":
                parser.print_help()
                return
            MBAApp().run()
            return
        if args.command == "create":
            print(json.dumps(sessions.create_session(args.name, template=args.config)))
        elif args.command == "run":
            name = args.name or (args.directory.name if args.directory else "run-" + uuid4().hex[:12])
            directory = args.directory.resolve() if args.directory else sessions.run_path(name, load_paths(args.config)["runs_root"])
            if args.name and args.directory and args.name != args.directory.name:
                raise ValueError("Run name must match --directory")
            if not directory.exists():
                sessions.create_session(name, root=directory.parent, template=args.config)
            elif args.config:
                raise ValueError("Existing Runs use their own config.toml")
            record = sessions.load_session(directory)
            draft = InputDraft.model_validate(record["draft"])
            draft.goal = args.goal or draft.goal or "Compile a traceable benchmark."
            if args.goal_kind is not None:
                draft.goal_kind = (args.plan_level or "scenarios") if args.goal_kind == "plan" else args.goal_kind
            elif args.plan_level is not None:
                draft.goal_kind = args.plan_level
            if args.sample_count is not None:
                draft.sample_count = args.sample_count
            if args.no_limited_delivery is not None:
                draft.allow_limited_delivery = not args.no_limited_delivery
            if args.dataset:
                draft = select_dataset(args.dataset, draft)
            elif args.source_dir:
                draft.source_dirs = [str(path.resolve()) for path in args.source_dir]
                draft.key_files = [SelectedFile(path=str(path.resolve())) for path in args.key_file]
                draft.canonical_manifest = draft.source_manifest = None
            elif args.source_manifest:
                from medbenchagent.data.task_data import SourceManifest
                path = args.source_manifest.resolve()
                source = SourceManifest.model_validate_json(path.read_bytes())
                roots = {s.source_id: (path.parent / s.root).resolve() for s in source.sources}
                draft.source_manifest, draft.canonical_manifest = str(path), None
                draft.dataset_id = source.dataset.dataset_id
                draft.source_dirs = [str(root) for root in roots.values()]
                draft.key_files = [SelectedFile(path=str(roots[f.source_id] / f.path), purpose=f.purpose) for f in source.key_files]
            elif args.dataset_manifest:
                draft.canonical_manifest = str(args.dataset_manifest.resolve())
                draft.source_dirs, draft.key_files, draft.source_manifest = [], [], None
            sessions.freeze_session(directory, draft)
            if args.prepare_only:
                print(directory.resolve())
                return
            raise SystemExit(_consume(execute_compilation(directory)))
        elif args.command == "batch":
            if len(set(args.dataset_ids)) != len(args.dataset_ids):
                raise ValueError("Batch dataset IDs must be distinct")
            status = 0
            for identifier in args.dataset_ids:
                name = "run-" + uuid4().hex[:12]
                record = sessions.create_session(name)
                draft = select_dataset(identifier, InputDraft(**record["draft"], goal=args.goal,
                    goal_kind=(args.plan_level or "scenarios") if args.goal_kind == "plan" else (args.goal_kind or "benchmark"),
                    sample_count=args.sample_count, allow_limited_delivery=not args.no_limited_delivery))
                sessions.freeze_session(record["directory"], draft)
                status = max(status, _consume(execute_compilation(record["directory"])))
            raise SystemExit(status)
        elif args.command == "resume":
            raise SystemExit(_consume(execute_compilation(args.directory, resume=True, answer=args.answer)))
        elif args.command == "inspect":
            print(json.dumps(inspect_compilation(args.directory), ensure_ascii=False, indent=2))
        elif args.command == "register":
            from medbenchagent.data.dataset_catalog import register_dataset
            print(json.dumps(register_dataset(args.dataset_id, args.manifest, kind=args.kind)))
        elif args.command == "export":
            from medbenchagent.generation.publication import export_benchmark
            print(export_benchmark(args.source, args.destination))
        elif args.command == "demo":
            from medbenchagent.compilation.demo import create_demo_source
            print(create_demo_source(args.directory))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
