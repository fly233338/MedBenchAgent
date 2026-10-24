from medbenchagent.cli import _parser


def test_parser_exposes_terminal_commands():
    parser = _parser()
    args = parser.parse_args(["run", "--source-manifest", "sources.json", "--goal-kind", "plan", "--plan-level", "scenarios"])
    assert args.goal_kind == "plan" and args.plan_level == "scenarios"
    assert parser.parse_args(["resume", "--directory", "saved", "--answer", "yes"]).answer == "yes"
    assert parser.parse_args(["register", "--dataset-id", "example", "--manifest", "dataset.json"]).kind == "dataset"


def test_cli_starts_saved_run_without_resetting_its_draft(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from medbenchagent.cli import main
    from medbenchagent.runtime import sessions
    from medbenchagent.runtime.intake import InputDraft, connection_values
    from medbenchagent.compilation.run import RunOptions, verify_input
    import medbenchagent.runtime.model_services as models
    import pytest
    monkeypatch.setattr(models, "resolve_model_service", lambda *_: "offline")
    source = tmp_path / "raw"
    source.mkdir()
    record = sessions.create_session("saved", root=tmp_path / "runs", runtime_root=tmp_path / "runtime")
    directory = Path(record["directory"])
    sessions.save_draft(directory, InputDraft(goal="Keep this goal", source_dirs=[str(source)],
                        sample_count=7, language="zh", allow_limited_delivery=False))
    main(["run", "--directory", str(directory), "--prepare-only"])
    options = RunOptions.model_validate(json.loads((directory / "run.json").read_text(encoding="utf-8")))
    assert options.goal == "Keep this goal" and options.benchmark_request == {"sample_count": 7, "language": "zh"}
    assert not options.goal_spec.allow_limited_delivery
    config = directory / "config.toml"
    config.write_text(config.read_text(encoding="utf-8").replace('model = ""', 'model = "different"'), encoding="utf-8")
    assert connection_values(config)["model"] == "offline"
    with Path(options.source_manifest).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="manifest changed"):
        verify_input(options)


def test_batch_creates_independent_runs_with_configured_defaults(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import pytest
    import tomlkit
    from medbenchagent import cli
    from medbenchagent.data.dataset_catalog import register_dataset
    from medbenchagent.data.task_data import SourceManifest
    from medbenchagent.runtime import sessions
    from medbenchagent.settings import DEFAULT_CONFIG_PATH
    import medbenchagent.runtime.model_services as models
    monkeypatch.setattr(models, "resolve_model_service", lambda *_: "offline")
    document = tomlkit.parse(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["paths"] = {"runs_root": str(tmp_path / "runs"), "runtime_root": str(tmp_path / "runtime/runs")}
    document["benchmark"]["default_language"] = "zh"
    config = tmp_path / "config.toml"
    config.write_text(tomlkit.dumps(document), encoding="utf-8")
    monkeypatch.setenv("MEDBENCH_CONFIG", str(config))
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        source = SourceManifest(dataset={"dataset_id": name, "revision": "r1"}, sources=[{"source_id": "raw", "root": str(root)}])
        path = root / "sources.json"
        path.write_text(source.model_dump_json(), encoding="utf-8")
        register_dataset(name, path, kind="source")
    monkeypatch.setattr(cli, "execute_compilation", lambda directory: iter(()))
    with pytest.raises(SystemExit) as stopped:
        cli.main(["batch", "first", "second", "--sample-count", "2"])
    assert stopped.value.code == 0
    records = sessions.list_runs(tmp_path / "runs")
    assert len(records) == 2 and len({row["runtime_directory"] for row in records}) == 2
    for row in records:
        options = json.loads((Path(row["directory"]) / "run.json").read_text(encoding="utf-8"))
        assert options["benchmark_request"] == {"sample_count": 2, "language": "zh"}
