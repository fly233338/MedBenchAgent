import json
from pathlib import Path
import pytest
from medbenchagent.agents import data_processing as preparation
from medbenchagent.data import preparation_services as pipeline
from medbenchagent.data import execution
from medbenchagent.runtime.dependencies import publish


def _context(tmp_path):
    context = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op")}
    for kind in ("Mapping", "ImageInventory"):
        publish(context, kind, {"path": "source"}, ())
    return context


def test_invalid_dataset_adapter_is_repairable(tmp_path, monkeypatch):
    from test_preparation_evidence import grounded_spec, setup_source

    context = setup_source(tmp_path, monkeypatch)
    context.update(pipeline.save_semantic_spec(context, spec=grounded_spec(context))["updates"])
    with pytest.raises(ValueError, match="Invalid Python"):
        pipeline.save_dataset_adapter(context, code="if :")
    assert pipeline.save_dataset_adapter(context, code="pass")["status"] == "success"


def test_preparation_observation_tracks_readiness_and_runtime(tmp_path, monkeypatch):
    from test_preparation_evidence import grounded_spec, setup_source
    context = setup_source(tmp_path, monkeypatch)
    context.update(pipeline.save_semantic_spec(context, spec=grounded_spec(context))["updates"])
    assert "save_dataset_adapter" in pipeline.available_tools(context)
    context["execution_runtime"]["requested_backend"] = "auto"
    observed = pipeline.describe(context)
    assert observed["source_readiness"]["one"]["SemanticPreparationSpec"]
    assert observed["pending_source_ids"] == ["one"]
    assert not observed["completion"]["can_finish"]
    assert not observed["execution_capabilities"]["generated_python"]
    assert "save_dataset_adapter" not in pipeline.available_tools(context)
    with pytest.raises(ValueError, match="explicit backend=local"):
        pipeline.save_dataset_adapter(context, code="pass")
    publish(context, "SourceFingerprint", {"revision": "changed"}, ("Source",))
    assert not pipeline.describe(context)["source_readiness"]["one"]["SemanticPreparationSpec"]


def test_execution_alone_cannot_publish_data(tmp_path):
    with pytest.raises(ValueError, match="Harmonization"):
        pipeline.finish(_context(tmp_path))


def test_ambiguous_display_requires_recorded_decision(tmp_path, monkeypatch):
    context = _context(tmp_path)
    path = tmp_path / "images.json"
    path.write_text(
        json.dumps(
            {"im": {"copy": False, "preserve_supported": False, "window": None, "invert": False}}
        )
    )
    context["image_inventory_path"] = str(path)
    with pytest.raises(ValueError, match="ask_user"):
        preparation.save_image_plan(context, overrides={})
    policy = {
        "encoding": "window",
        "low": 0.0,
        "high": 100.0,
        "invert": False,
        "decision_reference": "plan:1",
    }
    with pytest.raises(ValueError, match="answered user decision"):
        preparation.save_image_plan(context, overrides={"im": policy})
    from medbenchagent.runtime.clarifications import new_issue, record_answer, resolve_semantic_answer
    context.update(source_scope="s", sources=[{"id": "s"}], agent_role="DataProcessingAgent")
    publish(context, "Source", {}, ())
    publish(context, "ImageInventory", {"path": str(path)}, ("Source",))
    issue = new_issue("DataProcessingAgent", context, "display", "Window for im?", image_ids=["im"])
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", lambda *a: {"policies": {"im": {
        "encoding": "window", "low": 0.0, "high": 100.0, "invert": False}}})
    issue = resolve_semantic_answer(context, record_answer(context, issue, "Use window 0 to 100 without inversion"))
    context["issues"] = {issue["issue_id"]: issue}
    policy["decision_reference"] = issue["issue_id"]
    output = preparation.save_image_plan(context, overrides={"im": policy})
    assert output["status"] == "success"


def test_container_uses_fixed_image_readonly_mounts_and_no_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    context = _context(tmp_path)
    context["execution_runtime"] = {"backend": "docker", "requested_backend": "auto",
                                     "docker_server_version": "test"}
    context["source_root"] = str(tmp_path / "source")
    Path(context["source_root"]).mkdir()
    output = tmp_path / "output"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="sha256:fixed\n", stderr="")
        (output / "result.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execution.subprocess, "run", run)
    preparation._run(context, output, request={"operation": "table", "path": "a.csv"})
    command = commands[-1]
    assert "--network=none" in command and "--read-only" in command
    assert "--memory=4g" in command and "sha256:fixed" in command
    assert any("target=/source,readonly" in item for item in command)
    assert any("target=/project,readonly" in item for item in command)
    before = len(commands)
    preparation._run(context, output, request={"operation": "table", "path": "a.csv"})
    assert len(commands) == before  # frozen component identity reuses the completed operation
    with pytest.raises(ValueError, match="dependencies changed"):
        preparation._run(context, output, request={"operation": "table", "path": "b.csv"})
