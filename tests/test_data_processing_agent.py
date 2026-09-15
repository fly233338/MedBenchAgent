import json
from pathlib import Path
import pytest
from medbenchagent.agents import data_processing as preparation
from medbenchagent.agents import data_pipeline as pipeline
from medbenchagent.runtime.dependencies import publish


def _context(tmp_path):
    context = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op")}
    for kind in ("Mapping", "ImageInventory"):
        publish(context, kind, {"path": "source"}, ())
    return context


def test_invalid_annotation_script_is_repairable(tmp_path):
    context = _context(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"mapping": {"authoritative_population": "images", "expected_images": 1,
        "entity_meanings": {}, "evidence": ["README"]}, "fallback_reason": "custom binary source"}), encoding="utf-8")
    context["preparation_plan_path"] = str(path)
    with pytest.raises(ValueError, match="Invalid Python"):
        pipeline.write_fallback(context, code="if :")


def test_execution_alone_cannot_publish_data(tmp_path):
    with pytest.raises(ValueError, match="Harmonization"):
        pipeline.finish(_context(tmp_path))


def test_ambiguous_display_requires_recorded_decision(tmp_path):
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
    context["decisions"] = {
        json.dumps(["DataProcessingAgent", "plan:1", "window?"]): "Use 0 to 100"
    }
    output = preparation.save_image_plan(context, overrides={"im": policy})
    assert output["status"] == "success"


def test_container_uses_fixed_image_readonly_mounts_and_no_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    context = _context(tmp_path)
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

    monkeypatch.setattr(preparation.subprocess, "run", run)
    preparation._run(context, output, request={"operation": "table", "path": "a.csv"})
    command = commands[-1]
    assert "--network=none" in command and "--read-only" in command
    assert "--memory=4g" in command and "sha256:fixed" in command
    assert any("target=/source,readonly" in item for item in command)
    assert any("target=/project,readonly" in item for item in command)
    before = len(commands)
    preparation._run(context, output, request={"operation": "table", "path": "a.csv"})
    assert len(commands) == before + 1  # image identity checked, container not rerun
    with pytest.raises(ValueError, match="dependencies changed"):
        preparation._run(context, output, request={"operation": "table", "path": "b.csv"})
