import json
from pathlib import Path
import pytest
from medbenchagent.agents import data_processing as preparation
from medbenchagent.runtime.dependencies import publish


def _context(tmp_path):
    context = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op")}
    for kind in ("Mapping", "ImageInventory"):
        publish(context, kind, {"path": "source"}, ())
    return context


def test_invalid_annotation_script_is_repairable(tmp_path):
    with pytest.raises(ValueError, match="Invalid Python"):
        preparation.write_annotation_script(_context(tmp_path), code="if :")


def test_execution_alone_cannot_publish_data(tmp_path):
    with pytest.raises(ValueError, match="PreparedDataset"):
        preparation.finish(_context(tmp_path))


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


def test_generated_image_script_is_valid_python(tmp_path, monkeypatch):
    context = _context(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    context.update(image_plan_path=str(plan), prepared_annotation_path="annotation.jsonl")
    publish(context, "ImagePlan", {"path": str(plan)}, ())

    class StopAfterCompile(Exception):
        pass

    def inspect(context, output, *, script, annotation):
        compile(Path(script).read_text(), str(script), "exec")
        raise StopAfterCompile

    monkeypatch.setattr(preparation, "_run", inspect)
    with pytest.raises(StopAfterCompile):
        preparation.execute_images(context)
