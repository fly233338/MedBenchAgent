from pathlib import Path
import json
import pytest
from medbenchagent.agents import task_planning as planning
from medbenchagent.runtime.dependencies import publish, require


def _context(tmp_path):
    context = {
        "run_directory": str(tmp_path),
        "operation_directory": str(tmp_path / "op"),
        "benchmark_info": {"sample_count": 4},
    }
    for kind in (
        "Requirement",
        "DataConfig",
        "Dataset",
        "Materials",
        "PlanningConfig",
        "PlanningDraft",
    ):
        publish(context, kind, {"value": 1}, ())
    return context


def test_allocation_preserves_design_and_publishes_new_draft(tmp_path):
    context = _context(tmp_path)
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "view": {
                        "category": "image_acquisition_and_basic_information",
                        "generation_logic": "Recognize",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    context["working_blueprint_path"] = str(path)
    publish(context, "BlueprintDraft", {"path": str(path)}, ("PlanningDraft",))
    output = planning.allocate(context)
    blueprint = json.loads(Path(output["updates"]["working_blueprint_path"]).read_text())
    assert blueprint["tasks"]["view"]["sample_count"] == 4
    assert blueprint["tasks"]["view"]["generation_logic"] == "Recognize"


def test_draft_revision_invalidates_published_plan(tmp_path):
    context = _context(tmp_path)
    publish(context, "BlueprintDraft", {"path": "first"}, ("PlanningDraft",))
    publish(context, "Plan", {"path": "plan"}, ("BlueprintDraft",))
    publish(context, "PlanningDraft", {"path": "revised"}, ())
    with pytest.raises(ValueError, match="stale"):
        require(context, "Plan")


def test_family_screening_only_sends_selected_fields_and_preserves_other_drafts(
    tmp_path, monkeypatch
):
    context = _context(tmp_path)
    base = {"tasks": {"a": {"category": "one"}, "b": {"category": "two"}}}
    profile = {
        "total_images": 3,
        "fields": {"chosen": {"distinct_value_count": 2}, "unrelated": {}},
        "structures": [],
    }
    monkeypatch.setattr(planning, "_base", lambda c: base)
    monkeypatch.setattr(planning, "_profile", lambda c: profile)

    def screen(blueprint, subset, **kwargs):
        assert set(blueprint["tasks"]) == {"a"}
        assert set(subset["fields"]) == {"chosen"}
        return {"tasks": {"a": {"decision": "drop", "reason": "unsupported"}}}

    monkeypatch.setattr(planning, "generate_task_plan", screen)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps({"tasks": {"b": {"decision": "retain"}}}))
    context["draft_path"] = str(path)
    output = planning.screen_family(context, category="one", fields=["chosen"])
    draft = json.loads(Path(output["updates"]["draft_path"]).read_text())
    assert set(draft["tasks"]) == {"a", "b"}
    assert draft["tasks"]["b"] == {"decision": "retain"}
