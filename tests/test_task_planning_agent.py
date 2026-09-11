import json
import pytest
from medbenchagent.agents.task_planning import allocate, finish

def test_allocation_tool_preserves_task_design(tmp_path):
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps({"tasks": {"view": {"category": "image_acquisition_and_basic_information",
        "description": "Recognize view", "generation_logic": "Recognize"}}}))
    context = {"working_blueprint_path": str(path), "operation_directory": str(tmp_path / "op"),
               "benchmark_info": {"sample_count": 4}}
    output = allocate(context, {})
    blueprint = json.loads(open(output["updates"]["working_blueprint_path"], encoding="utf-8").read())
    assert blueprint["tasks"]["view"]["sample_count"] == 4
    assert blueprint["tasks"]["view"]["generation_logic"] == "Recognize"

def test_planning_finish_requires_complete_design(tmp_path):
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps({"tasks": {"view": {"sample_count": 1}}}))
    with pytest.raises(ValueError, match="medical design"):
        finish({"working_blueprint_path": str(path), "benchmark_info": {"sample_count": 1}}, {})

