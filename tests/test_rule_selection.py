import json
from pathlib import Path

from medbenchagent.agents import task_planning
from medbenchagent.planning.bir import RegistrySnapshot
from medbenchagent.planning.task_space import TaskPlanState

ROOT = Path(__file__).resolve().parents[1]


def test_rule_selection_is_persisted_once_for_dataset(tmp_path, monkeypatch):
    plan = TaskPlanState(dataset={"dataset_id": "breast-us"}, phase="rule_selection")
    plan_path = tmp_path / "task_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    captured = {}

    def fake_complete_json(prompt, user_prompt, schema):
        captured["payload"] = json.loads(user_prompt)
        return {"selected_rule_ids": ["breast-cancer-birads-boundary"], "reason": "Breast ultrasound dataset."}

    monkeypatch.setattr(task_planning, "_registry", lambda _context: registry)
    monkeypatch.setattr(task_planning, "complete_json", fake_complete_json)
    context = {
        "operation_directory": str(tmp_path),
        "task_plan_path": str(plan_path),
        "dataset_entry": {"dataset_id": "breast-us", "name": "Breast Ultrasound", "description": "Breast cancer ultrasound"},
        "benchmark_info": {"diseases": ["breast_cancer"], "modalities": ["ultrasound"]},
    }

    result = task_planning.select_planning_rules(context)
    saved = task_planning._load_plan(context)
    assert saved.phase == "axes"
    assert "clinical-target-boundaries" in saved.rule_selection["selected_rule_ids"]
    assert "breast-cancer-birads-boundary" in saved.rule_selection["selected_rule_ids"]
    assert captured["payload"]["dataset"]["name"] == "Breast Ultrasound"
    assert Path(result["updates"]["rule_selection_path"]).is_file()
