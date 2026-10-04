import json
from pathlib import Path

from medbenchagent.planning.bir import RegistrySnapshot
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.planning.extension_selection import select_extensions

ROOT = Path(__file__).resolve().parents[1]


def test_extension_selection_persists_empty_or_selected_ids(tmp_path):
    plan = TaskPlanState(dataset={"dataset_id": "breast-us"}, phase="axes")
    plan_path = tmp_path / "task_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    context = {
        "operation_directory": str(tmp_path),
        "task_plan_path": str(plan_path),
        "bir_path": str(ROOT / "experiments" / "test" / "BIR"),
        "dataset_scope": "breast-us",
        "run_directory": str(tmp_path),
        "artifact_versions": {},
    }
    # Use the checked-in Registry directly for this isolated selection test.
    from medbenchagent.planning import extension_selection
    extension_selection._catalog = lambda _context: registry.planning_extensions()
    extension_selection._snapshot = lambda _context, _dataset_id, refresh=False: (
        {"registry_digest": registry.digest, "candidates": {"breast_cancer": registry.planning_extensions().get("breast_cancer").metadata()}},
        tmp_path / "extension_candidates.json",
    )
    (tmp_path / "extension_candidates.json").write_text(json.dumps({"registry_digest": registry.digest, "candidates": {"breast_cancer": registry.planning_extensions().get("breast_cancer").metadata()}}), encoding="utf-8")
    result = select_extensions(
        context,
        dataset_id="breast-us",
        extension_ids=["breast_cancer"],
        selection_reason={"breast_cancer": "Human selected for breast-cancer planning."},
    )
    assert result["updates"]["planning_extension_selections"]["breast-us"]["selected_extension_ids"] == ["breast_cancer"]
    assert Path(result["updates"]["extension_selection_path"]).is_file()
