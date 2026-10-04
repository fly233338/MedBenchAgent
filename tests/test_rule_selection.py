import json
from pathlib import Path

from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.planning.extension_selection import discover_extension_candidates, select_extensions
from medbenchagent.planning.extensions import load_extension_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_extension_selection_persists_empty_or_selected_ids(tmp_path, monkeypatch):
    plan = TaskPlanState(dataset={"dataset_id": "breast-us"}, phase="axes")
    plan_path = tmp_path / "task_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    extension_root = tmp_path / "extensions"
    extension_root.mkdir()
    (extension_root / "demo.md").write_text(
        "---\nid: demo\ntype: disease\ndiseases: [demo]\nstages: [axis_analysis]\n---\nDemo extension\n",
        encoding="utf-8",
    )
    catalog = load_extension_catalog(extension_root)
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
    monkeypatch.setattr(extension_selection, "_catalog", lambda _context: catalog)
    monkeypatch.setattr(extension_selection, "_snapshot", lambda _context, _dataset_id, refresh=False: (
        {"registry_digest": "sha256:test", "candidates": {"demo": catalog.get("demo").metadata()}},
        tmp_path / "extension_candidates.json",
    ))
    (tmp_path / "extension_candidates.json").write_text(
        json.dumps({"registry_digest": "sha256:test", "candidates": {"demo": catalog.get("demo").metadata()}}),
        encoding="utf-8",
    )
    result = select_extensions(
        context,
        dataset_id="breast-us",
        extension_ids=["demo"],
        selection_reason={"demo": "Human selected for testing."},
    )
    assert result["updates"]["planning_extension_selections"]["breast-us"]["selected_extension_ids"] == ["demo"]
    assert Path(result["updates"]["extension_selection_path"]).is_file()


def test_discovery_without_extensions_auto_skips_human_selection(tmp_path):
    result = discover_extension_candidates(
        {"operation_directory": str(tmp_path)},
        dataset_id="dataset-a",
    )
    assert result["summary"] == "No planning extensions are registered"
    selection = result["updates"]["planning_extension_selections"]["dataset-a"]
    assert selection["selected_extension_ids"] == []
    assert selection["selection_reason"] == {}
