import shutil
from pathlib import Path
from compilation_fixtures import planning_context
from medbenchagent.planning.bir import BIRStore
from medbenchagent.planning.extension_selection import discover_extension_candidates, select_extensions


def test_extension_selection_persists_in_its_canonical_document(tmp_path):
    template = tmp_path / "template"
    shutil.copytree("config/bir_template", template)
    extensions = template / "registry/planning_extensions"
    extensions.mkdir(exist_ok=True)
    (extensions / "demo.md").write_text(
        "---\nid: demo\ntype: disease\ndiseases: [demo]\nstages: [axis_analysis]\n---\nDemo extension\n", encoding="utf-8")
    context = planning_context(tmp_path / "compilation", template=template)
    discover_extension_candidates(context)
    outcome = select_extensions(context, extension_ids=["demo"], selection_reason={"demo": "Human selected"})
    assert Path(outcome["updates"]["extension_selection_path"]).is_file()
    assert BIRStore(context["bir_path"]).load("planning_extensions.json")["selected_extension_ids"] == ["demo"]
    assert "planning_extension_selections" not in outcome["updates"]


def test_discovery_without_extensions_records_empty_selection(tmp_path):
    context = planning_context(tmp_path)
    result = discover_extension_candidates(context)
    assert result["summary"] == "No planning extensions are registered"
    selection = BIRStore(context["bir_path"]).load("planning_extensions.json")
    assert selection["selected_extension_ids"] == [] and selection["selection_reason"] == {}
