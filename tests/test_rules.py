from pathlib import Path

import pytest

from medbenchagent.planning.bir import RegistrySnapshot
from medbenchagent.planning.extensions import load_extension_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_registry_loads_extensions_and_renders_by_stage():
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    catalog = registry.planning_extensions()
    assert catalog.ids() == ()
    assert catalog.render([], stage="scenario_screening").startswith("No dataset-specific")


def test_extension_loader_rejects_invalid_metadata(tmp_path):
    root = tmp_path / "extensions"
    root.mkdir()
    body = "---\nid: demo\ntype: disease\nstages: [axis_analysis]\n---\nBody\n"
    (root / "a.md").write_text(body, encoding="utf-8")
    (root / "b.md").write_text(body.replace("id: demo", "id: demo2"), encoding="utf-8")
    assert set(load_extension_catalog(root).ids()) == {"demo", "demo2"}
    (root / "c.md").write_text(body.replace("type: disease", "type: invalid"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid type"):
        load_extension_catalog(root)
    (root / "c.md").write_text(body.replace("stages: [axis_analysis]", "stages: [invalid]"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid stages"):
        load_extension_catalog(root)


def test_extension_selection_rejects_unknown_ids():
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    with pytest.raises(ValueError, match="Unknown planning extension"):
        registry.planning_extensions().selected(["unknown"], stage="axis_analysis")
