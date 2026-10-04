from pathlib import Path

import pytest

from medbenchagent.planning.bir import RegistrySnapshot
from medbenchagent.planning.rules import load_rule_catalog, rule_selection_digest

ROOT = Path(__file__).resolve().parents[1]


def test_registry_loads_frozen_rules_and_renders_common_first():
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    catalog = registry.planning_rules()
    assert "clinical-target-boundaries" in catalog.ids()
    rendered = catalog.render(["ultrasound-spatial-calibration"], stage="scenario_screening")
    assert rendered.index("clinical-target-boundaries") < rendered.index("ultrasound-spatial-calibration")
    assert "{{PLANNING_RULES}}" not in rendered
    assert rule_selection_digest(catalog, ["ultrasound-spatial-calibration"]).startswith("sha256:")


def test_rule_catalog_rejects_duplicate_ids_and_invalid_stages(tmp_path):
    root = tmp_path / "rules"
    root.mkdir()
    body = "---\nid: duplicate\ndescription: One\nstages: [axis_analysis]\nscope: common\n---\nBody\n"
    (root / "a.md").write_text(body, encoding="utf-8")
    (root / "b.md").write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_rule_catalog(root)

    (root / "b.md").write_text(
        body.replace("id: duplicate", "id: invalid").replace("axis_analysis", "unknown_stage"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid stages"):
        load_rule_catalog(root)
