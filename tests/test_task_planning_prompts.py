from pathlib import Path

from medbenchagent.planning.bir import RegistrySnapshot

ROOT = Path(__file__).resolve().parents[1]


def test_core_prompts_are_complete_and_have_one_final_extension_slot():
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    for name in ("axis_analysis", "scenario_screening", "leaf_task_design", "scenario_review"):
        prompt = registry.prompt(name)
        assert prompt.count("{{DATASET_SPECIFIC_RULES}}") == 1
        assert prompt.rstrip().endswith("{{DATASET_SPECIFIC_RULES}}")
        for old in ("{{TARGET_SEMANTICS}}", "{{EVIDENCE_CONSTRAINTS}}", "{{REFERENCE_CONSTRAINTS}}", "{{DOMAIN_"):
            assert old not in prompt
    assert "BI-RADS stored as values 1–6 does not by itself support `quantify`." in registry.prompt("axis_analysis")
    assert "Each Leaf Task must select exactly one item from the Scenario's `validated_clinical_targets`." in registry.prompt("leaf_task_design")
    assert "When `decision` is `keep`, at least one validated clinical target is required." in registry.prompt("scenario_screening")
    assert "All required operations must exist in the registered Resolver capabilities." in registry.prompt("scenario_review")
