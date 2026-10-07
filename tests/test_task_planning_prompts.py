from pathlib import Path
from types import SimpleNamespace

from medbenchagent.agents import task_planning
from medbenchagent.planning.bir import RegistrySnapshot
from medbenchagent.planning.task_space import TaskPlanState, TaskScenario

ROOT = Path(__file__).resolve().parents[1]


def test_core_prompts_are_complete_and_have_one_final_extension_slot():
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    for name in ("axis_analysis", "scenario_screening", "leaf_task_design", "scenario_review"):
        prompt = registry.prompt(name)
        assert prompt.count("{{DATASET_SPECIFIC_RULES}}") == 1
        assert prompt.rstrip().endswith("{{DATASET_SPECIFIC_RULES}}")
        for old in ("{{TARGET_SEMANTICS}}", "{{EVIDENCE_CONSTRAINTS}}", "{{REFERENCE_CONSTRAINTS}}", "{{DOMAIN_"):
            assert old not in prompt
    assert "BI-RADS values `1–6` alone supporting `quantify`" in registry.prompt("axis_analysis")
    assert "Every Leaf must select exactly one target" in registry.prompt("leaf_task_design")
    assert "For `keep`, provide one or more `validated_clinical_targets`." in registry.prompt("scenario_screening")
    assert "produced through registered deterministic Resolver operations" in registry.prompt("scenario_review")


def test_leaf_instructions_render_registry_context_and_requirements(monkeypatch):
    registry = RegistrySnapshot.from_directory(ROOT / "config" / "bir_template" / "registry")
    scenario = TaskScenario(
        scenario_id="assessment-scenario",
        input_structure="single-image",
        operation="recognize",
        clinical_target="assessment",
        capability_category="Recognition",
    )
    plan = TaskPlanState(
        dataset={"dataset_id": "dataset"},
        phase="leaves",
        next_axis=None,
        current_scenario_id=scenario.scenario_id,
        scenarios={scenario.scenario_id: scenario},
    )
    context = {
        "question_template_count": 3,
        "benchmark_info": {"language": "zh"},
        "benchmark_input_policy": "image_only",
    }
    monkeypatch.setattr(task_planning, "_bir", lambda _: SimpleNamespace(registry=lambda: registry))
    monkeypatch.setattr(task_planning, "_load_plan", lambda _: plan)
    monkeypatch.setattr(task_planning, "_available_tools", lambda _: {"read_scenario_context"})

    prompt = task_planning._planning_instructions(context)

    assert "{{" not in prompt
    assert "# Task Planning Coordinator" in prompt
    assert "# Leaf Task Design" in prompt
    assert '"scenario_id": "assessment-scenario"' in prompt
    assert '"operation": ' in prompt
    assert '"id": "recognize"' in prompt
    assert '"id": "assessment"' in prompt
    assert '"resolver_policy": ' in prompt
    assert '"task_contracts": ' in prompt
    assert "Produce exactly `3`" in prompt
    assert "Benchmark output language:\n\n`zh`" in prompt
    assert "Effective benchmark input policy:\n\n`image_only`" in prompt
    assert task_planning._render_extensions(context, stage="leaf_task_design") in prompt
