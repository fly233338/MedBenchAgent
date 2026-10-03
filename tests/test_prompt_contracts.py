from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_data_preparation_prompt_describes_field_dispositions():
    prompt = (ROOT / "src/medbenchagent/prompts/data_preparation_agent.md").read_text(encoding="utf-8")

    assert "Case medical evidence" in prompt
    assert "Operational or administrative noise" in prompt
    assert "question" in prompt.lower()
    assert "mapping.json" in prompt


def test_task_planning_prompt_covers_visual_semantics_and_template_contract():
    source = (ROOT / "src/medbenchagent/agents/task_planning.py").read_text(encoding="utf-8")

    assert "patient or clinical context" in source
    assert "device/acquisition/administrative metadata" in source
    assert "question_template_count" in source
    prompt = (ROOT / "config/bir_template/registry/agent_prompts/task_planning.md").read_text(encoding="utf-8")
    for concept in ("Task Axis", "Axis Value", "Task Scenario", "Leaf Task"):
        assert concept in prompt
    assert "Resolver.inputs must be exactly identical to Leaf.input" in prompt
    assert "The model does not construct Task Scenarios" in prompt
    assert "Axis Phase Protocol" in prompt
    assert "next_axis` is `clinical_target`" in prompt
    assert "`analyze_axis` exactly once" in prompt
    assert "do not repeatedly request the same page" in prompt
    assert "discarded_leaf_tasks" in source
