import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "config/bir_template/registry/agent_prompts"


def test_data_preparation_prompt_describes_field_dispositions():
    prompt = (ROOT / "src/medbenchagent/prompts/data_preparation_agent.md").read_text(encoding="utf-8")

    assert "Case medical evidence" in prompt
    assert "Operational or administrative noise" in prompt
    assert "question" in prompt.lower()
    assert "mapping.json" in prompt
    assert "未标注”不能推成“阴性" in prompt


def test_task_planning_prompt_is_the_orchestrator_prompt():
    prompt = (PROMPTS / "task_planning.md").read_text(encoding="utf-8")
    source = (ROOT / "src/medbenchagent/agents/task_planning.py").read_text(encoding="utf-8")

    assert not (PROMPTS / "task_planning_orchestrator.md").exists()
    assert not (PROMPTS / "scenario_global_review.md").exists()
    assert 'prompt("task_planning")' in source
    assert "task_planning_orchestrator" not in source
    assert "Phase protocol:" in prompt
    assert "review_candidate_scenarios for one pending Scenario" in prompt
    assert "Do not perform the semantic work assigned to operation-specific calls" in prompt


def test_registry_contains_the_canonical_operation_definitions():
    axes = json.loads((PROMPTS.parent / "axes.json").read_text(encoding="utf-8"))
    operation_axis = next(axis for axis in axes["axes"] if axis["id"] == "operation")
    definitions = {item["id"]: item["description"] for item in operation_axis["values"]}

    assert definitions["recognize"].startswith("Identify the presence, absence, identity, type")
    assert definitions["localize"].startswith("Ground a specified anatomical structure")
    assert definitions["quantify"].startswith("Determine a count, magnitude, extent")
    assert definitions["compare"].startswith("Determine similarities, differences, ordering")
    assert definitions["align"].startswith("Establish correspondence or assess concordance")
    assert "An isolated label, label list" in definitions["describe"]


def test_stage_prompts_include_the_direct_semantic_constraints():
    axis = (PROMPTS / "axis_analysis.md").read_text(encoding="utf-8")
    leaf = (PROMPTS / "leaf_task_design.md").read_text(encoding="utf-8")
    screening = (PROMPTS / "scenario_screening.md").read_text(encoding="utf-8")
    review = (PROMPTS / "scenario_review.md").read_text(encoding="utf-8")
    source = (ROOT / "src/medbenchagent/agents/task_planning.py").read_text(encoding="utf-8")

    assert "supported 表示至少存在一种有临床意义" in axis
    assert "Classify the operation by the primary task requested" in leaf
    assert "Do not infer absence from missing annotations" in leaf
    assert "Tumor masks support finding localization" in screening
    assert "检查问句、可见输入、参考答案三者是否同一范围" in review
    assert "benchmark_brief" in source
    assert 'prompt("scenario_screening")' in source
