import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "config/bir_template/registry/agent_prompts"


def test_data_preparation_prompt_describes_field_dispositions():
    prompt = (ROOT / "src/medbenchagent/prompts/data_preparation_agent.md").read_text(encoding="utf-8")

    assert "Case medical evidence" in prompt
    assert "Operational or administrative noise" in prompt
    assert "question" in prompt.lower()
    assert "mapping.json" in prompt
    assert "Unannotated does not mean negative" in prompt


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

    assert "Supported means that at least one clinically meaningful task" in axis
    assert "Classify the operation by the primary task requested" in leaf
    assert "Do not infer absence from missing annotations" in leaf
    assert "Tumor masks support finding localization" in screening
    assert "Check whether the question, benchmark-model input, and reference answer cover the same scope" in review
    assert "describe_synthesis_allowed" in axis
    assert "direct report-level or structured narrative reference" in leaf
    assert "benchmark_brief" in source
    assert 'prompt("scenario_screening")' in source


def test_planning_prompts_and_registry_descriptions_are_english():
    paths = [*PROMPTS.glob("*.md"), ROOT / "src/medbenchagent/prompts/data_preparation_agent.md"]
    for path in paths:
        assert not re.search(r"[\u3400-\u9fff]", path.read_text(encoding="utf-8")), path
    axes = json.loads((PROMPTS.parent / "axes.json").read_text(encoding="utf-8"))
    for axis in axes["axes"]:
        assert not re.search(r"[\u3400-\u9fff]", axis.get("description", "")), axis["id"]
        for value in axis.get("values", []):
            assert not re.search(r"[\u3400-\u9fff]", value.get("description", "")), value["id"]
