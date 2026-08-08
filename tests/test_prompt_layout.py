import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from medbenchagent import task_screening
from medbenchagent.agents import capability, specification
from medbenchagent.rag import graph as rag_graph
from medbenchagent.rag import pdf as rag_pdf


@pytest.fixture(scope="module")
def prompt_content_hashes() -> dict[str, str]:
    path = Path(__file__).parent / "fixtures" / "prompt_content_hashes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_content_unchanged(
    prompt_content_hashes: dict[str, str], name: str, prompt: str
) -> None:
    collapsed = re.sub(r"\s+", " ", prompt).strip()
    assert hashlib.sha256(collapsed.encode("utf-8")).hexdigest() == prompt_content_hashes[name]


def _assert_call_unchanged(
    prompt_content_hashes: dict[str, str],
    name: str,
    call: tuple[str, str],
) -> None:
    _assert_content_unchanged(prompt_content_hashes, f"{name}_system", call[0])
    _assert_content_unchanged(prompt_content_hashes, f"{name}_user", call[1])


def test_task_revision_and_rag_design_prompts_only_change_whitespace(
    prompt_content_hashes: dict[str, str], monkeypatch
) -> None:
    revision = task_screening._task_revision_prompt(
        "view",
        {"response_format": "letter", "description": "Identify view."},
        {
            "decision": "retain",
            "supporting_fields": ["image.annotation.view"],
            "metric": "accuracy",
            "reason": "Present.",
        },
        {
            "fields": {
                "image.annotation.view": {
                    "top_values": ["CC", "MLO"],
                    "distinct_value_count": 2,
                }
            }
        },
        "Use clearer wording.",
    )

    _assert_content_unchanged(prompt_content_hashes, "task_revision", revision)
    assert "Return JSON only.\n\nReturn decision" in revision
    assert "do not qualify.\n\nfinding_presence" in revision
    assert "\n\nTask ID: view\nTask:" in revision
    assert "\n\nCurrent task plan:" in revision
    assert "\n\nUser feedback:" in revision
    assert "\n\nDataset profile:" in revision

    monkeypatch.setattr(
        rag_pdf,
        "get_settings",
        lambda: SimpleNamespace(generation=SimpleNamespace(question_template_count=3)),
    )
    task = {
        "response_format": "letter",
        "description": "Identify view.",
        "supporting_fields": ["image.annotation.view"],
    }
    system, user = rag_pdf._design_prompt(
        {"language": "en", "disease": "breast cancer"},
        "view",
        task,
        "[chunk-1, guide.pdf p.1] Clinical excerpt.",
        "Use concise wording.",
        {"question_templates": ["Which view?"]},
        {
            "image.annotation.view": {
                "values": [
                    {"value": "CC", "count": 2},
                    {"value": "MLO", "count": 2},
                ]
            }
        },
        allow_value_request=False,
    )

    _assert_call_unchanged(prompt_content_hashes, "rag_design", (system, user))
    assert "benchmark task.\n\nEvidence priority:\n(1)" in system
    assert "task may assess;\n(2)" in system
    assert "answer content;\n(3)" in system
    assert "complete design object and no other fields:\n{" in system
    assert "response format.\n\ngeneration_logic" in system
    assert "bbox fields.\n\nThe supplied task" in system
    assert "syntax only.\n\nLetter option rules:" in system
    assert "task valid.\n\nAnswer-mapping rules:" in system
    assert "\n\nTask ID: view\nTask:" in user
    assert "\n\nRequested revision:" in user
    assert "\n\nApproved field-value summaries:" in user
    assert "\n\nPDF excerpts:" in user


def test_capability_action_and_repair_prompts_only_change_whitespace(
    prompt_content_hashes: dict[str, str], monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def confirm(system: str, user: str) -> dict[str, str]:
        calls.append((system, user))
        return {"action": "confirm"}

    monkeypatch.setattr(capability, "complete_json", confirm)
    capability._dataset_review_action(
        "accept",
        [
            {
                "id": "demo",
                "name": "Demo",
                "description": "Dataset.",
                "annotation_path": "unused",
                "available": True,
            }
        ],
    )
    _assert_call_unchanged(prompt_content_hashes, "cap_dataset", calls[-1])
    assert "available dataset id\"}\n\nConfirm only" in calls[-1][1]
    assert "another available dataset.\n\nDataset catalog:" in calls[-1][1]
    assert "\n\nUser response: accept" in calls[-1][1]

    capability._review_action("accept", ["view", "report"])
    _assert_call_unchanged(prompt_content_hashes, "cap_task_plan", calls[-1])
    assert "requested change\"}\n\nConfirm only" in calls[-1][1]
    assert "change in feedback.\n\nAvailable task ids:" in calls[-1][1]
    assert "\n\nUser response: accept" in calls[-1][1]

    monkeypatch.setattr(
        capability,
        "get_settings",
        lambda: SimpleNamespace(
            category_weights={"image_acquisition_and_basic_information": 0.1}
        ),
    )
    sample_responses = iter([{}, {"action": "confirm"}])

    def sample_response(system: str, user: str) -> dict[str, object]:
        calls.append((system, user))
        return next(sample_responses)

    monkeypatch.setattr(capability, "complete_json", sample_response)
    capability._sample_count_review_action(
        "accept",
        {"view": 2},
        2,
        {"view": {"category": "image_acquisition_and_basic_information"}},
    )
    _assert_call_unchanged(prompt_content_hashes, "cap_sample_action", calls[-2])
    _assert_call_unchanged(prompt_content_hashes, "cap_sample_repair", calls[-1])
    assert "task id\":0}}\n\nFor adjust" in calls[-2][1]
    assert "non-negative integers.\n\nBenchmark total:" in calls[-2][1]
    assert "Current allocation: {\"view\": 2}\n\nUser response:" in calls[-2][1]
    assert "User response: accept\n\nPrevious response:" in calls[-1][1]
    assert "Previous response: {}\n\nContract error:" in calls[-1][1]

    material_responses = iter([{}, {"selected_files": ["Guideline/guide.pdf"]}])

    def material_response(system: str, user: str) -> dict[str, object]:
        calls.append((system, user))
        return next(material_responses)

    monkeypatch.setattr(capability, "complete_json", material_response)
    capability._material_selection_action(
        "choose guide",
        [{"category": "Guideline", "file": "Guideline/guide.pdf"}],
    )
    _assert_call_unchanged(prompt_content_hashes, "cap_material_action", calls[-2])
    _assert_call_unchanged(prompt_content_hashes, "cap_material_repair", calls[-1])
    assert "relative PDF path\"]}.\n\nSelect one or more files" in calls[-2][1]
    assert "invent filenames.\n\nScanned medical references:" in calls[-2][1]
    assert "\n\nUser response: choose guide" in calls[-2][1]
    assert "User response: choose guide\n\nPrevious response:" in calls[-1][1]
    assert "Previous response: {}\n\nContract error:" in calls[-1][1]


def test_specification_and_design_review_prompts_only_change_whitespace(
    prompt_content_hashes: dict[str, str], monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    def confirm(system: str, user: str) -> dict[str, str]:
        calls.append((system, user))
        return {"action": "confirm"}

    monkeypatch.setattr(specification, "complete_json", confirm)
    specification._benchmark_review_action(
        "accept", {"disease": "breast cancer", "sample_count": 2}
    )
    _assert_call_unchanged(prompt_content_hashes, "spec_review", calls[-1])
    assert "requested correction\"}\n\nConfirm only" in calls[-1][1]
    assert "correction in feedback.\n\nCurrent BenchmarkInfo:" in calls[-1][1]
    assert "\n\nUser response: accept" in calls[-1][1]

    review_responses = iter([{}, {"action": "confirm"}])

    def review_response(system: str, user: str) -> dict[str, object]:
        calls.append((system, user))
        return next(review_responses)

    monkeypatch.setattr(rag_graph, "complete_json", review_response)
    rag_graph._design_review_action(
        "accept",
        {
            "view": {
                "response_format": "letter",
                "description": "Identify view.",
                "supporting_fields": ["image.annotation.view"],
            }
        },
    )
    _assert_call_unchanged(prompt_content_hashes, "rag_review_action", calls[-2])
    _assert_call_unchanged(prompt_content_hashes, "rag_review_repair", calls[-1])
    assert "requested design change\"}]}\n\nConfirm only" in calls[-2][1]
    assert "task_type, or response_format.\n\nCurrent tasks:" in calls[-2][1]
    assert "\n\nUser response: accept" in calls[-2][1]
    assert "User response: accept\n\nPrevious response:" in calls[-1][1]
    assert "Previous response: {}\n\nContract error:" in calls[-1][1]
