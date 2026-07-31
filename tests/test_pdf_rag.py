import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import medbenchagent.rag.pdf as pdf_rag


def _task(response_format: str = "letter") -> dict:
    return {
        "description": "Identify the view",
        "task_type": "close_vqa",
        "response_format": response_format,
        "supporting_fields": ["image.annotation.view"],
    }


def _design(response_format: str = "letter") -> dict:
    design = {
        "question_templates": [
            "Which view is shown?",
            "Identify the view depicted in the image.",
            "What imaging view does this image represent?",
        ],
        "generation_logic": "Normalize the approved view annotation without inventing facts.",
        "medical_task_description": "This assesses view recognition; imaging guidance supports the terminology.",
    }
    if response_format == "letter":
        design["options"] = ["CC", "MLO"]
    return design


def _profile(*paths: str) -> dict:
    return {
        "fields": {
            path: {
                "path": path,
                "types": ["string"],
                "non_empty_images": 10,
                "distinct_value_count": 2,
                "top_values": ["CC", "MLO"],
            }
            for path in paths
        }
    }


def test_validate_task_design_accepts_only_embedded_rag_fields() -> None:
    assert pdf_rag.validate_task_design("view", _task(), _design())["options"] == ["CC", "MLO"]
    with pytest.raises(ValueError, match="exactly"):
        pdf_rag.validate_task_design("view", _task(), _design() | {"validation_rules": []})
    with pytest.raises(ValueError, match="options"):
        pdf_rag.validate_task_design("view", _task(), _design() | {"options": ["CC"]})
    with pytest.raises(ValueError, match="exactly 3"):
        pdf_rag.validate_task_design("view", _task(), _design() | {"question_templates": ["one", "two"]})
    with pytest.raises(ValueError, match="unique"):
        pdf_rag.validate_task_design(
            "view",
            _task(),
            _design() | {"question_templates": ["one", "one", "three"]},
        )
    assert pdf_rag.validate_task_design(
        "view", _task(), _design() | {"options": ["A. CC", "B) MLO"]}
    )["options"] == ["CC", "MLO"]
    with pytest.raises(ValueError, match="2-8"):
        pdf_rag.validate_task_design(
            "view",
            _task(),
            _design() | {"options": [f"option-{index}" for index in range(9)]},
        )


def test_validate_task_design_uses_configured_template_count(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf_rag,
        "get_settings",
        lambda: SimpleNamespace(generation=SimpleNamespace(question_template_count=2)),
    )
    design = _design() | {"question_templates": ["Which view?", "What view is shown?"]}

    assert pdf_rag.validate_task_design("view", _task(), design)["question_templates"] == [
        "Which view?",
        "What view is shown?",
    ]


def test_design_task_repairs_one_invalid_response(monkeypatch) -> None:
    responses = iter([_design() | {"validation_rules": []}, _design()])
    monkeypatch.setattr(pdf_rag, "complete_json", lambda *_args: next(responses))
    rag = SimpleNamespace(
        retrieve=lambda _query: [
            {"chunk_id": "guide-p1-c0", "pdf": "guide.pdf", "page": 1, "text": "Views include CC and MLO."}
        ]
    )

    result = pdf_rag.design_task(
        rag, {"language": "en"}, "view", _task(), dataset_profile=_profile("image.annotation.view")
    )

    assert result["options"] == ["CC", "MLO"]
    assert result["citations"] == [{"pdf": "guide.pdf", "page": 1, "chunk_id": "guide-p1-c0"}]


def test_design_task_requests_safe_field_values_only_when_needed(monkeypatch) -> None:
    responses = iter([{"requested_field_values": ["image.annotation.view"]}, _design()])
    calls: list[str] = []

    def complete(system: str, _user: str) -> dict:
        calls.append(system)
        return next(responses)

    monkeypatch.setattr(pdf_rag, "complete_json", complete)
    rag = SimpleNamespace(retrieve=lambda _query: [])

    result = pdf_rag.design_task(
        rag, {"language": "en"}, "view", _task(), dataset_profile=_profile("image.annotation.view")
    )

    assert result["options"] == ["CC", "MLO"]
    assert len(calls) == 2
    assert "requested_field_values" in calls[0]
    assert "requested_field_values" not in calls[1]


@pytest.mark.parametrize(
    ("requested_path", "task", "profile", "message"),
    [
        (
            "image.annotation.dicom_metadata.view_position",
            _task() | {"supporting_fields": ["image.annotation.dicom_metadata.view_position"]},
            _profile("image.annotation.dicom_metadata.view_position"),
            "not eligible",
        ),
        (
            "image.annotation.findings[*].bbox[*]",
            _task() | {"supporting_fields": ["image.annotation.findings[*].bbox[*]"]},
            _profile("image.annotation.findings[*].bbox[*]"),
            "not eligible",
        ),
        ("image.annotation.other", _task(), _profile("image.annotation.other"), "not a supporting field"),
        (
            "image.annotation.view",
            _task(),
            {"fields": {"image.annotation.view": {"path": "image.annotation.view"}}},
            "no safe top_values",
        ),
    ],
)
def test_design_task_rejects_unsafe_field_value_requests(
    monkeypatch, requested_path: str, task: dict, profile: dict, message: str
) -> None:
    monkeypatch.setattr(
        pdf_rag, "complete_json", lambda *_args: {"requested_field_values": [requested_path]}
    )

    with pytest.raises(ValueError, match=message):
        pdf_rag.design_task(SimpleNamespace(retrieve=lambda _query: []), {"language": "en"}, "view", task, dataset_profile=profile)


def test_discover_and_extract_selected_categorized_medical_pdfs(
    tmp_path: Path, monkeypatch
) -> None:
    selected = tmp_path / "Standard" / "selected.pdf"
    unselected = tmp_path / "Guideline" / "unselected.pdf"
    selected.parent.mkdir()
    unselected.parent.mkdir()
    selected.write_bytes(b"selected")
    unselected.write_bytes(b"unselected")

    class FakeReader:
        def __init__(self, path: str) -> None:
            self.pages = [SimpleNamespace(extract_text=lambda: f"{Path(path).stem} paragraph " * 40)]

    monkeypatch.setattr(pdf_rag, "PdfReader", FakeReader)
    materials = pdf_rag.discover_medical_materials(tmp_path)
    fingerprint, chunks = pdf_rag.extract_pdf_chunks(
        tmp_path,
        ["Standard/selected.pdf"],
        chunk_size=120,
        overlap=20,
    )
    unselected.write_bytes(b"changed unselected content")
    unchanged_fingerprint, _ = pdf_rag.extract_pdf_chunks(
        tmp_path,
        ["Standard/selected.pdf"],
        chunk_size=120,
        overlap=20,
    )

    assert materials == [
        {
            "category": "Guideline",
            "category_label": "Guideline",
            "file": "Guideline/unselected.pdf",
        },
        {
            "category": "Standard",
            "category_label": "Standard",
            "file": "Standard/selected.pdf",
        },
    ]
    assert len(fingerprint) == 64
    assert unchanged_fingerprint == fingerprint
    assert {chunk["pdf"] for chunk in chunks} == {
        "Standard/selected.pdf"
    }


def test_discover_medical_materials_rejects_uncategorized_pdf(tmp_path: Path) -> None:
    (tmp_path / "guide.pdf").write_bytes(b"guide")

    with pytest.raises(pdf_rag.RagUnavailable, match="category directory"):
        pdf_rag.discover_medical_materials(tmp_path)


def test_embedding_model_loads_once_per_resolved_path(tmp_path: Path, monkeypatch) -> None:
    loaded: list[str] = []

    class FakeEmbedder:
        def __init__(self, path: str) -> None:
            loaded.append(path)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeEmbedder),
    )
    pdf_rag.clear_embedding_cache()

    first = pdf_rag._get_embedder(str(tmp_path))
    second = pdf_rag._get_embedder(str(tmp_path))

    assert first is second
    assert loaded == [str(tmp_path.resolve())]

def test_design_prompt_keeps_letter_candidate_sets_compact() -> None:
    system, _ = pdf_rag._design_prompt(
        {"language": "en"},
        "view_identification",
        _task(),
        "",
        "",
        None,
    )

    assert "Use 2-4 options by default" in system
    assert "Return exactly 3 question templates" in system
    assert "Use 5-8 only when" in system
    assert "Do not enumerate an exhaustive differential diagnosis" in system
    assert "request the necessary supporting field values instead of inventing a taxonomy" in system
    assert "routes of spread" in system


@pytest.mark.parametrize("response_format", ["bbox", "free_text"])
def test_design_prompt_omits_letter_rules_for_other_formats(response_format: str) -> None:
    system, _ = pdf_rag._design_prompt(
        {"language": "en"},
        "non_letter_task",
        _task(response_format),
        "",
        "",
        None,
    )

    assert "Letter option rules" not in system
    assert "Use 2-4 options by default" not in system
    assert "categorical letter task" not in system
    assert '"options"' not in system
