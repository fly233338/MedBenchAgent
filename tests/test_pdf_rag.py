import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import medbenchagent.pdf_rag as pdf_rag


def _task(response_format: str = "letter") -> dict:
    return {
        "description": "Identify the view",
        "task_type": "close_vqa",
        "response_format": response_format,
        "supporting_fields": ["annotation.view"],
    }


def _design(response_format: str = "letter") -> dict:
    design = {
        "question_templates": ["Which view is shown?"],
        "generation_logic": "Normalize the approved view annotation without inventing facts.",
        "medical_task_description": "This assesses view recognition; imaging guidance supports the terminology.",
    }
    if response_format == "letter":
        design["options"] = ["CC", "MLO"]
    return design


def test_validate_task_design_accepts_only_embedded_rag_fields() -> None:
    assert pdf_rag.validate_task_design("view", _task(), _design())["options"] == ["CC", "MLO"]
    with pytest.raises(ValueError, match="exactly"):
        pdf_rag.validate_task_design("view", _task(), _design() | {"validation_rules": []})
    with pytest.raises(ValueError, match="options"):
        pdf_rag.validate_task_design("view", _task(), _design() | {"options": ["CC"]})


def test_write_task_design_replaces_only_rag_fields_atomically(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "view": _task()
                    | {
                        "question_templates": ["old"],
                        "validation_rules": ["legacy"],
                    },
                    "other": {"description": "unchanged"},
                }
            }
        ),
        encoding="utf-8",
    )

    pdf_rag.write_task_design(path, "view", _design() | {"citations": []})

    blueprint = json.loads(path.read_text(encoding="utf-8"))
    assert blueprint["tasks"]["view"]["question_templates"] == ["Which view is shown?"]
    assert "validation_rules" not in blueprint["tasks"]["view"]
    assert blueprint["tasks"]["other"] == {"description": "unchanged"}
    assert not path.with_suffix(".json.tmp").exists()


def test_design_task_repairs_one_invalid_response(monkeypatch) -> None:
    responses = iter([_design() | {"validation_rules": []}, _design()])
    monkeypatch.setattr(pdf_rag, "complete_json", lambda *_args: next(responses))
    rag = SimpleNamespace(
        retrieve=lambda _query: [
            {"chunk_id": "guide-p1-c0", "pdf": "guide.pdf", "page": 1, "text": "Views include CC and MLO."}
        ]
    )

    result = pdf_rag.design_task(rag, {"language": "en"}, "view", _task())

    assert result["options"] == ["CC", "MLO"]
    assert result["citations"] == [{"pdf": "guide.pdf", "page": 1, "chunk_id": "guide-p1-c0"}]


def test_extract_pdf_chunks_accepts_uppercase_and_ignores_auxiliary_files(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "one" / "guide.PDF"
    second = tmp_path / "two" / "guide.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    (tmp_path / "README.txt").write_text("notes", encoding="utf-8")

    class FakeReader:
        def __init__(self, path: str) -> None:
            name = Path(path).parent.name
            self.pages = [SimpleNamespace(extract_text=lambda: f"{name} paragraph " * 40)]

    monkeypatch.setattr(pdf_rag, "PdfReader", FakeReader)

    fingerprint, chunks = pdf_rag.extract_pdf_chunks(tmp_path, chunk_size=120, overlap=20)

    assert len(fingerprint) == 64
    assert {chunk["pdf"] for chunk in chunks} == {"one/guide.PDF", "two/guide.pdf"}
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


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
