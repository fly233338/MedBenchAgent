import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import medbenchagent.rag.pdf as pdf_rag
from pypdf.errors import PdfReadError

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
            "file": "Guideline/unselected.pdf",
        },
        {
            "category": "Standard",
            "file": "Standard/selected.pdf",
        },
    ]
    assert isinstance(fingerprint, str) and fingerprint
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


@pytest.mark.parametrize("during_extraction", [False, True])
def test_pdf_read_errors_identify_source_and_preserve_cause(tmp_path, monkeypatch, during_extraction):
    (tmp_path / "broken.pdf").write_bytes(b"broken")
    problem = PdfReadError("damaged stream")

    def fail():
        raise problem

    def reader(path):
        if not during_extraction:
            fail()
        return SimpleNamespace(pages=[SimpleNamespace(extract_text=fail)])

    monkeypatch.setattr(pdf_rag, "PdfReader", reader)
    with pytest.raises(pdf_rag.RagUnavailable, match="broken.pdf") as error:
        pdf_rag.extract_pdf_chunks(tmp_path)
    assert error.value.__cause__ is problem


def test_unexpected_pdf_programming_errors_propagate(tmp_path, monkeypatch):
    (tmp_path / "source.pdf").write_bytes(b"source")

    def reader(path):
        raise TypeError("unexpected programming error")

    monkeypatch.setattr(pdf_rag, "PdfReader", reader)
    with pytest.raises(TypeError, match="programming error"):
        pdf_rag.extract_pdf_chunks(tmp_path)
