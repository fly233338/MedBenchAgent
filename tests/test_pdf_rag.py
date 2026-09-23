import json
from pathlib import Path
from types import SimpleNamespace

import httpx
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


def test_rag_cache_isolated_by_embedding_endpoint_and_model(tmp_path, monkeypatch):
    rag = SimpleNamespace(embedding_base_url="http://embed-one/v1", embedding_model="model-one",
                          reranker_base_url="http://rerank/v1", reranker_model="reranker",
                          chunk_size=1200, chunk_overlap=200)
    monkeypatch.setattr(pdf_rag, "get_settings", lambda: SimpleNamespace(rag=rag))
    monkeypatch.setattr(pdf_rag, "extract_pdf_chunks", lambda *_args, **_kwargs: (
        "sources", [{"chunk_id": "one", "text": "text", "pdf": "a.pdf", "page": 1}]))
    root = tmp_path / "materials"
    root.mkdir()
    first = pdf_rag.LocalPdfRag(root, tmp_path / "cache")
    same = pdf_rag.LocalPdfRag(root, tmp_path / "cache")
    assert first.cache_dir == same.cache_dir
    rag.embedding_base_url = "http://embed-two/v1"
    other_endpoint = pdf_rag.LocalPdfRag(root, tmp_path / "cache")
    assert other_endpoint.cache_dir != first.cache_dir
    rag.embedding_model = "model-two"
    other_model = pdf_rag.LocalPdfRag(root, tmp_path / "cache")
    assert other_model.cache_dir not in {first.cache_dir, other_endpoint.cache_dir}


def test_remote_rag_retrieves_and_reranks_chroma_candidates(tmp_path, monkeypatch):
    rag = SimpleNamespace(embedding_base_url="http://embed/v1", embedding_model="embed",
                          reranker_base_url="http://rerank/v1", reranker_model="rerank",
                          chunk_size=1200, chunk_overlap=200, candidate_count=2,
                          reranker_max_length=8192, min_rerank_score=0.5, result_limit=2)
    monkeypatch.setattr(pdf_rag, "get_settings", lambda: SimpleNamespace(rag=rag))
    monkeypatch.setattr(pdf_rag, "extract_pdf_chunks", lambda *_args, **_kwargs: (
        "sources", [
            {"chunk_id": "a", "text": "first", "pdf": "a.pdf", "page": 1},
            {"chunk_id": "b", "text": "second", "pdf": "b.pdf", "page": 2},
        ]))
    original_client = httpx.Client

    def respond(request):
        payload = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            vectors = {"first": [1.0, 0.0], "second": [0.0, 1.0], "query": [0.0, 1.0]}
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": vectors[text]}
                for index, text in enumerate(payload["input"])
            ]})
        assert request.url.path == "/v1/rerank"
        return httpx.Response(200, json={"results": [
            {"index": index, "relevance_score": 0.9 if text == "first" else 0.2}
            for index, text in enumerate(payload["documents"])
        ]})

    monkeypatch.setattr(pdf_rag.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs))
    root = tmp_path / "materials"
    root.mkdir()
    reader = pdf_rag.LocalPdfRag(root, tmp_path / "cache")
    results = reader.retrieve("query")
    assert [(item["chunk_id"], item["rerank_score"]) for item in results] == [("a", 0.9)]


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
