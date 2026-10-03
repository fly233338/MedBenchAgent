"""Local PDF retrieval available throughout concept and leaf planning."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from medbenchagent.rag.remote import (
    REQUEST_TIMEOUT_SECONDS, RagUnavailable, embed_texts, rerank_documents,
)
from medbenchagent.settings import get_settings

CACHE_SCHEMA = "3"


def discover_medical_materials(materials_root: str | Path) -> list[dict[str, str]]:
    """List PDF references using their first-level directory as the category."""
    root = Path(materials_root).expanduser().resolve()
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    materials: list[dict[str, str]] = []
    for pdf in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"):
        relative_path = pdf.relative_to(root)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        if not category:
            raise RagUnavailable(
                f"Medical PDF must be stored in a category directory: {relative_path.as_posix()}"
            )
        materials.append(
            {
                "category": category,
                "file": relative_path.as_posix(),
            }
        )
    if not materials:
        raise RagUnavailable("No categorized PDF medical materials were found.")
    return materials


def _page_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pending = paragraph
        while pending:
            separator = "\n\n" if current else ""
            available = chunk_size - len(current) - len(separator)
            if available <= 0:
                chunks.append(current)
                current = current[-overlap:] if overlap else ""
                continue
            if len(pending) <= available:
                current += separator + pending
                pending = ""
                continue
            if current:
                chunks.append(current)
                current = current[-overlap:] if overlap else ""
                continue
            chunks.append(pending[:chunk_size])
            pending = pending[chunk_size - overlap :]
    if current and (not chunks or current != chunks[-1]):
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def extract_pdf_chunks(
    materials_root: str | Path,
    selected_files: list[str] | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[str, list[dict[str, object]]]:
    """Extract page-local chunks and return a description of sources and chunking rules."""
    root = Path(materials_root).expanduser().resolve()
    settings = get_settings().rag
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    if selected_files is None:
        pdfs = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
        )
    else:
        if not selected_files or len(selected_files) != len(set(selected_files)):
            raise RagUnavailable("Selected medical materials must be a non-empty unique file list.")
        pdfs = []
        for selected_file in selected_files:
            if not isinstance(selected_file, str):
                raise RagUnavailable("Selected medical material paths must be strings.")
            pdf = (root / selected_file).resolve()
            if (
                not pdf.is_relative_to(root)
                or not pdf.is_file()
                or pdf.suffix.lower() != ".pdf"
                or pdf.relative_to(root).as_posix() != selected_file
            ):
                raise RagUnavailable(f"Selected medical material is unavailable: {selected_file}")
            pdfs.append(pdf)
        pdfs.sort()
    if not pdfs:
        raise RagUnavailable("No PDF medical materials were found.")

    source_description = [CACHE_SCHEMA, str(chunk_size), str(overlap)]
    chunks: list[dict[str, object]] = []
    for pdf in pdfs:
        relative_path = pdf.relative_to(root).as_posix()
        source_description.append(relative_path)
        try:
            reader = PdfReader(str(pdf))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                for index, chunk_text in enumerate(_page_chunks(text, chunk_size, overlap)):
                    chunks.append(
                        {
                            "chunk_id": f"{relative_path}-p{page_number}-c{index}",
                            "text": chunk_text,
                            "pdf": relative_path,
                            "page": page_number,
                        }
                    )
        except (PyPdfError, OSError, ValueError) as exc:
            raise RagUnavailable(f"Cannot read PDF text: {relative_path}: {exc}") from exc
    if not chunks:
        raise RagUnavailable("PDF materials contain no extractable text.")
    source_identity = json.dumps(source_description)
    return source_identity, chunks


class LocalPdfRag:
    """Chroma retrieval using independently served embedding and reranking models."""

    def __init__(
        self,
        materials_root: str | Path,
        cache_root: str | Path,
        selected_files: list[str] | None = None,
        *, revision: str = "1", refresh: bool = False,
    ) -> None:
        settings = get_settings().rag
        if not all((settings.embedding_base_url, settings.embedding_model,
                    settings.reranker_base_url, settings.reranker_model)):
            raise RagUnavailable("Embedding and reranker API endpoints and models must be configured.")
        if not revision.isalnum():
            raise ValueError("RAG revision must be alphanumeric.")
        identity = {"root": str(Path(materials_root).resolve()), "sources": selected_files,
                    "chunk_size": settings.chunk_size, "overlap": settings.chunk_overlap,
                    "embedding_base_url": settings.embedding_base_url,
                    "embedding_model": settings.embedding_model, "revision": revision}
        cache = Path(cache_root) / "pdf-cache"
        cache.mkdir(parents=True, exist_ok=True)
        entries = sorted(cache.glob("*/sources.json"))
        match = next((p.parent for p in entries if json.loads(p.read_text(encoding="utf-8")) == identity), None)
        if match is None or refresh:
            from uuid import uuid4
            match = cache / str(uuid4())
            match.mkdir()
            _, self.chunks = extract_pdf_chunks(materials_root, selected_files)
            (match / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
            (match / "sources.json").write_text(json.dumps(identity), encoding="utf-8")
        else:
            self.chunks = json.loads((match / "chunks.json").read_text(encoding="utf-8"))
        self.cache_dir = match
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings

    def retrieve(self, query: str, limit: int | None = None) -> list[dict[str, object]]:
        try:
            import chromadb
        except ImportError as exc:
            raise RagUnavailable("PDF RAG dependencies are not installed.") from exc
        client = chromadb.PersistentClient(path=str(self.cache_dir))
        collection = client.get_or_create_collection("medical_references")
        settings = self.settings
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as api:
            if collection.count() != len(self.chunks):
                collection.upsert(
                    ids=[str(chunk["chunk_id"]) for chunk in self.chunks],
                    documents=[str(chunk["text"]) for chunk in self.chunks],
                    embeddings=embed_texts(api, settings.embedding_base_url, settings.embedding_model,
                                           [str(chunk["text"]) for chunk in self.chunks]),
                    metadatas=[
                        {"pdf": str(chunk["pdf"]), "page": int(chunk["page"])}
                        for chunk in self.chunks
                    ],
                )
            response = collection.query(
                query_embeddings=embed_texts(api, settings.embedding_base_url,
                                             settings.embedding_model, [query]),
                n_results=min(settings.candidate_count, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        candidates = [
            {
                "chunk_id": chunk_id,
                "text": document,
                "pdf": metadata["pdf"],
                "page": metadata["page"],
                "distance": distance,
            }
            for chunk_id, document, metadata, distance in zip(
                response["ids"][0],
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
                strict=True,
            )
        ]
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as api:
            scores = rerank_documents(
                api, settings.reranker_base_url, settings.reranker_model, query,
                [str(candidate["text"]) for candidate in candidates],
                max_length=settings.reranker_max_length,
            )
        threshold = settings.min_rerank_score
        accepted: list[dict[str, object]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            if float(score) >= threshold:
                accepted.append(candidate)
        return sorted(accepted, key=lambda item: float(item["rerank_score"]), reverse=True)[: limit or settings.result_limit]

