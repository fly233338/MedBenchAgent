"""Local PDF retrieval component with content-bound caches."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import httpx

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from medbenchagent.rag.remote import (
    REQUEST_TIMEOUT_SECONDS,
    RagUnavailable,
    embed_texts,
    rerank_documents,
)
from medbenchagent.settings import get_settings
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json

from filelock import FileLock


def discover_medical_materials(materials_root: str | Path) -> list[dict[str, str]]:
    """List PDF references using their first-level directory as the category."""
    root = Path(materials_root).expanduser().resolve()
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    materials: list[dict[str, str]] = []
    for pdf in sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    ):
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
    normalized = "\n\n".join(paragraphs)
    chunks, start = [], 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n\n", start + overlap + 1, end)
            if boundary != -1:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        # The paragraph boundary is chosen only beyond overlap: always consume text.
        start = end - overlap
    return chunks


def _pdf_sources(root: Path, selected_files):
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    if selected_files is None:
        pdfs = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
        )
    else:
        if (
            not selected_files
            or not all(isinstance(name, str) for name in selected_files)
            or len(selected_files) != len(set(selected_files))
        ):
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

    return pdfs


def _source_description(root, pdfs, chunk_size, overlap):
    return {
        "schema": "pdf-chunks",
        "strategy": "paragraph-window",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "documents": [
            {"path": pdf.relative_to(root).as_posix(), "digest": digest_file(pdf)} for pdf in pdfs
        ],
    }


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
    pdfs = _pdf_sources(root, selected_files)
    source_description = _source_description(root, pdfs, chunk_size, overlap)
    chunks: list[dict[str, object]] = []
    for pdf in pdfs:
        relative_path = pdf.relative_to(root).as_posix()
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

    if source_description != _source_description(root, pdfs, chunk_size, overlap):
        raise RagUnavailable(
            "Medical PDF content changed during extraction; retry with a stable source."
        )
    source_identity = json.dumps(source_description, sort_keys=True)
    return source_identity, chunks


class LocalPdfRag:
    """Chroma retrieval using independently served embedding and reranking models."""

    def __init__(
        self,
        materials_root: str | Path,
        cache_root: str | Path,
        selected_files: list[str] | None = None,
        *,
        refresh: bool = False,
    ) -> None:
        self.materials_root = Path(materials_root).expanduser().resolve()
        self.cache_root = Path(cache_root) / "pdf-cache"
        self.selected_files = list(selected_files) if selected_files is not None else None
        self._ensure_cache(refresh=refresh)

    def _ensure_cache(self, *, refresh=False):
        settings = get_settings().rag
        if not all(
            (
                settings.embedding_base_url,
                settings.embedding_model,
                settings.reranker_base_url,
                settings.reranker_model,
            )
        ):
            raise RagUnavailable(
                "Embedding and reranker API endpoints and models must be configured."
            )
        root = self.materials_root
        pdfs = _pdf_sources(root, self.selected_files)
        sources = _source_description(root, pdfs, settings.chunk_size, settings.chunk_overlap)
        identity = {
            "schema": "pdf-rag-cache",
            "root": str(root),
            "sources": sources,
            "embedding_base_url": settings.embedding_base_url,
            "embedding_model": settings.embedding_model,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        cache = self.cache_root / key
        cache.mkdir(parents=True, exist_ok=True)
        manifest, chunks_path = cache / "sources.json", cache / "chunks.json"
        with FileLock(str(cache / ".cache.lock")):
            saved = None
            if manifest.is_file():
                try:
                    saved = json.loads(manifest.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
            valid = (
                isinstance(saved, dict)
                and saved.get("identity") == identity
                and chunks_path.is_file()
                and saved.get("chunks_digest") == digest_file(chunks_path)
            )
            if refresh or not valid:
                _, chunks = extract_pdf_chunks(
                    root,
                    self.selected_files,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
                if sources != _source_description(
                    root,
                    pdfs,
                    settings.chunk_size,
                    settings.chunk_overlap,
                ):
                    raise RagUnavailable(
                        "Medical PDF content changed during extraction; retry with a stable source."
                    )
                atomic_write_json(chunks_path, chunks)
                atomic_write_json(
                    manifest,
                    {"identity": identity, "chunks_digest": digest_file(chunks_path)},
                )
                (cache / "indexed.json").unlink(missing_ok=True)
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.chunks_digest = digest_file(chunks_path)
        self.cache_dir, self.settings = cache, settings

    def retrieve(self, query: str, limit: int | None = None) -> list[dict[str, object]]:
        self._ensure_cache()
        try:
            import chromadb
        except ImportError as exc:
            raise RagUnavailable("PDF RAG dependencies are not installed.") from exc
        client = chromadb.PersistentClient(path=str(self.cache_dir))
        settings = self.settings
        with (
            FileLock(str(self.cache_dir / ".index.lock")),
            httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as api,
        ):
            collection = client.get_or_create_collection("medical_references")
            indexed = self.cache_dir / "indexed.json"
            complete = indexed.is_file() and indexed.read_text(
                encoding="utf-8"
            ).strip() == json.dumps(self.chunks_digest)
            if not complete or collection.count() != len(self.chunks):
                if collection.count():
                    client.delete_collection("medical_references")
                    collection = client.get_or_create_collection("medical_references")
                collection.upsert(
                    ids=[str(chunk["chunk_id"]) for chunk in self.chunks],
                    documents=[str(chunk["text"]) for chunk in self.chunks],
                    embeddings=embed_texts(
                        api,
                        settings.embedding_base_url,
                        settings.embedding_model,
                        [str(chunk["text"]) for chunk in self.chunks],
                    ),
                    metadatas=[
                        {"pdf": str(chunk["pdf"]), "page": int(chunk["page"])}
                        for chunk in self.chunks
                    ],
                )
                atomic_write_json(indexed, self.chunks_digest)
            response = collection.query(
                query_embeddings=embed_texts(
                    api,
                    settings.embedding_base_url,
                    settings.embedding_model,
                    [query],
                ),
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
                api,
                settings.reranker_base_url,
                settings.reranker_model,
                query,
                [str(candidate["text"]) for candidate in candidates],
                max_length=settings.reranker_max_length,
            )
        threshold = settings.min_rerank_score
        accepted: list[dict[str, object]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            if float(score) >= threshold:
                accepted.append(candidate)
        return sorted(accepted, key=lambda item: float(item["rerank_score"]), reverse=True)[
            : limit or settings.result_limit
        ]
