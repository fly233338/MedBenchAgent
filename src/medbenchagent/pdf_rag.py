"""Local, hash-cached PDF retrieval for question-design planning."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pypdf import PdfReader

from medbenchagent.llm import ModelUnavailable, complete_json


class RagUnavailable(RuntimeError):
    """Raised when local medical references or local BGE models are unavailable."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pdf_chunks(materials_root: str | Path, chunk_size: int = 1_200) -> tuple[str, list[dict[str, object]]]:
    root = Path(materials_root).expanduser().resolve()
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    pdfs = sorted(root.rglob("*.pdf"))
    other_files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() != ".pdf"]
    if other_files:
        raise RagUnavailable("Medical materials only support PDF files in the first release.")
    if not pdfs:
        raise RagUnavailable("No PDF medical materials were found.")
    chunks: list[dict[str, object]] = []
    hashes = []
    for pdf in pdfs:
        hashes.append(_file_hash(pdf))
        try:
            reader = PdfReader(str(pdf))
        except Exception as exc:  # PyPDF exceptions are version-specific.
            raise RagUnavailable(f"Cannot parse PDF: {pdf.name}") from exc
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for offset in range(0, len(text), chunk_size):
                chunk_id = f"{pdf.stem}-p{page_number}-c{offset // chunk_size}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text[offset : offset + chunk_size],
                        "pdf": pdf.name,
                        "page": page_number,
                    }
                )
    if not chunks:
        raise RagUnavailable("PDF materials contain no extractable text.")
    return hashlib.sha256("".join(hashes).encode()).hexdigest(), chunks


class LocalPdfRag:
    """Chroma + local BGE embedding/reranking, persisted by source-PDF hash."""

    def __init__(self, materials_root: str | Path, cache_root: str | Path) -> None:
        self.source_hash, self.chunks = extract_pdf_chunks(materials_root)
        embedding_path = Path(os.getenv("BGE_EMBEDDING_MODEL", ""))
        reranker_path = Path(os.getenv("BGE_RERANKER_MODEL", ""))
        if not embedding_path.is_dir() or not reranker_path.is_dir():
            raise RagUnavailable("Local BGE embedding and reranker model directories must be configured.")
        self.cache_dir = Path(cache_root) / self.source_hash
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_path = str(embedding_path)
        self.reranker_path = str(reranker_path)

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        try:
            import chromadb
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RagUnavailable("PDF RAG dependencies are not installed.") from exc
        embedder = SentenceTransformer(self.embedding_path)
        client = chromadb.PersistentClient(path=str(self.cache_dir))
        collection = client.get_or_create_collection("medical_references")
        if collection.count() == 0:
            collection.upsert(
                ids=[str(chunk["chunk_id"]) for chunk in self.chunks],
                documents=[str(chunk["text"]) for chunk in self.chunks],
                embeddings=embedder.encode([str(chunk["text"]) for chunk in self.chunks]).tolist(),
                metadatas=[{"pdf": str(chunk["pdf"]), "page": int(chunk["page"])} for chunk in self.chunks],
            )
        response = collection.query(
            query_embeddings=[embedder.encode(query).tolist()],
            n_results=min(12, collection.count()),
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
                response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0], strict=True
            )
        ]
        reranker = CrossEncoder(self.reranker_path)
        scores = reranker.predict([(query, str(candidate["text"])) for candidate in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
        return sorted(candidates, key=lambda item: float(item["rerank_score"]), reverse=True)[:limit]


def design_questions(
    rag: LocalPdfRag, benchmark_info: dict[str, object], task_plans: list[dict[str, object]], feedback: str = ""
) -> list[dict[str, object]]:
    """Ask DeepSeek to design, but not generate, questions grounded in retrieved chunks."""
    designs: list[dict[str, object]] = []
    for plan in task_plans:
        if plan["decision"] != "retain":
            continue
        query = " ".join(
            str(value)
            for value in (
                benchmark_info.get("disease"),
                benchmark_info.get("modality"),
                plan["task_id"],
                plan.get("supporting_fields", []),
            )
        )
        citations = rag.retrieve(query)
        context = "\n\n".join(
            f"[{item['chunk_id']}, {item['pdf']} p.{item['page']}] {item['text']}" for item in citations
        )
        try:
            design = complete_json(
                "Design a medical benchmark question specification. Use only the supplied PDF excerpts for "
                "medical phrasing. Do not alter ground-truth labels. Return JSON with question_templates, "
                "placeholder_bindings, answer_list, answer_field, answer_value_mapping, validation_rules. "
                "answer_list must be the fixed unique option strings. answer_field must be one supporting "
                "annotation field. answer_value_mapping maps serialized annotation values to answer_list values.",
                f"Benchmark: {json.dumps(benchmark_info)}\nTask plan: {json.dumps(plan)}\n"
                f"User design feedback: {feedback}\nPDF excerpts:\n{context}",
            )
        except ModelUnavailable:
            raise
        designs.append(
            {
                "task_id": plan["task_id"],
                "question_templates": design.get("question_templates", []),
                "placeholder_bindings": design.get("placeholder_bindings", {}),
                "answer_list": design.get("answer_list", []),
                "answer_field": design.get("answer_field"),
                "answer_value_mapping": design.get("answer_value_mapping", design.get("source_value_mapping", {})),
                "validation_rules": design.get("validation_rules", []),
                "citations": [{key: item[key] for key in ("pdf", "page", "chunk_id")} for item in citations],
            }
        )
    return designs
