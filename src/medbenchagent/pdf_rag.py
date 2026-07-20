"""Local PDF retrieval and Blueprint-embedded task design."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

from pypdf import PdfReader

from medbenchagent.llm import complete_json
from medbenchagent.qwen_reranker import score_pairs

CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200
CACHE_SCHEMA_VERSION = "2"
RAG_DESIGN_FIELDS = {
    "question_templates",
    "options",
    "generation_logic",
    "medical_task_description",
    "citations",
}
LEGACY_DESIGN_FIELDS = {
    "validation_rules",
    "placeholder_bindings",
    "answer_field",
    "answer_value_mapping",
    "instance_path",
    "target_text_field",
    "bbox_field",
    "bbox_format",
}


class RagUnavailable(RuntimeError):
    """Raised when local medical references or retrieval models are unavailable."""


class QuestionDesignError(ValueError):
    """Raised after a task design violates its contract twice."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[str, list[dict[str, object]]]:
    """Extract page-local chunks and return a fingerprint of inputs and chunking rules."""
    root = Path(materials_root).expanduser().resolve()
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    pdfs = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        raise RagUnavailable("No PDF medical materials were found.")

    fingerprint_items = [CACHE_SCHEMA_VERSION, str(chunk_size), str(overlap)]
    chunks: list[dict[str, object]] = []
    for pdf in pdfs:
        relative_path = pdf.relative_to(root).as_posix()
        fingerprint_items.extend((relative_path, _file_hash(pdf)))
        try:
            reader = PdfReader(str(pdf))
        except Exception as exc:  # PyPDF exceptions are version-specific.
            raise RagUnavailable(f"Cannot parse PDF: {relative_path}") from exc
        path_key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            for index, chunk_text in enumerate(_page_chunks(text, chunk_size, overlap)):
                chunks.append(
                    {
                        "chunk_id": f"{path_key}-p{page_number}-c{index}",
                        "text": chunk_text,
                        "pdf": relative_path,
                        "page": page_number,
                    }
                )
    if not chunks:
        raise RagUnavailable("PDF materials contain no extractable text.")
    fingerprint = hashlib.sha256("\0".join(fingerprint_items).encode("utf-8")).hexdigest()
    return fingerprint, chunks


_EMBEDDERS: dict[str, Any] = {}
_EMBEDDER_LOCK = Lock()


def _get_embedder(model_path: str) -> Any:
    key = str(Path(model_path).resolve())
    with _EMBEDDER_LOCK:
        if key not in _EMBEDDERS:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RagUnavailable("PDF RAG dependencies are not installed.") from exc
            _EMBEDDERS[key] = SentenceTransformer(key)
        return _EMBEDDERS[key]


def clear_embedding_cache() -> None:
    """Clear process-local embedding models; intended for isolated tests."""
    with _EMBEDDER_LOCK:
        _EMBEDDERS.clear()


def _model_fingerprint(path: Path) -> str:
    parts = [str(path.resolve())]
    for name in ("config.json", "modules.json", "sentence_bert_config.json"):
        config = path / name
        if config.is_file():
            parts.extend((name, _file_hash(config)))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class LocalPdfRag:
    """Chroma retrieval with process-cached embedding and reranking models."""

    def __init__(self, materials_root: str | Path, cache_root: str | Path) -> None:
        source_hash, self.chunks = extract_pdf_chunks(materials_root)
        embedding_value = os.getenv("BGE_EMBEDDING_MODEL")
        reranker_value = os.getenv("BGE_RERANKER_MODEL")
        if not embedding_value or not reranker_value:
            raise RagUnavailable("Local embedding and reranker model directories must be configured.")
        embedding_path = Path(embedding_value).expanduser().resolve()
        reranker_path = Path(reranker_value).expanduser().resolve()
        if not embedding_path.is_dir() or not reranker_path.is_dir():
            raise RagUnavailable("Local embedding and reranker model directories must be configured.")
        cache_key = hashlib.sha256(
            f"{source_hash}:{_model_fingerprint(embedding_path)}".encode("utf-8")
        ).hexdigest()
        self.cache_dir = Path(cache_root) / cache_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_path = str(embedding_path)
        self.reranker_path = str(reranker_path)

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        try:
            import chromadb
        except ImportError as exc:
            raise RagUnavailable("PDF RAG dependencies are not installed.") from exc
        embedder = _get_embedder(self.embedding_path)
        client = chromadb.PersistentClient(path=str(self.cache_dir))
        collection = client.get_or_create_collection("medical_references")
        if collection.count() != len(self.chunks):
            collection.upsert(
                ids=[str(chunk["chunk_id"]) for chunk in self.chunks],
                documents=[str(chunk["text"]) for chunk in self.chunks],
                embeddings=embedder.encode([str(chunk["text"]) for chunk in self.chunks]).tolist(),
                metadatas=[
                    {"pdf": str(chunk["pdf"]), "page": int(chunk["page"])}
                    for chunk in self.chunks
                ],
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
                response["ids"][0],
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
                strict=True,
            )
        ]
        scores = score_pairs(
            self.reranker_path,
            [(query, str(candidate["text"])) for candidate in candidates],
        )
        try:
            threshold = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.5"))
        except ValueError as exc:
            raise RagUnavailable("RAG_MIN_RERANK_SCORE must be a number between 0 and 1.") from exc
        if not 0 <= threshold <= 1:
            raise RagUnavailable("RAG_MIN_RERANK_SCORE must be between 0 and 1.")
        accepted: list[dict[str, object]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            if float(score) >= threshold:
                accepted.append(candidate)
        return sorted(accepted, key=lambda item: float(item["rerank_score"]), reverse=True)[:limit]


def _design_schema(response_format: object) -> str:
    common = (
        '{"question_templates":["non-empty template"],'
        '"generation_logic":"non-empty instructions for using supporting annotation",'
        '"medical_task_description":"one paragraph: assessed capability, then medical basis"}'
    )
    if response_format == "letter":
        return common[:-1] + ',"options":["2 to 26 unique medical options"]}'
    if response_format in {"bbox", "free_text"}:
        return common
    raise ValueError(f"Unsupported response_format for RAG design: {response_format}")


def validate_task_design(task_id: str, task: dict[str, Any], response: object) -> dict[str, Any]:
    """Validate one RAG contribution before embedding it in the Blueprint task."""
    if not isinstance(response, dict):
        raise ValueError(f"{task_id}: RAG design must be a JSON object.")
    response_format = task.get("response_format")
    if response_format not in {"letter", "bbox", "free_text"}:
        raise ValueError(f"{task_id}: unsupported response_format {response_format}.")
    expected = {"question_templates", "generation_logic", "medical_task_description"}
    if response_format == "letter":
        expected.add("options")
    if set(response) != expected:
        raise ValueError(f"{task_id}: RAG design must contain exactly {sorted(expected)}.")
    templates = response["question_templates"]
    if not isinstance(templates, list) or not templates or not all(
        isinstance(template, str) and template.strip() for template in templates
    ):
        raise ValueError(f"{task_id}: question_templates must be non-empty strings.")
    for field in ("generation_logic", "medical_task_description"):
        if not isinstance(response[field], str) or not response[field].strip():
            raise ValueError(f"{task_id}: {field} must be a non-empty string.")
    normalized: dict[str, Any] = {
        "question_templates": [template.strip() for template in templates],
        "generation_logic": response["generation_logic"].strip(),
        "medical_task_description": response["medical_task_description"].strip(),
    }
    if response_format == "letter":
        options = response["options"]
        normalized_options = [option.strip() for option in options] if isinstance(options, list) else []
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= 26
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len(set(normalized_options)) != len(normalized_options)
        ):
            raise ValueError(f"{task_id}: options must contain 2-26 unique non-empty strings.")
        normalized["options"] = normalized_options
    return normalized


def _design_prompt(
    benchmark_info: dict[str, object],
    task_id: str,
    task: dict[str, Any],
    context: str,
    feedback: str,
    current_design: dict[str, Any] | None,
) -> tuple[str, str]:
    schema = _design_schema(task.get("response_format"))
    system = (
        "Design the medical-content portion of exactly one medical VLM benchmark task. "
        f"Return exactly this JSON shape and no other fields: {schema}. "
        "Use the requested benchmark language. Medical references may determine clinically standard "
        "question wording, letter-task options, assessed capability, and medical basis. "
        "generation_logic must explain how a later model should use only supporting annotation to "
        "produce one question and answer without inventing facts. Do not return task_id, citations, "
        "validation_rules, placeholder bindings, answer fields, mappings, instance paths, or bbox fields. "
        "When no PDF excerpt is supplied, use generic wording and do not claim a specific guideline basis."
    )
    user = (
        f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"Task ID: {task_id}\nTask: {json.dumps(task, ensure_ascii=False)}\n"
        f"Requested revision: {feedback or 'none'}\n"
        f"Current RAG fields: {json.dumps(current_design or {}, ensure_ascii=False)}\n"
        f"PDF excerpts:\n{context or '[No sufficiently relevant PDF excerpt was retrieved.]'}"
    )
    return system, user


def design_task(
    rag: LocalPdfRag,
    benchmark_info: dict[str, object],
    task_id: str,
    task: dict[str, Any],
    *,
    feedback: str = "",
) -> dict[str, Any]:
    """Retrieve medical material and produce one validated Blueprint task contribution."""
    supporting_fields = task.get("supporting_fields")
    if not isinstance(supporting_fields, list) or not supporting_fields:
        raise ValueError(f"{task_id}: supporting_fields must be present before RAG design.")
    query = " ".join(
        str(value)
        for value in (
            benchmark_info.get("disease"),
            benchmark_info.get("modality"),
            task_id,
            task.get("description"),
            supporting_fields,
        )
    )
    citations = rag.retrieve(query)
    context = "\n\n".join(
        f"[{item['chunk_id']}, {item['pdf']} p.{item['page']}] {item['text']}" for item in citations
    )
    current = {field: task[field] for field in RAG_DESIGN_FIELDS if field in task and field != "citations"}
    system, user = _design_prompt(benchmark_info, task_id, task, context, feedback, current)
    response = complete_json(system, user)
    try:
        normalized = validate_task_design(task_id, task, response)
    except ValueError as first_error:
        repair_system = (
            f"{system}\nYour previous JSON violated the contract: {first_error}. "
            "Return one corrected object using the exact required shape."
        )
        repaired = complete_json(
            repair_system,
            f"Previous response: {json.dumps(response, ensure_ascii=False)}\n\n{user}",
        )
        try:
            normalized = validate_task_design(task_id, task, repaired)
        except ValueError as second_error:
            preview = json.dumps(repaired, ensure_ascii=False)[:500]
            raise QuestionDesignError(
                f"{task_id}: RAG design remained invalid after repair: {second_error}; response={preview}"
            ) from second_error
    normalized["citations"] = [
        {key: item[key] for key in ("pdf", "page", "chunk_id")} for item in citations
    ]
    return normalized


def write_task_design(
    blueprint_path: str | Path,
    task_id: str,
    design: dict[str, Any],
) -> None:
    """Atomically replace one task's embedded RAG fields in the working Blueprint."""
    path = Path(blueprint_path)
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(tasks.get(task_id), dict):
        raise ValueError(f"Working Blueprint does not contain task: {task_id}")
    task = tasks[task_id]
    for field in RAG_DESIGN_FIELDS | LEGACY_DESIGN_FIELDS:
        task.pop(field, None)
    task.update(design)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(blueprint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
