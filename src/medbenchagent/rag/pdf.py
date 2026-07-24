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
from medbenchagent.construction import event_operation
from medbenchagent.rag.reranker import score_pairs

CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200
CACHE_SCHEMA_VERSION = "2"
MATERIAL_CATEGORIES = {
    "clinical_guidelines": "临床指南",
    "imaging_reporting_and_classification": "影像报告与分类标准",
    "imaging_acquisition_and_quality": "影像采集与质量控制",
    "consensus_statements": "专家共识",
    "evidence_reviews": "证据综述",
    "regulations_and_policy": "法规与政策",
}
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
_VALUE_REQUEST_KEY = "requested_field_values"


class RagUnavailable(RuntimeError):
    """Raised when local medical references or retrieval models are unavailable."""


class QuestionDesignError(ValueError):
    """Raised after a task design violates its contract twice."""


def discover_medical_materials(materials_root: str | Path) -> list[dict[str, str]]:
    """List categorized PDF references relative to the fixed materials directory."""
    root = Path(materials_root).expanduser().resolve()
    if not root.is_dir():
        raise RagUnavailable(f"Medical-material directory does not exist: {root}")
    materials: list[dict[str, str]] = []
    for pdf in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"):
        relative_path = pdf.relative_to(root)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        if category not in MATERIAL_CATEGORIES:
            raise RagUnavailable(
                f"Medical PDF must be stored in a supported category directory: {relative_path.as_posix()}"
            )
        materials.append(
            {
                "category": category,
                "category_label": MATERIAL_CATEGORIES[category],
                "file": relative_path.as_posix(),
            }
        )
    if not materials:
        raise RagUnavailable("No categorized PDF medical materials were found.")
    return materials


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
    selected_files: list[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[str, list[dict[str, object]]]:
    """Extract page-local chunks and return a fingerprint of inputs and chunking rules."""
    root = Path(materials_root).expanduser().resolve()
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

    def __init__(
        self,
        materials_root: str | Path,
        cache_root: str | Path,
        selected_files: list[str] | None = None,
    ) -> None:
        source_hash, self.chunks = extract_pdf_chunks(materials_root, selected_files)
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
    if not isinstance(templates, list) or len(templates) != 1 or not all(
        isinstance(template, str) and template.strip() for template in templates
    ):
        raise ValueError(f"{task_id}: question_templates must contain exactly one non-empty string.")
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
        normalized_options = [
            re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", option).strip()
            for option in options
        ] if isinstance(options, list) else []
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= 26
            or not all(normalized_options)
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
    field_value_summaries: dict[str, dict[str, object]] | None = None,
    *,
    allow_value_request: bool = True,
) -> tuple[str, str]:
    schema = _design_schema(task.get("response_format"))
    system = (
        "Design the medical-content portion of exactly one medical VLM benchmark task. "
        f"Return exactly this JSON shape and no other fields: {schema}. "
        "Use the requested benchmark language. Medical references may guide clinical wording, "
        "option design, and medical basis. "
        "generation_logic must explain how a later model uses only approved supporting annotations "
        "to generate the question and answer without inventing facts. "
        "Do not return task_id, citations, validation_rules, placeholder bindings, answer fields, "
        "mappings, instance paths, or bbox fields. "

        "For letter tasks, return one question template and 2-26 options without A/B labels. "
        "The correct answer must be supported by annotations. Options must be semantically aligned "
        "with the same concept level and granularity as the answer. Distractors should be medically "
        "plausible and mutually exclusive. Choose the most appropriate compact candidate set, not options "
        "merely to add distractors. For categorical attributes, use concise category labels "
        "instead of verbose explanations or copied annotation descriptions. "

        "For modality_identification and anatomy_identification, a single supported value is valid. "
        "Use that value as the answer and add distractors from the same taxonomy and granularity. "

        "Do not create overly broad or clinically unrelated options. "
        "When no PDF excerpt is supplied, use generic wording and do not claim specific guideline support."
    )
    if allow_value_request:
        system += (
            " If the supplied task and supporting-field paths are insufficient to choose medically accurate "
            "wording or options, return instead exactly "
            '{"requested_field_values":["one supporting annotation path"]}. '
            "Request only fields whose categorical values are necessary; do not request DICOM metadata, "
            "bounding-box fields, raw annotations, or identifiers."
        )
    user = (
        f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"Task ID: {task_id}\nTask: {json.dumps(task, ensure_ascii=False)}\n"
        f"Requested revision: {feedback or 'none'}\n"
        f"Current RAG fields: {json.dumps(current_design or {}, ensure_ascii=False)}\n"
        f"Requested field value summaries: {json.dumps(field_value_summaries or {}, ensure_ascii=False)}\n"
        f"PDF excerpts:\n{context or '[No sufficiently relevant PDF excerpt was retrieved.]'}"
    )
    return system, user


def _field_value_summaries(
    task_id: str,
    task: dict[str, Any],
    dataset_profile: dict[str, Any],
    response: object,
) -> dict[str, dict[str, object]] | None:
    """Validate an optional on-demand request for safe categorical field summaries."""
    if not isinstance(response, dict) or _VALUE_REQUEST_KEY not in response:
        return None
    if set(response) != {_VALUE_REQUEST_KEY}:
        raise ValueError(f"{task_id}: field-value request must contain only {_VALUE_REQUEST_KEY}.")
    requested = response[_VALUE_REQUEST_KEY]
    if not isinstance(requested, list) or not requested or not all(isinstance(path, str) for path in requested):
        raise ValueError(f"{task_id}: field-value request must be a non-empty list of paths.")
    if len(requested) != len(set(requested)):
        raise ValueError(f"{task_id}: field-value request must not repeat paths.")
    supporting_fields = task.get("supporting_fields")
    if not isinstance(supporting_fields, list):
        raise ValueError(f"{task_id}: supporting_fields must be present before requesting values.")
    fields = dataset_profile.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"{task_id}: dataset profile must contain fields before requesting values.")

    summaries: dict[str, dict[str, object]] = {}
    for path in requested:
        segments = [segment.removesuffix("[*]") for segment in path.split(".")]
        if path not in supporting_fields:
            raise ValueError(f"{task_id}: requested field is not a supporting field: {path}")
        if path.startswith("annotation.dicom_metadata.") or "bbox" in segments:
            raise ValueError(f"{task_id}: requested field is not eligible for value summaries: {path}")
        profile_field = fields.get(path)
        if not isinstance(profile_field, dict) or not isinstance(profile_field.get("top_values"), list):
            raise ValueError(f"{task_id}: requested field has no safe top_values summary: {path}")
        summaries[path] = {
            key: profile_field[key]
            for key in ("path", "types", "non_empty_images", "distinct_value_count", "top_values")
            if key in profile_field
        }
    return summaries


def design_task(
    rag: LocalPdfRag,
    benchmark_info: dict[str, object],
    task_id: str,
    task: dict[str, Any],
    *,
    dataset_profile: dict[str, Any],
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
    field_value_summaries = _field_value_summaries(task_id, task, dataset_profile, response)
    if field_value_summaries is not None:
        system, user = _design_prompt(
            benchmark_info,
            task_id,
            task,
            context,
            feedback,
            current,
            field_value_summaries,
            allow_value_request=False,
        )
        response = complete_json(system, user)
    try:
        normalized = validate_task_design(task_id, task, response)
    except ValueError as first_error:
        repair_system = (
            f"{system}\nYour previous JSON violated the contract: {first_error}. "
            "Return one corrected object using the exact required shape."
        )
        with event_operation("repair"):
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
