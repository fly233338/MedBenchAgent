"""Local PDF retrieval and Blueprint-embedded task design."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

import duckdb
from pypdf import PdfReader

from medbenchagent.answer_mapping import canonical_answer_scalar
from medbenchagent.contracts import RAG_DESIGN_FIELDS
from medbenchagent.runtime.construction import event_operation
from medbenchagent.generation.context import scalar_evidence_candidates
from medbenchagent.runtime.llm import complete_json
from medbenchagent.rag.reranker import score_pairs
from medbenchagent.settings import get_settings

CACHE_SCHEMA_VERSION = "2"
MAX_RAG_LETTER_OPTIONS = 8
MAX_RAG_MAPPING_VALUES = 32
_VALUE_REQUEST_KEY = "requested_field_values"


class RagUnavailable(RuntimeError):
    """Raised when local medical references or retrieval models are unavailable."""


class QuestionDesignError(ValueError):
    """Raised after a task design violates its contract twice."""


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

    source_description = [CACHE_SCHEMA_VERSION, str(chunk_size), str(overlap)]
    chunks: list[dict[str, object]] = []
    for pdf in pdfs:
        relative_path = pdf.relative_to(root).as_posix()
        source_description.append(relative_path)
        try:
            reader = PdfReader(str(pdf))
        except Exception as exc:  # PyPDF exceptions are version-specific.
            raise RagUnavailable(f"Cannot parse PDF: {relative_path}") from exc
        path_key = relative_path
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
    source_identity = json.dumps(source_description)
    return source_identity, chunks


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




class LocalPdfRag:
    """Chroma retrieval with process-cached embedding and reranking models."""

    def __init__(
        self,
        materials_root: str | Path,
        cache_root: str | Path,
        selected_files: list[str] | None = None,
        *, revision: str = "1", refresh: bool = False,
    ) -> None:
        settings = get_settings().rag
        if not settings.embedding_model or not settings.reranker_model:
            raise RagUnavailable("Local embedding and reranker model directories must be configured.")
        embedding_path = settings.embedding_model
        reranker_path = settings.reranker_model
        if not embedding_path.is_dir() or not reranker_path.is_dir():
            raise RagUnavailable("Local embedding and reranker model directories must be configured.")
        if not revision.isalnum():
            raise ValueError("RAG revision must be alphanumeric.")
        identity = {"root": str(Path(materials_root).resolve()), "sources": selected_files,
                    "chunk_size": settings.chunk_size, "overlap": settings.chunk_overlap,
                    "model": str(embedding_path.resolve()), "revision": revision}
        cache = Path(cache_root) / ("pdf-v" + CACHE_SCHEMA_VERSION)
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
        self.embedding_path = str(embedding_path)
        self.reranker_path = str(reranker_path)

    def retrieve(self, query: str, limit: int | None = None) -> list[dict[str, object]]:
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
        settings = get_settings().rag
        response = collection.query(
            query_embeddings=[embedder.encode(query).tolist()],
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
        scores = score_pairs(
            self.reranker_path,
            [(query, str(candidate["text"])) for candidate in candidates],
            max_length=settings.reranker_max_length,
        )
        threshold = settings.min_rerank_score
        accepted: list[dict[str, object]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            if float(score) >= threshold:
                accepted.append(candidate)
        return sorted(accepted, key=lambda item: float(item["rerank_score"]), reverse=True)[: limit or settings.result_limit]


def _design_schema(response_format: object) -> str:
    common = (
        '{"generation_logic":"one semantically complete instruction sentence describing how to use approved annotation semantics and the selected question template to generate the question",'
        '"medical_task_description":"one paragraph: assessed capability, then medical basis"}'
    )
    if response_format == "letter":
        return (
            common[:-1]
            + f',"options":["2 to {MAX_RAG_LETTER_OPTIONS} unique medical options"],'
            '"answer_field":"one exact approved supporting field",'
            '"answer_value_mapping":[{"raw_value":"one exact JSON scalar",'
            '"option":"one exact returned option"}]}'
        )
    if response_format in {"bbox", "free_text"}:
        return common
    raise ValueError(f"Unsupported response_format for RAG design: {response_format}")


def validate_task_design(
    task_id: str,
    task: dict[str, Any],
    response: object,
    field_value_summaries: dict[str, dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate one RAG contribution before embedding it in the Blueprint task."""
    if not isinstance(response, dict):
        raise ValueError(f"{task_id}: RAG design must be a JSON object.")
    response_format = task.get("response_format")
    if response_format not in {"letter", "bbox", "free_text"}:
        raise ValueError(f"{task_id}: unsupported response_format {response_format}.")
    expected = {"generation_logic", "medical_task_description"}
    if response_format == "letter":
        expected.update({"options", "answer_field", "answer_value_mapping"})
    if set(response) != expected:
        raise ValueError(f"{task_id}: RAG design must contain exactly {sorted(expected)}.")
    for field in ("generation_logic", "medical_task_description"):
        if not isinstance(response[field], str) or not response[field].strip():
            raise ValueError(f"{task_id}: {field} must be a non-empty string.")
    normalized: dict[str, Any] = {
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
            or not 2 <= len(options) <= MAX_RAG_LETTER_OPTIONS
            or not all(normalized_options)
            or len(set(normalized_options)) != len(normalized_options)
        ):
            raise ValueError(
                f"{task_id}: options must contain 2-{MAX_RAG_LETTER_OPTIONS} "
                "unique non-empty strings."
            )
        normalized["options"] = normalized_options
        answer_field = response["answer_field"]
        supporting_fields = task.get("supporting_fields")
        if (
            not isinstance(answer_field, str)
            or not isinstance(supporting_fields, list)
            or answer_field not in supporting_fields
        ):
            raise ValueError(f"{task_id}: answer_field must be one approved supporting field.")
        summary = (field_value_summaries or {}).get(answer_field)
        summary_values = summary.get("values") if isinstance(summary, dict) else None
        if not isinstance(summary_values, list):
            raise ValueError(f"{task_id}: answer_field needs an approved field-value summary.")
        approved_values = {
            canonical
            for item in summary_values
            if isinstance(item, dict)
            and (canonical := canonical_answer_scalar(item.get("value"))) is not None
        }
        mappings = response["answer_value_mapping"]
        if not isinstance(mappings, list) or not mappings:
            raise ValueError(f"{task_id}: answer_value_mapping must be a non-empty list.")
        normalized_mapping: list[dict[str, object]] = []
        seen_values: set[str] = set()
        mapped_options: set[str] = set()
        for mapping in mappings:
            if not isinstance(mapping, dict) or set(mapping) != {"raw_value", "option"}:
                raise ValueError(
                    f"{task_id}: every answer mapping needs only raw_value and option."
                )
            raw_value, option = mapping["raw_value"], mapping["option"]
            canonical = canonical_answer_scalar(raw_value)
            if canonical is None:
                raise ValueError(f"{task_id}: mapped raw values must be non-empty JSON scalars.")
            if canonical in seen_values:
                raise ValueError(f"{task_id}: a raw value may be mapped only once.")
            if canonical not in approved_values:
                raise ValueError(f"{task_id}: mapped raw value is absent from the approved summary.")
            if not isinstance(option, str) or option not in normalized_options:
                raise ValueError(f"{task_id}: every mapped option must exactly match one option.")
            seen_values.add(canonical)
            mapped_options.add(option)
            normalized_mapping.append({"raw_value": raw_value, "option": option})
        if len(mapped_options) < 2:
            raise ValueError(
                f"{task_id}: answer mapping must support at least two correct-answer options."
            )
        normalized["answer_field"] = answer_field
        normalized["answer_value_mapping"] = normalized_mapping
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
    response_format = task.get("response_format")
    schema = _design_schema(response_format)
    system = (
        "Design the medical-content portion of exactly one medical VLM benchmark task.\n\n"
        "Evidence priority:\n"
        "(1) the task definition and approved supporting fields define what the "
        "task may assess;\n"
        "(2) supplied field-value summaries constrain dataset-backed answer content;\n"
        "(3) PDF excerpts may support clinical wording and plausibility, but must not define or expand "
        "the dataset's answer-label space."
    )
    if allow_value_request and response_format == "letter":
        system += (
            "\n\nBefore designing this letter task, return exactly this field-value request and no other fields:\n"
            '{"requested_field_values":["one answer-bearing supporting field path"]}.\n\n'
            "Request exactly one approved supporting field whose categorical values determine the answer. "
            "Do not request DICOM metadata, bounding-box fields, raw annotations, or identifiers."
        )
        user = (
            f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
            f"\nTask ID: {task_id}\nTask: {json.dumps(task, ensure_ascii=False)}\n"
            f"\nRequested revision: {feedback or 'none'}\n"
            f"Current design: {json.dumps(current_design or {}, ensure_ascii=False)}\n"
            f"\nPDF excerpts:\n{context or '[No sufficiently relevant PDF excerpt was retrieved.]'}"
        )
        return system, user
    system += f"\n\nReturn exactly this complete design object and no other fields:\n{schema}.\n\n"
    system += (
        "Use the requested benchmark language. generation_logic must be one semantically complete instruction sentence "
        "describing how to use approved annotation semantics and the selected question template to "
        "generate the question. It must not invent facts, infer an unsupported diagnosis, or use PDF "
        "text as annotation evidence.\n\n"
        "Do not return task_id, citations, validation_rules, placeholder bindings, instance paths, "
        "or bbox fields."
    )
    if response_format == "letter":
        system += (
            "\n\nThe supplied task is one fixed question type. Use only its declared semantic answer axis; "
            "do not create another question type or switch to a related property. All templates must use "
            "the same answer field, options, and answer mapping, and may vary wording and syntax only.\n\n"
            f"Letter option rules: return 2-{MAX_RAG_LETTER_OPTIONS} options without A/B labels. "
            "Use 2-4 options by default. Use 5-8 only when the approved field-value "
            "summaries or a standard closed scale required by the task establish a larger mutually "
            "exclusive label set. Correct-answer candidates must be supported by the approved annotations. "
            "Distractors must be medically plausible alternatives for the same question, imaging modality, "
            "semantic axis, evidence level, and diagnostic granularity. Options must be concise, unique, "
            "mutually exclusive, and written without letter prefixes. Do not enumerate an exhaustive "
            "differential diagnosis, copy a disease list from the PDF excerpts, or add rare entities merely "
            "to increase the number of distractors. Use one semantic axis and one level of granularity. Do not "
            "mix attributes, finding categories, assessments, diagnoses, or pathology results. For categorical "
            "attributes, use concise category labels rather than verbose explanations or copied annotation "
            "descriptions. Every letter task must have at least two dataset-supported "
            "semantic classes that can serve as correct answers; distractors do not make a constant-label "
            "task valid.\n\n"
            "Answer-mapping rules: select exactly one answer_field from the approved supporting fields. "
            "Copy raw values exactly from the approved field-value summary, preserving JSON type, case, "
            "spelling, and punctuation. Map each raw value to exactly one returned option, and support at "
            "least two distinct correct-answer options. Many-to-one mapping is allowed only within the same "
            "annotated variable when every raw value uniquely determines the declared coarser option. Do not "
            "map between clinical variables or ontology levels. Omit values that cannot produce one "
            "unambiguous option; omitted values are ineligible. generation_logic must not add or override "
            "the structured mapping."
        )
    system += (
        "\n\nWhen no relevant PDF excerpt is supplied, use generic medical wording and do not claim "
        "specific guideline support."
    )
    user = (
        f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"\nTask ID: {task_id}\nTask: {json.dumps(task, ensure_ascii=False)}\n"
        f"\nRequested revision: {feedback or 'none'}\n"
        f"Current design: {json.dumps(current_design or {}, ensure_ascii=False)}\n"
        f"\nApproved field-value summaries: {json.dumps(field_value_summaries or {}, ensure_ascii=False)}\n"
        f"\nPDF excerpts:\n{context or '[No sufficiently relevant PDF excerpt was retrieved.]'}"
    )
    return system, user


def _field_value_summaries(
    task_id: str,
    task: dict[str, Any],
    dataset_profile: dict[str, Any],
    dataset_access_manifest: dict[str, Any],
    response: object,
) -> dict[str, dict[str, object]] | None:
    """Validate an optional on-demand request for safe categorical field summaries."""
    if not isinstance(response, dict) or _VALUE_REQUEST_KEY not in response:
        return None
    if set(response) != {_VALUE_REQUEST_KEY}:
        raise ValueError(f"{task_id}: field-value request must contain only {_VALUE_REQUEST_KEY}.")
    requested = response[_VALUE_REQUEST_KEY]
    if not isinstance(requested, list) or len(requested) != 1 or not isinstance(requested[0], str):
        raise ValueError(f"{task_id}: field-value request must contain exactly one path.")
    supporting_fields = task.get("supporting_fields")
    if not isinstance(supporting_fields, list):
        raise ValueError(f"{task_id}: supporting_fields must be present before requesting values.")
    fields = dataset_profile.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"{task_id}: dataset profile must contain fields before requesting values.")

    database = dataset_access_manifest.get("duckdb_path")
    if not isinstance(database, str):
        raise ValueError(f"{task_id}: dataset access manifest lacks duckdb_path.")
    path = requested[0]
    segments = [segment.removesuffix("[*]") for segment in path.split(".")]
    if path not in supporting_fields:
        raise ValueError(f"{task_id}: requested field is not a supporting field: {path}")
    if path.startswith("image.annotation.dicom_metadata.") or "bbox" in segments:
        raise ValueError(f"{task_id}: requested field is not eligible for value summaries: {path}")
    profile_field = fields.get(path)
    if not isinstance(profile_field, dict) or not isinstance(profile_field.get("top_values"), list):
        raise ValueError(f"{task_id}: requested field has no safe categorical summary: {path}")

    counts: Counter[str] = Counter()
    raw_values: dict[str, object] = {}
    connection = duckdb.connect(database, read_only=True)
    try:
        rows = connection.execute("SELECT record FROM images ORDER BY _source_row").fetchall()
    finally:
        connection.close()
    for row in rows:
        record = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        if not isinstance(record, dict):
            continue
        for evidence in scalar_evidence_candidates(record, path):
            value = evidence["value"]
            if isinstance(value, str) and len(value) > 200:
                continue
            canonical = canonical_answer_scalar(value)
            if canonical is not None:
                counts[canonical] += 1
                raw_values[canonical] = value
    ordered = sorted(counts, key=lambda value: (-counts[value], value))
    selected = ordered[:MAX_RAG_MAPPING_VALUES]
    summaries: dict[str, dict[str, object]] = {
        path: {
            "field": path,
            "distinct_value_count": len(counts),
            "values": [
                {"value": raw_values[canonical], "count": counts[canonical]}
                for canonical in selected
            ],
            "truncated": len(ordered) > len(selected),
        }
    }
    return summaries


def design_task(
    rag: LocalPdfRag,
    benchmark_info: dict[str, object],
    task_id: str,
    task: dict[str, Any],
    *,
    dataset_profile: dict[str, Any],
    dataset_access_manifest: dict[str, Any],
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
    field_value_summaries = _field_value_summaries(
        task_id, task, dataset_profile, dataset_access_manifest, response
    )
    if task.get("response_format") == "letter" and field_value_summaries is None:
        raise QuestionDesignError(
            f"{task_id}: letter design must request one answer-bearing field before design."
        )
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
        normalized = validate_task_design(task_id, task, response, field_value_summaries)
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
            normalized = validate_task_design(task_id, task, repaired, field_value_summaries)
        except ValueError as second_error:
            preview = json.dumps(repaired, ensure_ascii=False)[:500]
            raise QuestionDesignError(
                f"{task_id}: RAG design remained invalid after repair: {second_error}; response={preview}"
            ) from second_error
    normalized["citations"] = [
        {key: item[key] for key in ("pdf", "page", "chunk_id")} for item in citations
    ]
    return normalized
