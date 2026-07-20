"""Local, hash-cached PDF retrieval for question-design planning."""

from __future__ import annotations

import hashlib
import json
import os
import string
from pathlib import Path
from typing import Any
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


def _design_instruction(plan: dict[str, object]) -> str:
    response_format = plan.get("response_format")
    common = (
        "Use only supporting annotation fields. question_templates must be non-empty strings; "
        "placeholder_bindings maps simple placeholder names to supporting annotation fields. "
    )
    if response_format == "letter":
        return common + (
            "Return JSON with question_templates, placeholder_bindings, answer_list, answer_field, "
            "answer_value_mapping, validation_rules. answer_list contains 2-26 fixed unique option strings. "
            "answer_field is one supporting field. answer_value_mapping maps serialized source values "
            "to answer_list values."
        )
    if response_format == "bbox":
        return common + (
            "Return JSON with question_templates, placeholder_bindings, instance_path, target_text_field, "
            "bbox_field, bbox_format, validation_rules. instance_path identifies the annotation object array "
            "containing each finding. target_text_field and bbox_field are fields under that same array. "
            "bbox_field may be a four-coordinate scalar leaf path ending in [*]. bbox_format must be xyxy. "
            "Every template must use {target_text}."
        )
    if response_format == "free_text":
        return common + (
            "Return JSON with question_templates, placeholder_bindings, answer_field, validation_rules. "
            "answer_field must identify paired expert-written text. Do not synthesize the reference answer. "
            "Use instance_path only when the paired text belongs to an object in an annotation array."
        )
    raise ValueError(f"Unsupported response_format for Question Design: {response_format}")


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _literal, field_name, format_spec, conversion in string.Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier() or format_spec or conversion:
            raise ValueError("Question templates support only simple {placeholder} fields.")
        fields.add(field_name)
    return fields


def _field_in_scope(field: object, supporting_fields: set[str]) -> bool:
    return isinstance(field, str) and field in supporting_fields


def validate_question_designs(
    task_plans: list[dict[str, object]], designs: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Validate and normalize one task-type-aware design per retained task."""
    retained = [plan for plan in task_plans if plan.get("decision") == "retain"]
    expected = [str(plan["task_id"]) for plan in retained]
    by_task: dict[str, dict[str, object]] = {}
    for design in designs:
        task_id = design.get("task_id")
        if not isinstance(task_id, str) or task_id in by_task:
            raise ValueError("Question Designs must have unique non-empty task_id values.")
        by_task[task_id] = design
    if set(by_task) != set(expected):
        raise ValueError("Question Designs must contain exactly every retained task.")

    normalized: list[dict[str, object]] = []
    for plan in retained:
        task_id = str(plan["task_id"])
        design = by_task[task_id]
        supporting = set(plan.get("supporting_fields", []))
        templates = design.get("question_templates")
        if not isinstance(templates, list) or not templates or not all(
            isinstance(template, str) and template.strip() for template in templates
        ):
            raise ValueError(f"{task_id}: question_templates must be non-empty strings.")
        bindings = design.get("placeholder_bindings", {})
        if not isinstance(bindings, dict) or not all(
            isinstance(name, str) and name.isidentifier() and _field_in_scope(field, supporting)
            for name, field in bindings.items()
        ):
            raise ValueError(f"{task_id}: placeholder_bindings must reference supporting fields.")
        placeholders = set().union(*(_template_fields(template) for template in templates))
        if placeholders - set(bindings):
            raise ValueError(f"{task_id}: every template placeholder needs a binding.")
        rules = design.get("validation_rules", [])
        if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
            raise ValueError(f"{task_id}: validation_rules must be strings.")

        item: dict[str, Any] = {
            "task_id": task_id,
            "question_templates": [template.strip() for template in templates],
            "placeholder_bindings": dict(bindings),
            "validation_rules": rules,
            "citations": list(design.get("citations", [])),
        }
        response_format = plan.get("response_format")
        if response_format == "letter":
            options = design.get("answer_list")
            answer_field = design.get("answer_field")
            mapping = design.get("answer_value_mapping", {})
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 26
                or not all(isinstance(option, str) and option for option in options)
                or len(set(options)) != len(options)
            ):
                raise ValueError(f"{task_id}: answer_list must contain 2-26 unique strings.")
            if not _field_in_scope(answer_field, supporting):
                raise ValueError(f"{task_id}: answer_field must be a supporting field.")
            if not isinstance(mapping, dict) or not all(
                isinstance(key, str) and isinstance(value, str) and value in options
                for key, value in mapping.items()
            ):
                raise ValueError(f"{task_id}: answer_value_mapping must map strings to answer_list.")
            item.update(answer_list=options, answer_field=answer_field, answer_value_mapping=mapping)
        elif response_format == "bbox":
            instance_path = design.get("instance_path")
            target_field = design.get("target_text_field")
            bbox_field = design.get("bbox_field")
            if not isinstance(instance_path, str) or not instance_path.endswith("[*]"):
                raise ValueError(f"{task_id}: instance_path must identify an annotation array.")
            if not _field_in_scope(target_field, supporting) or not _field_in_scope(bbox_field, supporting):
                raise ValueError(f"{task_id}: detection fields must be supporting fields.")
            prefix = instance_path + "."
            if not str(target_field).startswith(prefix) or not str(bbox_field).startswith(prefix):
                raise ValueError(f"{task_id}: detection fields must belong to instance_path.")
            if design.get("bbox_format") != "xyxy":
                raise ValueError(f"{task_id}: bbox_format must be xyxy.")
            if bindings.get("target_text") != target_field or any(
                "target_text" not in _template_fields(template) for template in templates
            ):
                raise ValueError(f"{task_id}: detection templates must bind {{target_text}}.")
            item.update(
                instance_path=instance_path,
                target_text_field=target_field,
                bbox_field=bbox_field,
                bbox_format="xyxy",
            )
        elif response_format == "free_text":
            answer_field = design.get("answer_field")
            if not _field_in_scope(answer_field, supporting):
                raise ValueError(f"{task_id}: answer_field must be a supporting field.")
            if answer_field in bindings.values():
                raise ValueError(f"{task_id}: a question placeholder must not expose the reference answer.")
            item["answer_field"] = answer_field
            instance_path = design.get("instance_path")
            if instance_path is not None:
                if not isinstance(instance_path, str) or not instance_path.endswith("[*]"):
                    raise ValueError(f"{task_id}: instance_path must identify an annotation array.")
                if not str(answer_field).startswith(instance_path + "."):
                    raise ValueError(f"{task_id}: answer_field must belong to instance_path.")
                item["instance_path"] = instance_path
        else:
            raise ValueError(f"{task_id}: unsupported response_format {response_format}.")
        normalized.append(item)
    return normalized


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
                "medical phrasing. Do not alter or invent ground-truth labels. " + _design_instruction(plan),
                f"Benchmark: {json.dumps(benchmark_info)}\nTask plan: {json.dumps(plan)}\n"
                f"User design feedback: {feedback}\nPDF excerpts:\n{context}",
            )
        except ModelUnavailable:
            raise
        designs.append(
            {
                "task_id": plan["task_id"],
                **design,
                "answer_value_mapping": design.get("answer_value_mapping", design.get("source_value_mapping", {})),
                "citations": [{key: item[key] for key in ("pdf", "page", "chunk_id")} for item in citations],
            }
        )
    return validate_question_designs(task_plans, designs)
