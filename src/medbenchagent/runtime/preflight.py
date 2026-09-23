"""Shared local capability checks, independent of the CLI entry point."""

import json
from urllib.parse import urlparse

from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.rag.pdf import RagUnavailable, discover_medical_materials


def rag_capability(settings):
    reasons = []
    for label, base_url, model in (
        ("Embedding", settings.rag.embedding_base_url, settings.rag.embedding_model),
        ("Reranker", settings.rag.reranker_base_url, settings.rag.reranker_model),
    ):
        if not base_url or not model:
            reasons.append(f"{label} endpoint and model must be configured")
        else:
            url = urlparse(base_url)
            if url.scheme not in {"http", "https"} or not url.netloc:
                reasons.append(f"{label} base_url must be an HTTP(S) URL")
    try:
        discover_medical_materials(settings.paths.materials_root)
    except (RagUnavailable, OSError) as exc:
        reasons.append(str(exc))
    return {"available": not reasons, "reasons": reasons}


def preflight_errors(settings, *, build=False):
    """Require only resources needed by this intent; retrieval remains optional."""
    errors = []
    if not settings.llm.model.strip():
        errors.append("LLM model must be configured")
    url = urlparse(settings.llm.base_url)
    if url.scheme not in {"http", "https"} or not url.netloc:
        errors.append("LLM base_url must be an HTTP(S) URL")
    try:
        load_dataset_catalog(settings.paths.dataset_catalog)
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
    if build:
        path = settings.paths.blueprint_template
        if not path.is_file():
            errors.append(f"Blueprint template does not exist: {path}")
        else:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict) or not isinstance(value.get("concepts"), dict) or not value["concepts"]:
                    errors.append("Blueprint template requires nonempty concepts")
            except (ValueError, OSError) as exc:
                errors.append(f"Cannot read Blueprint template: {exc}")
    return errors
