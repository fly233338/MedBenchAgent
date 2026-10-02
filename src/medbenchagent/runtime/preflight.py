"""Shared local capability checks, independent of the CLI entry point."""

from urllib.parse import urlparse

from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.rag.pdf import RagUnavailable, discover_medical_materials


def rag_capability(settings):
    reasons = []
    for label, base_url, model in (
        ("Embedding", settings.rag.embedding_base_url, settings.rag.embedding_model),
        ("Reranker", settings.rag.reranker_base_url, settings.rag.reranker_model),
    ):
        status = settings.service_status.get(label.lower())
        if status and status != "available":
            reasons.append(status)
        elif not base_url or not model:
            reasons.append(f"{label} endpoint or resolved model is unavailable")
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
    url = urlparse(settings.llm.base_url)
    if url.scheme not in {"http", "https"} or not url.netloc:
        errors.append("LLM base_url must be an HTTP(S) URL")
    try:
        load_dataset_catalog(settings.paths.dataset_catalog)
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
    if build:
        path = settings.paths.bir_template
        if not path.is_dir():
            errors.append(f"BIR template does not exist: {path}")
        else:
            try:
                from medbenchagent.planning.bir import BIRStore
                BIRStore.load_template(path)
            except (ValueError, OSError) as exc:
                errors.append(f"Cannot read BIR template: {exc}")
    return errors
