"""Canonical answer-mapping values shared by generation and RAG design."""

from __future__ import annotations

import json


def canonical_answer_scalar(value: object) -> str | None:
    """Serialize one non-empty JSON scalar for exact mapping comparisons."""
    if value is None or value == "" or isinstance(value, (dict, list)):
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return None
