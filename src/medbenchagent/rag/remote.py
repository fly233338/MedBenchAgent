"""HTTP clients for independently served embedding and reranking models."""

from __future__ import annotations

import math

import httpx

from medbenchagent.settings import get_rag_api_key

EMBEDDING_BATCH_SIZE = 64
REQUEST_TIMEOUT_SECONDS = 120


class RagUnavailable(RuntimeError):
    """Raised when medical references or remote retrieval models are unavailable."""


def _post(client: httpx.Client, base_url: str, route: str, payload: dict, kind: str) -> dict:
    key = get_rag_api_key(kind)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = client.post(f"{base_url.rstrip('/')}/{route}", json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RagUnavailable(f"{kind.capitalize()} API request failed: {exc}") from exc
    if not isinstance(body, dict):
        raise RagUnavailable(f"{kind.capitalize()} API returned an invalid response")
    return body


def embed_texts(
    client: httpx.Client, base_url: str, model: str, texts: list[str]
) -> list[list[float]]:
    """Embed in bounded batches, restoring the input order from response indices."""
    vectors: list[list[float]] = []
    dimension: int | None = None
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        body = _post(
            client,
            base_url,
            "embeddings",
            {"model": model, "input": batch, "encoding_format": "float"},
            "embedding",
        )
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise RagUnavailable("Embedding API returned the wrong number of vectors")
        ordered: list[list[float] | None] = [None] * len(batch)
        for item in data:
            if not isinstance(item, dict):
                raise RagUnavailable("Embedding API returned an invalid vector")
            index, values = item.get("index"), item.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(batch)
                or ordered[index] is not None
                or not isinstance(values, list)
                or not values
                or any(
                    not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
                    for v in values
                )
            ):
                raise RagUnavailable("Embedding API returned an invalid vector")
            if dimension is None:
                dimension = len(values)
            if len(values) != dimension:
                raise RagUnavailable("Embedding API returned inconsistent vector dimensions")
            ordered[index] = [float(v) for v in values]
        if any(vector is None for vector in ordered):
            raise RagUnavailable("Embedding API omitted a vector")
        vectors.extend(ordered)
    return vectors


def rerank_documents(
    client: httpx.Client,
    base_url: str,
    model: str,
    query: str,
    documents: list[str],
    *,
    max_length: int,
) -> list[float]:
    """Return one vLLM rerank score per candidate in its original order."""
    if not documents:
        return []
    body = _post(
        client,
        base_url,
        "rerank",
        {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "truncate_prompt_tokens": max_length,
        },
        "reranker",
    )
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(documents):
        raise RagUnavailable("Reranker API returned the wrong number of scores")
    scores: list[float | None] = [None] * len(documents)
    for item in results:
        if not isinstance(item, dict):
            raise RagUnavailable("Reranker API returned an invalid score")
        index, score = item.get("index"), item.get("relevance_score")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(documents)
            or scores[index] is not None
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
        ):
            raise RagUnavailable("Reranker API returned an invalid score")
        scores[index] = float(score)
    if any(score is None for score in scores):
        raise RagUnavailable("Reranker API omitted a score")
    return scores
