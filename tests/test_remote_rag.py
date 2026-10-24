import json

import httpx
import pytest

from medbenchagent.rag.remote import RagUnavailable, embed_texts, rerank_documents


def test_embedding_batches_restore_response_order(monkeypatch):
    monkeypatch.setattr("medbenchagent.rag.remote.EMBEDDING_BATCH_SIZE", 2)
    seen = []

    def respond(request):
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content)
        seen.append(payload["input"])
        data = [
            {"index": i, "embedding": [float(int(text)), 1.0]}
            for i, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        vectors = embed_texts(client, "http://embed/v1", "qwen-embed", ["1", "2", "3"])

    assert seen == [["1", "2"], ["3"]]
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_rerank_maps_sorted_results_back_to_candidates():
    def respond(request):
        assert request.url.path == "/v1/rerank"
        payload = json.loads(request.content)
        assert payload == {
            "model": "qwen-rerank",
            "query": "question",
            "documents": ["first", "second"],
            "top_n": 2,
            "truncate_prompt_tokens": 8192,
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        scores = rerank_documents(
            client,
            "http://rerank/v1",
            "qwen-rerank",
            "question",
            ["first", "second"],
            max_length=8192,
        )
    assert scores == [0.2, 0.9]


@pytest.mark.parametrize(
    "body",
    [
        {"results": [{"index": 0, "relevance_score": 0.5}]},
        {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.6}]},
        {
            "results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": float("nan")},
            ]
        },
    ],
)
def test_rerank_rejects_incomplete_or_invalid_scores(body):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(RagUnavailable, match="Reranker API"):
            rerank_documents(client, "http://rerank/v1", "model", "q", ["a", "b"], max_length=1024)


def test_configured_service_error_is_reported():
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable"))
    ) as client:
        with pytest.raises(RagUnavailable, match="Embedding API request failed"):
            embed_texts(client, "http://embed/v1", "model", ["text"])
