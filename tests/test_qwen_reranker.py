import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from medbenchagent.qwen_reranker import clear_reranker_cache, score_pairs


class _FakeTokenizer:
    loaded: tuple[str, dict] | None = None
    texts: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.loaded = (path, kwargs)
        return cls()

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"yes": 3, "no": 4}[token]

    def encode(self, text: str, **_kwargs) -> list[int]:
        return [91] if "system" in text else [92]

    def __call__(self, texts: list[str], **_kwargs) -> dict[str, list[list[int]]]:
        type(self).texts = texts
        return {"input_ids": [[10, 11], [12]]}

    def pad(self, inputs, **_kwargs):
        padded = [[0] * (4 - len(ids)) + ids for ids in inputs["input_ids"]]
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]),
        }


class _FakeModel:
    loaded_path: str | None = None
    load_count = 0
    device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, path: str):
        cls.loaded_path = path
        cls.load_count += 1
        return cls()

    def eval(self):
        return self

    def __call__(self, input_ids, **_kwargs):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 5))
        logits[0, -1, 3] = 2
        logits[1, -1, 4] = 2
        return SimpleNamespace(logits=logits)


def test_score_pairs_uses_qwen_causal_lm_yes_no_scoring(monkeypatch) -> None:
    clear_reranker_cache()
    _FakeModel.load_count = 0
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=_FakeModel,
        AutoTokenizer=_FakeTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    scores = score_pairs("/models/qwen-reranker", [("query one", "doc one"), ("query two", "doc two")])

    expected_path = str(Path("/models/qwen-reranker").resolve())
    assert _FakeModel.loaded_path == expected_path
    assert _FakeTokenizer.loaded == (expected_path, {"padding_side": "left"})
    assert "<Query>: query one\n<Document>: doc one" in _FakeTokenizer.texts[0]
    assert scores == pytest.approx([0.880797, 0.119203])

    score_pairs("/models/qwen-reranker", [("query one", "doc one")])
    assert _FakeModel.load_count == 1


def test_score_pairs_skips_model_loading_for_empty_input() -> None:
    assert score_pairs("/models/qwen-reranker", []) == []
