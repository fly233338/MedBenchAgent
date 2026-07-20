"""Qwen3 causal-LM reranking using the model's native yes/no scoring recipe."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

DEFAULT_RERANK_INSTRUCTION = (
    "Given a medical benchmark design query, retrieve relevant medical reference passages "
    "that answer the query"
)
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _format_instruction(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


class QwenReranker:
    """Loaded Qwen3 reranker reused by every retrieval in one worker process."""

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(Path(model_path).resolve())
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(path).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def score(
        self,
        pairs: list[tuple[str, str]],
        *,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
        max_length: int = 8_192,
    ) -> list[float]:
        if not pairs:
            return []

        false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix_tokens = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)
        texts = [_format_instruction(instruction, query, document) for query, document in pairs]
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with self._torch.no_grad():
            final_logits = self.model(**inputs).logits[:, -1, :]
            binary_logits = self._torch.stack(
                [final_logits[:, false_token_id], final_logits[:, true_token_id]], dim=1
            )
            return self._torch.nn.functional.log_softmax(binary_logits, dim=1)[:, 1].exp().tolist()


_RERANKERS: dict[str, QwenReranker] = {}
_RERANKER_LOCK = Lock()


def get_reranker(model_path: str | Path) -> QwenReranker:
    """Load one reranker per resolved model path and worker process."""
    key = str(Path(model_path).resolve())
    with _RERANKER_LOCK:
        if key not in _RERANKERS:
            _RERANKERS[key] = QwenReranker(key)
        return _RERANKERS[key]


def clear_reranker_cache() -> None:
    """Clear process-local models; intended for isolated tests."""
    with _RERANKER_LOCK:
        _RERANKERS.clear()


def score_pairs(
    model_path: str | Path,
    pairs: list[tuple[str, str]],
    *,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
    max_length: int = 8_192,
) -> list[float]:
    """Score pairs while reusing the process-local Qwen3 reranker."""
    if not pairs:
        return []
    return get_reranker(model_path).score(
        pairs,
        instruction=instruction,
        max_length=max_length,
    )
