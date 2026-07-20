"""Qwen3 causal-LM reranking using the model's native yes/no scoring recipe."""

from __future__ import annotations

from pathlib import Path

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


def score_pairs(
    model_path: str | Path,
    pairs: list[tuple[str, str]],
    *,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
    max_length: int = 8_192,
) -> list[float]:
    """Score query-document pairs with an unmodified local Qwen3-Reranker checkpoint."""
    if not pairs:
        return []

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    false_token_id = tokenizer.convert_tokens_to_ids("no")
    true_token_id = tokenizer.convert_tokens_to_ids("yes")
    prefix_tokens = tokenizer.encode(_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(_SUFFIX, add_special_tokens=False)
    texts = [_format_instruction(instruction, query, document) for query, document in pairs]
    inputs = tokenizer(
        texts,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for index, input_ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        final_logits = model(**inputs).logits[:, -1, :]
        binary_logits = torch.stack(
            [final_logits[:, false_token_id], final_logits[:, true_token_id]], dim=1
        )
        return torch.nn.functional.log_softmax(binary_logits, dim=1)[:, 1].exp().tolist()
