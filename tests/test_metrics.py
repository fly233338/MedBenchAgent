from medbenchagent.metrics import (
    is_metric_supported,
    metric_prompt_catalog,
    metrics_by_response_format,
)


def test_metric_registry_matches_current_response_format_contract() -> None:
    assert metrics_by_response_format() == {
        "bbox": {"miou"},
        "free_text": {"bleu"},
        "letter": {"accuracy", "macro_f1"},
    }
    assert is_metric_supported("letter", "accuracy")
    assert is_metric_supported("letter", "macro_f1")
    assert is_metric_supported("free_text", "bleu")
    assert is_metric_supported("bbox", "miou")
    assert not is_metric_supported("free_text", "accuracy")


def test_metric_prompt_catalog_contains_selection_guidance() -> None:
    prompt = metric_prompt_catalog("letter")

    assert "accuracy" in prompt
    assert "macro_f1" in prompt
    assert "balanced" in prompt
    assert "minority" in prompt
