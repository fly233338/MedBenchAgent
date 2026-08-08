"""Metric registry shared by task planning, export, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """One metric's evaluation contract and model-selection guidance."""

    name: str
    supported_response_formats: tuple[str, ...]
    description: str
    selection_rule: str


METRIC_REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="accuracy",
        supported_response_formats=("letter",),
        description="Exact-match correctness across all categorical answers.",
        selection_rule="Use as the default letter-task metric.",
    ),
    MetricSpec(
        name="macro_f1",
        supported_response_formats=("letter",),
        description="Unweighted mean of per-class F1 scores.",
        selection_rule=(
            "Choose only when the supplied most- and least-frequent class statistics show a large "
            "disparity, or when the task clearly represents a high-risk diagnostic decision for "
            "which class-wise performance matters; otherwise choose accuracy."
        ),
    ),
    MetricSpec(
        name="bleu",
        supported_response_formats=("free_text",),
        description="Corpus-level n-gram similarity between generated and reference text.",
        selection_rule="Use for free-text reference answers.",
    ),
    MetricSpec(
        name="miou",
        supported_response_formats=("bbox",),
        description="Mean intersection-over-union between predicted and reference boxes.",
        selection_rule="Use for bounding-box localization answers.",
    ),
)


def metrics_for_response_format(response_format: str) -> tuple[MetricSpec, ...]:
    """Return registered metrics compatible with one response format."""
    return tuple(
        spec
        for spec in METRIC_REGISTRY
        if response_format in spec.supported_response_formats
    )


def metrics_by_response_format() -> dict[str, set[str]]:
    """Return the evaluator's response-format-to-metric compatibility map."""
    formats = {
        response_format
        for spec in METRIC_REGISTRY
        for response_format in spec.supported_response_formats
    }
    return {
        response_format: {spec.name for spec in metrics_for_response_format(response_format)}
        for response_format in sorted(formats)
    }


def is_metric_supported(response_format: object, metric: object) -> bool:
    """Check a metric name without relying on model judgement."""
    return (
        isinstance(response_format, str)
        and isinstance(metric, str)
        and metric in {spec.name for spec in metrics_for_response_format(response_format)}
    )


def metric_prompt_catalog(response_format: str) -> str:
    """Render compatible metrics and their selection rules for an LLM prompt."""
    specs = metrics_for_response_format(response_format)
    if not specs:
        return f"No registered metric is compatible with response format `{response_format}`."
    lines = [f"Registered metrics compatible with response format `{response_format}`:"]
    for spec in specs:
        lines.append(f"- `{spec.name}`: {spec.description} {spec.selection_rule}")
    return "\n".join(lines)
