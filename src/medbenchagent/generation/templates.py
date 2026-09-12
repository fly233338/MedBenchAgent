"""Build reviewed templates once, then render questions without per-item model calls."""

from __future__ import annotations

import json
from string import Formatter

from medbenchagent.generation.context import image_modality
from medbenchagent.generation.grounding import _visual_target_text
from medbenchagent.llm import complete_json
from medbenchagent.settings import get_settings


def validate_templates(task_id: str, task: dict, templates: object) -> list[str]:
    count = get_settings().generation.question_template_count
    if (
        not isinstance(templates, list)
        or len(templates) != count
        or not all(isinstance(t, str) and t.strip() for t in templates)
        or len({t.strip() for t in templates}) != len(templates)
    ):
        raise ValueError(f"{task_id}: provide {count} distinct nonempty templates")
    allowed = {"modality"}
    if task_id == "visual_grounding":
        allowed.add("target")
    for template in templates:
        fields = set()
        for _, field, format_spec, conversion in Formatter().parse(template):
            if field is not None:
                if field not in allowed or format_spec or conversion:
                    raise ValueError(f"{task_id}: unsupported template slot")
                fields.add(field)
        if task_id == "visual_grounding" and "target" not in fields:
            raise ValueError(
                "Visual grounding templates must name the annotated target using {target}"
            )
    return [t.strip() for t in templates]


def build_templates(
    task_id: str,
    task: dict,
    benchmark_info: dict,
    feedback: str = "",
    *,
    temperature: float | None = None,
) -> list[str]:
    count = get_settings().generation.question_template_count
    prompt = (
        f"Create exactly {count} semantically equivalent but varied ready-to-render medical VLM question templates. "
        'Return {"templates":["question"]}. No instructions to a future writer. '
        "Only optional {modality} is supported; visual_grounding MUST use {target} to identify the finding. "
        "Other tasks must not use {target}. Never insert an answer or annotation label into the question. "
        "Detection asks for a generic lesion/abnormal region without naming its finding. "
        "Free text asks for the reference content without revealing it. Use the requested language. "
        "All templates must apply to every eligible record under this task; do not assume unprovided facts."
    )
    payload = {
        "task_id": task_id,
        "task": task,
        "benchmark_info": benchmark_info,
        "feedback": feedback,
    }
    for attempt in range(2):
        response = complete_json(
            prompt,
            json.dumps(payload, ensure_ascii=False),
            temperature=get_settings().generation.question_temperature
            if temperature is None
            else temperature,
        )
        try:
            templates = validate_templates(task_id, task, response.get("templates"))
            review = complete_json(
                "Review all supplied templates for task consistency, answer leakage, ambiguity, applicable slots, "
                'and meaningful expression diversity. Return {"approved":true|false,"reason":"specific reason"}.',
                json.dumps({**payload, "templates": templates}, ensure_ascii=False),
            )
            if review.get("approved") is True:
                return templates
            raise ValueError(str(review.get("reason", "Template review failed")))
        except ValueError as exc:
            if attempt:
                raise
            payload.update(previous=response, feedback=str(exc))
    raise ValueError("Template construction failed")


def render(candidate) -> str:
    target = _visual_target_text(candidate) if candidate.task_id == "visual_grounding" else ""
    if candidate.task_id == "visual_grounding" and not target:
        raise ValueError("Visual grounding requires one annotated target")
    return candidate.question_template.format(modality=image_modality(candidate), target=target)
