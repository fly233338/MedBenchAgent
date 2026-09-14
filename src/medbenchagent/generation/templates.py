"""Build reviewed templates once, then render questions without per-item model calls."""

from __future__ import annotations

import json
from string import Formatter

from medbenchagent.generation.context import image_modality
from medbenchagent.generation.grounding import _visual_target_text
from medbenchagent.runtime.llm import complete_json
from medbenchagent.settings import get_settings
from medbenchagent.planning.protocol import TemplateSpecification


def validate_templates(specification: TemplateSpecification, templates: object) -> list[str]:
    specification = TemplateSpecification.model_validate(specification)
    count = get_settings().generation.question_template_count
    if (
        not isinstance(templates, list)
        or len(templates) != count
        or not all(isinstance(t, str) and t.strip() for t in templates)
        or len({t.strip() for t in templates}) != len(templates)
    ):
        raise ValueError(f"Provide {count} distinct nonempty templates")
    allowed = set(specification.template_slots)
    for template in templates:
        fields = set()
        for _, field, format_spec, conversion in Formatter().parse(template):
            if field is not None:
                if field not in allowed or format_spec or conversion:
                    raise ValueError("Unsupported template slot")
                fields.add(field)
        if "target" in allowed and "target" not in fields:
            raise ValueError(
                "Visual grounding templates must name the annotated target using {target}"
            )
    return [t.strip() for t in templates]


def build_templates(
    specification: TemplateSpecification,
    feedback: str = "",
    *,
    temperature: float | None = None,
) -> list[str]:
    specification = TemplateSpecification.model_validate(specification)
    count = get_settings().generation.question_template_count
    prompt = (
        f"Create exactly {count} semantically equivalent but varied ready-to-render medical VLM question templates. "
        'Return {"templates":["question"]}. No instructions to a future writer. '
        "Use only the specification template_slots; when target is listed, MUST use {target}. "
        "Other tasks must not use {target}. Never insert an answer or annotation label into the question. "
        "Detection asks for a generic lesion/abnormal region without naming its finding. "
        "Free text asks for the reference content without revealing it. Use the requested language. "
        "All templates must apply to every eligible record under this task; do not assume unprovided facts."
    )
    payload = {
        "specification": specification.model_dump(),
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
            templates = validate_templates(specification, response.get("templates"))
            review_templates(specification, templates, payload["feedback"])
            return templates
        except ValueError as exc:
            if attempt:
                raise
            payload.update(previous=response, feedback=str(exc))
    raise ValueError("Template construction failed")


def review_templates(
    specification: TemplateSpecification, templates: list[str], feedback: str = ""
) -> None:
    """Review templates using only their narrow specification and editorial feedback."""
    specification = TemplateSpecification.model_validate(specification)
    review = complete_json(
        "Review all supplied templates for task consistency, answer leakage, ambiguity, applicable slots, "
        'and meaningful expression diversity. Return {"approved":true|false,"reason":"specific reason"}.',
        json.dumps(
            {
                "specification": specification.model_dump(),
                "templates": templates,
                "feedback": feedback,
            },
            ensure_ascii=False,
        ),
    )
    if review.get("approved") is not True:
        raise ValueError(str(review.get("reason", "Template review failed")))


def render(candidate) -> str:
    named = "target" in candidate.task["protocol"]["template_slots"]
    target = _visual_target_text(candidate) if named else ""
    if named and not target:
        raise ValueError("Visual grounding requires one annotated target")
    return candidate.question_template.format(modality=image_modality(candidate), target=target)
