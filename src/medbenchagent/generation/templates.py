"""Build reviewed templates once, then render questions without per-item model calls."""

from __future__ import annotations

import json
from string import Formatter
from typing import Literal

from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty

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


class TemplateReview(NoArguments):
    verdict: Literal["approved", "revise_templates", "revise_plan"]
    reason: NonEmpty


class TemplateSet(NoArguments):
    templates: list[NonEmpty]


def create_templates(specification: TemplateSpecification, feedback="", *, temperature=None):
    """Generate a template set; the separate template stage owns review and retries."""
    specification = TemplateSpecification.model_validate(specification)
    response = complete_json(
        f"Create exactly {get_settings().generation.question_template_count} varied, semantically equivalent medical VLM question templates. "
        'Return {"templates":["question"]}. Use only specification template_slots; target requires {target}. '
        "Never reveal answers or annotation labels. Detection asks for a generic lesion; free text asks for the reference "
        "without revealing it. Use the requested language. Every template must apply to every eligible record. "
        "Do not assume facts absent from the specification.",
        json.dumps({"specification": specification.model_dump(), "feedback": feedback}, ensure_ascii=False),
        TemplateSet, lambda parsed: validate_templates(specification, parsed.templates),
        temperature=get_settings().generation.question_temperature if temperature is None else temperature,
    )
    return validate_templates(specification, response.get("templates"))


def review_templates(specification: TemplateSpecification, templates: list[str], feedback="") -> dict:
    """Review a whole set once and distinguish wording corrections from invalid task specifications."""
    specification = TemplateSpecification.model_validate(specification)
    return TemplateReview.model_validate(complete_json(
        "Review all supplied templates for task consistency, answer leakage, ambiguity, slots, meaningful expression "
        "diversity and applicability to every eligible record described by the specification. "
        "Use revise_templates for wording/slot defects; use revise_plan when the task specification or constraints "
        "must change before valid questions can be written. Do not invent labels or research preferences. "
        'Return {"verdict":"approved|revise_templates|revise_plan","reason":"specific reason"}.',
        json.dumps({"specification": specification.model_dump(), "templates": templates, "feedback": feedback}, ensure_ascii=False),
        TemplateReview,
    )).model_dump()


def render(candidate) -> str:
    named = "target" in candidate.task["protocol"]["template_slots"]
    target = _visual_target_text(candidate) if named else ""
    if named and not target:
        raise ValueError("Visual grounding requires one annotated target")
    return candidate.question_template.format(modality=image_modality(candidate), target=target)
