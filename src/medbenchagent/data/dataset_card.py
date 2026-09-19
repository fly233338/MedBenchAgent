"""Deterministic dataset facts with validated semantic hints."""

import json
from pathlib import Path
from typing import Literal
from pydantic import Field
from medbenchagent.data.contracts import Contract, Text
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


class CapabilityHint(Contract):
    concept_id: Text
    fields: list[Text] = Field(min_length=1)
    reason: Text


class CardSemantics(Contract):
    description: Text
    clinical_domain: str | None = None
    field_roles: dict[str, str] = Field(default_factory=dict)
    capability_hints: list[CapabilityHint] = Field(default_factory=list)


class DatasetCard(Contract):
    schema_version: Literal["dataset-card.v1"] = "dataset-card.v1"
    dataset_id: Text
    name: Text
    annotation_digest: Text
    profile_digest: Text
    image_manifest_digest: Text
    blueprint_digest: Text
    total_images: int = Field(ge=0)
    summary: dict
    fields: dict
    structures: list[dict]
    semantics: CardSemantics


def validate_semantics(semantics, profile, concepts):
    value = CardSemantics.model_validate(semantics)
    paths = set(profile["fields"]) | {s["path"] for s in profile["structures"]}
    referenced = set(value.field_roles)
    for hint in value.capability_hints:
        if hint.concept_id not in concepts:
            raise ValueError(f"Unknown Card Concept: {hint.concept_id}")
        referenced.update(hint.fields)
    if referenced - paths:
        raise ValueError(f"Unknown Card annotation paths: {sorted(referenced - paths)}")
    return value


def build_card(dataset_id, bundle, *, name=None, description=""):
    bundle = Path(bundle).resolve()
    access = json.loads((bundle / "ready.json").read_text(encoding="utf-8"))["access"]
    profile_path = Path(access["profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    blueprint = get_settings().paths.blueprint_template
    concepts = json.loads(blueprint.read_text(encoding="utf-8"))["concepts"]
    identities = {"annotation_digest": access["annotation_digest"],
                  "profile_digest": digest_file(profile_path),
                  "image_manifest_digest": digest_file(bundle / "image_manifest.jsonl"),
                  "blueprint_digest": digest_file(blueprint)}
    destination = bundle / "dataset_card.json"
    if destination.is_file():
        card = DatasetCard.model_validate_json(destination.read_text(encoding="utf-8"))
        if card.dataset_id != dataset_id or any(getattr(card, k) != v for k, v in identities.items()):
            raise ValueError("Card inputs changed; prepare a new dataset version")
        return card
    semantics = complete_json(
        "Describe only this prepared medical dataset using supplied annotation facts and source description. "
        "Return the specified JSON schema. Unknown clinical_domain is null. Field roles and capability hints "
        "must cite existing paths. Hints are tentative, not validated Protocol support. Never invent fields, "
        "labels or clinical facts. Do not infer absent annotations as negatives.\n" + json.dumps(CardSemantics.model_json_schema()),
        json.dumps({"dataset_id": dataset_id, "description": description,
                    "summary": profile["summary"], "fields": profile["fields"],
                    "structures": profile["structures"], "concepts": concepts}, ensure_ascii=False),
    )
    card = DatasetCard(dataset_id=dataset_id, name=name or dataset_id, **identities,
                       total_images=profile["total_images"], summary=profile["summary"],
                       fields=profile["fields"], structures=profile["structures"],
                       semantics=validate_semantics(semantics, profile, concepts))
    atomic_write_json(destination, card.model_dump())
    return card


def card_summary(card):
    return {"dataset_id": card.dataset_id, "name": card.name, "total_images": card.total_images,
            "description": card.semantics.description, "clinical_domain": card.semantics.clinical_domain,
            "modality": card.summary.get("modality"), "anatomy": card.summary.get("anatomy"),
            "capability_hints": [h.model_dump() for h in card.semantics.capability_hints]}
