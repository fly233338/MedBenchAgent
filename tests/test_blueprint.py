import json

from medbenchagent.settings import get_settings


def test_blueprint_contains_static_core_concepts() -> None:
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    assert set(blueprint) == {"benchmark_info", "concepts"}
    concepts = blueprint["concepts"]
    assert concepts
    categories = {
        "image_acquisition_and_basic_information",
        "anatomy_and_global_tissue_characteristics",
        "finding_recognition_and_characterization",
        "clinical_assessment_and_diagnosis",
        "report_generation",
        "spatial_localization",
    }
    for concept_id, concept in concepts.items():
        assert concept_id.strip()
        assert set(concept) == {"description", "require", "category", "origin"}
        assert concept["origin"] == "core"
        assert concept["category"] in categories
        assert isinstance(concept["description"], str) and concept["description"].strip()
        assert isinstance(concept["require"], str) and concept["require"].strip()
    assert {concept["category"] for concept in concepts.values()} == categories
