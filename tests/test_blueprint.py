import json

from medbenchagent.settings import get_settings


def test_blueprint_defines_all_categories_and_spatial_tasks() -> None:
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    tasks = blueprint["tasks"]

    assert all("metric" not in task for task in tasks.values())

    assert tasks["visual_grounding"]["response_format"] == "bbox"
    assert tasks["detection"]["response_format"] == "bbox"
    assert tasks["detection"]["category"] == "spatial_localization"
    assert tasks["visual_grounding"]["category"] == "spatial_localization"
    assert tasks["visual_grounding"]["description"] == (
        "Locate the single image region corresponding to one clearly specified, localized, "
        "image-observable finding or imaging sign."
    )
    assert "mask" not in tasks["detection"]["require"].lower()
    assert "mask" not in tasks["visual_grounding"]["require"].lower()
    assert "do not make detection unsupported" in tasks["detection"]["require"]
    assert "structured finding label" in tasks["visual_grounding"]["require"]
    assert "natural-language finding description" in tasks["finding_description"]["require"]
    assert "do not support this task by themselves" in tasks["finding_description"]["require"]
    assert "pathology-confirmed diagnosis" in tasks["diagnosis"]["require"]
    assert "benign or malignant status" in tasks["diagnosis"]["require"]
    assert {task["category"] for task in tasks.values()} == {
        "image_acquisition_and_basic_information",
        "anatomy_and_global_tissue_characteristics",
        "finding_recognition_and_characterization",
        "clinical_assessment_and_diagnosis",
        "report_generation",
        "spatial_localization",
    }
    assert "at least two semantically distinct" in tasks["modality_identification"]["require"]
    assert "at least two semantically distinct" in tasks["anatomy_identification"]["require"]
    assert "binary presence or absence" in tasks["finding_presence"]["require"]
    assert "direct categorical label" in tasks["finding_presence"]["require"]
    assert "explicit positive and negative" in tasks["finding_presence"]["require"]
    assert tasks["assessment_category"]["description"] == (
        "Determine the standardized reporting category assigned to an imaging finding or case."
    )
    assert "formal clinical reporting or risk-stratification scheme" in tasks["assessment_category"]["require"]
    assert "predefined clinical meanings" in tasks["assessment_category"]["require"]
    assert tasks["severity_grading"]["description"] == (
        "Determine how severe or extensive an imaging abnormality or disease is, expressed as an ordered grade or stage."
    )
    assert "directly encodes the severity or extent" in tasks["severity_grading"]["require"]
    assert "laterality_identification" not in tasks
    assert "missing finding annotations or bounding boxes" in tasks["finding_presence"]["require"]
    assert "masks" not in tasks["finding_presence"]["require"]
