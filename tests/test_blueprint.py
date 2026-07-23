import json
from pathlib import Path


def test_blueprint_defines_all_categories_and_spatial_tasks() -> None:
    blueprint = json.loads((Path(__file__).parents[1] / "blueprint.json").read_text(encoding="utf-8"))
    tasks = blueprint["tasks"]

    assert tasks["visual_grounding"]["response_format"] == "bbox"
    assert tasks["detection"]["response_format"] == "bbox"
    assert tasks["detection"]["category"] == "spatial_localization"
    assert tasks["visual_grounding"]["category"] == "spatial_localization"
    assert {task["category"] for task in tasks.values()} == {
        "image_acquisition_and_basic_information",
        "anatomy_and_global_tissue_characteristics",
        "finding_recognition_and_characterization",
        "clinical_assessment_and_diagnosis",
        "report_generation",
        "spatial_localization",
    }
    assert "only one modality" in tasks["modality_identification"]["require"]
    assert "Missing bounding boxes" in tasks["finding_presence"]["require"]
