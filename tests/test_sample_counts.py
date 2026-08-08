import json
from pathlib import Path

from medbenchagent.sample_counts import (
    distribute_weighted_sample_counts,
    write_sample_counts,
)


def test_weighted_distribution_normalizes_active_categories_and_splits_ties() -> None:
    tasks = {
        "acquisition": {"category": "image_acquisition_and_basic_information"},
        "anatomy_a": {"category": "anatomy_and_global_tissue_characteristics"},
        "anatomy_b": {"category": "anatomy_and_global_tissue_characteristics"},
        "spatial": {"category": "spatial_localization"},
    }

    assert distribute_weighted_sample_counts(tasks, 13) == {
        "acquisition": 3,
        "anatomy_a": 3,
        "anatomy_b": 2,
        "spatial": 5,
    }
    assert distribute_weighted_sample_counts(tasks, 13, {"anatomy_a": 5}) == {
        "acquisition": 2,
        "anatomy_a": 5,
        "anatomy_b": 3,
        "spatial": 3,
    }


def test_expanded_question_types_share_their_parent_task_weight() -> None:
    category = "finding_recognition_and_characterization"
    tasks = {
        "finding_type": {"category": category},
        "finding_attribute.shape": {
            "category": category,
            "parent_task_id": "finding_attribute",
        },
        "finding_attribute.signal_intensity": {
            "category": category,
            "parent_task_id": "finding_attribute",
        },
        "finding_attribute.enhancement": {
            "category": category,
            "parent_task_id": "finding_attribute",
        },
    }

    assert distribute_weighted_sample_counts(tasks, 12) == {
        "finding_type": 6,
        "finding_attribute.shape": 2,
        "finding_attribute.signal_intensity": 2,
        "finding_attribute.enhancement": 2,
    }

def test_write_sample_counts_is_atomic_and_preserves_tasks(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"sample_count": 3},
                "tasks": {
                    "a": {"description": "A", "category": "report_generation"},
                    "b": {"description": "B", "category": "spatial_localization"},
                },
            }
        ),
        encoding="utf-8",
    )

    write_sample_counts(path, {"a": 2, "b": 1})

    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    assert tasks == {
        "a": {"description": "A", "category": "report_generation", "sample_count": 2},
        "b": {"description": "B", "category": "spatial_localization", "sample_count": 1},
    }
    assert not Path(f"{path}.tmp").exists()
