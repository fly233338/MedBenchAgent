import json
from pathlib import Path

import pytest

from medbenchagent.sample_counts import (
    distribute_sample_counts,
    distribute_weighted_sample_counts,
    write_sample_counts,
)


def test_distribution_is_stable_and_allows_zero_counts() -> None:
    assert distribute_sample_counts(["a", "b", "c"], 2) == {"a": 1, "b": 1, "c": 0}
    assert distribute_sample_counts(["a", "b", "c"], 7, {"b": 5}) == {
        "a": 1,
        "b": 5,
        "c": 1,
    }


def test_complete_fixed_counts_must_match_total() -> None:
    with pytest.raises(ValueError, match="must sum"):
        distribute_sample_counts(["a", "b"], 3, {"a": 1, "b": 1})


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
