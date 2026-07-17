import json
from pathlib import Path

import duckdb
import pytest

from medbenchagent.dataset import AnnotationValidationError, annotation_schema_path, prepare_annotation_dataset


def _write_annotation_dataset(tmp_path: Path, records: list[dict]) -> Path:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    return annotation_path


def _records() -> list[dict]:
    return [
        {
            "image_id": "img-001",
            "image_path": "images/img-001.png",
            "patient_id": None,
            "case_id": None,
            "study_id": "study-001",
            "annotation": {
                "laterality": "L",
                "breast_birads": "BI-RADS 4",
                "findings": [{"finding_type": "calcification", "bbox": [120, 80, 260, 210]}],
            },
        },
        {
            "image_id": "img-002",
            "image_path": "images/img-002.png",
            "patient_id": None,
            "case_id": None,
            "study_id": "study-001",
            "annotation": {
                "laterality": "R",
                "breast_birads": "BI-RADS 2",
                "findings": [
                    {"finding_type": "mass", "bbox": [1, 2, 3, 4]},
                    {"finding_type": "calcification", "bbox": [5, 6, 7, 8]},
                ],
            },
        },
    ]


def test_annotation_jsonl_creates_one_image_row_and_deterministic_profile(tmp_path: Path, monkeypatch) -> None:
    annotation_path = _write_annotation_dataset(tmp_path, _records())
    monkeypatch.setattr("medbenchagent.llm.complete_json", lambda *_args: pytest.fail("LLM must not be called"))

    profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")

    assert profile["total_images"] == 2
    assert profile["schema_version"] == "annotation-profile.v2"
    assert profile["fields"]["annotation.findings[*].finding_type"]["coverage_images"] == 2
    assert profile["fields"]["annotation.findings[*].finding_type"]["unique_value_count"] == 2
    assert profile["fields"]["annotation.breast_birads"]["coverage_rate"] == 1.0
    assert profile["fields"]["annotation.breast_birads"]["non_empty_count"] == 2
    assert manifest["cache_format_version"] == "annotation-cache.v2"
    connection = duckdb.connect(manifest["duckdb_path"])
    try:
        assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2
        assert connection.execute("SELECT image_path FROM images WHERE image_id = 'img-001'").fetchone() == (
            "images/img-001.png",
        )
        columns = {column[0] for column in connection.execute("DESCRIBE images").fetchall()}
        assert "resolved_image_path" not in columns
        stored = connection.execute("SELECT annotation::VARCHAR FROM images WHERE image_id = 'img-002'").fetchone()[0]
        assert stored.index('"mass"') < stored.index('"calcification"')
        assert connection.execute("SELECT patient_id, case_id, study_id FROM images LIMIT 1").fetchone() == (
            None,
            None,
            "study-001",
        )
    finally:
        connection.close()

    monkeypatch.setattr("medbenchagent.dataset.load_annotation_jsonl", lambda _path: pytest.fail("cache not used"))
    cached_profile, cached_manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    assert cached_profile == profile
    assert cached_manifest == manifest


def test_annotation_jsonl_accepts_unavailable_image_paths_and_profiles_non_empty_values(tmp_path: Path) -> None:
    records = [
        {
            **_records()[0],
            "image_path": "/unavailable/images/img-001.dicom",
            "annotation": {"empty": "", "nested": {"values": [0, False, None, [], {}]}},
        }
    ]
    annotation_path = _write_annotation_dataset(tmp_path, records)

    profile, _manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")

    assert profile["fields"]["annotation.empty"]["non_empty_count"] == 0
    assert profile["fields"]["annotation.nested.values[*]"]["non_empty_count"] == 2
    assert profile["fields"]["annotation.nested.values[*]"]["examples"] == [0, False]


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{key: value for key, value in _records()[0].items() if key != "study_id"}], "missing required fields"),
        ([_records()[0], {**_records()[1], "image_id": "img-001"}], "duplicate image_id"),
        ([{**_records()[0], "annotation": []}], "annotation must be a JSON object"),
        ([{**_records()[0], "image_path": ""}], "image_path must be a non-empty string"),
    ],
)
def test_annotation_jsonl_rejects_invalid_records(tmp_path: Path, records: list[dict], expected: str) -> None:
    annotation_path = _write_annotation_dataset(tmp_path, records)

    with pytest.raises(AnnotationValidationError, match=expected):
        prepare_annotation_dataset(annotation_path, tmp_path / "cache")


def test_annotation_jsonl_rejects_invalid_json_and_schema_is_versioned(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(AnnotationValidationError, match="invalid JSON"):
        prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    schema = json.loads(annotation_schema_path().read_text(encoding="utf-8"))
    assert schema["title"].endswith("v1")
