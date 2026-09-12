import json
from pathlib import Path

import duckdb
import pytest

from medbenchagent.data.dataset import (
    AnnotationValidationError,
    annotation_schema_path,
    prepare_annotation_dataset,
)


def _write_annotation_dataset(tmp_path: Path, records: list[dict]) -> Path:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return annotation_path


def _records() -> list[dict]:
    return [
        {
            "patient": {"patient_id": None, "annotation": {}},
            "case": {"case_id": "case-001", "annotation": {"status": "first"}},
            "study": {"study_id": "study-001", "annotation": {}},
            "image": {
                "image_id": "img-001",
                "image_path": "images/img-001.png",
                "modality": "Mammography",
                "anatomy": "Breast",
                "annotation": {
                    "laterality": "L",
                    "breast_birads": "BI-RADS 4",
                    "findings": [
                        {"finding_type": "calcification", "bbox": [120, 80, 260, 210]}
                    ],
                },
            },
        },
        {
            "patient": {"patient_id": None, "annotation": {}},
            "case": {"case_id": "case-001", "annotation": {"status": "second"}},
            "study": {"study_id": "study-001", "annotation": {}},
            "image": {
                "image_id": "img-002",
                "image_path": "images/img-002.png",
                "modality": "Mammography",
                "anatomy": "Breast",
                "annotation": {
                    "laterality": "R",
                    "breast_birads": "BI-RADS 2",
                    "findings": [
                        {"finding_type": "mass", "bbox": [1, 2, 3, 4]},
                        {"finding_type": "calcification", "bbox": [5, 6, 7, 8]},
                    ],
                },
            },
        },
    ]


def test_annotation_jsonl_creates_one_image_row_and_deterministic_profile(
    tmp_path: Path, monkeypatch
) -> None:
    annotation_path = _write_annotation_dataset(tmp_path, _records())
    monkeypatch.setattr(
        "medbenchagent.runtime.llm.complete_json", lambda *_args: pytest.fail("LLM must not be called")
    )

    profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")

    assert profile["total_images"] == 2
    assert profile["schema_version"] == "task-capability-profile.v3"
    assert (
        profile["fields"]["image.annotation.findings[*].finding_type"]["non_empty_images"] == 2
    )
    assert (
        profile["fields"]["image.annotation.findings[*].finding_type"]["distinct_value_count"]
        == 2
    )
    assert profile["fields"]["image.annotation.breast_birads"]["top_values"] == [
        "BI-RADS 2",
        "BI-RADS 4",
    ]
    assert profile["fields"]["image.annotation.findings[*].finding_type"][
        "value_distribution"
    ] == {
        "total_count": 3,
        "top": [
            {"value": "calcification", "count": 2, "proportion": 2 / 3},
            {"value": "mass", "count": 1, "proportion": 1 / 3},
        ],
        "least_frequent": {"value": "mass", "count": 1, "proportion": 1 / 3},
        "truncated": False,
    }
    assert profile["fields"]["case.annotation.status"]["top_values"] == ["first", "second"]
    assert profile["fields"]["image.modality"]["top_values"] == ["Mammography"]
    assert not any(path.endswith("_id") or path.endswith("_path") for path in profile["fields"])
    assert profile["structures"] == [
        {
            "path": "case.annotation",
            "kind": "object",
            "children": ["case.annotation.status"],
        },
        {
            "path": "image.annotation",
            "kind": "object",
            "children": [
                "image.annotation.breast_birads",
                "image.annotation.findings",
                "image.annotation.laterality",
            ],
        },
        {
            "path": "image.annotation.findings",
            "kind": "array",
            "item_path": "image.annotation.findings[*]",
            "children": [
                "image.annotation.findings[*].bbox",
                "image.annotation.findings[*].finding_type",
            ],
        },
        {
            "path": "image.annotation.findings[*].bbox",
            "kind": "array",
            "item_path": "image.annotation.findings[*].bbox[*]",
            "children": [],
        },
        {"path": "patient.annotation", "kind": "object", "children": []},
        {"path": "study.annotation", "kind": "object", "children": []},
    ]
    assert manifest["cache_format_version"] == "annotation-cache.v7"
    connection = duckdb.connect(manifest["duckdb_path"])
    try:
        assert connection.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2
        assert connection.execute(
            "SELECT image_path FROM images WHERE image_id = 'img-001'"
        ).fetchone() == ("images/img-001.png",)
        columns = {column[0] for column in connection.execute("DESCRIBE images").fetchall()}
        assert "record" in columns
        assert "annotation" not in columns
        stored = connection.execute(
            "SELECT record::VARCHAR FROM images WHERE image_id = 'img-002'"
        ).fetchone()[0]
        assert stored.index('"mass"') < stored.index('"calcification"')
        assert connection.execute(
            "SELECT patient_id, case_id, study_id FROM images LIMIT 1"
        ).fetchone() == (None, "case-001", "study-001")
    finally:
        connection.close()

    monkeypatch.setattr(
        "medbenchagent.data.dataset.load_annotation_jsonl", lambda _path: pytest.fail("cache not used")
    )
    assert prepare_annotation_dataset(annotation_path, tmp_path / "cache") == (profile, manifest)


def test_profile_omits_null_only_bbox_parent_but_keeps_bbox_leaf(
    tmp_path: Path,
) -> None:
    records = _records()
    records[0]["image"]["annotation"]["findings"].append(
        {"finding_type": "no_finding", "bbox": None}
    )
    profile, _manifest = prepare_annotation_dataset(
        _write_annotation_dataset(tmp_path, records), tmp_path / "cache"
    )

    assert "image.annotation.findings[*].bbox" not in profile["fields"]
    assert profile["fields"]["image.annotation.findings[*].bbox[*]"]["types"] == ["number"]
    assert any(
        structure["path"] == "image.annotation.findings[*].bbox"
        and structure["kind"] == "array"
        for structure in profile["structures"]
    )


def test_compact_profile_counts_non_empty_images_and_filters_raw_values(
    tmp_path: Path,
) -> None:
    record = _records()[0]
    record["image"] = {
        **record["image"],
        "image_path": "/unavailable/images/img-001.dicom",
        "annotation": {
            "empty": "",
            "nested": {"values": [0, False, None, [], {}]},
            "finding_uid": "secret-uid",
            "dicom_metadata": {"patient_age": "42"},
        },
    }
    profile, _manifest = prepare_annotation_dataset(
        _write_annotation_dataset(tmp_path, [record]), tmp_path / "cache"
    )

    assert profile["fields"]["image.annotation.empty"]["non_empty_images"] == 0
    assert "value_distribution" not in profile["fields"]["image.annotation.empty"]
    assert profile["fields"]["image.annotation.nested.values[*]"]["non_empty_images"] == 1
    assert profile["fields"]["image.annotation.nested.values[*]"]["top_values"] == [0, False]
    assert "top_values" not in profile["fields"]["image.annotation.finding_uid"]
    assert "value_distribution" not in profile["fields"]["image.annotation.finding_uid"]
    assert "top_values" not in profile["fields"]["image.annotation.dicom_metadata.patient_age"]


def test_compact_profile_deduplicates_repeated_findings_and_limits_values(
    tmp_path: Path,
) -> None:
    records = []
    imbalanced_categories = ["A"] * 5 + ["B"] * 3 + ["C"] * 2 + ["D", "E"]
    for index in range(12):
        records.append(
            {
                "patient": {"patient_id": None, "annotation": {}},
                "case": {"case_id": None, "annotation": {}},
                "study": {"study_id": None, "annotation": {}},
                "image": {
                    "image_id": f"img-{index}",
                    "image_path": f"images/img-{index}.png",
                    "modality": None,
                    "anatomy": None,
                    "annotation": {
                        "flag": index % 2 == 0,
                        "category": f"value-{index}",
                        "imbalanced_category": imbalanced_categories[index],
                        "score": index % 2,
                        "long_text": f"value-{index:02d}-" + "x" * 220,
                        "source_path": f"private/value-{index}.txt",
                        "findings": [{"kind": "mass"}, {"kind": "mass"}],
                        "bbox": [index, index + 1, index + 2, index + 3],
                    },
                },
            }
        )
    profile, _manifest = prepare_annotation_dataset(
        _write_annotation_dataset(tmp_path, records), tmp_path / "cache"
    )

    assert profile["fields"]["image.annotation.findings[*].kind"]["non_empty_images"] == 12
    assert profile["fields"]["image.annotation.flag"]["top_values"] == [False, True]
    assert profile["fields"]["image.annotation.category"]["top_values"] == [
        "value-0",
        "value-1",
        "value-10",
    ]
    assert profile["fields"]["image.annotation.category"]["value_distribution"] == {
        "total_count": 12,
        "top": [
            {"value": "value-0", "count": 1, "proportion": 1 / 12},
            {"value": "value-1", "count": 1, "proportion": 1 / 12},
            {"value": "value-10", "count": 1, "proportion": 1 / 12},
        ],
        "least_frequent": {"value": "value-0", "count": 1, "proportion": 1 / 12},
        "truncated": True,
    }
    assert profile["fields"]["image.annotation.imbalanced_category"][
        "value_distribution"
    ] == {
        "total_count": 12,
        "top": [
            {"value": "A", "count": 5, "proportion": 5 / 12},
            {"value": "B", "count": 3, "proportion": 3 / 12},
            {"value": "C", "count": 2, "proportion": 2 / 12},
        ],
        "least_frequent": {"value": "D", "count": 1, "proportion": 1 / 12},
        "truncated": True,
    }
    assert profile["fields"]["image.annotation.score"]["top_values"] == [0, 1]
    assert len(profile["fields"]["image.annotation.long_text"]["top_values"]) == 3
    assert all(
        len(value) == 200 and value.endswith("…")
        for value in profile["fields"]["image.annotation.long_text"]["top_values"]
    )
    assert "top_values" not in profile["fields"]["image.annotation.source_path"]
    assert "top_values" not in profile["fields"]["image.annotation.bbox[*]"]
    assert "value_distribution" not in profile["fields"]["image.annotation.bbox[*]"]


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{key: value for key, value in _records()[0].items() if key != "study"}], "only patient"),
        (
            [
                _records()[0],
                {
                    **_records()[1],
                    "image": {**_records()[1]["image"], "image_id": "img-001"},
                },
            ],
            "duplicate image.image_id",
        ),
        (
            [{**_records()[0], "case": {**_records()[0]["case"], "annotation": []}}],
            "case.annotation must be a JSON object",
        ),
        (
            [
                {
                    **_records()[0],
                    "image": {**_records()[0]["image"], "image_path": ""},
                }
            ],
            "image.image_path must be a non-empty string",
        ),
    ],
)
def test_annotation_jsonl_rejects_invalid_records(
    tmp_path: Path, records: list[dict], expected: str
) -> None:
    with pytest.raises(AnnotationValidationError, match=expected):
        prepare_annotation_dataset(
            _write_annotation_dataset(tmp_path, records), tmp_path / "cache"
        )


def test_annotation_jsonl_rejects_v1_and_invalid_json_and_schema_is_versioned(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(AnnotationValidationError, match="invalid JSON"):
        prepare_annotation_dataset(annotation_path, tmp_path / "cache")

    annotation_path.write_text(
        json.dumps(
            {
                "image_id": "old",
                "image_path": "old.png",
                "patient_id": None,
                "case_id": None,
                "study_id": None,
                "annotation": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AnnotationValidationError, match="only patient"):
        prepare_annotation_dataset(annotation_path, tmp_path / "cache")

    schema = json.loads(annotation_schema_path().read_text(encoding="utf-8"))
    assert schema["title"].endswith("v2")


def test_annotation_jsonl_allows_inconsistent_repeated_entity_annotations(
    tmp_path: Path,
) -> None:
    records = _records()
    records[0]["patient"] = {"patient_id": "patient", "annotation": {"age": 40}}
    records[1]["patient"] = {"patient_id": "patient", "annotation": {"age": 41}}
    records[0]["study"]["annotation"] = {"label": "A"}
    records[1]["study"]["annotation"] = {"label": "B"}

    profile, _manifest = prepare_annotation_dataset(
        _write_annotation_dataset(tmp_path, records), tmp_path / "cache"
    )

    assert profile["fields"]["patient.annotation.age"]["distinct_value_count"] == 2
    assert profile["fields"]["study.annotation.label"]["distinct_value_count"] == 2
