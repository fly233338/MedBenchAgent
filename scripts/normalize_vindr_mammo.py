"""Deterministically normalize the local VinDr-Mammo annotations."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_FILES = (
    "breast-level_annotations.csv",
    "finding_annotations.csv",
    "metadata.csv",
    "info.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def parse_categories(value: str) -> list[str]:
    # VinDr stores a Python-like list in this CSV column; retain each stated label.
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid finding_categories value: {value!r}")
    return parsed


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def source_inventory(
    breast_rows: list[dict[str, str]], finding_rows: list[dict[str, str]], metadata_rows: list[dict[str, str]]
) -> dict[str, Any]:
    def table_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
        columns = list(rows[0])
        return {
            "row_count": len(rows),
            "columns": columns,
            "null_counts": {column: sum(not row[column] for row in rows) for column in columns},
            "unique_counts": {column: len({row[column] for row in rows}) for column in columns},
        }

    breast_ids = {row["image_id"] for row in breast_rows}
    finding_ids = {row["image_id"] for row in finding_rows}
    metadata_ids = {row["SOP Instance UID"] for row in metadata_rows}
    finding_counts = Counter(row["image_id"] for row in finding_rows)
    return {
        "source_files": list(SOURCE_FILES),
        "tables": {
            "breast-level_annotations.csv": table_summary(breast_rows),
            "finding_annotations.csv": table_summary(finding_rows),
            "metadata.csv": table_summary(metadata_rows),
        },
        "authoritative_image_population": {
            "source": "breast-level_annotations.csv",
            "image_count": len(breast_ids),
            "duplicate_image_ids": sum(count > 1 for count in Counter(row["image_id"] for row in breast_rows).values()),
        },
        "join_measurements": {
            "breast_to_finding_missing_images": len(breast_ids - finding_ids),
            "finding_orphan_images": len(finding_ids - breast_ids),
            "breast_to_metadata_missing_images": len(breast_ids - metadata_ids),
            "metadata_orphan_images": len(metadata_ids - breast_ids),
            "finding_rows_per_image": {
                "minimum": min(finding_counts.values()),
                "maximum": max(finding_counts.values()),
                "images_with_multiple_rows": sum(count > 1 for count in finding_counts.values()),
            },
        },
        "image_files": {
            "local_existence_verified": False,
            "logical_path_rule": "<study_id>/<image_id>.dicom",
            "source_image_root": "/public/home/rxa-jj-wzk/data/fyl/BreastData/39_VinDr-Mammo/images",
        },
    }


def records(breast_rows: list[dict[str, str]], finding_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    findings_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in finding_rows:
        findings_by_image[row["image_id"]].append(
            {
                "finding_categories": parse_categories(row["finding_categories"]),
                "finding_birads": row["finding_birads"] or None,
                "bbox": {
                    "xmin": optional_float(row["xmin"]),
                    "ymin": optional_float(row["ymin"]),
                    "xmax": optional_float(row["xmax"]),
                    "ymax": optional_float(row["ymax"]),
                }
                if row["xmin"]
                else None,
            }
        )

    output = []
    for row in sorted(breast_rows, key=lambda item: item["image_id"]):
        output.append(
            {
                "image_id": row["image_id"],
                "image_path": f"{row['study_id']}/{row['image_id']}.dicom",
                "patient_id": None,
                "case_id": None,
                "study_id": row["study_id"],
                "annotation": {
                    "modality": "Mammography",
                    "anatomy": "Breast",
                    "series_id": row["series_id"],
                    "laterality": row["laterality"],
                    "view_position": row["view_position"],
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "breast_birads": row["breast_birads"],
                    "breast_density": row["breast_density"],
                    "split": row["split"],
                    "findings": findings_by_image[row["image_id"]],
                },
            }
        )
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_annotation(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for value in values:
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    breast_rows = read_rows(source / "breast-level_annotations.csv")
    finding_rows = read_rows(source / "finding_annotations.csv")
    metadata_rows = read_rows(source / "metadata.csv")
    normalized_records = records(breast_rows, finding_rows)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="vindr-normalize-") as temporary:
        temp_output = Path(temporary)
        annotation = temp_output / "annotation.jsonl"
        write_annotation(annotation, normalized_records)
        rerun = temp_output / "annotation.rerun.jsonl"
        write_annotation(rerun, records(breast_rows, finding_rows))
        if sha256(annotation) != sha256(rerun):
            raise ValueError("Determinism check failed")
        shutil.move(str(annotation), output / "annotation.jsonl")
    inventory = source_inventory(breast_rows, finding_rows, metadata_rows)
    plan = {
        "source_inputs": list(SOURCE_FILES),
        "authoritative_image_population": "one row per breast-level_annotations.csv image_id",
        "stable_output_order": "ascending image_id",
        "image_id_derivation": "breast-level_annotations.csv image_id",
        "image_path_derivation": "<study_id>/<image_id>.dicom relative to the supplied images root",
        "join_keys": ["image_id"],
        "measured_cardinalities": inventory["join_measurements"],
        "annotation_mapping": "breast-level fields map to annotation; finding rows are grouped in annotation.findings",
        "modality_anatomy_provenance": "dataset-level documentation (info.md)",
        "included_source_columns": {
            "breast-level_annotations.csv": list(breast_rows[0]),
            "finding_annotations.csv": list(finding_rows[0]),
        },
        "excluded_source_columns": {
            "metadata.csv": "not needed for image-level benchmark facts; retained as inventoried auxiliary DICOM metadata",
        },
        "unresolved_decisions": [],
    }
    write_json(output / "source_inventory.json", inventory)
    write_json(output / "conversion_plan.json", plan)
    report = {
        "input_file_hashes": {name: sha256(source / name) for name in SOURCE_FILES},
        "converter_hash": sha256(Path(__file__)),
        "output_file_hash": sha256(output / "annotation.jsonl"),
        "output_record_count": len(normalized_records),
        "reconciliation": {
            "authoritative_image_count": len(breast_rows),
            "output_record_count": len(normalized_records),
            "finding_rows_grouped": len(finding_rows),
        },
        "determinism_check": {"passed": True},
        "local_image_existence_verified": False,
    }
    write_json(output / "conversion_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
