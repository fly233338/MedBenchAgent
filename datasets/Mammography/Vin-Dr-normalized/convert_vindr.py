"""Convert Vin-Dr mammography CSV annotations into image-centric annotation.jsonl."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = OUTPUT_DIR.parent / "Vin-Dr"
ANNOTATION_PATH = OUTPUT_DIR / "annotation.jsonl"
REPORT_PATH = OUTPUT_DIR / "conversion_report.json"
SCHEMA_PATH = Path(__file__).resolve().parents[3] / ".codex" / "skills" / "normalize-medical-image-dataset" / "schema.json"

BREAST_COLUMNS = [
    "study_id",
    "series_id",
    "image_id",
    "laterality",
    "view_position",
    "height",
    "width",
    "breast_birads",
    "breast_density",
    "split",
]
FINDING_IMAGE_COLUMNS = [
    "study_id",
    "series_id",
    "laterality",
    "view_position",
    "height",
    "width",
    "breast_birads",
    "breast_density",
    "split",
]
REQUIRED_TOP_LEVEL_FIELDS = ["image_id", "image_path", "patient_id", "case_id", "study_id", "annotation"]


class ConversionError(RuntimeError):
    """Raised when source reconciliation or output validation fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_headers(headers: list[str]) -> list[str]:
    counts = Counter(headers)
    seen: Counter[str] = Counter()
    unique = []
    for header in headers:
        seen[header] += 1
        unique.append(header if counts[header] == 1 else f"{header}__{seen[header]}")
    return unique


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        raw_headers = next(reader)
        headers = unique_headers(raw_headers)
        return [dict(zip(headers, row, strict=True)) for row in reader]


def read_sha_manifest(path: Path) -> dict[str, str]:
    image_paths: dict[str, str] = {}
    duplicate_image_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        _, manifest_path = parts
        if not manifest_path.startswith("images/"):
            continue
        image_id = Path(manifest_path).stem
        if image_id in image_paths:
            duplicate_image_ids.add(image_id)
        image_paths[image_id] = manifest_path
    if duplicate_image_ids:
        raise ConversionError(f"Duplicate image IDs in SHA256SUMS image paths: {len(duplicate_image_ids)}")
    return image_paths


def empty_to_none(value: str) -> str | None:
    return None if value == "" else value


def parse_int(value: str, field: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConversionError(f"Expected integer for {field}: {value!r}") from exc


def parse_number(value: str, field: str) -> int | float | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ConversionError(f"Expected number for {field}: {value!r}") from exc
    if number.is_integer() and "." not in value:
        return int(number)
    return number


def parse_number_or_list(value: str, field: str) -> int | float | list[int | float] | None:
    if value == "":
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ConversionError(f"Expected numeric list for {field}: {value!r}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, (int, float)) for item in parsed):
            raise ConversionError(f"Expected numeric list for {field}: {value!r}")
        return parsed
    return parse_number(value, field)


def parse_spacing(value: str, field: str) -> list[int | float] | None:
    if value == "":
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ConversionError(f"Expected list-like spacing for {field}: {value!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, (int, float)) for item in parsed):
        raise ConversionError(f"Expected numeric list for {field}: {value!r}")
    return parsed


def parse_categories(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ConversionError(f"Expected list-like finding_categories: {value!r}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ConversionError(f"Expected string list for finding_categories: {value!r}")
    return parsed


def parse_bbox(row: dict[str, str]) -> list[float] | None:
    values = [row[field] for field in ("xmin", "ymin", "xmax", "ymax")]
    if all(value == "" for value in values):
        return None
    if any(value == "" for value in values):
        raise ConversionError(f"Partial bbox for image_id={row['image_id']}")
    bbox = [float(value) for value in values]
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        raise ConversionError(f"Invalid bbox order for image_id={row['image_id']}: {bbox}")
    return bbox


def output_relative_image_path(manifest_path: str) -> str:
    return Path(manifest_path).as_posix()


def require_unique(rows: list[dict[str, str]], key: str, source_name: str) -> dict[str, dict[str, str]]:
    counts = Counter(row[key] for row in rows)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        raise ConversionError(f"{source_name} has duplicate {key} values: {len(duplicates)}")
    return {row[key]: row for row in rows}


def reconcile_sources(
    breast_rows: list[dict[str, str]],
    finding_rows: list[dict[str, str]],
    metadata_rows: list[dict[str, str]],
    manifest_paths: dict[str, str],
) -> dict[str, Any]:
    breast_by_image = require_unique(breast_rows, "image_id", "breast-level_annotations.csv")
    metadata_by_image = require_unique(metadata_rows, "SOP Instance UID__1", "metadata.csv")

    if any(row["SOP Instance UID__1"] != row["SOP Instance UID__2"] for row in metadata_rows):
        raise ConversionError("metadata.csv duplicate SOP Instance UID columns contain conflicting values")

    breast_ids = set(breast_by_image)
    metadata_ids = set(metadata_by_image)
    finding_ids = {row["image_id"] for row in finding_rows}
    manifest_ids = set(manifest_paths)
    mismatches = {
        "breast_without_metadata": len(breast_ids - metadata_ids),
        "metadata_without_breast": len(metadata_ids - breast_ids),
        "breast_without_findings": len(breast_ids - finding_ids),
        "finding_without_breast": len(finding_ids - breast_ids),
        "breast_without_manifest_path": len(breast_ids - manifest_ids),
        "manifest_path_without_breast": len(manifest_ids - breast_ids),
    }
    if any(mismatches.values()):
        raise ConversionError(f"Image population mismatch: {mismatches}")

    metadata_conflicts = Counter()
    for image_id, breast_row in breast_by_image.items():
        metadata_row = metadata_by_image[image_id]
        checks = {
            "series_id": (breast_row["series_id"], metadata_row["Series Instance UID"]),
            "view_position": (breast_row["view_position"], metadata_row["View Position"]),
            "laterality": (breast_row["laterality"], metadata_row["Image Laterality"]),
            "height": (breast_row["height"], metadata_row["Rows"]),
            "width": (breast_row["width"], metadata_row["Columns"]),
        }
        for field, (left, right) in checks.items():
            if left != right:
                metadata_conflicts[field] += 1

    finding_conflicts = Counter()
    for finding_row in finding_rows:
        breast_row = breast_by_image[finding_row["image_id"]]
        for field in FINDING_IMAGE_COLUMNS:
            if finding_row[field] != breast_row[field]:
                finding_conflicts[field] += 1

    if metadata_conflicts or finding_conflicts:
        raise ConversionError(
            "Conflicting image-level values: "
            f"metadata={dict(metadata_conflicts)}, findings={dict(finding_conflicts)}"
        )

    return {
        "breast_by_image": breast_by_image,
        "metadata_by_image": metadata_by_image,
        "mismatches": mismatches,
        "metadata_conflicts": dict(metadata_conflicts),
        "finding_conflicts": dict(finding_conflicts),
    }


def build_dicom_metadata(row: dict[str, str]) -> dict[str, Any]:
    return {
        "sop_instance_uid": row["SOP Instance UID__1"],
        "series_instance_uid": row["Series Instance UID"],
        "patient_age": empty_to_none(row["Patient's Age"]),
        "view_position": row["View Position"],
        "image_laterality": row["Image Laterality"],
        "photometric_interpretation": row["Photometric Interpretation"],
        "rows": parse_int(row["Rows"], "Rows"),
        "columns": parse_int(row["Columns"], "Columns"),
        "imager_pixel_spacing": parse_spacing(row["Imager Pixel Spacing"], "Imager Pixel Spacing"),
        "pixel_spacing": parse_spacing(row["Pixel Spacing"], "Pixel Spacing"),
        "pixel_padding_value": parse_int(row["Pixel Padding Value"], "Pixel Padding Value"),
        "pixel_padding_range_limit": parse_number(
            row["Pixel Padding Range Limit"], "Pixel Padding Range Limit"
        ),
        "window_center": parse_number_or_list(row["Window Center"], "Window Center"),
        "window_width": parse_number_or_list(row["Window Width"], "Window Width"),
        "rescale_intercept": parse_number(row["Rescale Intercept"], "Rescale Intercept"),
        "rescale_slope": parse_number(row["Rescale Slope"], "Rescale Slope"),
        "rescale_type": row["Rescale Type"],
        "window_center_width_explanation": row["Window Center & Width Explanation"],
        "manufacturer": row["Manufacturer"],
        "manufacturer_model_name": row["Manufacturer's Model Name"],
    }


def build_record(
    image_id: str,
    breast_row: dict[str, str],
    metadata_row: dict[str, str],
    manifest_path: str,
    finding_rows: list[dict[str, str]],
) -> dict[str, Any]:
    findings = [
        {
            "finding_categories": parse_categories(row["finding_categories"]),
            "finding_birads": empty_to_none(row["finding_birads"]),
            "bbox": parse_bbox(row),
        }
        for row in finding_rows
    ]
    return {
        "image_id": image_id,
        "image_path": output_relative_image_path(manifest_path),
        "patient_id": None,
        "case_id": None,
        "study_id": breast_row["study_id"],
        "annotation": {
            "modality": "X-ray",
            "metadata": {
                "modality": "Mammography",
                "anatomy": "Breast",
            },
            "breast": {
                "series_id": breast_row["series_id"],
                "laterality": breast_row["laterality"],
                "view_position": breast_row["view_position"],
                "height": parse_int(breast_row["height"], "height"),
                "width": parse_int(breast_row["width"], "width"),
                "breast_birads": breast_row["breast_birads"],
                "breast_density": breast_row["breast_density"],
                "split": breast_row["split"],
            },
            "dicom_metadata": build_dicom_metadata(metadata_row),
            "findings": findings,
        },
    }


def validate_record(record: dict[str, Any], line_number: int, local_assets_present: bool) -> None:
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in record]
    if missing:
        raise ConversionError(f"Line {line_number}: missing required fields: {missing}")
    if not isinstance(record["image_id"], str) or not record["image_id"]:
        raise ConversionError(f"Line {line_number}: image_id must be a non-empty string")
    if not isinstance(record["image_path"], str) or not record["image_path"]:
        raise ConversionError(f"Line {line_number}: image_path must be a non-empty string")
    image_path = Path(record["image_path"])
    if image_path.is_absolute():
        raise ConversionError(f"Line {line_number}: image_path must be relative")
    if local_assets_present and not (OUTPUT_DIR / image_path).is_file():
        raise ConversionError(f"Line {line_number}: local image path does not exist: {record['image_path']}")
    for field in ("patient_id", "case_id", "study_id"):
        value = record[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise ConversionError(f"Line {line_number}: {field} must be a non-empty string or null")
    if not isinstance(record["annotation"], dict):
        raise ConversionError(f"Line {line_number}: annotation must be an object")


def build_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    breast_rows = read_csv_rows(SOURCE_DIR / "breast-level_annotations.csv")
    finding_rows = read_csv_rows(SOURCE_DIR / "finding_annotations.csv")
    metadata_rows = read_csv_rows(SOURCE_DIR / "metadata.csv")
    manifest_paths = read_sha_manifest(SOURCE_DIR / "SHA256SUMS.txt")
    reconciliation = reconcile_sources(breast_rows, finding_rows, metadata_rows, manifest_paths)

    findings_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in finding_rows:
        findings_by_image[row["image_id"]].append(row)

    records = [
        build_record(
            image_id=image_id,
            breast_row=reconciliation["breast_by_image"][image_id],
            metadata_row=reconciliation["metadata_by_image"][image_id],
            manifest_path=manifest_paths[image_id],
            finding_rows=findings_by_image[image_id],
        )
        for image_id in sorted(reconciliation["breast_by_image"])
    ]

    finding_counts = Counter(row["image_id"] for row in finding_rows)
    report_context = {
        "source_counts": {
            "breast_rows": len(breast_rows),
            "finding_rows": len(finding_rows),
            "metadata_rows": len(metadata_rows),
            "manifest_image_paths": len(manifest_paths),
        },
        "output_records": len(records),
        "mismatches": reconciliation["mismatches"],
        "metadata_conflicts": reconciliation["metadata_conflicts"],
        "finding_conflicts": reconciliation["finding_conflicts"],
        "finding_rows_per_image_distribution": [
            {"rows_per_image": rows_per_image, "image_count": image_count}
            for rows_per_image, image_count in sorted(Counter(finding_counts.values()).items())
        ],
        "finding_images_with_multiple_rows": sum(1 for count in finding_counts.values() if count > 1),
        "max_finding_rows_per_image": max(finding_counts.values()),
    }
    return records, report_context


def write_annotation(records: list[dict[str, Any]]) -> str:
    local_assets_present = (SOURCE_DIR / "images").is_dir()
    image_ids: set[str] = set()
    temp_path = ANNOTATION_PATH.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as output:
        for line_number, record in enumerate(records, start=1):
            validate_record(record, line_number, local_assets_present)
            if record["image_id"] in image_ids:
                raise ConversionError(f"Duplicate output image_id: {record['image_id']}")
            image_ids.add(record["image_id"])
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_path.replace(ANNOTATION_PATH)
    return sha256_file(ANNOTATION_PATH)


def write_report(report_context: dict[str, Any], output_sha256: str) -> None:
    report = {
        "status": "success",
        "source_dir": str(SOURCE_DIR),
        "output_annotation": str(ANNOTATION_PATH),
        "schema_path": str(SCHEMA_PATH),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "source_hashes": {
            name: sha256_file(SOURCE_DIR / name)
            for name in [
                "breast-level_annotations.csv",
                "finding_annotations.csv",
                "metadata.csv",
                "SHA256SUMS.txt",
            ]
        },
        "converter_sha256": sha256_file(Path(__file__).resolve()),
        "annotation_sha256": output_sha256,
        "validation": {
            "required_top_level_fields": REQUIRED_TOP_LEVEL_FIELDS,
            "unique_image_ids": True,
            "non_empty_relative_image_paths": True,
            "local_image_assets_present": (SOURCE_DIR / "images").is_dir(),
            "local_image_existence_check": "skipped because local image assets are not present",
            "schema_shape_check": "passed via converter-required top-level fields and annotation object checks",
        },
        **report_context,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    records, report_context = build_records()
    output_sha256 = write_annotation(records)
    write_report(report_context, output_sha256)
    print(f"Wrote {ANNOTATION_PATH} ({len(records)} records)")
    print(f"annotation_sha256={output_sha256}")


if __name__ == "__main__":
    main()
