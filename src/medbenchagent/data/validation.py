"""Public annotation, bbox and dataset-set validation without runtime dependencies."""

import math
import json
from typing import Any
from pathlib import Path


def validate_bboxes(value, *, width=None, height=None, path="annotation"):
    if isinstance(value, dict):
        for fields in ({"xmin", "ymin", "xmax", "ymax"}, {"x", "y", "width", "height"},
                       {"cx", "cy", "width", "height"}):
            if fields <= value.keys():
                raise ValueError(f"{path} contains non-canonical bbox coordinate fields")
        for key, nested in value.items():
            if key == "bbox" and nested is not None:
                if (
                    not isinstance(nested, list)
                    or len(nested) != 4
                    or any(type(n) not in (int, float) or not math.isfinite(n) for n in nested)
                    or not (0 <= nested[0] < nested[2] and 0 <= nested[1] < nested[3])
                    or (width is not None and nested[2] > width)
                    or (height is not None and nested[3] > height)
                ):
                    raise ValueError(
                        f"{path}.bbox must be a finite absolute xyxy box within the image"
                    )
            else:
                validate_bboxes(nested, width=width, height=height, path=f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            validate_bboxes(item, width=width, height=height, path=path)


def source_image_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Image reference must resolve beneath source root: {relative}")
    return path


def validate_dataset(records, images):
    """Check exact image population and canonical bbox geometry."""
    if {row["image"]["image_id"] for row in records} != set(images):
        raise ValueError("Annotations and image manifest must contain exactly the same images")
    for row in records:
        image = images[row["image"]["image_id"]]
        for entity in ("patient", "case", "study", "image"):
            validate_bboxes(row[entity]["annotation"], width=image.width, height=image.height)


RECORD_ENTITIES = ("patient", "case", "study", "image")
ENTITY_FIELDS = {
    "patient": {"patient_id", "annotation"},
    "case": {"case_id", "annotation"},
    "study": {"study_id", "annotation"},
    "image": {"image_id", "image_path", "modality", "anatomy", "annotation"},
}


class AnnotationValidationError(ValueError):
    """An annotation JSONL record does not meet the stable input contract."""


def _validate_optional_id(value: object, field: str, line_number: int) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise AnnotationValidationError(
            f"Line {line_number}: {field} must be a non-empty string or null."
        )


def load_annotation_jsonl(annotation_path: str | Path) -> list[dict[str, Any]]:
    """Validate entity-scoped image records without accessing image files."""
    path = Path(annotation_path).expanduser().resolve()
    if path.name != "annotation.jsonl":
        raise AnnotationValidationError("Input file must be named annotation.jsonl.")
    if not path.is_file():
        raise AnnotationValidationError(f"annotation.jsonl does not exist: {path}")

    records: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise AnnotationValidationError(f"Line {line_number}: blank lines are not allowed.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(f"Line {line_number}: invalid JSON.") from exc
            if not isinstance(record, dict):
                raise AnnotationValidationError(
                    f"Line {line_number}: record must be a JSON object."
                )
            if set(record) != set(RECORD_ENTITIES):
                raise AnnotationValidationError(
                    f"Line {line_number}: record must contain only patient, case, study, and image."
                )
            for entity_name in RECORD_ENTITIES:
                entity = record[entity_name]
                if not isinstance(entity, dict) or set(entity) != ENTITY_FIELDS[entity_name]:
                    expected = ", ".join(sorted(ENTITY_FIELDS[entity_name]))
                    raise AnnotationValidationError(
                        f"Line {line_number}: {entity_name} must contain only {expected}."
                    )
                if not isinstance(entity["annotation"], dict):
                    raise AnnotationValidationError(
                        f"Line {line_number}: {entity_name}.annotation must be a JSON object."
                    )

            patient = record["patient"]
            case = record["case"]
            study = record["study"]
            image = record["image"]
            _validate_optional_id(patient["patient_id"], "patient.patient_id", line_number)
            _validate_optional_id(case["case_id"], "case.case_id", line_number)
            _validate_optional_id(study["study_id"], "study.study_id", line_number)

            image_id = image["image_id"]
            if not isinstance(image_id, str) or not image_id.strip():
                raise AnnotationValidationError(
                    f"Line {line_number}: image.image_id must be a non-empty string."
                )
            if image_id in image_ids:
                raise AnnotationValidationError(
                    f"Line {line_number}: duplicate image.image_id: {image_id}."
                )
            image_ids.add(image_id)
            image_path = image["image_path"]
            if not isinstance(image_path, str) or not image_path.strip():
                raise AnnotationValidationError(
                    f"Line {line_number}: image.image_path must be a non-empty string."
                )
            if image_path in image_paths:
                raise AnnotationValidationError(
                    f"Line {line_number}: duplicate image.image_path: {image_path}."
                )
            image_paths.add(image_path)
            for field in ("modality", "anatomy"):
                value = image[field]
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise AnnotationValidationError(
                        f"Line {line_number}: image.{field} must be a non-empty string or null."
                    )
            for entity in RECORD_ENTITIES:
                validate_bboxes(record[entity]["annotation"])
            records.append({**record, "_source_row": line_number})
    if not records:
        raise AnnotationValidationError("annotation.jsonl contains no records.")
    return records
