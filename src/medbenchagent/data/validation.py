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
    seen = set()
    for row in records:
        if row["image"]["image_id"] not in images:
            raise ValueError("Annotations and image manifest must contain exactly the same images")
        seen.add(row["image"]["image_id"])
        image = images[row["image"]["image_id"]]
        for entity in ("patient", "case", "study", "image"):
            validate_bboxes(row[entity]["annotation"], width=image.width, height=image.height)
    if seen != set(images):
        raise ValueError("Annotations and image manifest must contain exactly the same images")


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


def validate_record(record, line_number=1):
    """Validate one record, leaving population checks to the streaming boundary."""
    if not isinstance(record, dict) or set(record) not in (
        set(RECORD_ENTITIES), set(RECORD_ENTITIES) | {"provenance"}
    ):
        raise AnnotationValidationError(
            f"Line {line_number}: record must contain only patient, case, study, image and optional provenance"
        )
    if "provenance" in record:
        from medbenchagent.data.contracts import Provenance
        Provenance.model_validate(record["provenance"])
    for name in RECORD_ENTITIES:
        entity = record[name]
        if not isinstance(entity, dict) or set(entity) != ENTITY_FIELDS[name]:
            raise AnnotationValidationError(f"Line {line_number}: invalid {name} fields")
        if not isinstance(entity["annotation"], dict):
            raise AnnotationValidationError(f"Line {line_number}: {name}.annotation must be a JSON object")
        identifier = entity[name + "_id"]
        _validate_optional_id(identifier, name + "_id", line_number)
        if name == "image" and identifier is None:
            raise AnnotationValidationError("image.image_id must be a non-empty string")
        validate_bboxes(entity["annotation"])
    image = record["image"]
    if not isinstance(image["image_path"], str) or not image["image_path"].strip():
        raise AnnotationValidationError("image.image_path must be a non-empty string")
    for field in ("modality", "anatomy"):
        _validate_optional_id(image[field], "image." + field, line_number)
    # Python's JSON parser accepts NaN/Infinity; reject them throughout annotations.
    try:
        json.dumps(record, allow_nan=False)
    except ValueError as exc:
        raise AnnotationValidationError("Annotation contains non-finite values") from exc
    return record


def validate_v4_record(record, line_number=1):
    """Validate the explicit finding contract used by new coding candidates."""
    validate_record(record, line_number)
    if "provenance" not in record:
        raise AnnotationValidationError(f"Line {line_number}: provenance is required by annotation.v4")
    findings = record["image"]["annotation"].get("findings", [])
    if not isinstance(findings, list):
        raise AnnotationValidationError(f"Line {line_number}: image.annotation.findings must be an array")
    seen = set()
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("finding_id"), str):
            raise AnnotationValidationError(f"Line {line_number}: finding_id is required")
        if finding["finding_id"] in seen:
            raise AnnotationValidationError(f"Line {line_number}: duplicate finding_id")
        seen.add(finding["finding_id"])
        if finding.get("status") not in {"known", "uncertain", "unknown", "missing"}:
            raise AnnotationValidationError(f"Line {line_number}: invalid finding status")
        if finding["finding_id"].strip() != finding["finding_id"]:
            raise AnnotationValidationError(f"Line {line_number}: finding_id must be trimmed")
        bbox = finding.get("bbox")
        if bbox is not None:
            if (not isinstance(bbox, list) or len(bbox) != 4 or
                    not all(type(v) in (int, float) and math.isfinite(v) for v in bbox)):
                raise AnnotationValidationError(f"Line {line_number}: invalid finding bbox")
            if not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3]):
                raise AnnotationValidationError(f"Line {line_number}: invalid finding bbox geometry")
        if finding.get("status") == "missing" and bbox is not None:
            raise AnnotationValidationError(f"Line {line_number}: missing finding cannot have a bbox")
    return record


def validate_candidate_sidecars(candidate_path: str | Path) -> None:
    """Validate the non-annotation files emitted by a Coding Session."""
    candidate = Path(candidate_path)
    fields = candidate / "annotation_fields.json"
    if not fields.is_file() or not isinstance(json.loads(fields.read_text(encoding="utf-8")), dict):
        raise AnnotationValidationError("annotation_fields.json must be a JSON object")
    provenance = candidate / "provenance.jsonl"
    if not provenance.is_file():
        raise AnnotationValidationError("provenance.jsonl is required")
    with provenance.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(f"Provenance line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict) or not value.get("source_id") or not value.get("record_ref"):
                raise AnnotationValidationError(f"Provenance line {line_number}: source_id and record_ref are required")


def iter_annotation_jsonl(annotation_path: str | Path):
    """Stream validated records using a disk-backed uniqueness index."""
    import sqlite3
    import tempfile

    path = Path(annotation_path).expanduser().resolve()
    if path.name != "annotation.jsonl":
        raise AnnotationValidationError("Input file must be named annotation.jsonl.")
    if not path.is_file():
        raise AnnotationValidationError(f"annotation.jsonl does not exist: {path}")
    with tempfile.TemporaryDirectory(prefix="medbench-validation-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "keys.sqlite"))
        try:
            connection.execute("CREATE TABLE seen (id TEXT PRIMARY KEY, path TEXT UNIQUE)")
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise AnnotationValidationError(f"Line {line_number}: blank lines are not allowed")
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise AnnotationValidationError(f"Line {line_number}: invalid JSON") from exc
                    validate_record(record, line_number)
                    image = record["image"]
                    try:
                        connection.execute("INSERT INTO seen VALUES (?, ?)",
                                           (image["image_id"], image["image_path"]))
                    except sqlite3.IntegrityError as exc:
                        raise AnnotationValidationError(
                            f"Line {line_number}: duplicate image.image_id or image.image_path"
                        ) from exc
                    count += 1
                    yield {**record, "_source_row": line_number}
            if not count:
                raise AnnotationValidationError("annotation.jsonl contains no records.")
        finally:
            connection.close()


def load_annotation_jsonl(annotation_path: str | Path) -> list[dict[str, Any]]:
    """Convenience materialization for small callers; processing uses the iterator."""
    return list(iter_annotation_jsonl(annotation_path))


def validate_dataset_files(annotation_path, manifest_path):
    """Join annotation and image dimensions on disk rather than loading the manifest."""
    from medbenchagent.data.image_assets import iter_image_manifest
    from medbenchagent.data.operators import connect
    with connect() as db:
        db.execute("CREATE TABLE manifest (id VARCHAR PRIMARY KEY, width BIGINT, height BIGINT, seen BOOLEAN DEFAULT false)")
        for entry in iter_image_manifest(Path(manifest_path)):
            db.execute("INSERT INTO manifest (id,width,height) VALUES (?,?,?)",
                       [entry.image_id, entry.width, entry.height])
        for row in iter_annotation_jsonl(annotation_path):
            key = row["image"]["image_id"]
            size = db.execute("SELECT width,height FROM manifest WHERE id=?", [key]).fetchone()
            if size is None:
                raise AnnotationValidationError("Annotations and manifest populations differ")
            for entity in RECORD_ENTITIES:
                validate_bboxes(row[entity]["annotation"], width=size[0], height=size[1])
            db.execute("UPDATE manifest SET seen=true WHERE id=?", [key])
        if db.execute("SELECT count(*) FROM manifest WHERE NOT seen").fetchone()[0]:
            raise AnnotationValidationError("Annotations and manifest populations differ")
