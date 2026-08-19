"""Prepare ISIC2018 test annotations in canonical annotation.v2 JSONL format.

The source tree is read-only. This converter is intentionally dataset-specific and
requires all paths to be supplied explicitly so it can be copied to the server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ATTRIBUTES = (
    "globules",
    "milia_like_cyst",
    "negative_network",
    "pigment_network",
    "streaks",
)
DIAGNOSIS_COLUMNS = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
TOP_LEVEL_KEYS = {"patient", "case", "study", "image"}
ENTITY_KEYS = {
    "patient": {"patient_id", "annotation"},
    "case": {"case_id", "annotation"},
    "study": {"study_id", "annotation"},
    "image": {"image_id", "image_path", "modality", "anatomy", "annotation"},
}


class ConversionError(ValueError):
    """Raised when the approved mapping cannot be applied safely."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_relative(path: Path, source_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise ConversionError(f"Path escapes source root: {path}") from exc
    return relative.as_posix()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ConversionError(f"{label} does not exist: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise ConversionError(f"{label} does not exist: {path}")
    return path


def read_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{label} must contain a JSON object: {path}")
    return value


def verify_approved_plan(plan_path: Path) -> str:
    plan = read_json(plan_path, "conversion plan")
    review = plan.get("review")
    if not isinstance(review, dict) or review.get("status") != "approved":
        raise ConversionError("conversion_plan.json must have review.status=approved")
    approved_hash = review.get("approved_mapping_sha256")
    if not isinstance(approved_hash, str) or len(approved_hash) != 64:
        raise ConversionError("conversion plan has no approved mapping hash")
    mapping = dict(plan)
    mapping.pop("review", None)
    calculated = sha256_bytes(canonical_json(mapping).encode("utf-8"))
    if calculated != approved_hash:
        raise ConversionError(
            "conversion plan approved_mapping_sha256 does not match its canonical mapping"
        )
    return approved_hash


def image_size(path: Path, expected_format: str) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.format != expected_format:
                raise ConversionError(
                    f"{path.name}: expected {expected_format}, detected {image.format}"
                )
            if getattr(image, "n_frames", 1) != 1:
                raise ConversionError(f"{path.name}: multi-frame image is unsupported")
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ConversionError(f"{path.name}: invalid image dimensions")
            return image.width, image.height
    except OSError as exc:
        raise ConversionError(f"Could not decode image: {path}") from exc


def mask_presence_and_bbox(
    path: Path,
    expected_size: tuple[int, int],
    *,
    derive_bbox: bool,
) -> tuple[bool, list[int] | None]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ConversionError(f"{path.name}: expected PNG, detected {image.format}")
            if getattr(image, "n_frames", 1) != 1:
                raise ConversionError(f"{path.name}: multi-frame mask is unsupported")
            image.load()
            if image.size != expected_size:
                raise ConversionError(
                    f"{path.name}: mask dimensions {image.size} do not match image {expected_size}"
                )
            grayscale = image.convert("L")
            maximum = grayscale.getextrema()[1]
            if not derive_bbox:
                return maximum > 0, None
            bbox = grayscale.getbbox()
            if bbox is None:
                raise ConversionError(f"{path.name}: all-zero segmentation mask")
            return maximum > 0, [int(value) for value in bbox]
    except OSError as exc:
        raise ConversionError(f"Could not decode mask: {path}") from exc


def load_task3_rows(csv_path: Path) -> dict[str, dict[str, Any]]:
    require_file(csv_path, "Task 3 ground-truth CSV")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image", *DIAGNOSIS_COLUMNS]:
            raise ConversionError(
                "Task 3 CSV columns must be image,MEL,NV,BCC,AKIEC,BKL,DF,VASC"
            )
        rows: dict[str, dict[str, Any]] = {}
        for line_number, row in enumerate(reader, start=2):
            image_id = row.get("image")
            if not isinstance(image_id, str) or not image_id:
                raise ConversionError(f"Task 3 CSV line {line_number}: invalid image id")
            if image_id in rows:
                raise ConversionError(f"Task 3 CSV has duplicate image id: {image_id}")
            scores: dict[str, float] = {}
            for column in DIAGNOSIS_COLUMNS:
                raw = row.get(column)
                try:
                    score = float(raw) if raw is not None else math.nan
                except (TypeError, ValueError) as exc:
                    raise ConversionError(
                        f"Task 3 CSV line {line_number}: {column} is not numeric"
                    ) from exc
                if not math.isfinite(score) or score not in {0.0, 1.0}:
                    raise ConversionError(
                        f"Task 3 CSV line {line_number}: {column} must be 0.0 or 1.0"
                    )
                scores[column] = score
            active = [column for column, score in scores.items() if score == 1.0]
            if len(active) != 1:
                raise ConversionError(
                    f"Task 3 CSV line {line_number}: expected exactly one active diagnosis"
                )
            rows[image_id] = {
                "diagnosis": active[0],
                "diagnosis_scores": scores,
            }
    if not rows:
        raise ConversionError("Task 3 CSV contains no data rows")
    return rows


def common_annotation(
    source_tasks: list[str], width: int, height: int
) -> dict[str, Any]:
    return {
        "source_dataset": "ISIC2018",
        "split": "test",
        "source_tasks": source_tasks,
        "width": width,
        "height": height,
    }


def task12_records(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    input_dir = require_directory(
        source_root
        / "ISIC2018_Task1-2_Test_Input"
        / "ISIC2018_Task1-2_Test_Input",
        "Task 1-2 test input directory",
    )
    task1_dir = require_directory(
        source_root
        / "ISIC2018_Task1_Test_GroundTruth"
        / "ISIC2018_Task1_Test_GroundTruth",
        "Task 1 ground-truth directory",
    )
    task2_dir = require_directory(
        source_root
        / "ISIC2018_Task2_Test_GroundTruth"
        / "ISIC2018_Task2_Test_GroundTruth",
        "Task 2 ground-truth directory",
    )
    image_paths = sorted(input_dir.glob("*.jpg"), key=lambda path: path.stem)
    if not image_paths:
        raise ConversionError("Task 1-2 test input contains no JPEG images")

    records: list[dict[str, Any]] = []
    task2_presence_counts: Counter[int] = Counter()
    for image_path in image_paths:
        image_id = image_path.stem
        width, height = image_size(image_path, "JPEG")
        task1_mask = require_file(
            task1_dir / f"{image_id}_segmentation.png",
            f"Task 1 mask for {image_id}",
        )
        _task1_present, bbox = mask_presence_and_bbox(
            task1_mask, (width, height), derive_bbox=True
        )
        if bbox is None:
            raise ConversionError(f"Task 1 mask for {image_id} did not produce a bbox")

        attribute_presence: dict[str, bool] = {}
        attribute_mask_paths: dict[str, str] = {}
        positive_attributes: list[str] = []
        for attribute in ATTRIBUTES:
            mask_path = require_file(
                task2_dir / f"{image_id}_attribute_{attribute}.png",
                f"Task 2 {attribute} mask for {image_id}",
            )
            present, _ = mask_presence_and_bbox(
                mask_path, (width, height), derive_bbox=False
            )
            attribute_presence[attribute] = present
            if present:
                positive_attributes.append(attribute)
            attribute_mask_paths[attribute] = source_relative(mask_path, source_root)
        task2_presence_counts[len(positive_attributes)] += 1

        annotation = common_annotation(["task1", "task2"], width, height)
        annotation.update(
            {
                "task1_segmentation_mask_path": source_relative(
                    task1_mask, source_root
                ),
                "findings": [{"finding": "lesion", "bbox": bbox}],
                "attributes": positive_attributes,
                "attribute_presence": attribute_presence,
                "task2_attribute_mask_paths": attribute_mask_paths,
            }
        )
        records.append(
            {
                "patient": {"patient_id": None, "annotation": {}},
                "case": {"case_id": None, "annotation": {}},
                "study": {"study_id": None, "annotation": {}},
                "image": {
                    "image_id": image_id,
                    "image_path": source_relative(image_path, source_root),
                    "modality": None,
                    "anatomy": None,
                    "annotation": annotation,
                },
            }
        )
    return records, {str(key): value for key, value in sorted(task2_presence_counts.items())}


def task3_records(source_root: Path) -> list[dict[str, Any]]:
    input_dir = require_directory(
        source_root
        / "ISIC2018_Task3_Test_Input"
        / "ISIC2018_Task3_Test_Input",
        "Task 3 test input directory",
    )
    csv_path = (
        source_root
        / "ISIC2018_Task3_Test_GroundTruth"
        / "ISIC2018_Task3_Test_GroundTruth"
        / "ISIC2018_Task3_Test_GroundTruth.csv"
    )
    csv_rows = load_task3_rows(csv_path)
    image_paths = sorted(input_dir.glob("*.jpg"), key=lambda path: path.stem)
    if not image_paths:
        raise ConversionError("Task 3 test input contains no JPEG images")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for image_path in image_paths:
        image_id = image_path.stem
        if image_id not in csv_rows:
            raise ConversionError(f"Task 3 image has no CSV row: {image_id}")
        seen_ids.add(image_id)
        width, height = image_size(image_path, "JPEG")
        annotation = common_annotation(["task3"], width, height)
        annotation.update(csv_rows[image_id])
        records.append(
            {
                "patient": {"patient_id": None, "annotation": {}},
                "case": {"case_id": None, "annotation": {}},
                "study": {"study_id": None, "annotation": {}},
                "image": {
                    "image_id": image_id,
                    "image_path": source_relative(image_path, source_root),
                    "modality": None,
                    "anatomy": None,
                    "annotation": annotation,
                },
            }
        )
    orphan_ids = sorted(set(csv_rows) - seen_ids)
    if orphan_ids:
        raise ConversionError(f"Task 3 CSV contains orphan image ids: {orphan_ids[:5]}")
    return records


def validate_bbox(bbox: object, width: int, height: int, label: str) -> None:
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in bbox
        )
        or not (0 <= bbox[0] < bbox[2] <= width)
        or not (0 <= bbox[1] < bbox[3] <= height)
    ):
        raise ConversionError(f"{label}: invalid canonical bbox {bbox!r}")


def validate_records(records: list[dict[str, Any]], source_root: Path) -> None:
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        if set(record) != TOP_LEVEL_KEYS:
            raise ConversionError(f"Output line {line_number}: invalid top-level keys")
        for entity, expected_keys in ENTITY_KEYS.items():
            value = record.get(entity)
            if not isinstance(value, dict) or set(value) != expected_keys:
                raise ConversionError(f"Output line {line_number}: invalid {entity} object")
            if not isinstance(value["annotation"], dict):
                raise ConversionError(f"Output line {line_number}: invalid {entity}.annotation")
        image = record["image"]
        image_id = image["image_id"]
        image_path_value = image["image_path"]
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ConversionError(f"Output line {line_number}: duplicate/invalid image_id")
        if (
            not isinstance(image_path_value, str)
            or not image_path_value
            or image_path_value in image_paths
            or Path(image_path_value).is_absolute()
        ):
            raise ConversionError(f"Output line {line_number}: duplicate/invalid image_path")
        image_ids.add(image_id)
        image_paths.add(image_path_value)
        resolved_image = (source_root / Path(image_path_value)).resolve()
        try:
            resolved_image.relative_to(source_root.resolve())
        except ValueError as exc:
            raise ConversionError(f"Output line {line_number}: image path escapes source root") from exc
        require_file(resolved_image, f"Referenced image for {image_id}")
        annotation = image["annotation"]
        width = annotation.get("width")
        height = annotation.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ConversionError(f"Output line {line_number}: invalid source dimensions")
        findings = annotation.get("findings")
        if findings is not None:
            if not isinstance(findings, list) or len(findings) != 1:
                raise ConversionError(f"Output line {line_number}: invalid findings list")
            finding = findings[0]
            if not isinstance(finding, dict) or set(finding) != {"finding", "bbox"}:
                raise ConversionError(f"Output line {line_number}: invalid finding object")
            if finding["finding"] != "lesion":
                raise ConversionError(f"Output line {line_number}: unexpected finding value")
            validate_bbox(finding["bbox"], width, height, f"line {line_number}")
        presence = annotation.get("attribute_presence")
        attributes = annotation.get("attributes")
        if presence is not None or attributes is not None:
            if not isinstance(presence, dict) or set(presence) != set(ATTRIBUTES):
                raise ConversionError(f"Output line {line_number}: invalid attribute_presence")
            if not all(isinstance(value, bool) for value in presence.values()):
                raise ConversionError(f"Output line {line_number}: attribute_presence must be boolean")
            expected_attributes = [attribute for attribute in ATTRIBUTES if presence[attribute]]
            if attributes != expected_attributes:
                raise ConversionError(f"Output line {line_number}: attributes do not match presence map")
        diagnosis = annotation.get("diagnosis")
        scores = annotation.get("diagnosis_scores")
        if diagnosis is not None or scores is not None:
            if diagnosis not in DIAGNOSIS_COLUMNS or not isinstance(scores, dict):
                raise ConversionError(f"Output line {line_number}: invalid diagnosis mapping")
            if tuple(scores) != DIAGNOSIS_COLUMNS:
                raise ConversionError(f"Output line {line_number}: diagnosis score column order changed")
            active = [column for column in DIAGNOSIS_COLUMNS if scores[column] == 1.0]
            if active != [diagnosis] or any(scores[column] not in {0.0, 1.0} for column in DIAGNOSIS_COLUMNS):
                raise ConversionError(f"Output line {line_number}: diagnosis scores are not one-hot")


def render_jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    return (
        "".join(f"{canonical_json(record)}\n" for record in records).encode("utf-8")
    )


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def tree_digest(path: Path, source_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    for file_path in files:
        if not file_path.is_file():
            continue
        relative = source_relative(file_path, source_root)
        file_hash = sha256_file(file_path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_all_records(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task12, presence_counts = task12_records(source_root)
    task3 = task3_records(source_root)
    records = sorted([*task12, *task3], key=lambda record: record["image"]["image_id"])
    if len(records) != len({record["image"]["image_id"] for record in records}):
        raise ConversionError("Task 1-2 and Task 3 populations contain duplicate image ids")
    validate_records(records, source_root)
    return records, {
        "task12_images": len(task12),
        "task3_images": len(task3),
        "task2_present_attribute_count_distribution": presence_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan-path", required=True, type=Path)
    parser.add_argument("--annotation-path", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Build the records twice and require byte-identical JSONL before publishing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plan_path = args.plan_path.expanduser().resolve()
    annotation_path = args.annotation_path.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()
    require_directory(source_root, "source root")
    if not annotation_path.is_relative_to(output_dir):
        raise ConversionError("annotation-path must remain beneath output-dir")
    if not report_path.is_relative_to(output_dir):
        raise ConversionError("report-path must remain beneath output-dir")

    approved_mapping_hash = verify_approved_plan(plan_path)
    records, reconciliation = build_all_records(source_root)
    rendered = render_jsonl(records)
    determinism_passed: bool | None = None
    if args.verify_determinism:
        second_records, second_reconciliation = build_all_records(source_root)
        second_rendered = render_jsonl(second_records)
        if reconciliation != second_reconciliation or rendered != second_rendered:
            raise ConversionError("Two unchanged converter passes were not byte-identical")
        determinism_passed = True

    write_atomic(annotation_path, rendered)
    source_inputs = {
        "task12_input": source_root / "ISIC2018_Task1-2_Test_Input" / "ISIC2018_Task1-2_Test_Input",
        "task1_ground_truth": source_root / "ISIC2018_Task1_Test_GroundTruth" / "ISIC2018_Task1_Test_GroundTruth",
        "task2_ground_truth": source_root / "ISIC2018_Task2_Test_GroundTruth" / "ISIC2018_Task2_Test_GroundTruth",
        "task3_input": source_root / "ISIC2018_Task3_Test_Input" / "ISIC2018_Task3_Test_Input",
        "task3_ground_truth_csv": source_root / "ISIC2018_Task3_Test_GroundTruth" / "ISIC2018_Task3_Test_GroundTruth" / "ISIC2018_Task3_Test_GroundTruth.csv",
    }
    input_hashes = {
        name: tree_digest(path, source_root) for name, path in source_inputs.items()
    }
    report = {
        "schema": "annotation.v2.schema.json",
        "source_root": str(source_root),
        "approved_mapping_sha256": approved_mapping_hash,
        "converter_hash": sha256_file(Path(__file__).resolve()),
        "input_file_hashes": input_hashes,
        "output_file_hash": sha256_file(annotation_path),
        "output_record_count": len(records),
        "source_to_output_reconciliation": reconciliation,
        "mapping_decisions": {
            "task1_bbox": "tight nonzero-mask extent in absolute source-image xyxy pixel coordinates",
            "task2_attribute": "filename attribute plus nonblack-mask presence; no bbox",
            "task3_diagnosis": "single active one-hot CSV column; no bbox",
            "image_paths": "source-relative JPEG paths; no image copied by this converter",
        },
        "schema_validation": {
            "passed": True,
            "validator": "deterministic structural and source-path checks in prepare_annotations.py",
        },
        "determinism_check": {
            "requested": args.verify_determinism,
            "passed": determinism_passed,
        },
        "unresolved_issues": [],
    }
    write_atomic(report_path, (canonical_json(report) + "\n").encode("utf-8"))
    print(
        json.dumps(
            {
                "annotation_path": str(annotation_path),
                "report_path": str(report_path),
                "records": len(records),
                "annotation_sha256": report["output_file_hash"],
                "determinism_check": report["determinism_check"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConversionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
