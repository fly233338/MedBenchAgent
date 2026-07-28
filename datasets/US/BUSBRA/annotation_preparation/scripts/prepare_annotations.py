"""Prepare canonical BUSBRA image annotations from the approved CSV mapping."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

COLUMNS = [
    "ID",
    "Case",
    "Histology",
    "Pathology",
    "BIRADS",
    "Device",
    "Width",
    "Height",
    "Side",
    "BBOX",
]


def _integer(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc


def _string(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _source_bbox(value: str) -> list[int]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid BBOX source value: {value}") from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 4
        or any(not isinstance(item, int) for item in parsed)
    ):
        raise ValueError(f"BBOX must contain four integers: {value}")
    x, y, width, height = parsed
    if x < 1 or y < 1 or width < 0 or height < 0:
        raise ValueError(f"BBOX is not a one-based MATLAB rectangle: {value}")
    return parsed


def _xyxy(source_bbox: list[int], image_width: int, image_height: int) -> list[int]:
    x, y, width, height = source_bbox
    bbox = [x - 1, y - 1, x + width - 1, y + height - 1]
    if not (
        0 <= bbox[0] <= bbox[2] < image_width
        and 0 <= bbox[1] <= bbox[3] < image_height
    ):
        raise ValueError(
            f"Converted bbox {bbox} is outside source dimensions "
            f"{image_width}x{image_height}."
        )
    return bbox


def _records(table_path: Path, source_root: Path) -> list[dict[str, object]]:
    with table_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 1875:
        raise ValueError(f"Expected 1875 BUSBRA source rows, found {len(rows)}.")
    rows.sort(key=lambda row: row["ID"])

    records: list[dict[str, object]] = []
    image_ids: set[str] = set()
    case_counts: Counter[int] = Counter()
    expected_images: set[str] = set()
    expected_masks: set[str] = set()
    for row in rows:
        image_id = _string(row, "ID")
        match = re.fullmatch(r"bus_(\d{4})-([lrs])", image_id)
        if match is None or image_id in image_ids:
            raise ValueError(f"Invalid or duplicate ID: {image_id}")
        image_ids.add(image_id)

        case = _integer(row, "Case")
        if int(match.group(1)) != case:
            raise ValueError(f"ID and Case disagree: {image_id}, {case}")
        case_counts[case] += 1

        side = _string(row, "Side")
        expected_side = {"l": "left", "r": "right", "s": "single"}[match.group(2)]
        if side != expected_side:
            raise ValueError(f"ID suffix and Side disagree: {image_id}, {side}")

        width = _integer(row, "Width")
        height = _integer(row, "Height")
        bbox_source = _string(row, "BBOX")
        bbox = _xyxy(_source_bbox(bbox_source), width, height)

        image_path = f"Images/{image_id}.png"
        mask_id = image_id.replace("bus_", "mask_", 1)
        mask_path = f"Masks/{mask_id}.png"
        expected_images.add(f"{image_id}.png")
        expected_masks.add(f"{mask_id}.png")

        annotation = {
            "modality": "ultrasound",
            "anatomy": "breast",
            "mask_path": mask_path,
            "histology": _string(row, "Histology"),
            "pathology": _string(row, "Pathology"),
            "birads": _integer(row, "BIRADS"),
            "device": _string(row, "Device"),
            "width": width,
            "height": height,
            "side": side,
            "bbox_source": bbox_source,
            "bbox": bbox,
        }
        records.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "patient_id": None,
                "case_id": str(case),
                "study_id": None,
                "annotation": annotation,
            }
        )

    if len(case_counts) != 1064 or Counter(case_counts.values()) != Counter({1: 253, 2: 811}):
        raise ValueError("Case-to-image cardinality does not match the approved mapping.")
    existing_images = {path.name for path in (source_root / "Images").glob("*.png")}
    existing_masks = {path.name for path in (source_root / "Masks").glob("*.png")}
    if expected_images != existing_images:
        raise ValueError("Image file population does not match bus_data.csv IDs.")
    if expected_masks != existing_masks:
        raise ValueError("Mask file population does not match bus_data.csv IDs.")
    return records


def _atomic_write(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as destination:
            for record in records:
                destination.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _records(args.table.resolve(), args.source_image_root.resolve())
    _atomic_write(records, args.output.resolve())
    print(json.dumps({"records": len(records)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
