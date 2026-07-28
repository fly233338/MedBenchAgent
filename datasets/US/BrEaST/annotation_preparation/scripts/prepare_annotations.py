"""Prepare canonical BrEaST image annotations from the approved XLSX mapping."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

SHEET_NAME = "BrEaST-Lesions-USG clinical dat"
PNG_DIRECTORY = PurePosixPath(
    "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023"
    "/BrEaST-Lesions_USG-images_and_masks"
)
COLUMNS = [
    "CaseID",
    "Image_filename",
    "Mask_tumor_filename",
    "Mask_other_filename",
    "Pixel_size",
    "Age",
    "Tissue_composition",
    "Signs",
    "Symptoms",
    "Shape",
    "Margin",
    "Echogenicity",
    "Posterior_features",
    "Halo",
    "Calcifications",
    "Skin_thickening",
    "Interpretation",
    "BIRADS",
    "Verification",
    "Diagnosis",
    "Classification",
]
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference}")
    result = 0
    for letter in match.group(1):
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    namespace = {"m": MAIN_NS}
    return ["".join(node.text or "" for node in item.findall(".//m:t", namespace)) for item in root]


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"m": MAIN_NS, "r": REL_NS}
    relationship_id = None
    for sheet in workbook.findall(".//m:sheets/m:sheet", namespace):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(f"{{{REL_NS}}}id")
            break
    if relationship_id is None:
        raise ValueError(f"XLSX sheet not found: {sheet_name}")

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = None
    for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib.get("Target")
            break
    if target is None:
        raise ValueError(f"XLSX relationship not found for sheet: {sheet_name}")
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join("xl", target))


def _numeric_value(raw: str) -> int | float:
    if not any(character in raw.lower() for character in (".", "e")):
        return int(raw)
    return float(raw)


def _cell_value(
    cell: ET.Element,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> object:
    if cell.find("m:f", namespace) is not None:
        raise ValueError("Formula cells are not supported in the approved source sheet.")
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//m:t", namespace))
    value = cell.find("m:v", namespace)
    if value is None or value.text is None:
        return None
    raw = value.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "e", "d"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    if cell_type == "n":
        return _numeric_value(raw)
    raise ValueError(f"Unsupported XLSX cell type: {cell_type}")


def _read_sheet(workbook_path: Path) -> list[dict[str, object]]:
    namespace = {"m": MAIN_NS}
    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = _shared_strings(archive)
        root = ET.fromstring(archive.read(_sheet_path(archive, SHEET_NAME)))

    parsed_rows: list[dict[int, object]] = []
    for row in root.findall(".//m:sheetData/m:row", namespace):
        parsed: dict[int, object] = {}
        for cell in row.findall("m:c", namespace):
            parsed[_column_index(cell.attrib["r"])] = _cell_value(
                cell,
                shared_strings,
                namespace,
            )
        parsed_rows.append(parsed)
    if not parsed_rows:
        raise ValueError(f"XLSX sheet is empty: {SHEET_NAME}")

    header_width = max(parsed_rows[0], default=-1) + 1
    headers = [parsed_rows[0].get(index) for index in range(header_width)]
    if headers != COLUMNS:
        raise ValueError(f"Unexpected XLSX columns: {headers}")

    rows = [
        {column: parsed.get(index) for index, column in enumerate(COLUMNS)}
        for parsed in parsed_rows[1:]
        if parsed
    ]
    if len(rows) != 256:
        raise ValueError(f"Expected 256 BrEaST source rows, found {len(rows)}.")
    return rows


def _relative_path(filename: str) -> str:
    return (PNG_DIRECTORY / filename).as_posix()


def _require_string(row: dict[str, object], field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _records(workbook_path: Path, source_root: Path) -> list[dict[str, object]]:
    rows = sorted(_read_sheet(workbook_path), key=lambda row: int(row["CaseID"]))
    png_root = source_root.joinpath(*PNG_DIRECTORY.parts)
    if not png_root.is_dir():
        raise ValueError(f"PNG directory does not exist: {png_root}")

    records: list[dict[str, object]] = []
    referenced_pngs: set[str] = set()
    image_ids: set[str] = set()
    case_ids: set[int] = set()
    for row in rows:
        case_id = row["CaseID"]
        if not isinstance(case_id, int) or case_id in case_ids:
            raise ValueError(f"Invalid or duplicate CaseID: {case_id}")
        case_ids.add(case_id)

        image_filename = _require_string(row, "Image_filename")
        if not image_filename.endswith(".png"):
            raise ValueError(f"Unexpected image filename: {image_filename}")
        image_id = image_filename[:-4]
        if image_id in image_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        image_ids.add(image_id)
        referenced_pngs.add(image_filename)

        tumor_value = row["Mask_tumor_filename"]
        if tumor_value is not None and not isinstance(tumor_value, str):
            raise ValueError("Mask_tumor_filename must be a string or blank.")
        tumor_names = [] if tumor_value is None else [tumor_value]
        referenced_pngs.update(tumor_names)

        other_value = row["Mask_other_filename"]
        if other_value is not None and not isinstance(other_value, str):
            raise ValueError("Mask_other_filename must be a string or blank.")
        other_names = [] if other_value is None else other_value.split("&")
        referenced_pngs.update(other_names)

        annotation = {
            "modality": "ultrasound",
            "anatomy": "breast",
            "tumor_mask_paths": [_relative_path(name) for name in tumor_names],
            "other_mask_paths": [_relative_path(name) for name in other_names],
            "pixel_size_cm": row["Pixel_size"],
            "age": row["Age"],
            "tissue_composition": _require_string(row, "Tissue_composition"),
            "signs": _require_string(row, "Signs"),
            "symptoms": _require_string(row, "Symptoms"),
            "shape": _require_string(row, "Shape"),
            "margin": _require_string(row, "Margin"),
            "echogenicity": _require_string(row, "Echogenicity"),
            "posterior_features": _require_string(row, "Posterior_features"),
            "halo": _require_string(row, "Halo"),
            "calcifications": _require_string(row, "Calcifications"),
            "skin_thickening": _require_string(row, "Skin_thickening"),
            "interpretation": _require_string(row, "Interpretation"),
            "birads": row["BIRADS"],
            "verification": _require_string(row, "Verification"),
            "diagnosis": _require_string(row, "Diagnosis"),
            "classification": _require_string(row, "Classification"),
        }
        records.append(
            {
                "image_id": image_id,
                "image_path": _relative_path(image_filename),
                "patient_id": None,
                "case_id": str(case_id),
                "study_id": None,
                "annotation": annotation,
            }
        )

    existing_pngs = {path.name for path in png_root.glob("*.png")}
    if referenced_pngs != existing_pngs:
        missing = sorted(referenced_pngs - existing_pngs)
        orphaned = sorted(existing_pngs - referenced_pngs)
        raise ValueError(f"PNG reconciliation failed; missing={missing}, orphaned={orphaned}")
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
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = _records(args.workbook.resolve(), args.source_image_root.resolve())
    _atomic_write(records, args.output.resolve())
    print(json.dumps({"records": len(records)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
