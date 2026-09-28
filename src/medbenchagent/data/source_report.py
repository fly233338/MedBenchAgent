"""Independently verify a coding worker's source inventory against original inputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.data.operators import columns, connect, quoted, read_relation
from medbenchagent.data.locators import json_objects
from medbenchagent.data.sources import digest_file, track_files, verify_source
from medbenchagent.data.validation import source_image_path


def _reconciliation_error(output: Path, *, kind: str, row_number: int,
                          source_locator: dict, expected: dict, actual: object,
                          message: str) -> ValueError:
    """Persist actionable source/candidate evidence before raising validation failure."""
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "reconciliation-report.v1",
        "status": "failed",
        "error_kind": kind,
        "row_number": row_number,
        "source_locator": source_locator,
        "expected": expected,
        "actual": actual,
        "field_diffs": [
            {"field": key, "expected": expected.get(key),
             "actual": actual.get(key) if isinstance(actual, dict) else None}
            for key in expected
            if not isinstance(actual, dict) or actual.get(key) != expected.get(key)
        ],
        "message": message,
        "repair_hint": "Compare the candidate row with the expected source row and change only the mapping or adapter logic responsible for the listed fields.",
    }
    (output / "reconciliation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ValueError(message + "; see reconciliation_report.json for expected/actual field diffs")


class ReportTable(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    sheet: str | None = None
    delimiter: str = ","
    rows: int = Field(ge=0)
    role: str


class PopulationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["table"] = "table"
    table_path: str
    sheet: str | None = None
    delimiter: str = ","
    row_id_field: str
    image_id_field: str
    image_path_field: str
    image_prefix: str = ""
    mask_fields: list[str] = Field(default_factory=list)
    mask_prefix: str = ""
    classification_field: str | None = None


class SpecialPopulationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["special"] = "special"
    source_files: list[str] = Field(min_length=1)
    required_operation: Literal["aggregate_filter", "nested_array_join", "derived_identity"]
    reason: str = Field(min_length=1)


class JsonPopulationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["json"] = "json"
    path: str
    collection: str = ""
    row_id_field: str
    image_id_field: str
    image_path_field: str
    image_prefix: str = ""
    mask_fields: list[str] = Field(default_factory=list)
    mask_prefix: str = ""
    classification_field: str | None = None


class SourceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["source-report.v1"] = "source-report.v1"
    tables: list[ReportTable] = Field(default_factory=list)
    population: Annotated[PopulationBinding | JsonPopulationBinding | SpecialPopulationBinding,
                          Field(discriminator="kind")]
    record_count: int = Field(ge=0)
    unique_images: int = Field(ge=1)
    mask_references: int = Field(ge=0)
    classification_counts: dict[str, int] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)


def _relative(prefix: str, value: str) -> str:
    path = Path(prefix) / value if prefix else Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("Source report paths must be source-relative")
    return path.as_posix()


def validate_source_report(context, report_path, candidate_population_path, output_directory):
    """Write a trusted population receipt only after a full independent table scan."""
    verify_source(context["source_manifest"])
    report = SourceReport.model_validate_json(Path(report_path).read_text(encoding="utf-8"))
    root = context["source_root"]
    binding = report.population
    selected = (next((table for table in report.tables if table.path == binding.table_path
                      and table.sheet == binding.sheet and table.delimiter == binding.delimiter), None)
                if isinstance(binding, PopulationBinding) else None)
    for table in report.tables:
        source_image_path(root, _relative("", table.path))
    tracked = [table.path for table in report.tables]
    if isinstance(binding, SpecialPopulationBinding):
        if report.classification_counts or report.record_count:
            raise ValueError("Special population cannot claim independently verified row or class counts")
        tracked.extend(binding.source_files)
        for path in binding.source_files:
            source_image_path(root, _relative("", path))
    elif isinstance(binding, JsonPopulationBinding):
        tracked.append(binding.path)
        source_image_path(root, _relative("", binding.path))
    track_files(context["source_manifest"], tracked)
    table_summaries = []
    for index, table in enumerate(report.tables):
        if table is selected:
            continue
        with connect() as other_db:
            read_relation(other_db, root, table.path, f"other_{index}", sheet=table.sheet,
                          delimiter=table.delimiter)
            actual = other_db.execute(f'SELECT count(*) FROM {quoted(f"other_{index}")}').fetchone()[0]
            if actual != table.rows:
                raise ValueError(f"Worker table row count differs from source: {table.path}")
            table_summaries.append({"path": table.path, "rows": actual, "role": table.role,
                                    "columns": [name for name in columns(other_db, f"other_{index}")
                                                if name != "_lineage"]})
    if isinstance(binding, SpecialPopulationBinding):
        verified = _validate_special_population(context, report, candidate_population_path, output_directory)
        verified["table_summaries"] = table_summaries
        verify_source(context["source_manifest"])
        return verified
    if isinstance(binding, JsonPopulationBinding):
        verified = _validate_json_population(context, report, candidate_population_path, output_directory)
        verified["table_summaries"] = table_summaries
        verify_source(context["source_manifest"])
        return verified
    if selected is None:
        raise ValueError("Population table is absent from the source report")
    if not report.record_count:
        raise ValueError("Tabular population requires a nonempty source table")
    required = {binding.row_id_field, binding.image_id_field, binding.image_path_field,
                *binding.mask_fields}
    if len(binding.mask_fields) != len(set(binding.mask_fields)):
        raise ValueError("Mask fields must be unique")
    if binding.classification_field:
        required.add(binding.classification_field)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trusted_path = output / "population.jsonl"
    rows = images = masks = 0
    class_counts: Counter[str] = Counter()
    image_ids: dict[str, str] = {}
    image_paths: set[str] = set()
    mask_owners: dict[str, str] = {}
    preview = []
    with connect() as db:
        read_relation(db, root, binding.table_path, "source", sheet=binding.sheet,
                      delimiter=binding.delimiter)
        fields = [name for name in columns(db, "source") if name != "_lineage"]
        absent = required - set(fields)
        if absent:
            raise ValueError(f"Report names unknown source columns: {sorted(absent)}")
        query = db.execute(f'SELECT * EXCLUDE (_lineage) FROM {quoted("source")} ORDER BY rowid')
        with trusted_path.open("w", encoding="utf-8") as trusted, \
                Path(candidate_population_path).open(encoding="utf-8") as candidate:
            while batch := query.fetchmany(1000):
                for values in batch:
                    rows += 1
                    record = dict(zip(fields, values, strict=True))
                    row_id = str(record[binding.row_id_field] or "")
                    image_id = str(record[binding.image_id_field] or "")
                    raw_path = str(record[binding.image_path_field] or "")
                    if not row_id or not image_id or not raw_path:
                        raise ValueError(f"Missing row/image identity at source row {rows}")
                    image_path = _relative(binding.image_prefix, raw_path)
                    source_image_path(root, image_path)
                    if image_id in image_ids and image_ids[image_id] != image_path:
                        raise ValueError(f"Image identity has conflicting paths at source row {rows}")
                    if image_id not in image_ids and image_path in image_paths:
                        raise ValueError(f"Duplicate image path: {image_path}")
                    first_image_row = image_id not in image_ids
                    image_ids[image_id] = image_path
                    image_paths.add(image_path)
                    mask_paths = []
                    for field in binding.mask_fields:
                        raw_mask = str(record[field] or "")
                        if raw_mask:
                            mask_path = _relative(binding.mask_prefix, raw_mask)
                            source_image_path(root, mask_path)
                            if mask_path == image_path:
                                raise ValueError(f"Mask and image are the same file at row {rows}")
                            owner = mask_owners.setdefault(mask_path, image_id)
                            if owner != image_id:
                                raise ValueError(f"Mask belongs to multiple images: {mask_path}")
                            if mask_path in mask_paths:
                                raise ValueError(f"Duplicate mask path at source row {rows}")
                            mask_paths.append(mask_path)
                            masks += 1
                    label = str(record[binding.classification_field] or "") if binding.classification_field else None
                    if label is not None:
                        class_counts[label] += 1
                    expected = {"source_id": context["source_scope"], "row_id": row_id,
                                "image_id": image_id, "image_path": image_path,
                                "mask_paths": mask_paths, "classification": label}
                    line = candidate.readline()
                    try:
                        claimed = json.loads(line)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Worker population is missing or invalid at row {rows}") from exc
                    if claimed != expected:
                        raise _reconciliation_error(
                            output, kind="row_mismatch", row_number=rows,
                            source_locator={"file": binding.table_path, "sheet": binding.sheet,
                                            "row": rows + 1}, expected=expected,
                            actual=claimed,
                            message=f"Worker population differs from source at row {rows}")
                    if len(preview) < 3:
                        preview.append(record)
                    if first_image_row:
                        trusted.write(json.dumps({"source_id": context["source_scope"],
                                                  "image_id": image_id, "image_path": image_path},
                                                 ensure_ascii=False) + "\n")
                        images += 1
            if candidate.readline():
                raise ValueError("Worker population contains extra rows")
    if (rows, images, masks) != (report.record_count, report.unique_images, report.mask_references):
        raise ValueError("Worker report counts differ from independently measured source counts")
    if dict(class_counts) != report.classification_counts:
        raise ValueError("Worker classification counts differ from source values")
    if selected.rows != rows:
        raise ValueError("Worker table row count differs from source")
    table_summaries.append({"path": selected.path, "rows": rows, "role": selected.role,
                            "columns": fields})
    verify_source(context["source_manifest"])
    return {"report": report, "rows": rows, "unique_images": images, "mask_references": masks,
            "classification_counts": dict(class_counts), "population_path": str(trusted_path),
            "population_digest": digest_file(trusted_path), "columns": fields, "samples": preview,
            "assurance": "independent", "table_summaries": table_summaries}


def _validate_special_population(context, report, candidate_population_path, output_directory):
    """Check a candidate list's structure and paths; completeness awaits human review."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trusted_path = output / "population.jsonl"
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    mask_owners: dict[str, str] = {}
    masks = 0
    with Path(candidate_population_path).open(encoding="utf-8") as candidate, \
            trusted_path.open("w", encoding="utf-8") as trusted:
        for number, line in enumerate(candidate, 1):
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid special population row {number}") from exc
            if not isinstance(item, dict) or set(item) != {"source_id", "image_id", "image_path", "mask_paths"}:
                raise ValueError(f"Invalid special population fields at row {number}")
            identifier, relative = item["image_id"], item["image_path"]
            if (item["source_id"] != context["source_scope"] or not isinstance(identifier, str)
                    or not identifier or not isinstance(relative, str) or not relative):
                raise ValueError(f"Invalid image identity at row {number}")
            if identifier in image_ids or relative in image_paths:
                raise ValueError(f"Duplicate image identity or path at row {number}")
            source_image_path(context["source_root"], _relative("", relative))
            image_ids.add(identifier)
            image_paths.add(relative)
            mask_list = item["mask_paths"]
            if (not isinstance(mask_list, list)
                    or any(not isinstance(mask, str) or not mask for mask in mask_list)
                    or len(mask_list) != len(set(mask_list))):
                raise ValueError(f"Invalid mask list at row {number}")
            for mask in mask_list:
                source_image_path(context["source_root"], _relative("", mask))
                if mask == relative or mask_owners.setdefault(mask, identifier) != identifier:
                    raise ValueError(f"Invalid mask relationship at row {number}")
                masks += 1
            trusted.write(json.dumps({"source_id": item["source_id"], "image_id": identifier,
                                      "image_path": relative}, ensure_ascii=False) + "\n")
    if not image_ids or len(image_ids) != report.unique_images or masks != report.mask_references:
        raise ValueError("Special population counts differ from the candidate list")
    return {"report": report, "rows": 0, "unique_images": len(image_ids), "mask_references": masks,
            "classification_counts": {}, "population_path": str(trusted_path),
            "population_digest": digest_file(trusted_path), "columns": [], "samples": [],
            "assurance": "human_review_required"}


def _validate_json_population(context, report, candidate_population_path, output_directory):
    """Enumerate direct JSON or JSONL object collections without using worker assertions."""
    binding = report.population
    if len(binding.mask_fields) != len(set(binding.mask_fields)):
        raise ValueError("Mask fields must be unique")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trusted_path = output / "population.jsonl"
    image_ids: dict[str, str] = {}
    image_paths: set[str] = set()
    mask_owners: dict[str, str] = {}
    classes: Counter[str] = Counter()
    rows = masks = 0
    fields: set[str] = set()
    preview = []
    with Path(candidate_population_path).open(encoding="utf-8") as candidate, \
            trusted_path.open("w", encoding="utf-8") as trusted:
        for locator, record in json_objects(context["source_root"], binding.path, binding.collection):
            rows += 1
            if not isinstance(record, dict):
                raise ValueError(f"JSON population member is not an object at row {rows}")
            fields.update(record)
            required = {binding.image_id_field, binding.image_path_field, *binding.mask_fields}
            required.add(binding.row_id_field)
            if binding.classification_field:
                required.add(binding.classification_field)
            if not required <= record.keys():
                raise ValueError(f"JSON population member lacks required fields at row {rows}")
            image_id = str(record[binding.image_id_field] or "")
            raw_path = str(record[binding.image_path_field] or "")
            row_id = str(record[binding.row_id_field] or "")
            if not image_id or not raw_path or not row_id:
                raise ValueError(f"Missing JSON row/image identity at row {rows}")
            image_path = _relative(binding.image_prefix, raw_path)
            source_image_path(context["source_root"], image_path)
            if image_id in image_ids and image_ids[image_id] != image_path:
                raise ValueError(f"JSON image identity has conflicting paths at row {rows}")
            if image_id not in image_ids and image_path in image_paths:
                raise ValueError(f"Duplicate JSON image path at row {rows}")
            first_image_row = image_id not in image_ids
            image_ids[image_id] = image_path
            image_paths.add(image_path)
            mask_paths = []
            for field in binding.mask_fields:
                raw_masks = record[field]
                for raw_mask in raw_masks if isinstance(raw_masks, list) else [raw_masks]:
                    if raw_mask:
                        mask_path = _relative(binding.mask_prefix, str(raw_mask))
                        source_image_path(context["source_root"], mask_path)
                        if mask_path == image_path or mask_owners.setdefault(mask_path, image_id) != image_id:
                            raise ValueError(f"Invalid JSON mask relationship at row {rows}")
                        if mask_path in mask_paths:
                            raise ValueError(f"Duplicate JSON mask path at row {rows}")
                        mask_paths.append(mask_path)
                        masks += 1
            label = str(record[binding.classification_field] or "") if binding.classification_field else None
            if label is not None:
                classes[label] += 1
            expected = {"source_id": context["source_scope"], "row_id": row_id,
                        "image_id": image_id, "image_path": image_path,
                        "mask_paths": mask_paths, "classification": label}
            try:
                claimed = json.loads(candidate.readline())
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Worker JSON population is missing or invalid at row {rows}") from exc
            if claimed != expected:
                raise ValueError(f"Worker JSON population differs from source at row {rows}")
            if len(preview) < 3:
                preview.append(record)
            if first_image_row:
                trusted.write(json.dumps({"source_id": context["source_scope"],
                                          "image_id": image_id, "image_path": image_path},
                                         ensure_ascii=False) + "\n")
        if candidate.readline():
            raise ValueError("Worker JSON population contains extra rows")
    if not rows or rows != report.record_count or len(image_ids) != report.unique_images \
            or masks != report.mask_references or dict(classes) != report.classification_counts:
        raise ValueError("Worker JSON report counts differ from independently measured source counts")
    return {"report": report, "rows": rows, "unique_images": len(image_ids), "mask_references": masks,
            "classification_counts": dict(classes), "population_path": str(trusted_path),
            "population_digest": digest_file(trusted_path), "columns": sorted(fields),
            "samples": preview, "assurance": "independent"}
