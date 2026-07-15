"""Local CSV/XLSX inventory, provenance-preserving DuckDB access, and task planning."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable

import duckdb
from openpyxl import load_workbook

from medbenchagent.llm import complete_json

TABULAR_SUFFIXES = {".csv", ".xlsx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".dcm"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_") or "column"


def _unique_columns(values: Iterable[object]) -> list[str]:
    used: dict[str, int] = {}
    result = []
    for index, value in enumerate(values, start=1):
        base = _safe_name(str(value or f"column_{index}"))
        used[base] = used.get(base, 0) + 1
        result.append(base if used[base] == 1 else f"{base}_{used[base]}")
    return result


def _read_csv(path: Path) -> tuple[str, list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, [])
        columns = _unique_columns(header)
        rows = [dict(zip(columns, row, strict=False)) for row in reader]
    return "csv", columns, rows


def _read_xlsx(path: Path) -> list[tuple[str, list[str], list[dict[str, object]]]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    results = []
    for sheet in workbook.worksheets:
        values = sheet.iter_rows(values_only=True)
        columns = _unique_columns(next(values, []))
        rows = [dict(zip(columns, row, strict=False)) for row in values]
        results.append((sheet.title, columns, rows))
    return results


def _column_stats(rows: list[dict[str, object]], column: str) -> dict[str, object]:
    values = [row.get(column) for row in rows]
    populated = [value for value in values if value not in (None, "")]
    text_values = [str(value) for value in populated]
    numeric_values: list[float] = []
    for value in populated:
        try:
            numeric_values.append(float(str(value)))
        except ValueError:
            pass
    inferred_type = "number" if populated and len(numeric_values) == len(populated) else "string"
    counts = Counter(text_values)
    # Values are hashed so inventory artifacts do not leak row-level medical data.
    labels = [
        {"value_hash": hashlib.sha256(value.encode()).hexdigest()[:12], "count": count}
        for value, count in counts.most_common(12)
    ]
    result: dict[str, object] = {
        "name": column,
        "inferred_type": inferred_type,
        "missing_rate": round(1 - len(populated) / max(1, len(rows)), 4),
        "unique_count": len(counts),
        "label_distribution": labels,
        "examples": [hashlib.sha256(value.encode()).hexdigest()[:12] for value in text_values[:3]],
    }
    if numeric_values:
        result["numeric_range"] = {"min": min(numeric_values), "max": max(numeric_values)}
    return result


def discover_dataset(root: str | Path) -> dict[str, object]:
    """Read supported tables and asset names without retaining raw row values in the inventory."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Dataset root does not exist: {root_path}")
    tables: list[dict[str, object]] = []
    images: list[str] = []
    raw_tables: list[dict[str, object]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(str(path))
        if path.suffix.lower() not in TABULAR_SUFFIXES:
            continue
        source_hash = _hash_file(path)
        source_tables = [_read_csv(path)] if path.suffix.lower() == ".csv" else _read_xlsx(path)
        for sheet, columns, rows in source_tables:
            table_id = f"table_{len(raw_tables)}"
            raw_tables.append(
                {
                    "table_id": table_id,
                    "file": str(path),
                    "sheet": sheet,
                    "columns": columns,
                    "rows": rows,
                }
            )
            tables.append(
                {
                    "table_id": table_id,
                    "file": str(path),
                    "sheet": sheet,
                    "source_hash": source_hash,
                    "row_count": len(rows),
                    "column_count": len(columns),
                    "columns": [_column_stats(rows, column) for column in columns],
                }
            )
    if not raw_tables:
        raise ValueError("No CSV or XLSX files were found under the dataset root.")
    return {
        "root": str(root_path),
        "tables": tables,
        "image_count": len(images),
        "image_paths": images,
        "_raw_tables": raw_tables,
    }


def _candidate_columns(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in tokens)]


def infer_schema(inventory: dict[str, object]) -> dict[str, object]:
    """Produce conservative mappings; unresolved primary/image keys require user confirmation."""
    raw_tables = inventory["_raw_tables"]
    assert isinstance(raw_tables, list)
    candidates: list[dict[str, object]] = []
    for table in raw_tables:
        columns = table["columns"]
        assert isinstance(columns, list)
        id_columns = [
            column
            for column in _candidate_columns(columns, ("case", "patient", "study", "image", "_id", "id"))
            if not any(token in column.lower() for token in ("path", "file", "filename"))
        ]
        image_columns = _candidate_columns(columns, ("image", "path", "file", "filename"))
        candidates.append(
            {
                "table_id": table["table_id"],
                "entity": "image" if image_columns else "case",
                "candidate_primary_keys": id_columns,
                "candidate_image_columns": image_columns,
                "columns": columns,
            }
        )
    canonical = max(candidates, key=lambda item: len(item["columns"]))
    key_options = canonical["candidate_primary_keys"]
    image_options = canonical["candidate_image_columns"]
    unknowns = []
    if len(key_options) != 1:
        unknowns.append("Select the stable case or image identifier column.")
    if len(image_options) > 1:
        unknowns.append("Select the column that resolves to image files.")
    return {
        "entities": candidates,
        "canonical_table": canonical["table_id"],
        "case_key": key_options[0] if len(key_options) == 1 else None,
        "image_column": image_options[0] if len(image_options) == 1 else None,
        "joins": [],
        "unknowns": unknowns,
    }


def propose_schema(inventory: dict[str, object], heuristic: dict[str, object]) -> dict[str, object]:
    """Ask DeepSeek for medical semantics while retaining only validated column mappings."""
    compact = {key: value for key, value in inventory.items() if key not in {"_raw_tables", "image_paths"}}
    proposal = complete_json(
        "Infer dataset entities, row granularity, safe joins, medical column semantics, and image-ID "
        "resolution rules from the supplied statistics. Return JSON with entities, joins, column_semantics, "
        "case_key, image_column, confidence, evidence, and conflicts. Never invent a column name.",
        f"Dataset inventory: {json.dumps(compact)}\nHeuristic candidates: {json.dumps(heuristic)}",
    )
    canonical = next(entity for entity in heuristic["entities"] if entity["table_id"] == heuristic["canonical_table"])
    columns = set(canonical["columns"])
    for field in ("case_key", "image_column"):
        suggested = proposal.get(field)
        if suggested in columns:
            heuristic[field] = suggested
    unresolved = []
    if not heuristic.get("case_key"):
        unresolved.append("Select the stable case or image identifier column.")
    if len(canonical["candidate_image_columns"]) > 1 and not heuristic.get("image_column"):
        unresolved.append("Select the column that resolves to image files.")
    heuristic["unknowns"] = unresolved
    heuristic["model_proposal"] = {
        key: proposal.get(key)
        for key in ("entities", "joins", "column_semantics", "confidence", "evidence", "conflicts")
    }
    return heuristic


def _build_image_index(image_paths: list[str], root: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    root_path = Path(root)
    for raw_path in image_paths:
        path = Path(raw_path)
        keys = {str(path), path.name, path.stem}
        try:
            keys.add(str(path.relative_to(root_path)))
        except ValueError:
            pass
        for key in keys:
            index.setdefault(key.replace("\\", "/"), []).append(str(path))
    return index


def build_access_layer(inventory: dict[str, object], schema_map: dict[str, object], run_root: str | Path) -> dict[str, object]:
    """Create a run-scoped DuckDB database with row-level source provenance."""
    raw_tables = inventory["_raw_tables"]
    assert isinstance(raw_tables, list)
    run_dir = Path(run_root) / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "dataset.duckdb"
    connection = duckdb.connect(str(db_path))
    table_map: dict[str, str] = {}
    try:
        for source in raw_tables:
            source_id = str(source["table_id"])
            sql_name = f"source_{source_id}"
            table_map[source_id] = sql_name
            columns = source["columns"]
            rows = source["rows"]
            assert isinstance(columns, list) and isinstance(rows, list)
            definitions = ", ".join(f'"{column}" VARCHAR' for column in columns)
            connection.execute(
                f'CREATE TABLE "{sql_name}" ({definitions}, _source_file VARCHAR, '
                "_source_sheet VARCHAR, _source_row BIGINT)"
            )
            placeholders = ", ".join("?" for _ in range(len(columns) + 3))
            records = [
                [*(None if row.get(column) is None else str(row.get(column)) for column in columns), source["file"], source["sheet"], index]
                for index, row in enumerate(rows, start=2)
            ]
            if records:
                connection.executemany(f'INSERT INTO "{sql_name}" VALUES ({placeholders})', records)
        canonical = str(schema_map["canonical_table"])
        connection.execute(f'CREATE VIEW cases AS SELECT * FROM "{table_map[canonical]}"')
    finally:
        connection.close()
    compact_inventory = {key: value for key, value in inventory.items() if key not in {"_raw_tables", "image_paths"}}
    inventory_path = run_dir / "dataset_inventory.json"
    schema_path = run_dir / "dataset_schema_map.json"
    image_index_path = run_dir / "image_index.json"
    inventory_path.write_text(json.dumps(compact_inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(schema_map, ensure_ascii=False, indent=2), encoding="utf-8")
    image_index_path.write_text(
        json.dumps(_build_image_index(inventory["image_paths"], str(inventory["root"])), ensure_ascii=False), encoding="utf-8"
    )
    return {
        "duckdb_path": str(db_path),
        "dataset_inventory_path": str(inventory_path),
        "dataset_schema_map_path": str(schema_path),
        "image_index_path": str(image_index_path),
        "views": {"cases": "cases"},
        "source_tables": table_map,
        "provenance_fields": ["_source_file", "_source_sheet", "_source_row"],
        "schema_map_hash": hashlib.sha256(json.dumps(schema_map, sort_keys=True).encode()).hexdigest(),
    }


TASK_TOKENS: dict[str, tuple[str, ...]] = {
    "modality_identification": ("modality",),
    "view_identification": ("view", "projection", "plane"),
    "laterality_identification": ("laterality", "side"),
    "image_quality_assessment": ("quality", "artifact"),
    "anatomy_identification": ("anatomy", "organ", "tissue"),
    "anatomy_attribute": ("density", "composition", "parenchymal", "anatom"),
    "finding_presence": ("finding", "abnormal", "normal"),
    "finding_type": ("finding_type", "lesion", "mass", "calcification"),
    "finding_attribute": ("shape", "margin", "distribution", "texture", "echogenicity"),
    "finding_description": ("description", "finding", "report"),
    "assessment_category": ("birads", "pi_rads", "assessment", "category"),
    "severity_grading": ("severity", "grade", "stage"),
    "diagnosis": ("diagnosis", "diagnostic"),
    "pathology_status": ("pathology", "malignan", "benign"),
    "report_generation": ("report",),
    "detection": ("bbox", "box", "bounding"),
}


def plan_task_capabilities(
    blueprint: dict[str, object], manifest: dict[str, object], schema_map: dict[str, object]
) -> list[dict[str, object]]:
    """Make executable candidate views using inspected column statistics, never model guesses alone."""
    database = str(manifest["duckdb_path"])
    connection = duckdb.connect(database)
    image_index = json.loads(Path(str(manifest["image_index_path"])).read_text(encoding="utf-8"))
    image_column = schema_map.get("image_column")
    image_resolution_rate: float | None = None
    if image_column:
        values = [item[0] for item in connection.execute(f'SELECT "{image_column}" FROM cases').fetchall()]
        populated = [str(value).replace("\\", "/") for value in values if value not in (None, "")]
        if populated:
            image_resolution_rate = round(sum(value in image_index for value in populated) / len(populated), 4)
    plans: list[dict[str, object]] = []
    try:
        columns = [item[0] for item in connection.execute("DESCRIBE cases").fetchall()]
        tasks = blueprint["tasks"]
        assert isinstance(tasks, dict)
        for task_id, task in tasks.items():
            candidates = _candidate_columns(columns, TASK_TOKENS.get(task_id, (task_id,)))
            source = candidates[0] if candidates else None
            if not source:
                plans.append(
                    {
                        "task_id": task_id,
                        "decision": "drop",
                        "decision_reason": "No matching answer column was found.",
                        "answer_sources": [],
                        "required_joins": [],
                        "candidate_view": None,
                        "eligible_case_count": 0,
                        "label_distribution": [],
                        "image_resolution_rate": image_resolution_rate,
                        "recommended_count": 0,
                        "evidence": {"candidate_columns": []},
                        "risks": ["Required task field is unavailable."],
                    }
                )
                continue
            quoted = f'"{source}"'
            count = connection.execute(f"SELECT COUNT(*) FROM cases WHERE {quoted} IS NOT NULL AND {quoted} != ''").fetchone()[0]
            distinct = connection.execute(f"SELECT COUNT(DISTINCT {quoted}) FROM cases WHERE {quoted} IS NOT NULL AND {quoted} != ''").fetchone()[0]
            labels = connection.execute(
                f"SELECT {quoted}, COUNT(*) FROM cases WHERE {quoted} IS NOT NULL AND {quoted} != '' GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
            ).fetchall()
            response_format = task.get("response_format") if isinstance(task, dict) else None
            decision = "retain" if count and (response_format == "free_text" or distinct >= 2) else "drop"
            view_name = f"task_candidates_{_safe_name(task_id)}"
            if decision == "retain":
                connection.execute(f"CREATE OR REPLACE VIEW \"{view_name}\" AS SELECT * FROM cases WHERE {quoted} IS NOT NULL AND {quoted} != ''")
            plans.append(
                {
                    "task_id": task_id,
                    "decision": decision,
                    "decision_reason": "Usable non-null labels found." if decision == "retain" else "Labels are absent or single-class.",
                    "answer_sources": [source],
                    "required_joins": [],
                    "candidate_view": view_name if decision == "retain" else None,
                    "eligible_case_count": count,
                    "label_distribution": [{"value": value, "count": size} for value, size in labels],
                    "image_resolution_rate": image_resolution_rate,
                    "recommended_count": min(count, 100),
                    "evidence": {"candidate_columns": candidates, "distinct_label_count": distinct},
                    "risks": [] if decision == "retain" else ["Task cannot provide a valid answer set."],
                }
            )
    finally:
        connection.close()
    return plans
