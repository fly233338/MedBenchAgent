"""A bounded relational operator engine; no generated SQL is accepted from agents."""

import json
import os
from pathlib import Path

import duckdb

from medbenchagent.data.contracts import PreparationPlan, Source
from medbenchagent.data.sources import namespace_id
from medbenchagent.data.validation import source_image_path


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def connect(path=":memory:", scratch=None):
    memory = os.environ.get("MEDBENCH_DATA_MEMORY")
    threads = os.environ.get("MEDBENCH_DATA_THREADS")
    if memory is None or threads is None:
        from medbenchagent.settings import get_settings

        settings = get_settings().data_processing
        memory = memory or settings["memory_limit"]
        threads = threads or settings["threads"]
    db = duckdb.connect(str(path))
    db.execute("SET memory_limit=" + literal(memory))
    db.execute("SET threads=" + str(int(threads)))
    scratch = scratch or os.environ.get("MEDBENCH_DATA_SCRATCH")
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
        db.execute("SET temp_directory=" + literal(str(scratch)))
    return db


def read_relation(db, root, path, table, *, sheet=None, delimiter=","):
    resolved = source_image_path(root, path)
    suffix = resolved.suffix.lower()
    location = literal(resolved.as_posix())
    if suffix in {".xlsx", ".xls"}:
        if resolved.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Excel exceeds 64 MiB; export the selected sheet to CSV/Parquet")
        import pandas as pd

        frame = pd.read_excel(resolved, sheet_name=sheet or 0, dtype=str, keep_default_na=False)
        db.register("_excel", frame)
        expression = "SELECT * FROM _excel"
    elif suffix == ".parquet":
        expression = f"SELECT * FROM read_parquet({location})"
    elif suffix in {".json", ".jsonl"}:
        expression = f"SELECT * FROM read_json_auto({location}, sample_size=-1)"
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else delimiter
        expression = (
            f"SELECT * FROM read_csv({location}, header=true, all_varchar=true, "
            f"delim={literal(sep)}, parallel=false)"
        )
    else:
        raise ValueError(f"Unsupported table format: {suffix}")
    db.execute(f"CREATE TABLE {quoted(table)} AS {expression}")
    if {"_lineage", "rowid"} & set(columns(db, table)):
        raise ValueError("Source columns _lineage and rowid are reserved; rename before import")
    # Physical input row locator is retained through every relational step.
    db.execute(f"ALTER TABLE {quoted(table)} ADD COLUMN _lineage VARCHAR[]")
    locator = path + (f"#sheet={sheet or 0}" if suffix in {".xls", ".xlsx"} else "")
    db.execute(
        f"UPDATE {quoted(table)} SET _lineage=[? || ':' || CAST(rowid+1 AS VARCHAR)]", [locator]
    )


def columns(db, table):
    return [r[0] for r in db.execute(f"DESCRIBE {quoted(table)}").fetchall()]


def require_columns(db, table, names):
    unknown = set(names) - set(columns(db, table))
    if unknown:
        raise ValueError(f"Unknown columns in {table}: {sorted(unknown)}")


def set_path(record, path, value):
    parts = path.split(".")
    node = record
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Overlapping output fields: {path}")
    if parts[-1] in node and node[parts[-1]] is not None:
        raise ValueError(f"Output field already populated: {path}")
    node[parts[-1]] = value


def empty_record():
    return {
        **{e: {e + "_id": None, "annotation": {}} for e in ("patient", "case", "study")},
        "image": {
            "image_id": None,
            "image_path": None,
            "modality": None,
            "anatomy": None,
            "annotation": {},
        },
    }


def inspect_table(root, path, sheet=None):
    with connect() as db:
        read_relation(db, root, path, "input", sheet=sheet)
        count = db.execute('SELECT count(*) FROM "input"').fetchone()[0]
        result = {}
        for name, dtype, *_ in db.execute('DESCRIBE "input"').fetchall():
            if name == "_lineage":
                continue
            nulls, unique = db.execute(
                f"SELECT count(*) FILTER (WHERE {quoted(name)} IS NULL), "
                f'count(DISTINCT {quoted(name)}) FROM "input"'
            ).fetchone()
            result[name] = {"type": dtype, "nulls": nulls, "unique": unique}
        rows = db.execute('SELECT * EXCLUDE (_lineage) FROM "input" LIMIT 5').fetchall()
        return {
            "path": path,
            "sheet": sheet,
            "rows": count,
            "columns": result,
            "samples": [dict(zip(result, row, strict=True)) for row in rows],
            "statistics": "exact",
            "identifier_policy": "CSV columns remain strings",
        }


def inspect_values(root, path, field, sheet=None, offset=0, limit=50):
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("Invalid value page")
    with connect() as db:
        read_relation(db, root, path, "input", sheet=sheet)
        require_columns(db, "input", [field])
        column = quoted(field)
        total = db.execute(
            f'SELECT count(*) FROM (SELECT DISTINCT {column} FROM "input")'
        ).fetchone()[0]
        values = db.execute(
            f'SELECT {column},count(*) FROM "input" GROUP BY {column} '
            f"ORDER BY {column} NULLS FIRST LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        return {
            "field": field,
            "total": total,
            "values": [{"value": v, "count": n} for v, n in values],
            "complete": offset + len(values) >= total,
            "next_offset": offset + len(values) if offset + len(values) < total else None,
        }


def join_stats(db, left, right, left_keys, right_keys):
    require_columns(db, left, left_keys)
    require_columns(db, right, right_keys)
    condition = " AND ".join(
        f"a.{quoted(a)}=b.{quoted(b)}" for a, b in zip(left_keys, right_keys, strict=True)
    )
    stats = {}
    for label, table, keys, alias in (
        ("left", left, left_keys, "a"),
        ("right", right, right_keys, "b"),
    ):
        null = " OR ".join(f"{quoted(k)} IS NULL" for k in keys)
        group = ",".join(map(quoted, keys))
        stats[label + "_nulls"] = db.execute(
            f"SELECT count(*) FROM {quoted(table)} WHERE {null}"
        ).fetchone()[0]
        stats[label + "_duplicates"] = db.execute(
            f"SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM {quoted(table)} "
            f"WHERE NOT ({null}) GROUP BY {group} HAVING count(*)>1)"
        ).fetchone()[0]
        other = right if label == "left" else left
        other_alias = "b" if label == "left" else "a"
        stats[label + "_missing"] = db.execute(
            f"SELECT count(*) FROM {quoted(table)} {alias} WHERE NOT EXISTS "
            f"(SELECT 1 FROM {quoted(other)} {other_alias} WHERE {condition})"
        ).fetchone()[0]
        stats["matched_" + label] = (
            db.execute(f"SELECT count(*) FROM {quoted(table)}").fetchone()[0]
            - stats[label + "_missing"]
        )
    return stats


def inspect_image_population(root, source, population, output):
    """Count actual distinct nonempty image identities after declared filters and joins."""
    from medbenchagent.data.contracts import Population
    population = Population.model_validate(population)
    plan = PreparationPlan.model_validate({
        "mapping": {"authoritative_population": "Population inspection", "expected_images": 1,
                    "entity_meanings": {}, "evidence": ["Deterministic inspection"]},
        "steps": [*(step.model_dump() for step in population.steps),
                  {"op": "emit", "input": population.input, "fields": {"image.image_id": population.image_id}}],
    })
    return execute_plan(root, source, plan, output, population_only=True)


def execute_plan(root, source: Source, plan: PreparationPlan, output, *, population_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    with connect(scratch=output / "scratch") as db:
        for index, step in enumerate(plan.steps):
            if hasattr(step, "field") and step.field == "_lineage":
                raise ValueError("Operators cannot modify provenance")
            if hasattr(step, "target") and step.target == "_lineage":
                raise ValueError("Operator target _lineage is reserved")
            if step.op == "read":
                read_relation(
                    db, root, step.path, step.output, sheet=step.sheet, delimiter=step.delimiter
                )
                continue
            table = quoted(step.input)
            if step.op == "emit":
                require_columns(db, step.input, step.fields.values())
                if population_only:
                    field = quoted(step.fields["image.image_id"])
                    rows, unique, missing = db.execute(
                        f"SELECT count(*), count(DISTINCT {field}), "
                        f"count(*) FILTER (WHERE {field} IS NULL OR trim(CAST({field} AS VARCHAR))='') FROM {table}"
                    ).fetchone()
                    if missing or not unique:
                        raise ValueError("Population requires nonempty image identities and at least one image")
                    return {"rows": rows, "unique_images": unique, "image_id": step.fields["image.image_id"]}
                names = columns(db, step.input)
                json_columns = {
                    r[0] for r in db.execute(f"DESCRIBE {table}").fetchall() if r[1] == "JSON"
                }
                cursor = db.execute(f"SELECT * FROM {table} ORDER BY _lineage")
                count = 0
                with (
                    (output / "annotation.jsonl").open("w", encoding="utf-8") as handle,
                    (output / "provenance.jsonl").open("w", encoding="utf-8") as provenance,
                ):
                    while batch := cursor.fetchmany(512):
                        for values in batch:
                            row = dict(zip(names, values, strict=True))
                            for column in json_columns:
                                row[column] = (
                                    json.loads(row[column]) if row[column] is not None else None
                                )
                            record = empty_record()
                            for path, column in step.fields.items():
                                set_path(record, path, row[column])
                            for path, value in step.constants.items():
                                set_path(record, path, value)
                            count += 1
                            if record["image"]["image_id"] is None:
                                record["image"]["image_id"] = json.dumps(row["_lineage"])
                            ref = f"record:{count}"
                            provenance.write(
                                json.dumps({"id": ref, "locators": row["_lineage"], "step": index})
                                + "\n"
                            )
                            # Provenance is attached at normalization, after raw record validation.
                            handle.write(
                                json.dumps(
                                    {"record": record, "record_ref": ref},
                                    ensure_ascii=False,
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                reports.append({"op": "emit", "images": count})
                if count != plan.mapping.expected_images:
                    raise ValueError("Emitted image count differs from authoritative population")
                continue
            before = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            select = "*"
            tail = ""
            if step.op == "select":
                if "_lineage" in step.columns:
                    raise ValueError("Provenance is retained automatically")
                require_columns(db, step.input, step.columns)
                select = ",".join(map(quoted, [*step.columns, "_lineage"]))
            elif step.op == "rename":
                require_columns(db, step.input, step.columns)
                targets = [step.columns.get(c, c) for c in columns(db, step.input)]
                if len(set(targets)) != len(targets) or "_lineage" in step.columns:
                    raise ValueError("Rename produces duplicate or reserved columns")
                select = ",".join(
                    f"{quoted(c)} AS {quoted(step.columns.get(c, c))}"
                    for c in columns(db, step.input)
                )
            elif step.op == "filter":
                require_columns(db, step.input, [step.field])
                values = [literal(v) for v in step.values if v is not None]
                predicate = f"{quoted(step.field)} IN ({','.join(values)})" if values else "false"
                if None in step.values:
                    predicate += f" OR {quoted(step.field)} IS NULL"
                tail = " WHERE " + ("" if step.keep else "NOT ") + f"coalesce(({predicate}), false)"
            elif step.op == "cast":
                require_columns(db, step.input, [step.field])
                select = f"* REPLACE (CAST({quoted(step.field)} AS {step.dtype}) AS {quoted(step.field)})"
            elif step.op in {"value_mapping", "missing_normalization"}:
                require_columns(db, step.input, [step.field])
                if step.op == "missing_normalization" and any(
                    m.target is not None for m in step.mapping
                ):
                    raise ValueError("Missing normalization targets must be null")
                # JSON comparison preserves scalar type; no implicit numeric/string label equivalence.
                branches, tests = [], []
                seen = set()
                for m in step.mapping:
                    key = json.dumps(m.value, allow_nan=False)
                    if key in seen:
                        raise ValueError("Duplicate value mapping")
                    seen.add(key)
                    test = f"coalesce(to_json({quoted(step.field)}), 'null') = {literal(key)}::JSON"
                    tests.append(test)
                    target = literal(json.dumps(m.target, allow_nan=False)) + "::JSON"
                    branches.append(f"WHEN {test} THEN {target}")
                for test in tests:
                    if not db.execute(f"SELECT count(*) FROM {table} WHERE {test}").fetchone()[0]:
                        raise ValueError(f"Mapping references an absent value in {step.field}")
                if (
                    step.unmapped == "error"
                    and db.execute(
                        f"SELECT count(*) FROM {table} WHERE NOT ({' OR '.join(tests)})"
                    ).fetchone()[0]
                ):
                    raise ValueError(f"Unmapped values in {step.field}")
                select = (
                    f"* REPLACE (CASE {' '.join(branches)} ELSE to_json({quoted(step.field)}) "
                    f"END AS {quoted(step.field)})"
                )
            elif step.op == "join":
                stats = join_stats(db, step.input, step.right, step.left_keys, step.right_keys)
                if (
                    step.cardinality in {"one_to_one", "one_to_many"}
                    and stats["left_duplicates"]
                    or step.cardinality in {"one_to_one", "many_to_one"}
                    and stats["right_duplicates"]
                ):
                    raise ValueError("Join cardinality violated")
                if step.null_keys == "error" and (stats["left_nulls"] or stats["right_nulls"]):
                    raise ValueError("Null join keys")
                if step.unmatched == "error" and (stats["left_missing"] or stats["right_missing"]):
                    raise ValueError("Unmatched join rows")
                left_cols = columns(db, step.input)
                right_cols = [
                    c
                    for c in columns(db, step.right)
                    if c not in step.right_keys and c != "_lineage"
                ]
                if set(left_cols) & set(right_cols):
                    raise ValueError("Rename overlapping columns before joining")
                condition = " AND ".join(
                    f"a.{quoted(a)}=b.{quoted(b)}"
                    for a, b in zip(step.left_keys, step.right_keys, strict=True)
                )
                select = "a.* EXCLUDE (_lineage)" + "".join(f",b.{quoted(c)}" for c in right_cols)
                select += (
                    ",list_sort(list_distinct(list_concat(a._lineage,b._lineage))) AS _lineage"
                )
                table += f" a {step.how.upper()} JOIN {quoted(step.right)} b ON {condition}"
                reports.append({"op": "join", **stats})
            elif step.op == "group_findings":
                require_columns(db, step.input, [*step.keys, *step.fields.values()])
                fields = ",".join(f"{quoted(k)} := {quoted(v)}" for k, v in step.fields.items())
                keys = ",".join(map(quoted, step.keys))
                select = (
                    f"{keys}, list(struct_pack({fields}) ORDER BY _lineage) AS {quoted(step.target)}, "
                    "list_sort(list_distinct(flatten(list(_lineage)))) AS _lineage"
                )
                tail = f" GROUP BY {keys}"
            elif step.op in {"bbox", "mask_to_bbox", "namespace_ids"}:
                _row_transform(db, root, source, step)
                reports.append({"op": step.op, "input": before, "output": before})
                continue
            else:
                raise ValueError(f"Unsupported operation: {step.op}")
            db.execute(f"CREATE TABLE {quoted(step.output)} AS SELECT {select} FROM {table}{tail}")
            after = db.execute(f"SELECT count(*) FROM {quoted(step.output)}").fetchone()[0]
            reports.append(
                {
                    "op": step.op,
                    "input": before,
                    "output": after,
                    **({"reason": step.reason} if step.op == "filter" else {}),
                }
            )
    (output / "operations.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def _row_transform(db, root, source, step):
    import numpy as np

    target = step.field if step.op == "namespace_ids" else step.target
    required = [step.field] if step.op != "bbox" else step.fields
    if step.op == "mask_to_bbox" or step.op == "bbox" and step.convention == "normalized_xyxy":
        required = [*required, step.width_field, step.height_field]
    require_columns(db, step.input, required)
    cols = columns(db, step.input)
    if step.op != "namespace_ids" and target in cols:
        raise ValueError(f"Output column exists: {target}")
    db.execute(f"CREATE TABLE {quoted(step.output)} AS SELECT * FROM {quoted(step.input)}")
    if step.op != "namespace_ids":
        db.execute(f"ALTER TABLE {quoted(step.output)} ADD COLUMN {quoted(target)} DOUBLE[]")
        if step.op == "mask_to_bbox" and step.instances == "by_value":
            db.execute(
                f"ALTER TABLE {quoted(step.output)} ALTER COLUMN {quoted(target)} TYPE DOUBLE[][]"
            )
    cursor = db.cursor()
    cursor.execute(
        f"SELECT rowid,{','.join(map(quoted, cols))} FROM {quoted(step.output)} ORDER BY rowid"
    )
    try:
        while batch := cursor.fetchmany(512):
            updates = []
            for values in batch:
                row = dict(zip(cols, values[1:], strict=True))
                if step.op == "namespace_ids":
                    # Emit receives raw IDs; namespace is idempotent at the normalization boundary.
                    result = namespace_id(source, step.entity, row[step.field])
                elif step.op == "bbox":
                    result = [float(row[f]) for f in step.fields]
                    a, b, c, d = result
                    if step.convention == "xywh":
                        result = [a, b, a + c, b + d]
                    elif step.convention == "cxcywh":
                        result = [a - c / 2, b - d / 2, a + c / 2, b + d / 2]
                    elif step.convention == "normalized_xyxy":
                        w, h = float(row[step.width_field]), float(row[step.height_field])
                        result = [a * w, b * h, c * w, d * h]
                else:
                    from medbenchagent.data.image_conversion import _pixels

                    pixels = np.asarray(_pixels(source_image_path(root, row[step.field]))[0])
                    if pixels.ndim != 2 or pixels.shape != (
                        int(row[step.height_field]),
                        int(row[step.width_field]),
                    ):
                        raise ValueError("Mask geometry differs from image; resizing is forbidden")
                    groups = (
                        [step.foreground]
                        if step.instances == "single"
                        else [[v] for v in step.foreground]
                    )
                    boxes = []
                    for group in groups:
                        y, x = np.where(np.isin(pixels, group))
                        if len(x):
                            boxes.append(
                                [
                                    float(x.min()),
                                    float(y.min()),
                                    float(x.max() + 1),
                                    float(y.max() + 1),
                                ]
                            )
                    result = (boxes[0] if boxes else None) if step.instances == "single" else boxes
                updates.append((result, values[0]))
            db.executemany(
                f"UPDATE {quoted(step.output)} SET {quoted(target)}=? WHERE rowid=?", updates
            )
    finally:
        cursor.close()
