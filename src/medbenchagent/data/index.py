"""Disk-backed profiling, preserving array witnesses and bounded Python buffers."""

import json
from pathlib import Path
from uuid import uuid4

from medbenchagent.data.operators import connect
from medbenchagent.data.sources import digest_file
from medbenchagent.data.validation import (
    AnnotationValidationError,
    RECORD_ENTITIES,
    iter_annotation_jsonl,
)
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


def create_tables(db):
    db.execute(
        "CREATE TABLE images (image_id VARCHAR PRIMARY KEY, image_path VARCHAR UNIQUE, "
        "patient_id VARCHAR, case_id VARCHAR, study_id VARCHAR, record JSON, _source_row BIGINT, source_id VARCHAR)"
    )
    db.execute(
        "CREATE TABLE field_values (image_id VARCHAR, source_id VARCHAR, path VARCHAR, "
        "entity_indices VARCHAR, type VARCHAR, value VARCHAR, nonempty BOOLEAN)"
    )
    db.execute(
        "CREATE TABLE arrays (image_id VARCHAR, path VARCHAR, indices VARCHAR, length BIGINT)"
    )


def ingest(db, records):
    from medbenchagent.data.dataset import _is_non_empty, _value_type

    structures = {}
    image_rows, facts, arrays = [], [], []

    def visit(value, path, image_id, source, indices=()):
        if isinstance(value, dict):
            item = structures.setdefault(path, {"path": path, "kind": "object", "children": set()})
            for key, child in value.items():
                if "." in key or "[*]" in key:
                    raise AnnotationValidationError(
                        "Annotation keys cannot contain path syntax; rename explicitly"
                    )
                child_path = path + "." + key
                item["children"].add(child_path)
                visit(child, child_path, image_id, source, indices)
        elif isinstance(value, list):
            item = structures.setdefault(
                path, {"path": path, "kind": "array", "item_path": path + "[*]", "children": set()}
            )
            arrays.append((image_id, path, json.dumps(indices), len(value)))
            for index, child in enumerate(value):
                if isinstance(child, dict):
                    for key, nested in child.items():
                        if "." in key or "[*]" in key:
                            raise AnnotationValidationError(
                                "Annotation keys cannot contain path syntax"
                            )
                        child_path = path + "[*]." + key
                        item["children"].add(child_path)
                        visit(nested, child_path, image_id, source, (*indices, index))
                else:
                    visit(child, path + "[*]", image_id, source, (*indices, index))
        else:
            entity_indices = indices[:-1] if path.endswith("[*]") else indices
            facts.append(
                (
                    image_id,
                    source,
                    path,
                    json.dumps(entity_indices),
                    _value_type(value),
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    _is_non_empty(value),
                )
            )

    def flush():
        if image_rows:
            db.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?,?)", image_rows)
            image_rows.clear()
        if facts:
            db.executemany("INSERT INTO field_values VALUES (?,?,?,?,?,?,?)", facts)
            facts.clear()
        if arrays:
            db.executemany("INSERT INTO arrays VALUES (?,?,?,?)", arrays)
            arrays.clear()

    for index, row in enumerate(records, 1):
        image = row["image"]
        source = row.get("provenance", {}).get("source_id", "unknown")
        payload = {k: v for k, v in row.items() if k != "_source_row"}
        image_rows.append(
            (
                image["image_id"],
                image["image_path"],
                row["patient"]["patient_id"],
                row["case"]["case_id"],
                row["study"]["study_id"],
                json.dumps(payload, ensure_ascii=False),
                row.get("_source_row", index),
                source,
            )
        )
        for entity in RECORD_ENTITIES:
            visit(row[entity]["annotation"], entity + ".annotation", image["image_id"], source)
        for field in ("modality", "anatomy"):
            visit(image[field], "image." + field, image["image_id"], source)
        if len(image_rows) >= 256 or len(facts) >= 8192:
            flush()
    flush()
    return [
        {**{k: v for k, v in item.items() if k != "children"}, "children": sorted(item["children"])}
        for _, item in sorted(structures.items())
    ]


def profile_database(db, structures):
    from medbenchagent.data.dataset import _profile_top_value, _should_include_top_values

    settings = get_settings().dataset_profile
    total, patients, cases, studies = db.execute(
        "SELECT count(*),count(DISTINCT patient_id),count(DISTINCT case_id),count(DISTINCT study_id) FROM images"
    ).fetchone()
    fields = {}
    for path, types, covered, distinct in db.execute(
        "SELECT path,list(DISTINCT type ORDER BY type),count(DISTINCT image_id) FILTER (WHERE nonempty),"
        "count(DISTINCT value) FILTER (WHERE nonempty) FROM field_values GROUP BY path ORDER BY path"
    ).fetchall():
        if (
            path.endswith(".bbox")
            and types == ["null"]
            and any(s["path"] == path and s["kind"] == "array" for s in structures)
        ):
            continue
        sources = db.execute(
            "SELECT i.source_id,count(DISTINCT i.image_id),count(DISTINCT f.image_id) FILTER (WHERE f.nonempty) "
            "FROM images i LEFT JOIN field_values f ON i.image_id=f.image_id AND f.path=? GROUP BY i.source_id ORDER BY i.source_id",
            [path],
        ).fetchall()
        field = {
            "path": path,
            "types": types,
            "non_empty_images": covered,
            "distinct_value_count": distinct,
            "entity_scope": path.split(".")[0],
            "coverage_rate": covered / total if total else 0,
            "denominator": "images",
            "value_kind": types[0] if len(types) == 1 else "mixed",
            "structure_type": "array_element" if "[*]" in path else "scalar",
            "per_source": {
                s: {"images": n, "non_empty_images": c, "coverage_rate": c / n}
                for s, n, c in sources
            },
        }
        if path.endswith(".bbox[*]"):
            field["value_kind"] = "bbox_coordinate"
        if _should_include_top_values(path):
            top = db.execute(
                "SELECT value,count(*) n FROM field_values WHERE path=? AND nonempty "
                "GROUP BY value ORDER BY n DESC,value LIMIT ?",
                [path, settings.max_top_values],
            ).fetchall()
            field["top_values"] = [
                _profile_top_value(v, settings.max_top_value_chars) for v, _ in top
            ]
            if top:
                count = db.execute(
                    "SELECT count(*) FROM field_values WHERE path=? AND nonempty", [path]
                ).fetchone()[0]
                least, least_count = db.execute(
                    "SELECT value,count(*) n FROM field_values WHERE path=? AND nonempty "
                    "GROUP BY value ORDER BY n,value LIMIT 1",
                    [path],
                ).fetchone()
                field["value_distribution"] = {
                    "total_count": count,
                    "top": [
                        {
                            "value": _profile_top_value(v, settings.max_top_value_chars),
                            "count": n,
                            "proportion": n / count,
                        }
                        for v, n in top
                    ],
                    "least_frequent": {
                        "value": _profile_top_value(least, settings.max_top_value_chars),
                        "count": least_count,
                        "proportion": least_count / count,
                    },
                    "truncated": distinct > len(top),
                }
        fields[path] = field
    sources = dict(
        db.execute(
            "SELECT source_id,count(*) FROM images GROUP BY source_id ORDER BY source_id"
        ).fetchall()
    )
    summary = {
        "images": total,
        "patients": patients,
        "cases": cases,
        "studies": studies,
        "sources": sources,
        "entity_count_policy": "distinct known IDs; null excluded",
    }
    for field in ("modality", "anatomy"):
        summary[field] = [
            {"value": json.loads(v), "images": n}
            for v, n in db.execute(
                "SELECT value,count(DISTINCT image_id) FROM field_values WHERE path=? AND nonempty GROUP BY value ORDER BY value",
                ["image." + field],
            ).fetchall()
        ]
    array_index = {
        path: {
            "occurrences": n,
            "images": images,
            "minimum": low,
            "maximum": high,
            "zero": zero,
            "one": one,
            "many": many,
            "denominator": "array occurrences",
        }
        for path, n, images, low, high, zero, one, many in db.execute(
            "SELECT path,count(*),count(DISTINCT image_id),min(length),max(length),"
            "count(*) FILTER (WHERE length=0),count(*) FILTER (WHERE length=1),"
            "count(*) FILTER (WHERE length>1) FROM arrays GROUP BY path ORDER BY path"
        ).fetchall()
    }
    return {
        "schema_version": "task-capability-profile.v4",
        "total_images": total,
        "summary": summary,
        "arrays": array_index,
        "structures": structures,
        "fields": fields,
    }


def build_profile(records):
    with connect() as db:
        create_tables(db)
        structures = ingest(db, records)
        return profile_database(db, structures)


def prepare_dataset(annotation_path, cache_root, *, revision="1", refresh=False):
    source = Path(annotation_path).resolve()
    if source.name != "annotation.jsonl" or not source.is_file():
        raise AnnotationValidationError("Input must be an existing annotation.jsonl")
    if not revision or not revision.isalnum():
        raise ValueError("Cache revision must be alphanumeric")
    settings = get_settings().dataset_profile
    folder = (
        Path(cache_root)
        / "annotation-cache.v9"
        / f"top-{settings.max_top_values}-chars-{settings.max_top_value_chars}"
        / revision
    )
    if refresh:
        folder = folder / str(uuid4())
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "dataset_access_manifest.json"
    profile_path = folder / "dataset_profile.json"
    database_path = folder / "dataset.duckdb"
    digest = digest_file(source)
    if manifest_path.exists():
        access = json.loads(manifest_path.read_text(encoding="utf-8"))
        if access["annotation_path"] != str(source) or access["annotation_digest"] != digest:
            raise ValueError("Dataset changed; create a new version/cache")
        from medbenchagent.data.integrity import verify_cache
        verify_cache(access)
        return json.loads(profile_path.read_text(encoding="utf-8")), access
    temporary = folder / "building.duckdb"
    temporary.unlink(missing_ok=True)
    with connect(temporary, folder / "scratch") as db:
        create_tables(db)
        structures = ingest(db, iter_annotation_jsonl(source))
        profile = profile_database(db, structures)
    quality = source.parent / "quality_report.json"
    if quality.is_file():
        profile["summary"]["quality"] = json.loads(quality.read_text(encoding="utf-8"))
    if digest_file(source) != digest:
        raise ValueError("Annotation changed during indexing")
    temporary.replace(database_path)
    atomic_write_json(profile_path, profile)
    access = {
        "annotation_path": str(source),
        "annotation_digest": digest,
        "revision": revision,
        "cache_format_version": "annotation-cache.v9",
        "duckdb_path": str(database_path),
        "duckdb_digest": digest_file(database_path),
        "table": "images",
        "profile_path": str(profile_path),
    }
    atomic_write_json(manifest_path, access)
    return profile, access
