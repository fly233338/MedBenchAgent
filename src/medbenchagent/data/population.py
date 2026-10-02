"""Compare the inspected image population with either execution backend's intermediate."""

import json
from pathlib import Path


def connect(*, scratch=None):
    """Return an isolated in-memory DuckDB connection for population checks."""
    import duckdb
    return duckdb.connect(":memory:")


def _pointer(value, path):
    current = value
    for part in path.lstrip("/").split("/"):
        current = current[part.replace("~1", "/").replace("~0", "~")]
    return current


def _json_objects(root, path, collection):
    payload = json.loads(Path(root, path).read_text(encoding="utf-8"))
    current = _pointer(payload, collection) if collection else payload
    if not isinstance(current, list):
        raise ValueError("JSON population collection must be an array")
    for index, item in enumerate(current):
        yield {"kind": "json", "path": path, "row": index + 1}, item


def identity(root, identifier, path):
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Population image identity must be an explicit nonempty string")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Population image path must be explicit")
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if Path(path).is_absolute() or not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Population image path must stay beneath the source root")
    return identifier, resolved.relative_to(root).as_posix()


def verify_population(root, expected_path, annotation, source_id):
    with connect() as db:
        db.execute("CREATE TABLE expected (id VARCHAR PRIMARY KEY, path VARCHAR UNIQUE)")
        db.execute("CREATE TABLE emitted (id VARCHAR PRIMARY KEY, path VARCHAR UNIQUE)")
        with Path(expected_path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or not {"source_id", "image_id", "image_path"} <= row.keys():
                    raise ValueError("Population rows require source_id, image_id and image_path")
                if row["source_id"] != source_id:
                    raise ValueError("Inspected population belongs to another source")
                key, path = identity(root, row["image_id"], row["image_path"])
                if db.execute("SELECT 1 FROM expected WHERE id=? OR path=?", [key, path]).fetchone():
                    raise ValueError("Duplicate expected population identity/path")
                db.execute("INSERT INTO expected VALUES (?,?)", [key, path])
        with Path(annotation).open(encoding="utf-8") as handle:
            for line in handle:
                image = json.loads(line)["record"]["image"]
                key, path = identity(root, image.get("image_id"), image.get("image_path"))
                if db.execute("SELECT count(*) FROM emitted WHERE id=? OR path=?", [key, path]).fetchone()[0]:
                    raise ValueError(f"Duplicate emitted population identity/path: {key}")
                db.execute("INSERT INTO emitted VALUES (?,?)", [key, path])
        missing = db.execute("SELECT * FROM (SELECT * FROM expected EXCEPT SELECT * FROM emitted) ORDER BY id LIMIT 10").fetchall()
        extra = db.execute("SELECT * FROM (SELECT * FROM emitted EXCEPT SELECT * FROM expected) ORDER BY id LIMIT 10").fetchall()
        if missing or extra:
            raise ValueError("Emitted population differs from inspection: " + json.dumps({"missing": missing, "extra_or_changed": extra}))


def inspect_json_population(root, source_id, population, output):
    from medbenchagent.data.contracts import JsonPopulation
    spec = JsonPopulation.model_validate(population)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count, included = 0, 0
    with connect(scratch=output / "scratch") as db, (output / "population.jsonl").open("w", encoding="utf-8") as handle:
        db.execute("CREATE TABLE identities (id VARCHAR PRIMARY KEY, path VARCHAR UNIQUE)")
        for locator, item in _json_objects(root, spec.path, spec.collection):
            count += 1
            if not isinstance(item, dict):
                raise ValueError("Population collection entries must be objects")
            def field(name):
                return _pointer(item, name) if name.startswith("/") else item[name]
            try:
                if spec.field:
                    raw = field(spec.field)
                    match = any(type(raw) is type(v) and raw == v for v in spec.values)
                    if match != spec.keep:
                        continue
                key, path = identity(root, field(spec.image_id), field(spec.image_path))
            except KeyError as exc:
                raise ValueError(f"Population field is absent: {exc}") from exc
            prior = db.execute("SELECT path FROM identities WHERE id=?", [key]).fetchone()
            if prior and prior[0] == path:
                continue
            if prior or db.execute("SELECT 1 FROM identities WHERE path=?", [path]).fetchone():
                raise ValueError("Population identities and paths must be one-to-one")
            db.execute("INSERT INTO identities VALUES (?,?)", [key, path])
            handle.write(json.dumps({"source_id": source_id, "image_id": key, "image_path": path,
                                     "locator": locator}, ensure_ascii=False) + "\n")
            included += 1
    if not included:
        raise ValueError("Population has no images")
    return {"rows": count, "unique_images": included, "coverage": "full", "population_file": "population.jsonl"}
