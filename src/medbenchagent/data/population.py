"""Compare the inspected image population with either execution backend's intermediate."""

import json
from pathlib import Path
from medbenchagent.data.operators import connect


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
                if row["source_id"] != source_id:
                    raise ValueError("Inspected population belongs to another source")
                db.execute("INSERT INTO expected VALUES (?,?)", identity(root, row["image_id"], row["image_path"]))
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
