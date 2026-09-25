"""Fixed source readers for independently resolving extraction provenance."""

import json

from medbenchagent.data.contracts import SourceLocator
from medbenchagent.data.validation import source_image_path


def pointer(value, address):
    if address == "":
        return value
    if not address.startswith("/"):
        raise ValueError("JSON Pointer must start with /")
    try:
        for token in address[1:].split("/"):
            import re
            if re.search(r"~(?![01])", token):
                raise ValueError("Invalid JSON Pointer escape")
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not token.isdigit() or str(int(token)) != token:
                    raise ValueError("Invalid JSON Pointer array index")
                value = value[int(token)]
            else:
                value = value[token]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Source object does not exist: {address}") from exc
    return value


def json_objects(root, path, collection=""):
    file = source_image_path(root, path)
    if file.suffix.lower() == ".jsonl":
        with file.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle, 1):
                value = pointer(json.loads(line), collection)
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        yield SourceLocator(kind="jsonl", path=path, row=row, pointer=collection + "/" + str(index)), item
                else:
                    yield SourceLocator(kind="jsonl", path=path, row=row, pointer=collection), value
    elif file.suffix.lower() == ".json":
        if file.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("JSON exceeds 64 MiB; export the object collection to JSONL")
        value = pointer(json.loads(file.read_text(encoding="utf-8")), collection)
        if not isinstance(value, (list, dict)):
            raise ValueError("JSON population collection must be an array or object")
        entries = enumerate(value) if isinstance(value, list) else value.items()
        for key, item in entries:
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield SourceLocator(kind="json", path=path, pointer=collection + "/" + escaped), item
    else:
        raise ValueError("JSON inspection requires .json or .jsonl")


class LocatorReader:
    """Cache parsed source rows on disk, never retain the full source in agent state."""

    def __init__(self, root, db):
        self.root, self.db = root, db
        db.execute("CREATE TABLE locator_readers (reader VARCHAR PRIMARY KEY)")
        db.execute("CREATE TABLE locator_objects (reader VARCHAR, row BIGINT, payload VARCHAR, PRIMARY KEY(reader,row))")

    def resolve(self, locator):
        loc = SourceLocator.model_validate(locator) if isinstance(locator, dict) else locator
        file = source_image_path(self.root, loc.path)
        key = json.dumps([loc.kind, loc.path, loc.sheet, loc.delimiter])
        if not self.db.execute("SELECT 1 FROM locator_readers WHERE reader=?", [key]).fetchone():
            self._load(loc, file, key)
            self.db.execute("INSERT INTO locator_readers VALUES (?)", [key])
        found = self.db.execute("SELECT payload FROM locator_objects WHERE reader=? AND row=?",
                                [key, loc.row or 0]).fetchone()
        if found is None:
            raise ValueError(f"Source row does not exist: {loc.path}:{loc.row}")
        value = json.loads(found[0])
        if loc.kind == "mask":
            if loc.instance is not None and loc.instance not in value["labels"]:
                raise ValueError(f"Mask instance does not exist: {loc.instance}")
            return value
        return pointer(value, loc.pointer)

    def _load(self, loc, file, key):
        if loc.kind in {"json", "jsonl"} and file.suffix.lower() != "." + loc.kind:
            raise ValueError("JSON locator kind must match the source format")
        def insert(number, value):
            self.db.execute("INSERT INTO locator_objects VALUES (?,?,?)",
                            [key, number, json.dumps(value, ensure_ascii=False, default=str)])
        if loc.kind == "table":
            from medbenchagent.data.operators import read_relation, columns
            read_relation(self.db, self.root, loc.path, "locator_input", sheet=loc.sheet, delimiter=loc.delimiter)
            names = [c for c in columns(self.db, "locator_input") if c != "_lineage"]
            from medbenchagent.data.operators import quoted
            fields = ",".join(f"{quoted(name)} := {quoted(name)}" for name in names)
            self.db.execute("INSERT INTO locator_objects SELECT ?, "
                            "CAST(json_extract_string(_lineage[1], '$.row') AS BIGINT), "
                            f"to_json(struct_pack({fields})) FROM locator_input", [key])
            self.db.execute("DROP TABLE locator_input")
        elif loc.kind == "jsonl":
            with file.open(encoding="utf-8") as handle:
                batch = []
                for number, line in enumerate(handle, 1):
                    batch.extend((key, number, json.dumps(json.loads(line), ensure_ascii=False)))
                    if len(batch) == 512 * 3:
                        self.db.execute("INSERT INTO locator_objects VALUES " + ",".join(["(?,?,?)"] * 512), batch)
                        batch = []
                if batch:
                    self.db.execute("INSERT INTO locator_objects VALUES " + ",".join(["(?,?,?)"] * (len(batch) // 3)), batch)
        elif loc.kind == "json":
            if file.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("JSON exceeds 64 MiB; export to JSONL")
            insert(0, json.loads(file.read_text(encoding="utf-8")))
        else:
            import numpy as np
            from medbenchagent.data.image_conversion import _pixels
            pixels = np.asarray(_pixels(file)[0])
            if pixels.ndim != 2:
                raise ValueError("Mask locator requires a two-dimensional label image")
            insert(0, {"labels": np.unique(pixels).tolist(), "height": pixels.shape[0], "width": pixels.shape[1]})


def finding_objects(value, address=""):
    """Concrete object locations preserve annotation array entity boundaries."""
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from finding_objects(item, address + "/" + escaped)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            target = address + "/" + str(index)
            if isinstance(item, dict):
                yield target, item
            yield from finding_objects(item, target)


def extract_findings(record):
    findings = []
    for target, item in finding_objects(record):
        refs = item.pop("__medbench_locators", None)
        if refs:
            findings.append({"target": target, "locators": [json.loads(ref) for ref in refs]})
    return findings
