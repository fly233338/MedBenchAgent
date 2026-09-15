"""Read-only source identities and explicit annotation provenance."""

import hashlib
import json
import os
import sqlite3
import filecmp
import shutil
from pathlib import Path

from medbenchagent.data.contracts import Source
from medbenchagent.data.validation import source_image_path
from medbenchagent.runtime.run_artifacts import atomic_write_json


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare_outputs(first, second):
    def files(folder):
        return sorted(
            p.relative_to(folder)
            for p in folder.rglob("*")
            if p.is_file()
            and not {"scratch", "cache"}.intersection(p.relative_to(folder).parts)
            and p.name not in {"upstream_provenance.jsonl", "preparation.json", "fallback.py"}
        )

    names = files(first)
    if names != files(second) or any(
        not filecmp.cmp(first / p, second / p, shallow=False) for p in names
    ):
        raise ValueError("Preparation is not deterministic")


def remove_work_directory(path, parent):
    target, root = Path(path).resolve(), Path(parent).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Temporary directory must be beneath its operation directory")
    shutil.rmtree(target)


def namespace_id(source, entity, value):
    if value is None:
        return None
    return json.dumps(
        [source.id, source.version, entity, str(value)], ensure_ascii=False, separators=(",", ":")
    )


def inventory_source(source: Source, destination):
    root = Path(source.root).resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    listing = destination / "files.jsonl"
    if all(
        (destination / name).is_file() for name in ("files.jsonl", "source.json", "used.sqlite")
    ):
        saved = Source.model_validate_json(
            (destination / "source.json").read_text(encoding="utf-8")
        )
        if saved != source:
            raise ValueError("Source identity changed; use a new input version")
        verify_source(destination)
        return destination
    with listing.open("w", encoding="utf-8") as output:
        for directory, directories, files in os.walk(root):
            directories.sort()
            for name in sorted(files):
                path = Path(directory) / name
                resolved = source_image_path(root, path.relative_to(root))
                stat = resolved.stat()
                output.write(
                    json.dumps(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    )
                    + "\n"
                )
    atomic_write_json(destination / "source.json", source.model_dump())
    with sqlite3.connect(destination / "used.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS used (path TEXT PRIMARY KEY, digest TEXT NOT NULL)")
    return destination


def track_files(manifest, paths):
    manifest = Path(manifest)
    source = Source.model_validate_json((manifest / "source.json").read_text(encoding="utf-8"))
    with sqlite3.connect(manifest / "used.sqlite") as db:
        for relative in paths:
            key = Path(relative).as_posix()
            digest = digest_file(source_image_path(source.root, relative))
            saved = db.execute("SELECT digest FROM used WHERE path=?", [key]).fetchone()
            if saved and saved[0] != digest:
                raise ValueError(f"Source changed: {key}; rebuild dependent artifacts")
            db.execute("INSERT OR IGNORE INTO used VALUES (?,?)", [key, digest])


def verify_source(manifest):
    manifest = Path(manifest)
    source = Source.model_validate_json((manifest / "source.json").read_text(encoding="utf-8"))
    root = Path(source.root)
    count = 0
    with (manifest / "files.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            path = source_image_path(root, item["path"])
            stat = path.stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"Source changed: {item['path']}; start a new input version")
            count += 1
    if sum(p.is_file() for p in root.rglob("*")) != count:
        raise ValueError("Source file population changed")
    with sqlite3.connect(manifest / "used.sqlite") as db:
        for relative, digest in db.execute("SELECT path,digest FROM used ORDER BY path"):
            if digest_file(source_image_path(root, relative)) != digest:
                raise ValueError(f"Source content changed: {relative}")


def attach_provenance(record, source, locator, plan_ref):
    original = {
        entity: record[entity][entity + "_id"] for entity in ("patient", "case", "study", "image")
    }
    for entity, value in original.items():
        if isinstance(value, str) and value.startswith("["):
            try:
                components = json.loads(value)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(components, list)
                and len(components) == 4
                and components[:3] == [source.id, source.version, entity]
            ):
                original[entity] = components[3]
    record["provenance"] = {
        "source_id": source.id,
        "source_version": source.version,
        "original_ids": original,
        "original_image_path": record["image"]["image_path"],
        "record_ref": locator,
        "plan_ref": plan_ref,
    }
    for entity, value in original.items():
        record[entity][entity + "_id"] = namespace_id(source, entity, value)
    return record
