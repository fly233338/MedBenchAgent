"""Read the canonical published package without construction runtime or Prepared state."""
import json
from pathlib import Path
import hashlib

PACKAGE_FILES = ("benchmark.jsonl", "benchmark_manifest.json", "audit_report.json", "quarantine.jsonl")

def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_package(root):
    root = Path(root).resolve()
    marker = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or marker.get("schema_version") != "benchmark-delivery":
        raise ValueError("Unsupported delivery completion marker")
    files = marker.get("files", {})
    if not isinstance(files, dict) or not set(PACKAGE_FILES) <= set(files) or "compilation_report.md" not in files:
        raise ValueError("Delivery completion marker is incomplete")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Delivery file changed or is missing: {name}")
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    if marker.get("benchmark_id") != manifest["benchmark_id"]:
        raise ValueError("Delivery identity mismatch")
    return manifest



def load_benchmark_package(directory):
    root = Path(directory).resolve()
    manifest = verify_package(root)
    if manifest.get("schema_version") != "benchmark-manifest":
        raise ValueError("Unsupported benchmark manifest")
    items = []
    ids = set()
    for line in (root / "benchmark.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["item_id"] in ids:
            raise ValueError("Duplicate benchmark item")
        ids.add(item["item_id"])
        if f"{item['dataset_id']}::{item['leaf_task_id']}" not in manifest["tasks"]:
            raise ValueError("Benchmark task is missing from manifest")
        if not item.get("input_images"):
            raise ValueError("Benchmark item has no model image inputs")
        for asset in item["input_images"]:
            path = Path(asset["path"])
            if manifest.get("package_mode") == "portable" and (path.is_absolute() or not (root / path).resolve().is_relative_to(root)):
                raise ValueError("Portable image reference escapes package")
            path = path if path.is_absolute() else root / path
            if not path.is_file() or digest_file(path) != asset["file_digest"]:
                raise ValueError("Benchmark image changed or is missing")
            asset["path"] = str(path.resolve())
        items.append(item)
    if len(items) != manifest["item_count"]:
        raise ValueError("Benchmark item count differs from manifest")
    return {"manifest": manifest, "items": items}
