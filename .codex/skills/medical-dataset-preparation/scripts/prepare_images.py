"""Convert source images, build the exact mapping, and validate the result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Preparation command returned invalid JSON: {command[0]}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Preparation command returned an invalid result: {command[0]}")
    return result


def prepare(source_root: Path, artifact_dir: Path, processed_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    artifact_dir = artifact_dir.resolve()
    processed_root = processed_root.resolve()
    if processed_root == artifact_dir:
        raise ValueError("Processed-image directory must be a child of the artifact directory.")
    try:
        processed_root.relative_to(artifact_dir)
    except ValueError as exc:
        raise ValueError(
            "Processed-image directory must be inside the artifact directory."
        ) from exc

    annotation = artifact_dir / "annotation.jsonl"
    if not annotation.is_file():
        raise ValueError(f"annotation.jsonl does not exist: {annotation}")
    script_dir = Path(__file__).resolve().parent
    converter = script_dir / "convert_images.py"
    builder = script_dir / "build_image_manifest.py"
    validator = script_dir / "validate_prepared_dataset.py"
    for script in (converter, builder, validator):
        if not script.is_file():
            raise ValueError(f"Required preparation script does not exist: {script}")

    conversion_manifest = processed_root / "image_conversion_manifest.jsonl"
    image_manifest = artifact_dir / "image_manifest.jsonl"
    _run(
        [
            "uv",
            "run",
            "--locked",
            "--script",
            str(converter),
            "--input",
            str(source_root),
            "--output",
            str(processed_root),
        ]
    )
    _run(
        [
            sys.executable,
            str(builder),
            "--annotation",
            str(annotation),
            "--conversion-manifest",
            str(conversion_manifest),
            "--png-root",
            str(processed_root),
            "--output",
            str(image_manifest),
        ]
    )
    validation = _run(
        [
            sys.executable,
            str(validator),
            "--annotation",
            str(annotation),
            "--image-manifest",
            str(image_manifest),
        ]
    )
    scripts = (Path(__file__).resolve(), converter, builder, validator)
    report = {
        "schema_version": "image-preparation-report.v1",
        "records": validation["records"],
        "modes": validation["modes"],
        "annotation_sha256": validation["annotation_sha256"],
        "image_manifest_sha256": validation["image_manifest_sha256"],
        "scripts": {script.name: _sha256(script) for script in scripts},
    }
    report_path = artifact_dir / "image_preparation_report.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--processed-image-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source_image_root, args.artifact_dir, args.processed_image_dir),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
