"""Publish an already validated package; business validation belongs to FinalizationAgent."""

from pathlib import Path

from medbenchagent.runtime.run_artifacts import atomic_write_json

PACKAGE_FILES = ("benchmark.jsonl", "benchmark_manifest.json", "audit_report.json", "quarantine.jsonl")


def publish_package(staging, destination, completion):
    """Make a complete package visible with one same-filesystem directory rename."""
    staging, destination = Path(staging), Path(destination)
    if destination.exists():
        raise ValueError("Delivery already exists; never overwrite a published package")
    atomic_write_json(staging / "complete.json", completion)
    staging.rename(destination)
    return destination
