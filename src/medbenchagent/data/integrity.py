"""Integrity checks for published cache and frozen asset indexes."""

from pathlib import Path
from medbenchagent.data.sources import digest_file


def verify_cache(access):
    path = Path(access["duckdb_path"])
    if not access.get("duckdb_digest") or not path.is_file() or digest_file(path) != access["duckdb_digest"]:
        raise ValueError("Prepared DuckDB cache changed or lacks integrity binding; prepare a new version")


def verify_prepared_indexes(access):
    verify_cache(access)
    if access.get("schema_version") == "annotation":
        from medbenchagent.data.bundle_catalog import verify_bundle
        verify_bundle(Path(access["cases_path"]).parent)
        return
    raise ValueError("Only annotation prepared bundles are supported; re-prepare this dataset")

