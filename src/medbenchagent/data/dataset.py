"""Canonical dataset indexing."""

from pathlib import Path
from medbenchagent.data.index import prepare_dataset


def annotation_schema_path():
    return Path(__file__).parents[1] / "schemas" / "annotation.schema.json"


def prepare_annotation_dataset(
    cases_path, cache_root, *, revision="1", refresh=False, scratch_root=None
):
    cases = Path(cases_path)
    return prepare_dataset(
        cases,
        cases.parent / "asset_manifest.jsonl",
        cache_root,
        revision=revision,
        refresh=refresh,
    )

