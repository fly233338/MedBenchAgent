"""Canonical v5 dataset indexing."""

from pathlib import Path
from medbenchagent.data.v5_index import prepare_v5_dataset


def annotation_schema_path():
    return Path(__file__).parents[1] / "schemas" / "annotation.v5.schema.json"


def prepare_annotation_dataset(
    annotation_path, cache_root, *, revision="1", refresh=False, scratch_root=None
):
    annotation = Path(annotation_path)
    return prepare_v5_dataset(
        annotation,
        annotation.parent / "asset_manifest.jsonl",
        cache_root,
        revision=revision,
        refresh=refresh,
    )
