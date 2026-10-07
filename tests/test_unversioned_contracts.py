import json

import pytest

from medbenchagent.data.bundle_catalog import verify_bundle
from medbenchagent.data.index import DatasetIndex


def test_old_contract_markers_are_rejected(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "ready.json").write_text(
        json.dumps({"schema_version": "dataset-bundle.v4", "annotation_schema": "annotation.v5"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        verify_bundle(bundle)

    with pytest.raises(ValueError, match="Only annotation is supported"):
        DatasetIndex({"schema_version": "annotation.v5"})

