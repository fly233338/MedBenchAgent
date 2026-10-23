import pytest

from medbenchagent.data.sources import digest_path
from medbenchagent.data.models import AssetEntry, CaseRecord, Evidence, validate_case


def test_canonical_case_validates_refs_and_asset_links(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    asset = AssetEntry(asset_id="a1", source_id="s", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(image), width=10, height=10)
    case = CaseRecord(case_id="c1", patient_id=None, evidence=[
        Evidence(id="a", type="asset", name="view", value="a1"),
        Evidence(id="r", type="region", name="lesion", value=[0.1, 0.2, 0.5, 0.8], ref=["a"]),
    ])
    assert validate_case(case, {"a1": asset}).case_id == "c1"

    with pytest.raises(ValueError):
        validate_case(case.model_copy(update={"evidence": [case.evidence[0], case.evidence[0]]}), {"a1": asset})
