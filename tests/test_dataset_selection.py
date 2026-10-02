import pytest

from medbenchagent.planning.dataset_selection import Selection, _validate_selection


def test_selection_requires_nonempty_unique_ids_and_reasons():
    _validate_selection(Selection(selected_dataset_ids=["ct"], selection_reason={"ct": "Explicitly selected"}), {"ct": {}})

    with pytest.raises(ValueError, match="unique prepared candidates"):
        _validate_selection(Selection(selected_dataset_ids=["ct", "ct"],
                                     selection_reason={"ct": "duplicate"}), {"ct": {}})
    with pytest.raises(ValueError, match="unique prepared candidates"):
        _validate_selection(Selection(selected_dataset_ids=["unknown"],
                                     selection_reason={"unknown": "not frozen"}), {"ct": {}})
