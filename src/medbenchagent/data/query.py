"""Canonical, version-bound Case query API."""
from medbenchagent.data.index import DatasetIndex


class DatasetQuery(DatasetIndex):
    def __init__(self, access):
        if access.get("schema_version") != "annotation":
            raise ValueError("Only annotation is supported; prepare a new dataset")
        super().__init__(access)


def iter_records(access, batch_size=512):
    yield from DatasetQuery(access).iter_cases()

