"""Canonical, version-bound Case query API."""
from medbenchagent.data.v5_index import V5DatasetQuery


class DatasetQuery(V5DatasetQuery):
    def __init__(self, access):
        if access.get("schema_version") != "annotation.v5":
            raise ValueError("Only annotation.v5 is supported; prepare a new dataset")
        super().__init__(access)


def iter_records(access, batch_size=512):
    yield from DatasetQuery(access).iter_cases()
