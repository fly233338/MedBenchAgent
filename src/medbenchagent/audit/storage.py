"""Bounded batch reads and disposable disk indexes for finalization."""

from itertools import islice
from pathlib import Path

from medbenchagent.data.image_assets import ImageEntry, iter_image_manifest
from medbenchagent.data.operators import connect
from medbenchagent.data.query import DatasetQuery

BATCH_SIZE = 256


def batches(rows):
    iterator = iter(rows)
    while batch := list(islice(iterator, BATCH_SIZE)):
        yield batch


def query_rows(db, sql, parameters=()):
    cursor = db.cursor()
    try:
        cursor.execute(sql, parameters)
        while rows := cursor.fetchmany(BATCH_SIZE):
            yield from rows
    finally:
        cursor.close()


class ReferencedInputs:
    """Index a manifest once on disk, then load only the current sample batch."""

    def __init__(self, access, manifest_path, directory):
        self.query = DatasetQuery(access)
        self.manifest = Path(manifest_path)
        self.directory = Path(directory)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = connect(self.directory / "images.duckdb", self.directory / "scratch")
        try:
            self.db.execute("CREATE OR REPLACE TABLE images (id VARCHAR PRIMARY KEY, path VARCHAR, width INTEGER, height INTEGER)")
            for batch in batches(iter_image_manifest(self.manifest)):
                self.db.executemany("INSERT INTO images VALUES (?,?,?,?)", [
                    (e.image_id, e.relative_png_path, e.width, e.height) for e in batch])
        except BaseException:
            self.db.close()
            raise
        return self

    def read(self, image_ids):
        keys = list(dict.fromkeys(key for key in image_ids if isinstance(key, str)))
        records = self.query.records(keys)
        rows = self.db.execute("SELECT * FROM images WHERE id IN (SELECT unnest(?))", [keys]).fetchall()
        root = self.manifest.parent.resolve()
        images = {key: ImageEntry(key, (root / path).resolve(), path, width, height) for key, path, width, height in rows}
        return records, images

    def __exit__(self, *exc):
        self.db.close()
