"""Version-bound, bounded reads of dataset facts and shared protocol eligibility."""

import json
from collections import Counter

import duckdb


def iter_records(access, batch_size=512):
    with duckdb.connect(access["duckdb_path"], read_only=True) as db:
        cursor = db.execute("SELECT record, _source_row FROM images ORDER BY _source_row")
        while rows := cursor.fetchmany(batch_size):
            for payload, source_row in rows:
                yield {**json.loads(payload), "_source_row": source_row}


def _page(rows, total, offset, limit, revision):
    return {
        "dataset_version": revision,
        "total": total,
        "values": rows,
        "complete": offset + len(rows) >= total,
        "next_offset": offset + len(rows) if offset + len(rows) < total else None,
    }


class DatasetQuery:
    def __init__(self, access):
        self.access = access

    def _connect(self):
        return duckdb.connect(self.access["duckdb_path"], read_only=True)

    def records(self, image_ids):
        """Fetch only a bounded batch of referenced canonical records."""
        if len(image_ids) > 512 or any(not isinstance(key, str) for key in image_ids):
            raise ValueError("Record lookup requires at most 512 string image IDs")
        if not image_ids:
            return {}
        with self._connect() as db:
            rows = db.execute("SELECT image_id,record FROM images WHERE image_id IN (SELECT unnest(?))", [image_ids]).fetchall()
        return {key: json.loads(payload) for key, payload in rows}

    def values(self, field, offset=0, limit=20, source_id=None):
        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("Invalid query page")
        with self._connect() as db:
            if not db.execute("SELECT count(*) FROM field_values WHERE path=?", [field]).fetchone()[
                0
            ]:
                raise ValueError(f"Unknown field: {field}")
            where = "path=? AND nonempty"
            args = [field]
            if source_id is not None:
                where += " AND source_id=?"
                args.append(source_id)
            total = db.execute(
                f"SELECT count(DISTINCT value) FROM field_values WHERE {where}", args
            ).fetchone()[0]
            rows = db.execute(
                f"SELECT value,count(*) FROM field_values WHERE {where} "
                "GROUP BY value ORDER BY value LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
            return _page(
                [{"value": json.loads(v), "count": n} for v, n in rows],
                total,
                offset,
                limit,
                self.access["revision"],
            )

    def by_source(self, field):
        with self._connect() as db:
            if not db.execute("SELECT count(*) FROM field_values WHERE path=?", [field]).fetchone()[
                0
            ]:
                raise ValueError(f"Unknown field: {field}")
            rows = db.execute(
                "SELECT i.source_id,count(DISTINCT i.image_id),"
                "count(DISTINCT f.image_id) FILTER (WHERE f.nonempty),"
                "count(DISTINCT f.value) FILTER (WHERE f.nonempty) "
                "FROM images i LEFT JOIN field_values f ON i.image_id=f.image_id AND f.path=? "
                "GROUP BY i.source_id ORDER BY i.source_id",
                [field],
            ).fetchall()
            return {
                "dataset_version": self.access["revision"],
                "field": field,
                "sources": [
                    {
                        "source_id": s,
                        "images": n,
                        "non_empty_images": c,
                        "coverage_rate": c / n,
                        "distinct_value_count": d,
                    }
                    for s, n, c, d in rows
                ],
            }

    def sample(self, offset=0, limit=5, source_id=None):
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError("Invalid sample page")
        with self._connect() as db:
            where = " WHERE source_id=?" if source_id else ""
            args = [source_id] if source_id else []
            count = db.execute("SELECT count(*) FROM images" + where, args).fetchone()[0]
            rows = db.execute(
                "SELECT record FROM images" + where + " ORDER BY _source_row LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
            # Do not expose raw source identifiers through planner samples.
            records = [
                {k: v for k, v in json.loads(row[0]).items() if k != "provenance"} for row in rows
            ]
            return _page(records, count, offset, limit, self.access["revision"])

    def pair(self, left, right):
        with self._connect() as db:
            for field in (left, right):
                if not db.execute(
                    "SELECT count(*) FROM field_values WHERE path=?", [field]
                ).fetchone()[0]:
                    raise ValueError(f"Unknown field: {field}")
            # Entity identity includes the concrete full ancestor-array index tuple.
            same_parent = left.rsplit(".", 1)[0] == right.rsplit(".", 1)[0]
            sql = (
                "SELECT count(DISTINCT a.image_id),count(*) FILTER (WHERE a.entity_indices=b.entity_indices) "
                "FROM (SELECT DISTINCT image_id,entity_indices FROM field_values WHERE path=? AND nonempty) a "
                "JOIN (SELECT DISTINCT image_id,entity_indices FROM field_values WHERE path=? AND nonempty) b "
                "ON a.image_id=b.image_id"
            )
            images, pairs = db.execute(sql, [left, right]).fetchone()
            matched, ambiguous = db.execute(
                "SELECT count(*),count(*) FILTER (WHERE n>1) FROM (SELECT a.image_id,count(*) n "
                "FROM (SELECT DISTINCT image_id,entity_indices FROM field_values WHERE path=? AND nonempty) a "
                "JOIN (SELECT DISTINCT image_id,entity_indices FROM field_values WHERE path=? AND nonempty) b "
                "ON a.image_id=b.image_id AND a.entity_indices=b.entity_indices GROUP BY a.image_id)",
                [left, right],
            ).fetchone()
            return {
                "dataset_version": self.access["revision"],
                "same_image_count": images,
                "same_entity_pairs": pairs if same_parent else 0,
                "paired_images": matched if same_parent else 0,
                "multiple_pair_images": ambiguous if same_parent else 0,
                "compatible_parent": same_parent,
                "note": "Co-occurrence is not protocol eligibility; query protocol capacity.",
            }

    def array(self, path):
        with self._connect() as db:
            rows = db.execute(
                "SELECT length,count(*) FROM arrays WHERE path=? GROUP BY length ORDER BY length",
                [path],
            ).fetchall()
            if not rows:
                raise ValueError(f"Unknown array: {path}")
            return {
                "dataset_version": self.access["revision"],
                "path": path,
                "distribution": [{"length": n, "count": c} for n, c in rows],
                "zero": sum(c for n, c in rows if n == 0),
                "one": sum(c for n, c in rows if n == 1),
                "many": sum(c for n, c in rows if n > 1),
                "denominator": "array occurrences",
            }

    def capacity(self, protocol):
        from medbenchagent.planning.protocol import Protocol, resolve_eligibility

        protocol = Protocol.model_validate(protocol).model_dump()
        reasons = Counter()
        sources = Counter()
        for row in iter_records(self.access):
            result = resolve_eligibility(row, protocol)
            reasons[result["reason"]] += 1
            if result["eligible"]:
                sources[row.get("provenance", {}).get("source_id", "unknown")] += 1
        return {
            "dataset_version": self.access["revision"],
            "eligible": reasons["eligible"],
            "reasons": dict(reasons),
            "per_source": dict(sources),
        }
