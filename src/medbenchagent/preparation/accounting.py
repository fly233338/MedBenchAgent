"""Runtime-only converter counters, independently reconciled by the host."""
import csv
import json
from pathlib import Path
from medbenchagent.data.asset_adapters import contained_path


def records(mapping, source_roots):
    for source in mapping.record_sources:
        path = contained_path(source_roots[source.source_id], source.path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if source.format == "csv":
                yield from csv.DictReader(stream)
            elif source.format == "jsonl":
                yield from (json.loads(line) for line in stream if line.strip())
            else:
                value = json.load(stream)
                if not isinstance(value, list):
                    raise ValueError("Accounting JSON source must be an array of records")
                yield from value


class TextAccounting:
    """Call consume/exclude/skip once per seen record; emit counts output Cases."""
    def __init__(self, mapping, source_roots):
        self.mapping, self.source_roots = mapping, source_roots
        self.seen = self.consumed = self.emitted = 0
        self.excluded, self.skipped = {}, {}

    def __iter__(self):
        for row in records(self.mapping, self.source_roots):
            self.seen += 1
            yield row

    def consume(self):
        self.consumed += 1

    def exclude(self, reason):
        self.excluded[reason] = self.excluded.get(reason, 0) + 1

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def emit(self, stream, case):
        stream.write(case.model_dump_json() + "\n")
        self.emitted += 1

    def write(self, path):
        Path(path).write_text(json.dumps({"records_seen": self.seen, "records_consumed": self.consumed,
            "cases_emitted": self.emitted, "excluded": self.excluded, "skipped": self.skipped}), encoding="utf-8")


def validate_accounting(path, mapping, source_roots, case_count):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    total = sum(1 for _ in records(mapping, source_roots))
    counts = [report[key] for key in ("records_seen", "records_consumed", "cases_emitted")]
    for name in ("excluded", "skipped"):
        for reason, count in report[name].items():
            if not reason.strip() or not mapping.population_policy.exclusion:
                raise ValueError("Excluded/skipped records require an accepted exclusion rule and reason")
            counts.append(count)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Accounting counts must be nonnegative integers")
    if report["records_seen"] != total or report["cases_emitted"] != case_count:
        raise ValueError("Accounting differs from source population or emitted Cases")
    if report["records_consumed"] + sum(report["excluded"].values()) + sum(report["skipped"].values()) != total:
        raise ValueError("Unaccounted source records")
