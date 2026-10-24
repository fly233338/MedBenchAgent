"""Explicit tiny source fixture for development, never implicit production input."""

import json
from pathlib import Path

from medbenchagent.data.task_data import write_once


def create_demo_source(directory):
    root = Path(directory).resolve()
    write_once(
        root / "sources" / "cases.jsonl",
        b'{"case_id":"example-1","observation":"Synthetic demonstration","label":"example"}\n',
    )
    value = {
        "dataset": {"dataset_id": "demo", "revision": "demo"},
        "sources": [{"source_id": "source", "root": "sources"}],
    }
    path = root / "sources.json"
    write_once(path, json.dumps(value, indent=2).encode())
    return path
