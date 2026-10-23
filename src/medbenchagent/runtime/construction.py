"""Append-only diagnostics; checkpoints and artifacts remain authoritative."""
import json
from datetime import UTC, datetime
from pathlib import Path

def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def record_event(path: str | Path, event_type: str, **values: object) -> dict[str, object]:
    """Append one construction event as compact UTF-8 JSON."""
    event = {"timestamp": _timestamp(), "event_type": event_type} | {
        key: value for key, value in values.items() if value is not None
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    return event
