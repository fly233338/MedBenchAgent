"""Append-only diagnostics; checkpoints and artifacts remain authoritative."""
import json
import os
import re
from uuid import uuid4
from datetime import UTC, datetime
from pathlib import Path
from filelock import FileLock

def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sanitize(value, *, redactions=(), limit=4000, depth=0):
    secrets = [v for k, v in os.environ.items() if v and (k.endswith("API_KEY") or k.endswith("ACCESS_TOKEN"))]
    secrets.extend(v for v in redactions if v)
    if depth > 6:
        return "[内容省略]"
    if isinstance(value, dict):
        return {str(k): "[redacted]" if re.search(r"api.?key|authorization|password|secret|access_token|reasoning|thought|thinking", str(k), re.I)
                else sanitize(v, redactions=secrets, limit=limit, depth=depth + 1)
                for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, redactions=secrets, limit=limit, depth=depth + 1) for v in value[:40]]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[redacted]", text)
    return text if len(text) <= limit else text[:limit] + "\n…[已截断]"


def record_event(path: str | Path, event_type: str, *, redactions=(), **values: object) -> dict[str, object]:
    """Append one construction event as compact UTF-8 JSON."""
    event = {"event_id": uuid4().hex, "timestamp": _timestamp(), "event_type": event_type} | {
        key: value for key, value in values.items() if value is not None
    }
    event = sanitize(event, redactions=redactions)
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock", timeout=2), destination.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except (OSError, TimeoutError):
        # Diagnostics are not an execution ledger and cannot invalidate accepted work.
        pass
    return event
