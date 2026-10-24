"""Append-only diagnostics; checkpoints and artifacts remain authoritative."""
import json
import os
import re
from uuid import uuid4
from datetime import UTC, datetime
from pathlib import Path
from filelock import FileLock


def payload_summary(value, *, key="", depth=0):
    """Keep structural receipts; arbitrary tool text may be patient data or a prompt."""
    if depth > 4:
        return "[omitted]"
    if isinstance(value, dict):
        return {str(k): payload_summary(v, key=str(k), depth=depth + 1) for k, v in list(value.items())[:24]
                if not re.search(r"reasoning|thought|thinking|password|secret|token|api.?key|authorization", str(k), re.I)}
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "items": [payload_summary(v, key=key, depth=depth + 1) for v in value[:4]]}
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if key in {"command", "cmd"}:
        return {"program": sanitize(text.split(maxsplit=1)[0] if text.strip() else "", limit=80), "characters": len(text)}
    if re.fullmatch(r".*(?:_id|_ids|_digest)|kind|status|outcome|operation|revision|name|tool_name|backend", key):
        return sanitize(text, limit=160)
    return {"characters": len(text)}


def result_summary(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    return payload_summary(value)

def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sanitize(value, *, redactions=(), limit=4000, depth=0):
    secrets = [v for k, v in os.environ.items() if v and (k.endswith("API_KEY") or k.endswith("ACCESS_TOKEN"))]
    secrets.extend(v for v in redactions if v)
    if depth > 6:
        return "[omitted]"
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
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def record_event(path: str | Path, event_type: str, *, redactions=(), **values: object) -> dict[str, object]:
    """Append one construction event as compact UTF-8 JSON."""
    event = {"event_id": uuid4().hex, "timestamp": _timestamp(), "event_type": event_type} | {
        key: value for key, value in values.items() if value is not None
    }
    try:
        event = sanitize(event, redactions=redactions)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock", timeout=2), destination.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        # Diagnostics are not an execution ledger and cannot invalidate accepted work.
        pass
    return event
