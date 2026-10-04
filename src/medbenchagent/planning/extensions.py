"""Lightweight, human-selected planning Prompt extensions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


EXTENSION_TYPES = frozenset({"disease", "dataset", "disease_domain"})
EXTENSION_STAGES = frozenset({
    "axis_analysis",
    "scenario_screening",
    "leaf_task_design",
    "scenario_review",
})
EXTENSION_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class PlanningExtension:
    id: str
    type: str
    diseases: tuple[str, ...]
    datasets: tuple[str, ...]
    domains: tuple[str, ...]
    stages: tuple[str, ...]
    body: str
    path: str
    digest: str

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "diseases": list(self.diseases),
            "datasets": list(self.datasets),
            "domains": list(self.domains),
            "stages": list(self.stages),
        }


class PlanningExtensionCatalog:
    """Immutable, deterministic catalog of extension Markdown files."""

    def __init__(self, extensions: Iterable[PlanningExtension] = ()):
        ordered = sorted(extensions, key=lambda item: (item.id, item.path))
        ids = [item.id for item in ordered]
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError(f"Planning extension IDs must be unique: {duplicates}")
        self._extensions = tuple(ordered)
        self._by_id = {item.id: item for item in self._extensions}

    def __iter__(self):
        return iter(self._extensions)

    def __len__(self) -> int:
        return len(self._extensions)

    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self._extensions)

    def get(self, extension_id: str) -> PlanningExtension:
        try:
            return self._by_id[extension_id]
        except KeyError as exc:
            raise ValueError(f"Unknown planning extension: {extension_id}") from exc

    def metadata(self) -> list[dict[str, Any]]:
        return [item.metadata() for item in self._extensions]

    def selected(self, selected_ids: Iterable[str], *, stage: str | None = None) -> tuple[PlanningExtension, ...]:
        if stage is not None and stage not in EXTENSION_STAGES:
            raise ValueError(f"Unknown planning extension stage: {stage}")
        ids = list(dict.fromkeys(selected_ids))
        selected = [self.get(extension_id) for extension_id in ids]
        if stage is not None:
            selected = [item for item in selected if stage in item.stages]
        return tuple(sorted(selected, key=lambda item: (item.id, item.path)))

    def render(self, selected_ids: Iterable[str], *, stage: str) -> str:
        extensions = self.selected(selected_ids, stage=stage)
        if not extensions:
            return "No dataset-specific planning extensions are selected for this stage."
        return "\n\n".join(item.body.strip() for item in extensions if item.body.strip())

    def selection_digest(self, selected_ids: Iterable[str]) -> str:
        payload = [{"id": item.id, "digest": item.digest} for item in self.selected(selected_ids)]
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Planning extension requires YAML front matter: {path}")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid planning extension front matter: {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Planning extension front matter must be a mapping: {path}")
    return metadata, match.group(2).strip()


def _strings(metadata: dict[str, Any], key: str, path: Path, *, required: bool = False) -> tuple[str, ...]:
    value = metadata.get(key, [])
    if value is None:
        value = []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Planning extension {path} field {key!r} must be a list of non-empty strings")
    if required and not value:
        raise ValueError(f"Planning extension {path} requires {key}")
    return tuple(item.strip() for item in value)


def load_extension_catalog(root: str | Path) -> PlanningExtensionCatalog:
    root = Path(root)
    if not root.exists():
        return PlanningExtensionCatalog()
    if not root.is_dir():
        raise ValueError(f"Planning extension root is not a directory: {root}")
    extensions: list[PlanningExtension] = []
    for path in sorted(root.rglob("*.md"), key=lambda item: item.relative_to(root).as_posix()):
        metadata, body = _front_matter(path)
        extension_id = metadata.get("id")
        extension_type = metadata.get("type")
        stages = _strings(metadata, "stages", path, required=True)
        if not isinstance(extension_id, str) or not EXTENSION_ID.fullmatch(extension_id):
            raise ValueError(f"Planning extension {path} has an invalid id")
        if not isinstance(extension_type, str) or extension_type not in EXTENSION_TYPES:
            raise ValueError(f"Planning extension {path} has an invalid type: {extension_type!r}")
        if not set(stages) <= EXTENSION_STAGES:
            raise ValueError(f"Planning extension {path} has invalid stages: {stages}")
        if not body:
            raise ValueError(f"Planning extension {path} has an empty body")
        extensions.append(PlanningExtension(
            id=extension_id,
            type=extension_type,
            diseases=tuple(item.casefold() for item in _strings(metadata, "diseases", path)),
            datasets=tuple(item.casefold() for item in _strings(metadata, "datasets", path)),
            domains=tuple(item.casefold() for item in _strings(metadata, "domains", path)),
            stages=stages,
            body=body,
            path=path.relative_to(root).as_posix(),
            digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        ))
    return PlanningExtensionCatalog(extensions)
