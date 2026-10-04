"""Frozen Markdown planning rules loaded from the experiment Registry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


RULE_STAGES = frozenset({
    "axis_analysis",
    "scenario_screening",
    "leaf_task_design",
    "scenario_review",
})
RULE_SCOPES = frozenset({"common", "disease", "modality", "example"})
RULE_SCOPE_ORDER = {"common": 0, "disease": 1, "modality": 2, "example": 3}
RULE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class PlanningRule:
    """One non-executable Markdown rule and its validated metadata."""

    id: str
    description: str
    stages: tuple[str, ...]
    scope: str
    diseases: tuple[str, ...]
    modalities: tuple[str, ...]
    body: str
    path: str
    digest: str

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "stages": list(self.stages),
            "scope": self.scope,
            "diseases": list(self.diseases),
            "modalities": list(self.modalities),
        }


class PlanningRuleCatalog:
    """Immutable, deterministic view of the rules in one BIR Registry."""

    def __init__(self, rules: Iterable[PlanningRule] = ()):
        ordered = sorted(rules, key=lambda item: (RULE_SCOPE_ORDER[item.scope], item.id, item.path))
        ids = [item.id for item in ordered]
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError(f"Planning rule IDs must be unique: {duplicates}")
        self._rules = tuple(ordered)
        self._by_id = {item.id: item for item in self._rules}

    def __iter__(self):
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self._rules)

    def get(self, rule_id: str) -> PlanningRule:
        try:
            return self._by_id[rule_id]
        except KeyError as exc:
            raise ValueError(f"Unknown planning rule: {rule_id}") from exc

    def common(self) -> tuple[PlanningRule, ...]:
        return tuple(item for item in self._rules if item.scope == "common")

    def selected(self, selected_ids: Iterable[str], *, stage: str | None = None) -> tuple[PlanningRule, ...]:
        if stage is not None and stage not in RULE_STAGES:
            raise ValueError(f"Unknown planning rule stage: {stage}")
        ids = list(dict.fromkeys(selected_ids))
        rules = [self.get(rule_id) for rule_id in ids]
        # Common rules are always active; accepting them in a persisted
        # selection is harmless but they must not be duplicated in output.
        rules.extend(item for item in self.common() if item.id not in {rule.id for rule in rules})
        if stage is not None:
            rules = [item for item in rules if stage in item.stages]
        return tuple(sorted(rules, key=lambda item: (RULE_SCOPE_ORDER[item.scope], item.id, item.path)))

    def render(self, selected_ids: Iterable[str], *, stage: str) -> str:
        rules = self.selected(selected_ids, stage=stage)
        if not rules:
            return "No additional planning rules are registered for this stage."
        chunks = []
        for rule in rules:
            chunks.append(f"### Rule: {rule.id}\n\n{rule.body.strip()}")
        return "\n\n".join(chunks)

    def selection_metadata(self) -> list[dict[str, Any]]:
        return [item.metadata() for item in self._rules]


def _front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Planning rule requires YAML front matter: {path}")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid planning rule front matter: {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Planning rule front matter must be a mapping: {path}")
    return metadata, match.group(2).strip()


def _strings(metadata: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = metadata.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Planning rule {path} field {key!r} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def load_rule_catalog(root: str | Path) -> PlanningRuleCatalog:
    root = Path(root)
    if not root.exists():
        return PlanningRuleCatalog()
    if not root.is_dir():
        raise ValueError(f"Planning rule root is not a directory: {root}")
    rules: list[PlanningRule] = []
    for path in sorted(root.rglob("*.md"), key=lambda item: item.relative_to(root).as_posix()):
        metadata, body = _front_matter(path)
        rule_id = metadata.get("id")
        description = metadata.get("description")
        stages = _strings(metadata, "stages", path)
        scope = metadata.get("scope")
        if not isinstance(rule_id, str) or not RULE_ID.fullmatch(rule_id):
            raise ValueError(f"Planning rule {path} has an invalid id")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Planning rule {path} requires a description")
        if not stages or not set(stages) <= RULE_STAGES:
            raise ValueError(f"Planning rule {path} has invalid stages: {stages}")
        if not isinstance(scope, str) or scope not in RULE_SCOPES:
            raise ValueError(f"Planning rule {path} has invalid scope: {scope!r}")
        if not body:
            raise ValueError(f"Planning rule {path} has an empty body")
        relative = path.relative_to(root).as_posix()
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        rules.append(PlanningRule(
            id=rule_id,
            description=description.strip(),
            stages=stages,
            scope=scope,
            diseases=tuple(item.casefold() for item in _strings(metadata, "diseases", path)),
            modalities=tuple(item.casefold() for item in _strings(metadata, "modalities", path)),
            body=body,
            path=relative,
            digest=digest,
        ))
    return PlanningRuleCatalog(rules)


def rule_selection_digest(catalog: PlanningRuleCatalog, selected_ids: Iterable[str]) -> str:
    """Digest the selected rule bodies for persisted planning provenance."""
    payload = [
        {"id": rule.id, "digest": rule.digest}
        for rule in catalog.selected(selected_ids)
    ]
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
