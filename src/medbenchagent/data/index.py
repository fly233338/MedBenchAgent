"""DuckDB index for annotation Case/Evidence/Asset bundles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from medbenchagent.data.asset_manifest import iter_asset_manifest
from medbenchagent.data.sources import digest_file
from medbenchagent.data.models import CaseRecord, validate_case


def _manifest_access(root: Path) -> dict[str, str]:
    """Return optional prepared manifest paths and their content digests."""
    values: dict[str, str] = {}
    for name in ("presentation", "split", "sampling"):
        path = root / f"{name}_manifest.jsonl"
        if path.is_file():
            values[f"{name}_manifest_path"] = str(path)
            values[f"{name}_manifest_digest"] = digest_file(path)
    return values


def prepare_dataset(cases_path, manifest_path, cache_root, *, revision="1", refresh=False,
                       source_root=None, source_roots=None):
    cases_file = Path(cases_path).resolve()
    manifest = Path(manifest_path).resolve()
    cache = Path(cache_root) / "annotation-cache" / revision
    if refresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / "dataset.duckdb"
    profile_path = cache / "dataset_profile.json"
    access_path = cache / "dataset_access_manifest.json"
    manifest_access = _manifest_access(cases_file.parent)
    if access_path.is_file():
        access = json.loads(access_path.read_text(encoding="utf-8"))
        if (
            access.get("annotation_digest") == digest_file(cases_file)
            and access.get("asset_manifest_digest") == digest_file(manifest)
            and all(access.get(key) == value for key, value in manifest_access.items())
            and all(key in manifest_access for key in access if key.endswith("_manifest_path") or key.endswith("_manifest_digest"))
        ):
            return json.loads(profile_path.read_text(encoding="utf-8")), access
    assets = {entry.asset_id: entry for entry in iter_asset_manifest(
        manifest, source_root=source_root, source_roots=source_roots
    )}
    cases = []
    with cases_file.open(encoding="utf-8") as source:
        for line in source:
            case = CaseRecord.model_validate_json(line)
            cases.append(validate_case(case, assets))
    temporary = cache / "building.duckdb"
    temporary.unlink(missing_ok=True)
    with duckdb.connect(str(temporary)) as db:
        db.execute("CREATE TABLE cases (case_id VARCHAR PRIMARY KEY, patient_id VARCHAR, record JSON)")
        db.execute("CREATE TABLE assets (asset_id VARCHAR PRIMARY KEY, kind VARCHAR, format VARCHAR, source_path VARCHAR, file_digest VARCHAR, metadata JSON)")
        db.execute("CREATE TABLE evidence (case_id VARCHAR, evidence_id VARCHAR, type VARCHAR, name VARCHAR, value JSON, time DOUBLE, unit VARCHAR, PRIMARY KEY(case_id,evidence_id))")
        db.execute("CREATE TABLE case_assets (case_id VARCHAR, asset_id VARCHAR, evidence_id VARCHAR, PRIMARY KEY(case_id,asset_id,evidence_id))")
        db.execute("CREATE TABLE evidence_refs (case_id VARCHAR, evidence_id VARCHAR, ref_id VARCHAR, PRIMARY KEY(case_id,evidence_id,ref_id))")
        for asset in assets.values():
            db.execute("INSERT INTO assets VALUES (?,?,?,?,?,?)", [asset.asset_id, asset.kind, asset.format, asset.source_path, asset.file_digest, json.dumps(asset.metadata)])
        for case in cases:
            db.execute("INSERT INTO cases VALUES (?,?,?)", [case.case_id, case.patient_id, case.model_dump_json()])
            for evidence in case.evidence:
                db.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?)", [case.case_id, evidence.id, evidence.type, evidence.name, json.dumps(evidence.value), evidence.time, evidence.unit])
                for ref in evidence.ref:
                    db.execute("INSERT INTO evidence_refs VALUES (?,?,?)", [case.case_id, evidence.id, ref])
                if evidence.type == "asset":
                    db.execute("INSERT INTO case_assets VALUES (?,?,?)", [case.case_id, evidence.value, evidence.id])
    temporary.replace(db_path)
    profile = _profile(cases, assets)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    access = {"schema_version": "annotation", "revision": revision, "cases_path": str(cases_file), "asset_manifest_path": str(manifest), "profile_path": str(profile_path), "duckdb_path": str(db_path), "annotation_digest": digest_file(cases_file), "asset_manifest_digest": digest_file(manifest), "profile_digest": digest_file(profile_path), "duckdb_digest": digest_file(db_path), **manifest_access}
    access_path.write_text(json.dumps(access, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return profile, access


def _profile(cases, assets):
    from collections import Counter, defaultdict
    evidence = defaultdict(lambda: {"cases": 0, "types": Counter(), "value_types": Counter(), "units": Counter(), "has_time": 0, "has_ref": 0})
    for case in cases:
        names = set()
        for item in case.evidence:
            row = evidence[item.name]
            row["cases"] += item.name not in names
            names.add(item.name)
            row["types"][item.type] += 1
            value = item.value
            row["value_types"]["boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "string" if isinstance(value, str) else "array" if isinstance(value, list) else "object"] += 1
            row["has_time"] += item.time is not None
            row["has_ref"] += bool(item.ref)
            if item.unit is not None:
                row["units"][item.unit] += 1
    per_case = Counter(sum(item.type == "asset" for item in case.evidence) for case in cases)
    return {"schema_version": "dataset-profile", "case_count": len(cases), "patient_count": len({c.patient_id for c in cases if c.patient_id is not None}), "cases_per_patient": dict(Counter(c.patient_id for c in cases if c.patient_id is not None)), "evidence": [{"name": name, **row, "types": dict(row["types"]), "value_types": dict(row["value_types"]), "units": dict(row["units"])} for name, row in sorted(evidence.items())], "assets": {"kinds": dict(Counter(a.kind for a in assets.values())), "formats": dict(Counter(a.format for a in assets.values())), "assets_per_case": dict(per_case), "volume_properties": {a.asset_id: a.metadata for a in assets.values() if a.kind == "volume"}}}


class DatasetIndex:
    def __init__(self, access):
        self.access = access

    def _db(self):
        return duckdb.connect(self.access["duckdb_path"], read_only=True)

    def records(self, case_ids):
        if len(case_ids) > 512:
            raise ValueError("At most 512 Case IDs can be requested")
        with self._db() as db:
            rows = db.execute("SELECT case_id,record FROM cases WHERE case_id IN (SELECT unnest(?))", [case_ids]).fetchall()
        return {case_id: json.loads(record) for case_id, record in rows}

    def sample(self, offset=0, limit=5, *, case_id=None, evidence_name=None, asset_kind=None):
        filters, args = [], []
        if case_id is not None:
            filters.append("case_id=?")
            args.append(case_id)
        if evidence_name is not None:
            filters.append("case_id IN (SELECT case_id FROM evidence WHERE name=?)")
            args.append(evidence_name)
        if asset_kind is not None:
            filters.append("case_id IN (SELECT ca.case_id FROM case_assets ca JOIN assets a ON a.asset_id=ca.asset_id WHERE a.kind=?)")
            args.append(asset_kind)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._db() as db:
            total = db.execute("SELECT count(*) FROM cases" + where, args).fetchone()[0]
            rows = db.execute("SELECT record FROM cases" + where + " ORDER BY case_id LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
        return {"dataset_version": self.access["revision"], "total": total, "values": [json.loads(row[0]) for row in rows], "complete": offset + len(rows) >= total, "next_offset": offset + len(rows) if offset + len(rows) < total else None}

    def evidence(self, name=None, evidence_type=None):
        where, args = [], []
        if name is not None:
            where.append("name=?")
            args.append(name)
        if evidence_type is not None:
            where.append("type=?")
            args.append(evidence_type)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._db() as db:
            return [dict(case_id=row[0], evidence_id=row[1], type=row[2], name=row[3], value=json.loads(row[4]), time=row[5], unit=row[6]) for row in db.execute("SELECT case_id,evidence_id,type,name,value,time,unit FROM evidence" + clause, args).fetchall()]

    def assets(self, kind=None, format_name=None):
        where, args = [], []
        if kind is not None:
            where.append("kind=?")
            args.append(kind)
        if format_name is not None:
            where.append("format=?")
            args.append(format_name)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._db() as db:
            return [dict(asset_id=row[0], kind=row[1], format=row[2], source_path=row[3], file_digest=row[4], metadata=json.loads(row[5])) for row in db.execute("SELECT asset_id,kind,format,source_path,file_digest,metadata FROM assets" + clause, args).fetchall()]

    def iter_cases(self):
        offset = 0
        while True:
            page = self.sample(offset=offset, limit=512)
            yield from page["values"]
            if page["complete"]:
                break
            offset = page["next_offset"]

    def facts(self):
        from medbenchagent.data.task_facts import build_dataset_facts
        assets = {row["asset_id"]: row for row in self.assets()}
        profile = json.loads(Path(self.access["profile_path"]).read_text(encoding="utf-8"))
        facts = build_dataset_facts(self.iter_cases(), assets, profile=profile, revision=self.access.get("revision"))
        try:
            from medbenchagent.data.manifests import load_manifest, PresentationRecord, SamplingManifest, verify_manifest_digest
            for name, model in (("presentation", PresentationRecord), ("sampling", SamplingManifest)):
                path = self.access.get(f"{name}_manifest_path")
                digest = self.access.get(f"{name}_manifest_digest")
                if not path:
                    continue
                verify_manifest_digest(path, digest)
                manifest = load_manifest(path, model)
                if name == "presentation":
                    records = manifest if isinstance(manifest, list) else [manifest]
                    ordered = len({record.view_id for record in records if record.view_id})
                else:
                    ordered = len(manifest.units)
                facts.setdefault("view_panel_capabilities", {})["ordered"] = ordered
                facts.setdefault("derived_capabilities", {}).setdefault("view_panel", {})["ordered"] = ordered
        except (OSError, ValueError):
            pass
        return facts

    def node_context(self, node, *, sample_limit=5, evidence_limit=100, registry=None):
        from medbenchagent.data.task_facts import case_facts
        from medbenchagent.planning.node_builder import NodeConstraintRegistry
        from medbenchagent.planning.task_space import TaskScenario
        node = TaskScenario.model_validate(node)
        constraint_registry = NodeConstraintRegistry(registry)
        assets = {row["asset_id"]: row for row in self.assets()}
        samples, count = [], 0
        names = set()
        for case in self.iter_cases():
            if constraint_registry.check(node, case_facts(case, assets))[0]:
                count += 1
                names.update(row["name"] for row in case["evidence"])
                if len(samples) < sample_limit:
                    samples.append(case)
        profile = json.loads(Path(self.access["profile_path"]).read_text(encoding="utf-8"))
        return {"dataset_version": self.access["revision"], "scenario_id": node.scenario_id,
                "scenario": node.model_dump(),
                "evidence": [row for row in profile["evidence"] if row["name"] in names][:evidence_limit],
                "capacity_hint": count, "relations": self.facts()["relations"],
                "representative_cases": samples}

    def values(self, field, offset=0, limit=20, source_id=None):
        with self._db() as db:
            total = db.execute("SELECT count(DISTINCT value) FROM evidence WHERE name=?", [field]).fetchone()[0]
            rows = db.execute("SELECT value,count(*) FROM evidence WHERE name=? GROUP BY value ORDER BY value LIMIT ? OFFSET ?", [field, limit, offset]).fetchall()
        return {"values": [{"value": json.loads(v), "count": n} for v, n in rows], "total": total,
                "dataset_version": self.access["revision"], "complete": offset + len(rows) >= total}

