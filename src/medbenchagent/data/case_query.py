"""Data-side Case/Evidence/Asset profile and query API."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from medbenchagent.data.v5_models import AssetProfile, CaseRecord, DatasetProfile, EvidenceProfile, validate_case


class CaseQuery:
    def __init__(self, cases_path, asset_manifest_path):
        self.assets = {item.asset_id: item for item in self._assets(asset_manifest_path)}
        self.cases = []
        with Path(cases_path).open(encoding="utf-8") as handle:
            for line in handle:
                case = CaseRecord.model_validate_json(line)
                self.cases.append(validate_case(case, self.assets))

    @staticmethod
    def _assets(path):
        from medbenchagent.data.asset_manifest import iter_asset_manifest
        return list(iter_asset_manifest(path))

    def case_count(self):
        return len(self.cases)

    def patient_count(self):
        return len({case.patient_id for case in self.cases if case.patient_id is not None})

    def cases_per_patient(self):
        counts = Counter(case.patient_id for case in self.cases if case.patient_id is not None)
        return dict(sorted(counts.items()))

    def sample(self, *, case_id=None, evidence_name=None, evidence_type=None, asset_kind=None, limit=5):
        values = self.cases
        if case_id is not None:
            values = [case for case in values if case.case_id == case_id]
        if evidence_name is not None:
            values = [case for case in values if any(e.name == evidence_name for e in case.evidence)]
        if evidence_type is not None:
            values = [case for case in values if any(e.type == evidence_type for e in case.evidence)]
        if asset_kind is not None:
            values = [case for case in values if any(e.type == "asset" and self.assets[e.value].kind == asset_kind for e in case.evidence)]
        return values[:limit]

    def evidence_profile(self):
        stats = defaultdict(lambda: {"cases": 0, "types": Counter(), "units": Counter(), "has_time": 0, "has_ref": 0})
        for case in self.cases:
            seen = set()
            for evidence in case.evidence:
                key = evidence.name
                if key in seen:
                    continue
                seen.add(key)
                item = stats[key]
                item["cases"] += 1
                item["types"][evidence.type] += 1
                if evidence.unit is not None:
                    item["units"][evidence.unit] += 1
                item["has_time"] += evidence.time is not None
                item["has_ref"] += bool(evidence.ref)
        return {name: {**value, "types": dict(value["types"]), "units": dict(value["units"])} for name, value in sorted(stats.items())}

    def asset_profile(self):
        counts = Counter(asset.kind for asset in self.assets.values())
        formats = Counter(asset.format for asset in self.assets.values())
        per_case = Counter(sum(1 for e in case.evidence if e.type == "asset") for case in self.cases)
        return {"kinds": dict(counts), "formats": dict(formats), "assets_per_case": dict(sorted(per_case.items()))}

    def profile(self) -> DatasetProfile:
        evidence = [EvidenceProfile(name=name, **value) for name, value in self.evidence_profile().items()]
        assets = AssetProfile.model_validate(self.asset_profile())
        return DatasetProfile(case_count=self.case_count(), patient_count=self.patient_count(),
                              cases_per_patient=self.cases_per_patient(), evidence=evidence, assets=assets)


class EvidenceQuery(CaseQuery):
    """Evidence-scoped view of the data-side query API."""

    def by_name(self, name):
        return [e for case in self.cases for e in case.evidence if e.name == name]

    def by_type(self, evidence_type):
        return [e for case in self.cases for e in case.evidence if e.type == evidence_type]


class AssetQuery(CaseQuery):
    """Asset-scoped view of the data-side query API."""

    def by_kind(self, kind):
        return [asset for asset in self.assets.values() if asset.kind == kind]

    def by_format(self, format_name):
        return [asset for asset in self.assets.values() if asset.format == format_name]
