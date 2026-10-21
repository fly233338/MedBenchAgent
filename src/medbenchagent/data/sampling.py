"""Deterministic examples covering observed categories, missingness and evidence relations."""
import json
from collections import Counter


def representative_samples(cases, limit=6):
    candidates, frequency, first = {}, Counter(), {}
    numeric = {}
    for case in cases:
        features = set()
        for row in case.get("evidence", []):
            name, value = row.get("name", ""), row.get("value")
            features.add(("field", name))
            if value is None:
                features.add(("missing", name))
            elif isinstance(value, bool) or isinstance(value, str) and len(value) <= 160 and row.get("type") != "asset":
                features.add(("value", name, json.dumps(value, ensure_ascii=False)))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                previous = numeric.get(name)
                if previous is None or value < previous[0]:
                    numeric[name] = (value, case, previous[2] if previous else value, previous[3] if previous else case)
                if value > numeric[name][2]:
                    numeric[name] = (*numeric[name][:2], value, case)
            features.add(("relation", name, "linked" if row.get("ref") else "unlinked"))
        features.add(("fields", tuple(sorted(row.get("name", "") for row in case.get("evidence", [])))))
        frequency.update(features)
        for feature in features:
            if feature not in first:
                first[feature] = case["case_id"]
                candidates[case["case_id"]] = (case, features)
    for name, (_, low, _, high) in numeric.items():
        for label, case in (("minimum", low), ("maximum", high)):
            feature = (label, name)
            frequency[feature] = 1
            if case["case_id"] not in candidates:
                candidates[case["case_id"]] = (case, set())
            candidates[case["case_id"]][1].add(feature)
    selected, covered = [], set()
    while candidates and len(selected) < limit:
        key = min(candidates, key=lambda key: (-sum(1 / frequency[f] for f in candidates[key][1] - covered), str(key)))
        case, features = candidates.pop(key)
        selected.append(case)
        covered.update(features)
    return selected
