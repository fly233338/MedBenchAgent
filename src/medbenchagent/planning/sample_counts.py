"""Deterministic capacity-aware leaf allocation."""


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("Leaf weights must sum to a positive value.")
    quotas = {category: total * weight / weight_total for category, weight in weights.items()}
    allocated = {category: int(quota) for category, quota in quotas.items()}
    for category in sorted(
        weights,
        key=lambda category: (
            -(quotas[category] - allocated[category]),
            list(weights).index(category),
        ),
    )[: total - sum(allocated.values())]:
        allocated[category] += 1
    return allocated


def allocate_leaf_counts(weights: dict[str, float], capacities: dict[str, int], total: int) -> dict:
    """Cap weighted quotas and redistribute before reporting a final shortfall."""
    import math

    if type(total) is not int or total <= 0 or not weights or set(weights) != set(capacities):
        raise ValueError("Provide a positive total and matching leaf weights/capacities")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Weights must be nonnegative finite numbers")
    if any(type(v) is not int or v < 0 for v in capacities.values()):
        raise ValueError("Capacities must be nonnegative integers")
    if not any(weights.values()):
        raise ValueError("At least one weight must be positive")
    counts = dict.fromkeys(sorted(weights), 0)
    remaining = total
    while remaining:
        active = {k: weights[k] for k in counts if counts[k] < capacities[k] and weights[k] > 0}
        if not active:
            break
        scale = max(active.values())
        quotas = _largest_remainder(remaining, {k: v / scale for k, v in active.items()})
        added = 0
        for key, quota in quotas.items():
            amount = min(quota, capacities[key] - counts[key])
            counts[key] += amount
            added += amount
        remaining -= added
        if not added:
            break
    return {
        "counts": counts,
        "requested_count": total,
        "allocated_count": total - remaining,
        "planning_shortfall": remaining,
        "capacities": capacities,
        "weights": weights,
    }
