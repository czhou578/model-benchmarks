"""Summary statistics for benchmark samples."""

from __future__ import annotations

import statistics


def percentile(data: list[float], pct: float) -> float:
    """Return the *pct* percentile with linear interpolation, or NaN for no data."""
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def stat_summary(values: list[float], suffix: str = "s") -> dict[str, float | None]:
    """Compute avg / median / p95 / min / max for a list of floats."""
    if not values:
        return {
            f"avg_{suffix}": None,
            f"median_{suffix}": None,
            f"p95_{suffix}": None,
            f"min_{suffix}": None,
            f"max_{suffix}": None,
        }
    p95 = percentile(values, 95)
    return {
        f"avg_{suffix}": round(statistics.mean(values), 4),
        f"median_{suffix}": round(statistics.median(values), 4),
        f"p95_{suffix}": round(p95, 4),
        f"min_{suffix}": round(min(values), 4),
        f"max_{suffix}": round(max(values), 4),
    }

