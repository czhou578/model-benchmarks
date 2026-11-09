"""GPU telemetry: power samples to energy, and per-window summaries."""

from __future__ import annotations

import bisect
import statistics
from typing import Any


def _power_at(points: list[tuple[float, float]], t: float) -> float:
    """Return the power at time ``t`` by linear interpolation.

    ``points`` is a sorted list of ``(t, power)`` pairs. Before the first
    point and after the last point, the function holds the nearest value.
    """
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    i = bisect.bisect_right([p[0] for p in points], t)
    (t0, p0), (t1, p1) = points[i - 1], points[i]
    return p0 + (p1 - p0) * (t - t0) / (t1 - t0)


def gpu_energy_wh(
    samples: list[dict[str, Any]],
    t_start: float | None = None,
    t_end: float | None = None,
) -> float | None:
    """Integrate GPU power over time with the trapezoid rule.

    The function uses the real timestamp ``t`` of each sample, not the
    nominal interval. Without a window, it integrates from the first sample
    to the last sample. With ``t_start`` and ``t_end``, it integrates over
    that window only, and it interpolates the power at each edge.
    It returns ``None`` when no sample has a power value or the span is 0.
    """
    points = sorted(
        (s["t"], s["gpu_power_w"]) for s in samples
        if s.get("t") is not None and s.get("gpu_power_w") is not None
    )
    if not points:
        return None
    lo = points[0][0] if t_start is None else t_start
    hi = points[-1][0] if t_end is None else t_end
    if hi <= lo:
        return None
    knots = [(lo, _power_at(points, lo))]
    knots += [(t, p) for t, p in points if lo < t < hi]
    knots.append((hi, _power_at(points, hi)))
    joules = sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(knots, knots[1:]))
    return joules / 3600.0


def decode_energy_summary(
    samples: list[dict[str, Any]],
    decode: dict[str, Any],
    t_start: float,
    t_end: float,
) -> dict[str, Any]:
    """Return the GPU energy per output token of the decode benchmark.

    The energy covers only the decode window, from the start of the first
    decode request to the end of the last one. The token count is the sum
    of ``actual_output_tokens`` in ``decode``.
    """
    tokens = sum(v.get("actual_output_tokens") or 0 for v in decode.values())
    energy = gpu_energy_wh(samples, t_start, t_end)
    per_token = round(energy / tokens, 8) if energy is not None and tokens else None
    return {
        "window_start_unix": round(t_start, 3),
        "window_end_unix": round(t_end, 3),
        "duration_s": round(t_end - t_start, 2),
        "energy_wh": round(energy, 4) if energy is not None else None,
        "output_tokens": tokens,
        "energy_per_output_token_wh": per_token,
    }


def _window_summary(
    samples: list[dict[str, Any]],
    idle_samples: list[dict[str, Any]],
    all_samples: list[dict[str, Any]],
    t_start: float,
    t_end: float,
) -> dict[str, Any]:
    """Compute per-window GPU telemetry.

    Args:
        samples: samples that the monitor took while the window was open.
        idle_samples: baseline samples recorded before the benchmark started.
        all_samples: all samples of the run. The energy integral uses them
            to interpolate the power at the window edges.
        t_start, t_end: Unix times of the window start and the window stop.

    Returns a dict with avg/peak power, utilization, memory, energy (Wh),
    and incremental energy above idle.
    """
    if not samples:
        return {}

    def _vals(key: str) -> list[float]:
        return [s[key] for s in samples if s.get(key) is not None]

    powers = _vals("gpu_power_w")
    utils = _vals("gpu_util_pct")
    mem_utils = _vals("gpu_mem_util_pct")
    mems = _vals("gpu_mem_used_mib")

    idle_powers = [s["gpu_power_w"] for s in idle_samples if s.get("gpu_power_w") is not None]
    idle_base = round(statistics.mean(idle_powers), 2) if idle_powers else None

    duration_s = t_end - t_start
    energy_wh = gpu_energy_wh(all_samples, t_start, t_end)
    if energy_wh is not None:
        energy_wh = round(energy_wh, 6)
    energy_above_wh = None
    if idle_base is not None and energy_wh is not None:
        energy_above_wh = round(energy_wh - idle_base * duration_s / 3600.0, 6)

    result: dict[str, Any] = {
        "num_samples": len(samples),
        "duration_s": round(duration_s, 2),
        "gpu_util_avg_pct": round(statistics.mean(utils), 2) if utils else None,
        "gpu_util_peak_pct": round(max(utils), 2) if utils else None,
        "gpu_power_avg_w": round(statistics.mean(powers), 2) if powers else None,
        "gpu_power_peak_w": round(max(powers), 2) if powers else None,
        "gpu_mem_util_avg_pct": round(statistics.mean(mem_utils), 2) if mem_utils else None,
        "gpu_mem_util_peak_pct": round(max(mem_utils), 2) if mem_utils else None,
        "gpu_mem_used_avg_mib": round(statistics.mean(mems), 1) if mems else None,
        "gpu_mem_used_peak_mib": round(max(mems), 1) if mems else None,
        "energy_wh": energy_wh,
    }
    if idle_base is not None:
        result["idle_base_power_w"] = idle_base
        result["energy_above_idle_wh"] = energy_above_wh
    return result

