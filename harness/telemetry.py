"""GPU telemetry: the nvidia-smi sampler, energy, and per-window summaries."""

from __future__ import annotations

import bisect
import csv
import shutil
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Optional

from harness.environment import run_command
from harness.paths import repo_relative


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


class GpuMonitor:
    """Background thread that samples GPU utilization, memory, and power.

    It waits *interval_s* seconds (default 1) after each nvidia-smi call.
    """

    FIELDS = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    # Keys that map to the CSV fields above
    KEYS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_power_w")

    def __init__(self, out_dir: Path, interval_s: float = 1.0):
        self.out_dir = out_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[dict[str, Any]] = []
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        # Per-length window tracking
        self._windows: dict[str, list[dict[str, Any]]] = {}
        self._window_starts: dict[str, float] = {}
        self._idle_samples: list[dict[str, Any]] = []

    def _parse_row(self, out: str) -> dict[str, Any]:
        """Parse one nvidia-smi CSV line into a row dict."""
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        row: dict[str, Any] = {}
        for k, v in zip(self.KEYS, parts):
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                row[k] = None
        return row

    def _sample_once(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": time.time()}
        if self.has_nvidia_smi:
            out = run_command([
                "nvidia-smi",
                f"--query-gpu={self.FIELDS}",
                "--format=csv,noheader,nounits",
            ])
            if out:
                parsed = self._parse_row(out)
                row.update(parsed)
        return row

    def _loop(self):
        while not self._stop.is_set():
            sample = self._sample_once()
            self.samples.append(sample)
            # Also record in every active window
            for window in list(self._windows.values()):
                window.append(sample)
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Window management for per-length telemetry
    # ------------------------------------------------------------------ #

    def start_idle(self) -> None:
        """Begin recording an idle baseline before benchmark work."""
        self._idle_samples = []

    def record_idle(self) -> None:
        """Take one nvidia-smi sample now and add it to the idle baseline."""
        sample = self._sample_once()
        self._idle_samples.append(sample)

    def start_window(self, name: str) -> None:
        """Start a new telemetry window by name."""
        self._window_starts[name] = time.time()
        self._windows[name] = []

    def stop_window(self, name: str) -> dict[str, Any] | None:
        """Stop the named window and return a summary.  Returns ``None`` if the
        window is empty (e.g. the benchmark loop was short and no sample was
        taken)."""
        t_end = time.time()
        t_start = self._window_starts.pop(name, None)
        window = self._windows.pop(name, None)
        if not window or t_start is None:
            return None
        return _window_summary(window, self._idle_samples, list(self.samples), t_start, t_end)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

        csv_path = self.out_dir / "gpu_samples.csv"
        if self.samples:
            keys = sorted({k for row in self.samples for k in row.keys()})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.samples)

        powers = [s["gpu_power_w"] for s in self.samples if s.get("gpu_power_w") is not None]
        mem = [s["gpu_mem_used_mib"] for s in self.samples if s.get("gpu_mem_used_mib") is not None]
        util = [s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None]
        mem_util = [s["gpu_mem_util_pct"] for s in self.samples if s.get("gpu_mem_util_pct") is not None]
        summary: dict[str, Any] = {
            "samples_csv": repo_relative(csv_path) if self.samples else None,
            "num_samples": len(self.samples),
            "gpu_util_avg_pct": round(statistics.mean(util), 2) if util else None,
            "gpu_util_peak_pct": round(max(util), 2) if util else None,
            "gpu_power_avg_w": round(statistics.mean(powers), 2) if powers else None,
            "gpu_power_peak_w": round(max(powers), 2) if powers else None,
            "gpu_mem_used_avg_mib": round(statistics.mean(mem), 1) if mem else None,
            "gpu_mem_used_peak_mib": round(max(mem), 1) if mem else None,
            "gpu_mem_util_avg_pct": round(statistics.mean(mem_util), 2) if mem_util else None,
            "gpu_mem_util_peak_pct": round(max(mem_util), 2) if mem_util else None,
        }
        energy_wh = gpu_energy_wh(self.samples)
        if energy_wh is not None:
            summary["duration_s"] = round(self.samples[-1]["t"] - self.samples[0]["t"], 2)
            summary["energy_wh"] = round(energy_wh, 4)
        return summary

