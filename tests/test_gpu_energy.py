"""Tests for the GPU energy integral and the decode energy per token."""

from __future__ import annotations

import pytest

from harness.telemetry import _window_summary, decode_energy_summary, gpu_energy_wh

# Uneven gaps: 1 s, 2 s, 0.5 s. Trapezoids: 15 J + 40 J + 15 J = 70 J.
SAMPLES = [
    {"t": 100.0, "gpu_power_w": 10.0},
    {"t": 101.0, "gpu_power_w": 20.0},
    {"t": 103.0, "gpu_power_w": 20.0},
    {"t": 103.5, "gpu_power_w": 40.0},
]


def test_energy_uses_the_real_timestamps() -> None:
    # Mean power x samples x 1 s gives 90 J. The integral must give 70 J.
    assert gpu_energy_wh(SAMPLES) == pytest.approx(70.0 / 3600)
    # The integral sorts the samples by time.
    assert gpu_energy_wh(list(reversed(SAMPLES))) == pytest.approx(70.0 / 3600)


def test_energy_skips_samples_without_power() -> None:
    rows = SAMPLES + [{"t": 104.0, "gpu_power_w": None}]
    assert gpu_energy_wh(rows) == pytest.approx(70.0 / 3600)
    assert gpu_energy_wh([{"t": 1.0, "gpu_power_w": None}]) is None
    assert gpu_energy_wh(SAMPLES[:1]) is None


def test_energy_window_interpolates_at_the_edges() -> None:
    # p(100.5) = 15 W and p(103.25) = 30 W: 8.75 J + 40 J + 6.25 J = 55 J.
    assert gpu_energy_wh(SAMPLES, 100.5, 103.25) == pytest.approx(55.0 / 3600)
    # Outside the samples, the integral holds the nearest power value.
    assert gpu_energy_wh(SAMPLES, 99.0, 100.0) == pytest.approx(10.0 / 3600)
    assert gpu_energy_wh(SAMPLES, 103.5, 105.5) == pytest.approx(80.0 / 3600)


def test_decode_energy_covers_only_the_decode_window() -> None:
    decode = {
        "512": {"actual_output_tokens": 500},
        "1024": {"actual_output_tokens": 1000},
        "2048": {"actual_output_tokens": None},
    }
    result = decode_energy_summary(SAMPLES, decode, 101.0, 103.0)

    assert result["output_tokens"] == 1500
    assert result["duration_s"] == 2.0
    assert result["energy_wh"] == round(40.0 / 3600, 4)
    assert result["energy_per_output_token_wh"] == round(40.0 / 3600 / 1500, 8)


def test_decode_energy_without_tokens_has_no_per_token_value() -> None:
    result = decode_energy_summary(SAMPLES, {"512": {}}, 101.0, 103.0)
    assert result["output_tokens"] == 0
    assert result["energy_per_output_token_wh"] is None


def test_window_summary_uses_the_idle_samples_as_baseline() -> None:
    idle = [{"t": 0.0, "gpu_power_w": 5.0}, {"t": 1.0, "gpu_power_w": 5.0}]
    window = SAMPLES[1:3]

    result = _window_summary(window, idle, SAMPLES, 101.0, 103.0)

    assert result["duration_s"] == 2.0
    assert result["gpu_power_avg_w"] == 20.0
    assert result["energy_wh"] == pytest.approx(40.0 / 3600, abs=1e-6)
    assert result["idle_base_power_w"] == 5.0
    # 40 J - 5 W x 2 s = 30 J above idle.
    assert result["energy_above_idle_wh"] == pytest.approx(30.0 / 3600, abs=1e-6)
