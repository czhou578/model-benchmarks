"""Tests for the nvidia-smi row parser of GpuMonitor."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.telemetry import GpuMonitor

KEYS = GpuMonitor.KEYS


def test_a_normal_row_gives_every_key_and_no_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monitor = GpuMonitor(tmp_path)

    row = monitor._parse_row("95, 40, [N/A], [N/A], 61.25\n")

    assert row == {
        "gpu_util_pct": 95.0,
        "gpu_mem_util_pct": 40.0,
        "gpu_mem_used_mib": None,
        "gpu_mem_total_mib": None,
        "gpu_power_w": 61.25,
    }
    assert capsys.readouterr().out == ""


def test_a_short_row_gives_none_for_the_missing_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monitor = GpuMonitor(tmp_path)

    row = monitor._parse_row("95, 40, 1024")

    assert tuple(row) == KEYS
    assert row["gpu_mem_used_mib"] == 1024.0
    assert row["gpu_mem_total_mib"] is None
    assert row["gpu_power_w"] is None
    out = capsys.readouterr().out
    assert "[telemetry] warning: nvidia-smi gave 3 columns, expected 5" in out


def test_a_long_row_drops_the_extra_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monitor = GpuMonitor(tmp_path)

    row = monitor._parse_row("95, 40, 1024, 2048, 61.25, 7")

    assert row == dict(zip(KEYS, [95.0, 40.0, 1024.0, 2048.0, 61.25], strict=True))
    assert "nvidia-smi gave 6 columns, expected 5" in capsys.readouterr().out


def test_the_monitor_warns_only_one_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monitor = GpuMonitor(tmp_path)

    monitor._parse_row("95, 40")
    monitor._parse_row("95, 40, 1024, 2048, 61.25, 7")

    assert capsys.readouterr().out.count("warning") == 1
