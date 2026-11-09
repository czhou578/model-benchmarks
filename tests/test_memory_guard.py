"""Tests for memory_guard.py: the meminfo parser and the run wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from memory_guard import EXIT_LOW_MEMORY, parse_meminfo

GUARD = Path(__file__).resolve().parents[1] / "memory_guard.py"


def test_parse_meminfo_reports_gib() -> None:
    text = (
        "MemTotal:       127604736 kB\n"
        "MemFree:         1000000 kB\n"
        "MemAvailable:   104857600 kB\n"
        "SwapTotal:      16777216 kB\n"
        "SwapFree:       15728640 kB\n"
    )
    mem = parse_meminfo(text)
    assert mem["mem_available_gib"] == 100.0
    assert mem["swap_used_gib"] == 1.0
    assert mem["mem_used_gib"] == round(mem["mem_total_gib"] - 100.0, 2)


def _run_guard(tmp_path: Path, run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    child = f"print('[core_runner] writing results to {run_dir}', flush=True)"
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    return subprocess.run(
        [sys.executable, str(GUARD), "--min-start-gib", "0", "--interval-s", "0.2",
         *extra, "--", sys.executable, "-c", child],
        capture_output=True, text=True, timeout=60, cwd=cwd,
        env={**os.environ, "TMPDIR": str(temp_dir)},
    )


def test_guard_moves_temporary_samples_into_the_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proc = _run_guard(tmp_path, run_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "memory_samples.csv").read_text().startswith("elapsed_s,timestamp,")
    # No stray file stays in the current directory or the temporary directory.
    assert list((tmp_path / "cwd").iterdir()) == []
    assert list((tmp_path / "tmp").iterdir()) == []


def test_guard_keeps_an_explicit_csv_and_copies_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proc = _run_guard(tmp_path, run_dir, "--csv", str(tmp_path / "mem.csv"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "mem.csv").read_text().startswith("elapsed_s,timestamp,")
    assert (run_dir / "memory_samples.csv").read_text().startswith("elapsed_s,timestamp,")
    summary = json.loads((run_dir / "memory_guard.json").read_text())
    assert summary["stopped_for_memory"] is False
    assert summary["floor_gib"] == 8.0


def test_guard_refuses_to_start_without_enough_memory(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(GUARD), "--min-start-gib", "1000000",
         "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )
    assert proc.returncode == EXIT_LOW_MEMORY
    assert list(tmp_path.iterdir()) == []
    assert "refusing to start" in proc.stdout
