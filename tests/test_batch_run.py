"""Tests for batch_run.py. A fake subprocess.Popen replaces core_runner and vLLM."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import pytest

import batch_run


class FakeProc:
    """A stand-in for subprocess.Popen that replays one scripted attempt."""

    def __init__(self, lines: list[str], returncode: int, block: bool = False) -> None:
        self._lines = lines
        self._returncode = returncode
        self._killed = threading.Event()
        self._block = block
        self.stdout = self._read()

    def _read(self):
        yield from self._lines
        if self._block:
            self._killed.wait(5)

    def kill(self) -> None:
        self._killed.set()

    def wait(self) -> int:
        return -9 if self._killed.is_set() else self._returncode


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(batch_run, "ROOT", tmp_path)
    monkeypatch.setattr(batch_run, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(batch_run.time, "sleep", lambda s: None)
    return tmp_path / "results"


def make_run(results: Path, ts: str, status: str | None) -> Path:
    run_dir = results / "m" / ts
    run_dir.mkdir(parents=True)
    if status is not None:
        (run_dir / "summary.json").write_text(json.dumps({"status": status, "error": f"{status} error"}))
    return run_dir


def script(monkeypatch: pytest.MonkeyPatch, results: Path, attempts: list[tuple[str | None, str | None, int]]) -> list[list[str]]:
    """Make each Popen call replay one (timestamp, summary status, exit code) attempt."""
    calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        ts, status, rc = attempts[len(calls)]
        calls.append(cmd)
        lines = ["[core_runner] starting\n"]
        if ts is not None:
            run_dir = make_run(results, ts, status)
            lines.append(f"{batch_run.RUN_DIR_PREFIX}{run_dir}\n")
        return FakeProc(lines, rc)

    monkeypatch.setattr(batch_run.subprocess, "Popen", fake_popen)
    return calls


def model_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "m.yml"
    path.write_text("name: m\n")
    return path


def test_failed_run_does_not_report_an_older_completed_run(results, tmp_path, monkeypatch) -> None:
    make_run(results, "20260101_000000", "completed")
    script(monkeypatch, results, [("20260201_000000", "failed", 1)])
    r = batch_run.run_one_model(model_yaml(tmp_path), Path("core_runner.py"), retry=False, log_dir=tmp_path / "logs")
    assert r["status"] == "fail"
    assert r["run_dir"] == "results/m/20260201_000000"
    assert r["error"] == "failed error"


def test_nonzero_exit_without_a_new_run_dir_is_an_error(results, tmp_path, monkeypatch) -> None:
    make_run(results, "20990101_000000", "completed")  # newer by name than any real run
    script(monkeypatch, results, [(None, None, 2)])
    r = batch_run.run_one_model(model_yaml(tmp_path), Path("core_runner.py"), retry=False, log_dir=tmp_path / "logs")
    assert r["status"] == "error"
    assert r["run_dir"] is None
    assert r["summary"] is None
    assert "exit code 2" in r["error"]


def test_missing_run_dir_line_falls_back_to_the_new_dir(results, tmp_path, monkeypatch) -> None:
    make_run(results, "20260101_000000", "failed")

    def fake_popen(cmd, **kwargs):
        make_run(results, "20250101_000000", "completed")  # older name, but new
        return FakeProc(["no run-dir line\n"], 0)

    monkeypatch.setattr(batch_run.subprocess, "Popen", fake_popen)
    r = batch_run.run_one_model(model_yaml(tmp_path), Path("core_runner.py"), retry=False, log_dir=tmp_path / "logs")
    assert r["status"] == "pass"
    assert r["run_dir"] == "results/m/20250101_000000"


def test_retry_then_success(results, tmp_path, monkeypatch) -> None:
    calls = script(monkeypatch, results, [("20260101_000000", "failed", 1), ("20260101_010000", "completed", 0)])
    r = batch_run.run_one_model(model_yaml(tmp_path), Path("core_runner.py"), max_retries=2, log_dir=tmp_path / "logs")
    assert len(calls) == 2
    assert r["status"] == "pass"
    assert r["attempts"] == 2
    assert r["error"] is None
    assert r["run_dir"] == "results/m/20260101_010000"
    assert calls[0][-2:] == ["--results-dir", str(results)]
    text = Path(r["log"]).read_text()
    assert r["log"] == str(tmp_path / "logs" / "m.log")
    assert "[core_runner] starting" in text
    assert "attempt 2" in text


@pytest.mark.parametrize(("retry", "expected_calls"), [(False, 1), (True, 3)])
def test_retry_count(results, tmp_path, monkeypatch, retry, expected_calls) -> None:
    calls = script(monkeypatch, results, [(None, None, 1)] * 4)
    r = batch_run.run_one_model(model_yaml(tmp_path), Path("core_runner.py"), retry=retry, max_retries=2, log_dir=tmp_path / "logs")
    assert len(calls) == expected_calls
    assert r["status"] == "error"


def test_attempt_timeout_kills_the_process(results, tmp_path, monkeypatch) -> None:
    proc = FakeProc(["[core_runner] starting\n"], 0, block=True)
    monkeypatch.setattr(batch_run.subprocess, "Popen", lambda cmd, **kwargs: proc)
    out = batch_run.run_attempt(model_yaml(tmp_path), Path("core_runner.py"), "m", io.StringIO(), 0.05)
    assert out["status"] == "timeout"


def test_discovery_skips_subfolders(tmp_path, monkeypatch, capsys) -> None:
    models = tmp_path / "models"
    (models / "legacy").mkdir(parents=True)
    (models / "top.yml").write_text("name: top\n")
    (models / "legacy" / "old.yml").write_text("name: old\n")
    monkeypatch.setattr(batch_run, "MODELS_DIR", models)
    batch_run.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "top" in out
    assert "old" not in out
