"""Tests for the step order and the run-dir files of core_runner.main().

Fake benchmark functions replace the real ones. No server, no GPU.
"""

from __future__ import annotations

import json
import sys

import pytest
import yaml

import benchmarks.architecture_flops
import benchmarks.concurrency
import benchmarks.decode
import benchmarks.latency
import benchmarks.prefill
import benchmarks.reasoning
import benchmarks.tool_calling
import benchmarks.ttft_breakdown
import core_runner

ALL_STEPS = [
    "latency", "deep_context", "ttft_breakdown", "decode", "reasoning",
    "concurrency", "prefill_scaling", "flops_analysis", "tool_calling",
]


class _FakeMonitor:
    idle_records = 0

    def __init__(self, run_dir, interval_s=1.0):
        self.samples = []

    def start(self):
        pass

    def start_idle(self):
        pass

    def record_idle(self):
        _FakeMonitor.idle_records += 1

    def stop(self):
        return {"samples": 0}


class _FakeClient:
    base_url = "http://127.0.0.1:1"

    def wait_until_ready(self, **kwargs):
        pass


def _patch(monkeypatch, module, name, fn):
    """Patch *name* in its module and in core_runner, wherever core_runner reads it."""
    monkeypatch.setattr(module, name, fn)
    if hasattr(core_runner, name):
        monkeypatch.setattr(core_runner, name, fn)


def _run_main(tmp_path, monkeypatch, *flags, tool_calling_error=None, extra_cfg=None, seen=None):
    calls: list[str] = []
    _FakeMonitor.idle_records = 0

    def fake(step, result):
        def run(*args, **kwargs):
            calls.append(step)
            if seen is not None:
                seen[step] = kwargs
            return result
        return run

    def fake_tool_calling(*args, **kwargs):
        calls.append("tool_calling")
        if tool_calling_error:
            raise tool_calling_error
        return {"composite_score": 1.0, "total_correct": 1, "total_tasks": 1}

    monkeypatch.setattr(core_runner, "make_client", lambda cfg: _FakeClient())
    monkeypatch.setattr(core_runner, "GpuMonitor", _FakeMonitor)
    monkeypatch.setattr(core_runner, "collect_environment", lambda: {"gpu_name": "fake"})
    monkeypatch.setattr(core_runner, "_get_server_model_config", lambda client: {})
    monkeypatch.setattr(core_runner, "fetch_server_version", lambda *args: "0.33.0")
    monkeypatch.setattr(core_runner.time, "sleep", lambda s: None)
    _patch(monkeypatch, benchmarks.latency, "run_latency_sweep", fake("latency", {}))
    _patch(monkeypatch, benchmarks.latency, "run_deep_context", fake("deep_context", {}))
    _patch(monkeypatch, benchmarks.ttft_breakdown, "run_ttft_breakdown", fake("ttft_breakdown", {}))
    _patch(monkeypatch, benchmarks.decode, "run_decode_speed", fake("decode", {}))
    _patch(monkeypatch, benchmarks.reasoning, "run_reasoning_benchmark", fake("reasoning", {}))
    _patch(monkeypatch, benchmarks.concurrency, "run_concurrency_test", fake("concurrency", {}))
    _patch(monkeypatch, benchmarks.prefill, "run_prefill_scaling", fake("prefill_scaling", {"per_length": {}}))
    _patch(monkeypatch, benchmarks.architecture_flops, "run_flops_analysis",
           fake("flops_analysis", {"estimate_status": "unsupported"}))
    _patch(monkeypatch, benchmarks.tool_calling, "run_tool_calling_benchmark", fake_tool_calling)

    config = tmp_path / "model.yml"
    config.write_text(yaml.safe_dump({
        "name": "fake-model",
        "server": {"mode": "external"},
        "endpoint": {"base_url": "http://127.0.0.1:1"},
        "context_lengths": [32],
        **(extra_cfg or {}),
    }))
    results = tmp_path / "results"
    argv = ["core_runner.py", "--model", str(config), "--results-dir", str(results), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    core_runner.main()
    (run_dir,) = (results / "fake-model").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    return calls, run_dir, summary


def test_main_runs_all_steps_in_order(tmp_path, monkeypatch):
    calls, run_dir, summary = _run_main(tmp_path, monkeypatch)
    assert calls == ALL_STEPS
    assert summary["status"] == "completed"
    assert summary["tool_calling"]["status"] == "completed"
    for step in ALL_STEPS:
        assert (run_dir / f"{step}.json").exists(), step
    for name in ("model_config.yml", "environment.json"):
        assert (run_dir / name).exists(), name


def test_skip_flags_leave_out_their_steps(tmp_path, monkeypatch):
    flags = ["--skip-latency", "--skip-ttft", "--skip-decode", "--skip-reasoning",
             "--skip-concurrency", "--skip-prefill"]
    calls, _, summary = _run_main(tmp_path, monkeypatch, *flags)
    assert calls == ["deep_context", "flops_analysis", "tool_calling"]
    assert "latency" not in summary and "decode" not in summary


def test_new_skip_flags_leave_out_their_steps(tmp_path, monkeypatch):
    flags = ["--skip-deep-context", "--skip-flops", "--skip-tool-calling"]
    calls, run_dir, summary = _run_main(tmp_path, monkeypatch, *flags)
    assert calls == [s for s in ALL_STEPS if s not in ("deep_context", "flops_analysis", "tool_calling")]
    for step in ("deep_context", "flops_analysis", "tool_calling"):
        assert step not in summary
        assert not (run_dir / f"{step}.json").exists()


def test_config_keys_set_deep_context_tokens_and_idle_baseline(tmp_path, monkeypatch):
    seen: dict = {}
    _run_main(tmp_path, monkeypatch, seen=seen)
    assert seen["deep_context"]["output_tokens"] == 64
    assert _FakeMonitor.idle_records == 5

    extra = {"deep_context_output_tokens": 16, "idle_baseline_s": 2.0, "monitor_interval_s": 0.5}
    (tmp_path / "custom").mkdir()
    _run_main(tmp_path / "custom", monkeypatch, extra_cfg=extra, seen=seen)
    assert seen["deep_context"]["output_tokens"] == 16
    assert _FakeMonitor.idle_records == 4


def test_zero_monitor_interval_fails_before_the_idle_baseline(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="monitor_interval_s"):
        _run_main(tmp_path, monkeypatch, extra_cfg={"monitor_interval_s": 0})


def test_tool_calling_failure_is_recorded_and_the_run_completes(tmp_path, monkeypatch):
    _, _, summary = _run_main(tmp_path, monkeypatch, tool_calling_error=RuntimeError("boom"))
    assert summary["status"] == "completed"
    assert summary["tool_calling"] == {"status": "failed", "error": "RuntimeError: boom"}


def test_compare_spec_needs_managed_mode(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="managed server mode"):
        _run_main(tmp_path, monkeypatch, "--compare-spec")
    (run_dir,) = (tmp_path / "results" / "fake-model").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "failed"


def test_summary_and_environment_record_the_provenance(tmp_path, monkeypatch):
    harness = {"schema_version": 3, "git_commit": "abc123", "git_dirty": False}
    monkeypatch.setattr(core_runner, "harness_provenance", lambda: harness)
    _, run_dir, summary = _run_main(tmp_path, monkeypatch, extra_cfg={"run_label": "prefill rerun"})
    assert summary["harness"] == harness
    assert summary["run_label"] == "prefill rerun"
    env = json.loads((run_dir / "environment.json").read_text())
    assert env["gpu_name"] == "fake" and env["vllm_server_version"] == "0.33.0"


def test_run_label_is_absent_when_not_set(tmp_path, monkeypatch):
    _, _, summary = _run_main(tmp_path, monkeypatch)
    assert "run_label" not in summary
    assert summary["harness"]["schema_version"] == 3
