"""Tests for the prompt calibration and GPU windows of the TTFT and prefill benchmarks.

A fake client replaces the server. No HTTP, no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import benchmarks.prefill
import benchmarks.ttft_breakdown
from benchmarks.prefill import run_prefill_scaling
from benchmarks.ttft_breakdown import TtftRequestResult, run_ttft_breakdown
from core_runner import prefix_caching_flag

TEMPLATE_TOKENS = 12


class _FakeClient:
    """Counts 12 template tokens plus 1 token per 4 characters, like a chat server."""

    def __init__(self, events: list[str], fail_tokenize: bool = False) -> None:
        self.events = events
        self.fail_tokenize = fail_tokenize
        self.prompts: list[str] = []

    def preflight_cache_salt(self) -> bool:
        return True

    def tokenize_prompt(self, prompt: str) -> SimpleNamespace:
        self.events.append("tokenize")
        if self.fail_tokenize:
            raise RuntimeError("tokenize failed")
        return SimpleNamespace(count=TEMPLATE_TOKENS + len(prompt) // 4)

    def generate(self, prompt: str, max_tokens: int, cache_salt: str) -> SimpleNamespace:
        self.events.append("generate")
        self.prompts.append(prompt)
        return SimpleNamespace(
            prompt_tokens=TEMPLATE_TOKENS + len(prompt) // 4, prompt_tokens_exact=True,
            output_tokens=1, ttft_s=0.1, total_time_s=0.1, cached_tokens=0,
            queue_time_s=0.001, prefill_time_s=None, time_to_first_token_s=0.09,
            post_queue_time_s=0.09,
        )


class _FakeMonitor:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start_window(self, name: str) -> None:
        self.events.append(f"start:{name}")

    def stop_window(self, name: str) -> None:
        self.events.append(f"stop:{name}")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(benchmarks.prefill.time, "sleep", lambda s: None)
    monkeypatch.setattr(benchmarks.ttft_breakdown.time, "sleep", lambda s: None)


def _window_holds_no_tokenize(events: list[str]) -> None:
    start = next(i for i, e in enumerate(events) if e.startswith("start:"))
    stop = next(i for i, e in enumerate(events) if e.startswith("stop:"))
    assert "tokenize" in events[:start]
    assert "tokenize" not in events[start:stop]


def test_ttft_prompts_have_the_exact_server_length():
    events: list[str] = []
    client = _FakeClient(events)
    result = run_ttft_breakdown(client, prompt_lengths=[128, 512], repetitions=2,
                                gpu_monitor=_FakeMonitor(events))

    for length in (128, 512):
        entry = result["per_length"][str(length)]
        assert entry["status"] == "success"
        assert entry["actual_tokens"] == length
        assert entry["prompt_length_tolerance"] == 0
        assert {r["prompt_tokens"] for r in entry["per_request"]} == {length}
    assert "hello world" not in client.prompts[0]
    _window_holds_no_tokenize(events)


def test_ttft_calibration_error_opens_no_window():
    events: list[str] = []
    client = _FakeClient(events, fail_tokenize=True)
    result = run_ttft_breakdown(client, prompt_lengths=[128], repetitions=1,
                                gpu_monitor=_FakeMonitor(events))

    assert result["per_length"]["128"] == {"status": "request_error", "error": "tokenize failed"}
    assert not [e for e in events if e.startswith(("start:", "stop:"))]


def test_ttft_failed_result_keeps_the_error_field():
    failed = TtftRequestResult.failed(index=3, exc=RuntimeError("boom"))
    assert failed.error == "boom"
    assert not failed.success


def test_prefill_window_opens_after_calibration_and_closes():
    events: list[str] = []
    client = _FakeClient(events)
    result = run_prefill_scaling(client, [512], repetitions=2,
                                 gpu_monitor=_FakeMonitor(events), prefix_caching=False)

    assert events.count("start:length_512") == events.count("stop:length_512") == 1
    _window_holds_no_tokenize(events)
    config = result["config"]
    assert config["server_cache_settings"]["prefix_caching"] is False
    assert config["start_time"].endswith("+00:00")


def test_prefill_calibration_error_opens_no_window():
    events: list[str] = []
    client = _FakeClient(events, fail_tokenize=True)
    result = run_prefill_scaling(client, [512], repetitions=1, gpu_monitor=_FakeMonitor(events))

    assert result["per_length"]["512"]["status"] == "request_error"
    assert result["config"]["server_cache_settings"]["prefix_caching"] is None
    assert not [e for e in events if e.startswith(("start:", "stop:"))]


@pytest.mark.parametrize(("command", "expected"), [
    (["vllm", "serve", "m", "--enable-prefix-caching"], True),
    (["vllm", "serve", "m", "--no-enable-prefix-caching"], False),
    (["vllm", "serve", "m"], None),
    (["vllm", "serve", "m", "--enable-prefix-caching", "--no-enable-prefix-caching"], False),
])
def test_prefix_caching_flag(command, expected):
    assert prefix_caching_flag(command) is expected
