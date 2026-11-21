"""Tests for the model client: cache salt placement and vLLM metrics parsing."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

import harness.client
from benchmarks import latency
from harness.client import GenerationResult, ModelClient, _parse_server_metrics


class _FakeResponse:
    """Minimal stand-in for a streaming ``requests`` response."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")


def _sse(obj: dict[str, Any]) -> str:
    return "data: " + json.dumps(obj)


def _stream_with_metrics() -> list[str]:
    """A vLLM stream: two deltas, then the final usage chunk with metrics."""
    return [
        _sse({"choices": [{"delta": {"reasoning": "think"}}]}),
        _sse({"choices": [{"delta": {"content": "hello"}}]}),
        _sse({
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
            "metrics": {
                "time_to_first_token_ms": 250.0,
                "queue_time_ms": 50.0,
                "generation_time_ms": 400.0,
                "mean_itl_ms": 13.0,
                "tokens_per_second": 75.0,
            },
        }),
        "data: [DONE]",
    ]


def _patch_post(monkeypatch, lines: list[str], captured: list[dict]) -> None:
    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        captured.append(json)
        return _FakeResponse(lines)

    monkeypatch.setattr(harness.client.requests, "post", fake_post)


def test_generate_sends_cache_salt_at_top_level(monkeypatch):
    captured: list[dict] = []
    _patch_post(monkeypatch, _stream_with_metrics(), captured)
    client = ModelClient("http://127.0.0.1:8000", "m")

    client.generate("hi", max_tokens=4, cache_salt="abc123")

    payload = captured[0]
    assert payload["cache_salt"] == "abc123"
    assert "extra_body" not in payload
    assert payload["messages"][0]["content"] == "hi"


def test_generate_reads_vllm_metrics_and_cached_tokens(monkeypatch):
    captured: list[dict] = []
    _patch_post(monkeypatch, _stream_with_metrics(), captured)
    client = ModelClient("http://127.0.0.1:8000", "m")

    gen = client.generate("hi", max_tokens=4)

    assert gen.time_to_first_token_s == pytest.approx(0.25)
    assert gen.queue_time_s == pytest.approx(0.05)
    # vLLM TTFT starts after the queue, so post-queue time equals TTFT.
    assert gen.post_queue_time_s == pytest.approx(0.25)
    assert gen.generation_time_s == pytest.approx(0.40)
    assert gen.prefill_time_s is None
    assert gen.cached_tokens == 7
    assert gen.prompt_tokens == 12 and gen.prompt_tokens_exact
    assert gen.output_tokens == 2 and gen.output_tokens_exact
    assert gen.reasoning_text == "think" and gen.answer_text == "hello"
    assert gen.server_metrics["tokens_per_second"] == 75.0


def test_parse_server_metrics_does_not_subtract_queue_time():
    # A long queue must not change the post-queue time.
    parsed = _parse_server_metrics({"time_to_first_token_ms": 300.0, "queue_time_ms": 900.0})

    assert parsed["queue_time_s"] == pytest.approx(0.9)
    assert parsed["time_to_first_token_s"] == pytest.approx(0.3)
    assert parsed["post_queue_time_s"] == pytest.approx(0.3)


def test_ttft_breakdown_post_queue_equals_server_ttft(monkeypatch):
    from benchmarks import ttft_breakdown

    gen = GenerationResult(
        prompt_tokens=128, ttft_s=0.32, total_time_s=0.33, output_text="x",
        output_tokens=1, **_parse_server_metrics({"time_to_first_token_ms": 300.0, "queue_time_ms": 900.0}),
    )

    class _Client:
        def preflight_cache_salt(self) -> bool:
            return True

        def tokenize_prompt(self, prompt: str) -> Any:
            return harness.client.TokenizedPrompt(count=len(prompt) // 4, tokens=[])

        def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
            return gen

    monkeypatch.setattr(ttft_breakdown.time, "sleep", lambda s: None)
    result = ttft_breakdown.run_ttft_breakdown(_Client(), prompt_lengths=[128], repetitions=1)

    row = result["per_length"]["128"]["per_request"][0]
    assert row["scheduler_delay_ms"] == pytest.approx(900.0)
    assert row["post_queue_ms"] == pytest.approx(300.0)


def test_parse_server_metrics_without_object():
    parsed = _parse_server_metrics(None)
    assert all(value is None for value in parsed.values())


def test_latency_and_deep_context_use_fresh_salts():
    salts: list[str | None] = []

    class _Client:
        def generate(self, prompt, max_tokens=256, temperature=0.0, cache_salt=None):
            salts.append(cache_salt)
            return GenerationResult(
                prompt_tokens=10, ttft_s=0.1, total_time_s=0.2,
                output_text="x", output_tokens=1, prompt_tokens_exact=True,
            )

    latency.run_latency_sweep(_Client(), prompt_lengths=[32], repeats=3)
    latency.run_deep_context(_Client(), context_lengths=[64], repeats=2)

    assert len(salts) == 5
    assert all(salts)
    assert len(set(salts)) == 5


def _patch_timed_post(monkeypatch, lines: list[str]) -> None:
    """Serve *lines* with a fake clock that moves 0.1 s at each line.

    ``time.time`` returns 0.0 before the first line. A chunk read from line
    N thus has the time 0.1 * N, however often the client reads the clock.
    """
    clock = {"now": 0.0}

    class _TimedResponse(_FakeResponse):
        def iter_lines(self):
            for line in self._lines:
                clock["now"] += 0.1
                yield line.encode("utf-8")

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        return _TimedResponse(lines)

    monkeypatch.setattr(harness.client.requests, "post", fake_post)
    monkeypatch.setattr(harness.client.time, "time", lambda: clock["now"])


def _hidden_first_token_stream() -> list[str]:
    """A ``max_tokens=1`` stream from a reasoning parser that hides the token.

    The role chunk and the finish chunk have no text. Only the final usage
    chunk (line 3) shows that the server made one output token.
    """
    return [
        _sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        _sse({
            "choices": [],
            "usage": {"prompt_tokens": 512, "completion_tokens": 1},
            "metrics": {"time_to_first_token_ms": 200.0, "queue_time_ms": 10.0},
        }),
        "data: [DONE]",
    ]


def test_hidden_first_token_gives_finite_client_ttft(monkeypatch):
    _patch_timed_post(monkeypatch, _hidden_first_token_stream())

    gen = ModelClient("http://127.0.0.1:8000", "m").generate("hi", max_tokens=1)

    assert gen.ttft_s == pytest.approx(0.3)
    assert gen.output_text == ""
    assert gen.output_tokens == 1
    assert gen.per_token_times == []


def test_text_event_keeps_priority_over_usage_time(monkeypatch):
    # With per-chunk usage, every chunk counts tokens. The first text event
    # (line 2) sets the TTFT, as before. The usage of line 1 does not.
    lines = [
        _sse({"choices": [{"delta": {}}], "usage": {"completion_tokens": 1}}),
        _sse({"choices": [{"delta": {"content": "a"}}], "usage": {"completion_tokens": 2}}),
        _sse({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}),
        "data: [DONE]",
    ]
    _patch_timed_post(monkeypatch, lines)

    gen = ModelClient("http://127.0.0.1:8000", "m").generate("hi", continuous_usage=True)

    assert gen.ttft_s == pytest.approx(0.2)


def test_stream_without_output_token_keeps_nan_ttft(monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {}, "finish_reason": "abort"}]}),
        _sse({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 0}}),
        "data: [DONE]",
    ]
    _patch_timed_post(monkeypatch, lines)

    gen = ModelClient("http://127.0.0.1:8000", "m").generate("hi", max_tokens=1)

    assert math.isnan(gen.ttft_s)


def test_prefill_scaling_rate_with_hidden_first_token(monkeypatch):
    from benchmarks import prefill

    class _Client(ModelClient):
        def preflight_cache_salt(self) -> bool:
            return True

        def tokenize_prompt(self, prompt: str) -> Any:
            return harness.client.TokenizedPrompt(count=len(prompt) // 4, tokens=[])

    _patch_timed_post(monkeypatch, _hidden_first_token_stream())
    monkeypatch.setattr(prefill.time, "sleep", lambda s: None)

    result = prefill.run_prefill_scaling(
        _Client("http://127.0.0.1:8000", "m"), target_lengths=[512], repetitions=1
    )

    row = result["per_length"]["512"]["per_request"][0]
    assert row["client_ttft_s"] == pytest.approx(0.3)
    assert row["effective_prefill_tps"] == pytest.approx(512 / 0.3, rel=1e-3)
