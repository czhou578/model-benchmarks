"""Tests for the decode rates: per-chunk usage, the rate method, and the stop flag."""

from __future__ import annotations

import json
from typing import Any

import pytest

import dashboard
import harness.client
from benchmarks import decode
from harness.client import GenerationResult, ModelClient


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")


def _sse(obj: dict[str, Any]) -> str:
    return "data: " + json.dumps(obj)


def _patch(monkeypatch, lines: list[str], captured: list[dict]) -> None:
    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        captured.append(json)
        return _FakeResponse(lines)

    monkeypatch.setattr(harness.client.requests, "post", fake_post)
    # start, three text events 0.1 s apart, end.
    clock = iter([0.0, 1.0, 1.1, 1.2, 1.3])
    monkeypatch.setattr(harness.client.time, "time", lambda: next(clock))


def _usage(n: int) -> dict[str, int]:
    return {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}


def _mtp_stream() -> list[str]:
    """vLLM with continuous usage: each chunk holds the cumulative token count."""
    return [
        _sse({"choices": [{"delta": {"role": "assistant", "content": ""}}], "usage": _usage(0)}),
        _sse({"choices": [{"delta": {"reasoning": "a"}}], "usage": _usage(1)}),
        _sse({"choices": [{"delta": {"content": "bcd"}}], "usage": _usage(4)}),
        _sse({"choices": [{"delta": {"content": "e"}, "finish_reason": "length"}], "usage": _usage(5)}),
        _sse({"choices": [], "usage": _usage(5)}),
        "data: [DONE]",
    ]


def test_default_request_has_no_continuous_usage(monkeypatch):
    captured: list[dict] = []
    _patch(monkeypatch, _mtp_stream(), captured)
    ModelClient("http://x", "m").generate("hi", max_tokens=5)
    assert captured[0]["stream_options"] == {"include_usage": True}


def test_continuous_usage_gives_token_deltas_and_finish_reason(monkeypatch):
    captured: list[dict] = []
    _patch(monkeypatch, _mtp_stream(), captured)
    gen = ModelClient("http://x", "m").generate("hi", max_tokens=5, continuous_usage=True)

    assert captured[0]["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert gen.per_event_token_deltas == [3, 1]
    assert gen.per_token_times == pytest.approx([0.1, 0.1])
    assert gen.finish_reason == "length"
    assert gen.output_tokens == 5


def test_stream_without_per_chunk_usage_has_no_deltas(monkeypatch):
    # A server that ignores continuous_usage_stats: usage only at the end.
    lines = [
        _sse({"choices": [{"delta": {"content": "a"}}]}),
        _sse({"choices": [{"delta": {"content": "b"}}]}),
        _sse({"choices": [{"delta": {"content": "c"}, "finish_reason": "stop"}]}),
        _sse({"choices": [], "usage": _usage(3)}),
        "data: [DONE]",
    ]
    _patch(monkeypatch, lines, [])
    gen = ModelClient("http://x", "m").generate("hi", max_tokens=5, continuous_usage=True)
    assert gen.per_event_token_deltas is None
    assert gen.finish_reason == "stop"


def test_event_rates_methods():
    rates, method = decode.event_rates([0.1, 0.2], [3, 1])
    assert method == "token_delta" and rates == pytest.approx([30.0, 5.0])
    rates, method = decode.event_rates([0.1, 0.2], None)
    assert method == "event_gap" and rates == pytest.approx([10.0, 5.0])


class _Client:
    def __init__(self, gen: GenerationResult) -> None:
        self.gen = gen
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        self.calls.append(kwargs)
        return self.gen


def _gen(output_tokens: int, deltas: list[int] | None, finish: str) -> GenerationResult:
    return GenerationResult(
        prompt_tokens=128, ttft_s=0.5, total_time_s=1.5, output_text="x" * 10,
        output_tokens=output_tokens, answer_text="x" * 10,
        per_token_times=[0.1, 0.1, 0.2], per_event_token_deltas=deltas, finish_reason=finish,
    )


def test_run_decode_speed_salts_and_uses_token_rates():
    client = _Client(_gen(512, [3, 4, 2], "length"))
    row = decode.run_decode_speed(client, [512, 1024])

    salts = [c["cache_salt"] for c in client.calls]
    assert all(salts) and len(set(salts)) == 2
    assert all(c["continuous_usage"] for c in client.calls)
    first, second = row["512"], row["1024"]
    assert first["rate_method"] == "token_delta"
    assert first["tok_per_sec_peak"] == 40.0 and first["tok_per_sec_min"] == 10.0
    assert first["stopped_early"] is False and first["finish_reason"] == "length"
    assert second["stopped_early"] is True
    assert "output_truncated" not in first


def test_run_decode_speed_falls_back_to_event_gap():
    row = decode.run_decode_speed(_Client(_gen(512, None, "length")), [512])["512"]
    assert row["rate_method"] == "event_gap"
    assert row["tok_per_sec_peak"] == 10.0


def test_dashboard_reads_old_decode_rows_as_event_gap():
    rows = dashboard.decode_rows({"512": {"tok_per_sec_avg": 20.0, "tok_per_sec_peak": 9.0}})
    assert rows[0]["rate_method"] == "event_gap"
    rows = dashboard.decode_rows({"512": {"tok_per_sec_peak": 40.0, "rate_method": "token_delta"}})
    assert rows[0]["rate_method"] == "token_delta"
