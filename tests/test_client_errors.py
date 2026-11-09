"""Tests for client HTTP errors, the cache_salt mode, and completion-mode payloads."""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

import core_runner
from core_runner import GenerationResult, ModelClient, ServerHTTPError


class _Resp:
    """A stand-in for a ``requests`` response, streaming or not."""

    def __init__(self, status_code: int = 200, body: str = "", lines: list[str] | None = None) -> None:
        self.status_code = status_code
        self.text = body
        self._lines = lines or []

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")


def _ok_stream() -> _Resp:
    chunk = {"choices": [{"delta": {"content": "ok"}, "text": "ok"}],
             "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    return _Resp(lines=["data: " + json.dumps(chunk), "data: [DONE]"])


def _patch_post(monkeypatch, responses: list[_Resp], captured: list[tuple[str, dict]]) -> None:
    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        captured.append((url, json))
        return responses.pop(0)

    monkeypatch.setattr(core_runner.requests, "post", fake_post)


def test_http_400_raises_with_status_and_body(monkeypatch):
    body = json.dumps({"error": {"message": "This model's maximum context length is 4096 tokens."}})
    captured: list[tuple[str, dict]] = []
    _patch_post(monkeypatch, [_Resp(400, body)], captured)
    client = ModelClient("http://127.0.0.1:8000", "m")

    with pytest.raises(ServerHTTPError) as info:
        client.generate("hi", cache_salt="abc")

    message = str(info.value)
    assert "400" in message and "maximum context length is 4096" in message
    assert isinstance(info.value, requests.HTTPError)
    # A 400 that does not name cache_salt sends no text-salt retry.
    assert len(captured) == 1
    assert client.cache_salt_supported is None


def test_error_body_is_truncated(monkeypatch):
    _patch_post(monkeypatch, [_Resp(500, "x" * 5000)], [])
    client = ModelClient("http://127.0.0.1:8000", "m")

    with pytest.raises(ServerHTTPError) as info:
        client.generate("hi")

    assert len(info.value.body) == core_runner.MAX_ERROR_BODY_CHARS


def test_cache_salt_fallback_when_server_rejects_field(monkeypatch):
    reject = json.dumps({"detail": [{"loc": ["body", "cache_salt"], "msg": "Extra inputs are not permitted"}]})
    captured: list[tuple[str, dict]] = []
    _patch_post(monkeypatch, [_Resp(400, reject), _ok_stream(), _ok_stream()], captured)
    client = ModelClient("http://127.0.0.1:8000", "m")

    first = client.generate("hi", cache_salt="s1")
    second = client.generate("hi", cache_salt="s2")

    assert first.answer_text == "ok" and second.answer_text == "ok"
    assert client.cache_salt_supported is False
    payloads = [p for _, p in captured]
    assert payloads[0]["cache_salt"] == "s1"
    assert "cache_salt" not in payloads[1]
    assert payloads[1]["messages"][0]["content"] == "hi salt=s1"
    # The mode is decided once: the second call goes straight to the text salt.
    assert len(payloads) == 3
    assert "cache_salt" not in payloads[2]
    assert payloads[2]["messages"][0]["content"] == "hi salt=s2"


def test_known_cache_salt_mode_does_not_fall_back(monkeypatch):
    reject = json.dumps({"error": "cache_salt is bad"})
    _patch_post(monkeypatch, [_ok_stream(), _Resp(400, reject)], [])
    client = ModelClient("http://127.0.0.1:8000", "m")

    client.generate("hi", cache_salt="s1")
    assert client.cache_salt_supported is True
    with pytest.raises(ServerHTTPError):
        client.generate("hi", cache_salt="s2")


def test_preflight_sets_cache_salt_mode(monkeypatch):
    captured: list[tuple[str, dict]] = []
    _patch_post(monkeypatch, [_Resp(400, "no"), _ok_stream()], captured)
    client = ModelClient("http://127.0.0.1:8000", "m", chat=False)

    assert client.preflight_cache_salt() is False
    client.generate("hi", cache_salt="s1")

    assert captured[0][1]["prompt"] == "hi" and "messages" not in captured[0][1]
    assert captured[1][1]["prompt"] == "hi salt=s1"
    assert "cache_salt" not in captured[1][1]


def test_completion_mode_warmup_sends_prompt(monkeypatch):
    monkeypatch.setattr(core_runner.requests, "get", lambda *a, **k: _Resp(200))
    captured: list[tuple[str, dict]] = []
    _patch_post(monkeypatch, [_Resp(200)], captured)
    client = ModelClient("http://127.0.0.1:8000", "m", chat=False)

    client.wait_until_ready(timeout_s=5, poll_s=0)

    url, payload = captured[0]
    assert url.endswith("/v1/completions")
    assert payload["prompt"] == "hi"
    assert "messages" not in payload


def test_completion_mode_generate_sends_prompt(monkeypatch):
    captured: list[tuple[str, dict]] = []
    _patch_post(monkeypatch, [_ok_stream()], captured)
    client = ModelClient("http://127.0.0.1:8000", "m", chat=False)

    gen = client.generate("hello", cache_salt="s1")

    url, payload = captured[0]
    assert url.endswith("/v1/completions")
    assert payload["prompt"] == "hello" and "messages" not in payload
    assert gen.answer_text == "ok"


def test_deep_context_records_failure_and_runs_next_length():
    calls: list[int] = []

    class _Client:
        def generate(self, prompt, max_tokens=256, temperature=0.0, cache_salt=None):
            calls.append(len(prompt))
            if len(calls) == 1:
                raise RuntimeError("HTTP 400: prompt is longer than max-model-len")
            return GenerationResult(
                prompt_tokens=10, ttft_s=0.1, total_time_s=0.2,
                output_text="x", output_tokens=1, prompt_tokens_exact=True,
            )

    result = core_runner.run_deep_context(_Client(), context_lengths=[256, 32], repeats=2)

    assert result["256"]["status"] == "request_error"
    assert "max-model-len" in result["256"]["error"]
    assert result["256"]["n"] == 0
    assert result["32"]["status"] == "success"
    assert "error" not in result["32"]
    assert result["32"]["n"] == 2
    assert len(calls) == 3


def test_deep_context_marks_oom_and_does_not_swallow_interrupt():
    class _OomClient:
        def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
            raise requests.exceptions.HTTPError("HTTP 500: CUDA out of memory")

    result = core_runner.run_deep_context(_OomClient(), context_lengths=[32], repeats=1)
    assert result["32"]["status"] == "oom"
    assert "out of memory" in result["32"]["error"]

    class _StopClient:
        def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        core_runner.run_deep_context(_StopClient(), context_lengths=[32], repeats=1)
