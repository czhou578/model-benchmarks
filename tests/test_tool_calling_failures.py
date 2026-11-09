"""Tests for tool-calling timeouts, request settings, the currency mock, and task paths."""

from __future__ import annotations

import requests

import benchmarks.tool_calling as tc
from harness.client import ModelClient
from harness.paths import ROOT

_TOOLS = tc._build_tools_definition([{
    "name": "get_stock_price",
    "description": "Get the stock price",
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}])
_SINGLE = {
    "id": "t", "category": "stock_basic", "prompt": "p",
    "expected": {"tool": "get_stock_price", "params": {"symbol": "NVDA"}},
}
_MULTI = {
    "id": "m", "category": "multi_tool", "prompt": "p", "expected_turns": 1,
    "expected_sequence": [{"step": 1, "tool": "get_stock_price", "params": {"symbol": "NVDA"}}],
}


class _RecordingPost:
    """Records the timeout and body of each call, then returns one response or raises."""

    def __init__(self, response: dict | None = None, exc: Exception | None = None) -> None:
        self.response = response or {"choices": [{"finish_reason": "stop",
                                                  "message": {"content": "hi"}}]}
        self.exc = exc
        self.calls: list[dict] = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"timeout": timeout, "json": json})
        if self.exc:
            raise self.exc
        return self

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.response


def _client() -> ModelClient:
    return ModelClient("http://127.0.0.1:8000", "m")


def test_currency_mock_uses_amount_from_call():
    usd_jpy = {"from_currency": "USD", "to_currency": "JPY"}
    assert tc._mock_tool_response("convert_currency", {**usd_jpy, "amount": 100})["result"] == 100 * 149.5
    assert tc._mock_tool_response("convert_currency", {**usd_jpy, "amount": 230})["result"] == 230 * 149.5
    gbp_eur = tc._mock_tool_response(
        "convert_currency", {"from_currency": "GBP", "to_currency": "EUR", "amount": 250})
    assert gbp_eur == {"amount": 250.0, "rate": 1.18, "result": 295.0}
    assert "error" in tc._mock_tool_response("convert_currency", {**usd_jpy, "amount": "x"})
    assert "error" in tc._mock_tool_response(
        "convert_currency", {"from_currency": "JPY", "to_currency": "CHF", "amount": 1})


def test_single_task_timeout_is_a_visible_failure(monkeypatch):
    post = _RecordingPost(exc=requests.exceptions.ReadTimeout("Read timed out. (read timeout=7)"))
    monkeypatch.setattr(tc.requests, "post", post)

    result = tc._run_task(_client(), _SINGLE, _TOOLS, timeout_s=7)

    assert result.correct is False
    assert result.details.startswith("Request timed out after 7 s")
    agg = tc._aggregate([result], {})
    assert agg.failure_modes["request_timeout"] == 1
    assert agg.failure_modes["request_failed"] == 0


def test_multi_turn_timeout_is_a_visible_failure(monkeypatch):
    post = _RecordingPost(exc=requests.exceptions.ReadTimeout("Read timed out."))
    monkeypatch.setattr(tc.requests, "post", post)

    result = tc._run_multi_turn_task(_client(), _MULTI, _TOOLS, timeout_s=9)

    assert result.correct is False
    assert result.details.startswith("Request timed out after 9 s")
    assert tc._aggregate([result], {}).failure_modes["request_timeout"] == 1


def test_other_request_error_counts_as_request_failed(monkeypatch):
    post = _RecordingPost(exc=requests.exceptions.ConnectionError("refused"))
    monkeypatch.setattr(tc.requests, "post", post)

    result = tc._run_task(_client(), _SINGLE, _TOOLS)

    assert result.details == "Request failed: refused"
    modes = tc._aggregate([result], {}).failure_modes
    assert modes["request_failed"] == 1 and modes["request_timeout"] == 0


def test_benchmark_passes_timeout_and_temperature(monkeypatch, tmp_path):
    post = _RecordingPost()
    monkeypatch.setattr(tc.requests, "post", post)
    monkeypatch.setattr(tc.time, "sleep", lambda _s: None)
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text(
        "tasks:\n"
        "  - {id: a, category: stock_basic, prompt: p, expected: {tool: null}}\n"
        "  - {id: b, category: multi_tool, prompt: p, expected: {tool: null},\n"
        "     expected_sequence: [], expected_turns: 0}\n"
    )

    out = tc.run_tool_calling_benchmark(
        _client(), tasks_path=tasks, temperature=0.3, request_timeout_s=42.0,
    )

    assert len(post.calls) == 2
    assert all(c["timeout"] == 42.0 for c in post.calls)
    assert all(c["json"]["temperature"] == 0.3 for c in post.calls)
    assert out["config"]["request_timeout_s"] == 42.0
    assert out["config"]["temperature"] == 0.3


def test_default_timeout_covers_a_full_answer_at_slow_decode():
    # 4096 tokens at 12.3 tok/s (Qwen3.8-27B) take about 333 s.
    assert tc.DEFAULT_REQUEST_TIMEOUT_S >= 4096 / 12.3


def test_relative_tasks_path_resolves_from_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    resolved = tc.resolve_tasks_path("datasets/tool_calling_tasks.yaml")

    assert resolved == ROOT / "datasets/tool_calling_tasks.yaml"
    assert resolved.exists()
    local = tmp_path / "local.yaml"
    local.write_text("tasks: []\n")
    assert tc.resolve_tasks_path("local.yaml") == local.relative_to(tmp_path)
