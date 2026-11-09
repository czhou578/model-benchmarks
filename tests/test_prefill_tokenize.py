"""Tests for ModelClient.tokenize_prompt, which the prefill benchmark needs."""

from __future__ import annotations

from typing import Any

import harness.client
from benchmarks.prefill import prepare_exact_prompt
from harness.client import ModelClient

TEMPLATE_TOKENS = 12


class _FakeJsonResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


def _patch_tokenize(monkeypatch, captured: list[dict], data=None) -> None:
    """Fake /tokenize: the chat template adds 12 tokens, text costs 4 chars per token."""

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured.append({"url": url, "json": json})
        if data is not None:
            return _FakeJsonResponse(data)
        text = json["messages"][0]["content"] if "messages" in json else json["prompt"]
        return _FakeJsonResponse({"count": TEMPLATE_TOKENS + len(text) // 4, "tokens": []})

    monkeypatch.setattr(harness.client.requests, "post", fake_post)


def test_tokenize_prompt_renders_chat_template(monkeypatch):
    captured: list[dict] = []
    _patch_tokenize(monkeypatch, captured, data={"count": 42, "tokens": [1, 2]})
    client = ModelClient("http://127.0.0.1:8000/", "org/model", chat=True)

    result = client.tokenize_prompt("hello")

    assert result.count == 42
    assert captured[0]["url"] == "http://127.0.0.1:8000/tokenize"
    assert captured[0]["json"] == {
        "model": "org/model",
        "messages": [{"role": "user", "content": "hello"}],
        "add_generation_prompt": True,
    }


def test_tokenize_prompt_completion_mode(monkeypatch):
    captured: list[dict] = []
    _patch_tokenize(monkeypatch, captured, data={"tokens": [5, 6, 7]})
    client = ModelClient("http://127.0.0.1:8000", "org/model", chat=False)

    result = client.tokenize_prompt("hello")

    assert result.count == 3
    assert captured[0]["json"] == {
        "model": "org/model",
        "prompt": "hello",
        "add_special_tokens": True,
    }


def test_prepare_exact_prompt_works_with_model_client(monkeypatch):
    # Before the fix, ModelClient had no tokenize_prompt, and every prefill
    # length failed with AttributeError before it sent a request.
    captured: list[dict] = []
    _patch_tokenize(monkeypatch, captured)
    client = ModelClient("http://127.0.0.1:8000", "org/model", chat=True)

    prompt = prepare_exact_prompt(client, 512)

    assert prompt.exact
    assert prompt.actual_tokens == 512
    assert TEMPLATE_TOKENS + len(prompt.text) // 4 == 512
