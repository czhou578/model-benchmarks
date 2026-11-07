"""Tests for tool-calling request construction and score aggregation."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

import benchmarks.tool_calling as tc
from core_runner import ModelClient

_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a symbol",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def _tool_call_response(name: str, args: dict[str, Any], completion_tokens: int = 40) -> dict:
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "srv",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            },
        }],
        "usage": {"completion_tokens": completion_tokens},
    }


def _text_response(text: str | None, finish_reason: str = "stop", completion_tokens: int = 12) -> dict:
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": text},
        }],
        "usage": {"completion_tokens": completion_tokens},
    }


class _FakeHttp:
    """Replays canned responses and records every request payload."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._current: dict = {}
        self.payloads: list[dict] = []
        self.status_code = 200

    def post(self, url, headers=None, json=None, timeout=None):
        # Snapshot the body: requests serializes it at call time, while the
        # multi-turn loop keeps mutating the same ``messages`` list.
        self.payloads.append(copy.deepcopy(json))
        self._current = self._responses.pop(0)
        return self

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._current


def _client() -> ModelClient:
    return ModelClient("http://127.0.0.1:8000", "m")


def test_multi_turn_history_puts_assistant_call_before_tool_result(monkeypatch):
    http = _FakeHttp([
        _tool_call_response("get_stock_price", {"symbol": "NVDA"}),
        _tool_call_response("calculate", {"expression": "150 * 1.2"}),
        _text_response("Yes, a 20% increase gives 180."),
    ])
    monkeypatch.setattr(tc.requests, "post", http.post)
    task = {
        "id": "chain", "category": "multi_tool", "prompt": "p", "expected_turns": 2,
        "expected_sequence": [
            {"step": 1, "tool": "get_stock_price", "params": {"symbol": "NVDA"}},
            {"step": 2, "tool": "calculate", "params": {"expression": "150 * 1.2"}},
        ],
        "expected_intermediate_values": {"step_1_result": 150, "step_2_input_from_step_1": 150},
    }

    result = tc._run_multi_turn_task(
        _client(), task, tc._build_tools_definition(_TOOLS), max_tokens=64,
    )

    second_turn = http.payloads[1]["messages"]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "tool"]
    assert second_turn[2]["tool_call_id"] == second_turn[1]["tool_calls"][0]["id"]
    third_turn = http.payloads[2]["messages"]
    assert [m["role"] for m in third_turn] == ["user", "assistant", "tool", "assistant", "tool"]
    assert third_turn[4]["tool_call_id"] == third_turn[3]["tool_calls"][0]["id"]
    assert third_turn[3]["tool_calls"][0]["id"] != second_turn[1]["tool_calls"][0]["id"]
    assert result.correct and result.orchestration_correct and result.turns_correct
    assert result.turn_finish_reasons == ["tool_calls", "tool_calls", "stop"]
    assert result.completion_tokens == 92


def test_run_task_records_truncation_and_template_kwargs(monkeypatch):
    http = _FakeHttp([_text_response(None, finish_reason="length", completion_tokens=64)])
    monkeypatch.setattr(tc.requests, "post", http.post)
    task = {
        "id": "t", "category": "stock_basic", "prompt": "p",
        "expected": {"tool": "get_stock_price", "params": {"symbol": "NVDA"}},
    }

    result = tc._run_task(
        _client(), task, tc._build_tools_definition(_TOOLS), max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert result.correct is False
    assert result.finish_reason == "length" and result.completion_tokens == 64
    assert "max_tokens" in result.details
    payload = http.payloads[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 64
    assert payload["cache_salt"] and "extra_body" not in payload


def _single(task_id: str, category: str, *, correct: bool, tool_correct: bool,
            complete: float = 1.0, valid: bool = True, finish: str | None = None) -> tc.TaskResult:
    return tc.TaskResult(
        task_id=task_id, category=category, prompt="p", expected_tool="search",
        actual_tool="search" if tool_correct else None, tool_call_id=None,
        actual_params={"query": "q"} if tool_correct else None, correct=correct,
        tool_correct=tool_correct, params_complete=complete, params_valid=valid,
        composite_score=float(tool_correct) * complete * float(valid), finish_reason=finish,
    )


def test_single_tool_rollup_feeds_composite_and_failure_modes():
    refusal = tc.TaskResult(
        task_id="r", category="refusal", prompt="p", expected_tool=None, actual_tool=None,
        tool_call_id=None, actual_params=None, correct=True, tool_correct=True,
        params_complete=1.0, params_valid=True, composite_score=1.0,
    )
    results = [
        _single("a", "search_basic", correct=True, tool_correct=True),
        _single("b", "search_basic", correct=False, tool_correct=False,
                complete=0.0, valid=False, finish="length"),
        _single("c", "weather_enum", correct=True, tool_correct=True),
        _single("d", "email_basic", correct=False, tool_correct=True, complete=0.5),
        refusal,
    ]

    agg = tc._aggregate(results, config={})

    single = agg.category_scores["single_tool"]
    assert single["total"] == 4
    assert single["source_categories"] == ["email_basic", "search_basic", "weather_enum"]
    assert single["tool_accuracy"] == pytest.approx(0.75)
    assert single["param_completeness_avg"] == pytest.approx(0.625)
    components = agg.composite_components
    assert components["tool_accuracy"] == pytest.approx(0.75)
    assert components["param_completeness"] == pytest.approx(0.625)
    assert components["refusal"] == pytest.approx(1.0)
    assert agg.failure_modes["truncated_output"] == 1
    assert agg.failure_modes["no_tool_call"] == 1
    assert agg.failure_modes["missing_required_param"] == 1
    assert agg.per_task[1]["finish_reason"] == "length"
    assert agg.per_task[1]["completion_tokens"] is None
