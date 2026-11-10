"""Multi-turn tasks: the tool-call loop with mock tool results, and its scoring."""

from __future__ import annotations

import json
import uuid
from typing import Any

import requests

from benchmarks.tool_calling.mocks import _mock_tool_response
from benchmarks.tool_calling.protocol import (
    DEFAULT_REQUEST_TIMEOUT_S,
    _build_completion_url,
    _get_tool_schema,
    _parse_all_tool_calls,
    _request_error_details,
    _response_metadata,
)
from benchmarks.tool_calling.results import MultiTurnTaskResult, ToolCallTrace
from benchmarks.tool_calling.scoring import _score_params, _tools_match
from harness.client import ModelClient


def _run_multi_turn_task(
    client: ModelClient,
    task: dict,
    tools_def: list[dict],
    max_tokens: int = 4096,
    chat_template_kwargs: dict[str, Any] | None = None,
    temperature: float = 0.0,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> MultiTurnTaskResult:
    """Execute a multi-tool chain task.

    The loop runs up to ``max(expected_turns, 5)`` rounds:
      1. Send the conversation history (prompt + tool calls + responses).
      2. The model may return 0, 1, or multiple tool calls.
      3. Simulate tool responses via the mock executor.
      4. Feed responses back and repeat.

    Scoring compares the actual tool-call sequence against
    ``task["expected_sequence"]``.
    """

    task_id = task["id"]
    category = task.get("category", "unknown")
    prompt = task["prompt"]
    expected_sequence = task.get("expected_sequence", [])
    expected_intermediate = task.get("expected_intermediate_values", {})
    expected_turns = task.get("expected_turns", len(expected_sequence))
    max_turns = max(expected_turns, 5)  # safety cap

    # vLLM URL
    url = _build_completion_url(client)

    # Initialise the conversation with just the user prompt.
    messages = [{"role": "user", "content": prompt}]
    trace: list[ToolCallTrace] = []
    actual_sequence: list[dict[str, Any]] = []
    response: dict[str, Any] = {}
    tool_calls: list[tuple[str, dict[str, Any] | None]] = []

    # Track intermediate values that flow between turns.
    # e.g. {"step_1_result": 150, "step_2_input_from_step_1": 150}
    intermediate_results: dict[str, Any] = {}
    turn_finish_reasons: list[str | None] = []
    completion_tokens_total: int | None = None
    request_error: str | None = None

    for turn_idx in range(1, max_turns + 1):
        cache_salt = uuid.uuid4().hex
        payload = {
            "model": client.model_name,
            "messages": messages,
            "tools": tools_def,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        payload["cache_salt"] = cache_salt
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)

        try:
            resp = requests.post(
                url, headers=client.headers, json=payload, timeout=timeout_s
            )
            resp.raise_for_status()
            response = resp.json()
            finish_reason, completion_tokens = _response_metadata(response)
            turn_finish_reasons.append(finish_reason)
            if completion_tokens is not None:
                completion_tokens_total = (completion_tokens_total or 0) + completion_tokens
        except Exception as exc:
            request_error = _request_error_details(exc, timeout_s)
            trace.append(ToolCallTrace(
                step=turn_idx,
                tool_name=None,
                params=None,
                tool_response=None,
            ))
            break

        # Extract all tool calls the model wants to make this turn.
        tool_calls = _parse_all_tool_calls(response)

        if not tool_calls:
            # Model stopped calling tools.
            trace.append(ToolCallTrace(
                step=turn_idx,
                tool_name=None,
                params=None,
                tool_response=None,
            ))
            break

        # Execute each call, then append the history in OpenAI order: the
        # assistant tool_calls message first, then one tool message per call
        # with a matching id. The chat template renders messages in list
        # order, so a tool message before its call shows the model the
        # result before the request.
        tool_call_entries: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        for i, (call_name, call_params) in enumerate(tool_calls):
            tool_call_id = f"call_{turn_idx}_{i + 1}"
            mock_resp = _mock_tool_response(call_name, call_params or {})
            step_key = f"step_{turn_idx}_result"
            intermediate_results[step_key] = mock_resp

            trace.append(ToolCallTrace(
                step=turn_idx,
                tool_name=call_name,
                params=call_params,
                tool_response=mock_resp,
            ))
            actual_sequence.append({
                "step": turn_idx,
                "tool": call_name,
                "params": call_params,
            })
            tool_call_entries.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": call_name,
                    "arguments": json.dumps(call_params or {}),
                },
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(mock_resp),
            })

        messages.append({
            "role": "assistant",
            "tool_calls": tool_call_entries,
        })
        messages.extend(tool_messages)

    # --- Scoring ---
    orchestration_correct = True
    data_flow_correct = True
    turns_correct = True
    parallel_detected = len(tool_calls) > 1 if tool_calls else False

    if not expected_sequence:
        # Expected no tool calls.
        orchestration_correct = len(actual_sequence) == 0
        turns_correct = orchestration_correct
    else:
        # Check that the tool sequence matches.
        actual_tools = [c["tool"] for c in actual_sequence]
        expected_tools = [c["tool"] for c in expected_sequence]

        # The actual sequence must start with the expected sequence
        # (the model may call extra tools after the expected ones).
        if actual_tools[:len(expected_tools)] != expected_tools:
            orchestration_correct = False

        # Check intermediate value flow.
        if expected_intermediate:
            # The expected_intermediate map contains keys like
            # "step_1_result" and "step_2_input_from_step_1".
            for key in expected_intermediate:
                if key.endswith("_result"):
                    step_num = int(key.split("_")[1])
                    actual_result = intermediate_results.get(key)
                    if actual_result is not None:
                        # The model's next step should use this value.
                        next_step_key = f"step_{step_num + 1}_input_from_step_{step_num}"
                        if next_step_key in expected_intermediate:
                            next_input = expected_intermediate[next_step_key]
                            actual_next: dict[str, Any] = next(
                                (
                                    c["params"]
                                    for c in actual_sequence
                                    if c["step"] == step_num + 1
                                ),
                                {},
                            )
                            if actual_next:
                                # Check if the expression contains the expected number.
                                expr = actual_next.get("expression", "")
                                if str(next_input) not in str(expr):
                                    data_flow_correct = False

        # Check turn count.
        if len(actual_sequence) != expected_turns:
            turns_correct = False

    # Build composite score for multi-tool tasks.
    multi_score = int(orchestration_correct) * int(data_flow_correct) * int(turns_correct)

    # Single-tool baseline: need both orchestration and data flow to pass.
    single_correct = orchestration_correct and data_flow_correct

    details = ""
    if not orchestration_correct:
        expected_display = [c["tool"] for c in expected_sequence]
        actual_display = [c["tool"] for c in actual_sequence]
        details = f"Tool sequence mismatch. Expected: {expected_display}, Got: {actual_display}"
    elif not data_flow_correct:
        details = "Model did not use previous tool output as input"
    elif not turns_correct:
        details = f"Expected {expected_turns} turns, got {len(actual_sequence)}"
    if request_error:
        # A failed request ends the loop early; never score it as an answer.
        details = request_error

    # For single-tool tasks within multi-tool category, also compute the
    # legacy fields so that old scoring logic still works.
    first_tool = actual_sequence[0]["tool"] if actual_sequence else None
    first_params = actual_sequence[0]["params"] if actual_sequence else None
    expected_tool = expected_sequence[0]["tool"] if expected_sequence else None
    first_tool_schema = _get_tool_schema(tools_def, expected_tool or "")
    params_complete, params_valid = _score_params(first_params, first_tool_schema)
    tool_correct = _tools_match(expected_tool, first_tool)
    composite = int(tool_correct) * params_complete * int(params_valid)
    correct = single_correct

    # ── Error-recovery scoring ─────────────────────────────────────────────
    recovery_ok = True
    self_corrected = True
    recovery_details = ""

    if task.get("category") == "error_recovery":
        # Check if model made at least as many calls as expected
        if len(actual_sequence) != expected_turns:
            recovery_ok = False
            recovery_details = f"expected {expected_turns} turns, got {len(actual_sequence)}"

        # Check if the model's last call had non-empty, valid params
        if actual_sequence:
            last_params = actual_sequence[-1].get("params", {}) or {}
            for _, v in last_params.items():
                if isinstance(v, str) and not v.strip():
                    self_corrected = False
                    recovery_ok = False
                    recovery_details += "; last call has empty param"

        if not recovery_ok or not self_corrected:
            correct = False

    if request_error:
        correct = False

    result = MultiTurnTaskResult(
        task_id=task_id,
        category=category,
        prompt=prompt,
        expected_tool=expected_tool,
        actual_tool=first_tool,
        tool_call_id=None,
        actual_params=first_params,
        correct=correct,
        tool_correct=tool_correct,
        params_complete=round(params_complete, 4),
        params_valid=params_valid,
        composite_score=round(composite, 4),
        details=details,
        raw_response=response,
        finish_reason=turn_finish_reasons[-1] if turn_finish_reasons else None,
        completion_tokens=completion_tokens_total,
        trace=trace,
        turn_finish_reasons=turn_finish_reasons,
        orchestration_correct=orchestration_correct,
        data_flow_correct=data_flow_correct,
        turns_correct=turns_correct,
        parallel_detected=parallel_detected,
        multi_score=round(multi_score, 4),
    )
    return result
