"""Tool-calling benchmark.

The default task file has 85 tasks in these groups:

Single-tool invocation
  One-shot tool calls. Tests tool selection and parameter validation.
  The group also holds the ``ambiguous`` and ``refusal`` tasks, where the
  correct answer can be a question or no tool call.

Multi-tool chaining (``multi_tool``)
  Multi-turn chains where the model must orchestrate multiple tools.
  The output of one tool feeds into the next.

Schema compliance (``schema_compliance``)
  Strict JSON Schema constraint validation (minLength, maxLength, pattern,
  enum, required, nested objects).

Error recovery (``error_recovery``)
  Multi-turn tasks. The model must correct its call after a tool error.

The composite score is the plain sum of six components, from 0 to 6
(``_compute_weighted_composite``). docs/tool-calling.md describes the scoring.

Input : ``datasets/tool_calling_tasks.yaml``
Output: ``tool_calling.json`` inside the run directory. The standalone CLI
writes ``results/tool_calling.json`` unless --output names another file.

Usage::

    python -m benchmarks.tool_calling --tasks datasets/tool_calling_tasks.yaml --base-url http://localhost:8000/v1 --model <model-name>

Or import ``run_tool_calling_benchmark(client, tasks_path=...)`` from benchmarks.tool_calling.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmarks.tool_calling.mocks import _mock_tool_response
from benchmarks.tool_calling.protocol import (
    DEFAULT_REQUEST_TIMEOUT_S,
    _build_completion_url,
    _build_tools_definition,
    _get_tool_schema,
    _parse_all_tool_calls,
    _request_error_details,
    _response_metadata,
)
from benchmarks.tool_calling.results import (
    BenchmarkResult,
    MultiTurnTaskResult,
    TaskResult,
    ToolCallTrace,
)
from benchmarks.tool_calling.scoring import _score_params, _tools_match
from benchmarks.tool_calling.single_turn import _attach_overlay_defaults, _run_task
from benchmarks.tool_calling.tasks import load_tasks, resolve_tasks_path
from harness.client import ModelClient
from harness.paths import ROOT


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
                            actual_next = next(
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


# --------------------------------------------------------------------------- #
# Weighted composite score
# --------------------------------------------------------------------------- #

# Categories with their own scoring. Every other category is a single-tool
# category and rolls up into ``category_scores["single_tool"]``.
_NON_SINGLE_TOOL_CATEGORIES = frozenset({
    "multi_tool", "error_recovery", "schema_compliance", "ambiguous", "refusal",
})

# The six components of the composite score. Each key maps to the category
# and the ``category_scores`` field that gives the component value. The
# composite applies no weights: it is the sum of the six values.
_WEIGHTS: dict[str, tuple[str, str]] = {
    "tool_accuracy": ("single_tool", "tool_accuracy"),
    "param_completeness": ("single_tool", "param_completeness_avg"),
    "param_correctness": ("single_tool", "param_correctness_avg"),
    "multi_tool": ("multi_tool", "multi_tool_score_avg"),
    "schema_compliance": ("schema_compliance", "strict_compliance"),
    "refusal": ("refusal", "correct_refusal"),
}


def _compute_weighted_composite(
    category_scores: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Compute the composite score: the plain sum of the six components.

    Each component is a ``category_scores`` field with a value from 0 to 1,
    so the composite is from 0 to 6. The function applies no weights. When
    the field is missing, the component is the ``pass_rate`` of the
    category. When the category is missing, the component is 0.

    Returns:
        ``(composite, components)`` where *components* maps each component
        name to its value.
    """
    components: dict[str, float] = {}
    composite = 0.0
    for key, (cat, score_key) in _WEIGHTS.items():
        score = category_scores.get(cat, {}).get(score_key)
        if score is None:
            # Fallback: use pass_rate if the category exists but the
            # specific field is missing.  If the category is entirely
            # absent, the component is 0.0.
            fallback = category_scores.get(cat, {}).get("pass_rate")
            score = fallback if fallback is not None else 0.0
        components[key] = round(float(score), 4)
        composite += components[key]

    composite = round(composite, 4)
    return composite, components


def _aggregate(results: list[TaskResult], config: dict) -> BenchmarkResult:
    """Compute category scores, failure modes, and the composite score."""

    # Category-level scores
    categories: dict[str, list[TaskResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_scores: dict[str, dict[str, Any]] = {}
    for cat, cat_results in sorted(categories.items()):
        total = len(cat_results)
        correct = sum(1 for r in cat_results if r.correct)
        tool_correct_count = sum(1 for r in cat_results if r.tool_correct)
        avg_completeness = (
            round(statistics.mean(r.params_complete for r in cat_results), 4)
            if cat_results else None
        )
        avg_validity = sum(1 for r in cat_results if r.params_valid) / total if total else 0
        avg_composite = round(statistics.mean(r.composite_score for r in cat_results), 4)

        cat_score: dict[str, Any] = {
            "pass_count": correct,
            "total": total,
            "pass_rate": round(correct / total, 4) if total else None,
            "tool_accuracy": round(tool_correct_count / total, 4) if total else None,
            "param_completeness_avg": avg_completeness,
            "param_correctness_avg": round(avg_validity, 4),
            "composite_avg": avg_composite,
        }

        # Add multi-tool-specific fields for the multi_tool category.
        if cat == "multi_tool":
            orchestration_pass = sum(
                1 for r in cat_results
                if isinstance(r, MultiTurnTaskResult) and r.orchestration_correct
            )
            data_flow_pass = sum(
                1 for r in cat_results
                if isinstance(r, MultiTurnTaskResult) and r.data_flow_correct
            )
            turns_pass = sum(
                1 for r in cat_results
                if isinstance(r, MultiTurnTaskResult) and r.turns_correct
            )
            multi_scores = [
                r.multi_score for r in cat_results
                if isinstance(r, MultiTurnTaskResult)
            ]
            cat_score.update({
                "orchestration_accuracy": round(
                    orchestration_pass / total, 4
                ) if total else None,
                "data_flow_correct": round(
                    data_flow_pass / total, 4
                ) if total else None,
                "turns_correct": round(
                    turns_pass / total, 4
                ) if total else None,
                "multi_tool_score_avg": round(
                    statistics.mean(multi_scores), 4
                ) if multi_scores else None,
                "avg_chain_length": (
                    round(
                        statistics.mean([len(getattr(r, "trace", [])) for r in cat_results]),
                        2,
                    ) if cat_results else None
                ),
            })

        # Add ambiguous-task-specific fields for the ambiguous category.
        if cat == "ambiguous":
            correct_default = sum(1 for r in cat_results if r.correct)
            no_halluc = sum(
                1 for r in cat_results
                if getattr(r, "no_hallucinated_params", True)
            )
            # A clarifying response is one where the model didn't hallucinate
            # params and either clarified (underspecified) or made a valid default.
            asked_clarification = sum(
                1 for r in cat_results
                if getattr(r, "ambig_strategy", None) == "underspecified"
                and r.actual_tool is None
                and getattr(r, "ambig_details", "") != "no clarification"
            )
            cat_score.update({
                "correct_refusal_or_default": round(
                    correct_default / total, 4
                ) if total else None,
                "no_hallucinated_params": round(
                    no_halluc / total, 4
                ) if total else None,
                "clarification_rate": round(
                    asked_clarification / total, 4
                ) if total else None,
            })

        # A correct refusal is a reply with no tool call. A truncated reply
        # or a failed request also has no tool call, but it is no refusal.
        if cat == "refusal":
            refused = sum(
                1 for r in cat_results
                if r.actual_tool is None and r.tool_correct
                and r.finish_reason != "length"
            )
            cat_score["correct_refusal"] = round(refused / total, 4) if total else None

        # Schema-compliance-specific fields.
        if cat == "schema_compliance":
            total_compliant = sum(1 for r in cat_results if r.correct)
            no_halluc = sum(
                1 for r in cat_results
                if getattr(r, "no_hallucinated_params", True)
            )
            req_present = sum(
                1 for r in cat_results
                if getattr(r, "schema_required_present", True)
            )
            length_ok = sum(
                1 for r in cat_results
                if getattr(r, "schema_length_valid", True)
            )
            enum_ok = sum(
                1 for r in cat_results
                if getattr(r, "schema_enum_valid", True)
            )
            pattern_ok = sum(
                1 for r in cat_results
                if getattr(r, "schema_pattern_valid", True)
            )
            nested_ok = sum(
                1 for r in cat_results
                if getattr(r, "schema_nested_valid", True)
            )
            cat_score.update({
                "strict_compliance": round(
                    total_compliant / total, 4
                ) if total else None,
                "no_hallucinated_params": round(
                    no_halluc / total, 4
                ) if total else None,
                "required_present": round(
                    req_present / total, 4
                ) if total else None,
                "length_valid": round(
                    length_ok / total, 4
                ) if total else None,
                "enum_valid": round(
                    enum_ok / total, 4
                ) if total else None,
                "pattern_valid": round(
                    pattern_ok / total, 4
                ) if total else None,
                "nested_valid": round(
                    nested_ok / total, 4
                ) if total else None,
            })

        # Error-recovery-specific fields.
        if cat == "error_recovery":
            total_recovery = sum(1 for r in cat_results if r.correct)
            no_halluc = sum(
                1 for r in cat_results
                if getattr(r, "no_hallucinated_params", True)
            )
            # Count how many had their expected turn count
            turns_matched = sum(
                1 for r in cat_results
                if isinstance(r, MultiTurnTaskResult) and r.turns_correct
            )
            cat_score.update({
                "error_recovery": round(
                    total_recovery / total, 4
                ) if total else None,
                "self_corrected": round(
                    no_halluc / total, 4
                ) if total else None,
                "turns_matched": round(
                    turns_matched / total, 4
                ) if total else None,
            })

        category_scores[cat] = cat_score

    # Roll every single-tool category (search_basic, weather_enum, ...) into
    # one ``single_tool`` entry. The composite reads tool_accuracy and the
    # parameter averages from it; without the entry those three components
    # were always 0.
    single_results = [
        r for r in results if r.category not in _NON_SINGLE_TOOL_CATEGORIES
    ]
    if single_results:
        n_single = len(single_results)
        single_pass = sum(1 for r in single_results if r.correct)
        category_scores["single_tool"] = {
            "pass_count": single_pass,
            "total": n_single,
            "pass_rate": round(single_pass / n_single, 4),
            "tool_accuracy": round(
                sum(1 for r in single_results if r.tool_correct) / n_single, 4
            ),
            "param_completeness_avg": round(
                statistics.mean(r.params_complete for r in single_results), 4
            ),
            "param_correctness_avg": round(
                sum(1 for r in single_results if r.params_valid) / n_single, 4
            ),
            "composite_avg": round(
                statistics.mean(r.composite_score for r in single_results), 4
            ),
            "source_categories": sorted({r.category for r in single_results}),
        }

    total_correct = sum(1 for r in results if r.correct)
    # Compute the composite score (a plain sum) from the category scores.
    composite, _weighted_components = (
        _compute_weighted_composite(category_scores)
    ) if category_scores else (0.0, {})

    # Failure modes
    failure_modes: dict[str, int] = {
        "wrong_tool_selected": 0,
        "no_tool_call": 0,
        "truncated_output": 0,
        "missing_required_param": 0,
        "invalid_enum_value": 0,
        "wrong_param_type": 0,
        "incorrect_refusal": 0,
        "unnecessary_tool_call": 0,
        "request_failed": 0,
        "request_timeout": 0,
        "wrong_tool_sequence": 0,
        "data_flow_error": 0,
        "wrong_turn_count": 0,
        "hallucinated_param_value": 0,
        "schema_constraint_violation": 0,
        "error_no_retry": 0,
        "error_retry_invalid": 0,
    }
    for r in results:
        if not r.correct:
            if r.finish_reason == "length":
                failure_modes["truncated_output"] += 1
            if r.details.startswith("Request timed out"):
                failure_modes["request_timeout"] += 1
            elif "failed:" in r.details:
                failure_modes["request_failed"] += 1
            elif r.expected_tool is None and r.actual_tool is not None:
                failure_modes["unnecessary_tool_call"] += 1
            elif r.expected_tool is not None and r.actual_tool is not None and r.actual_tool != r.expected_tool:
                failure_modes["wrong_tool_selected"] += 1
            elif r.expected_tool is not None and r.actual_tool is None:
                failure_modes["no_tool_call"] += 1
            elif r.tool_correct and r.params_complete < 1.0:
                failure_modes["missing_required_param"] += 1

            # Multi-tool specific failure modes
            if isinstance(r, MultiTurnTaskResult):
                if not r.orchestration_correct:
                    failure_modes["wrong_tool_sequence"] += 1
                if not r.data_flow_correct:
                    failure_modes["data_flow_error"] += 1
                if not r.turns_correct:
                    failure_modes["wrong_turn_count"] += 1

            # Track hallucinated params for ambiguous tasks.
            if getattr(r, "no_hallucinated_params", True) is False:
                failure_modes["hallucinated_param_value"] += 1

            # Schema compliance failure modes.
            if r.category == "schema_compliance":
                if not getattr(r, "schema_required_present", True):
                    failure_modes["schema_constraint_violation"] += 1
                elif not getattr(r, "schema_length_valid", True):
                    failure_modes["schema_constraint_violation"] += 1
                elif not getattr(r, "schema_enum_valid", True):
                    failure_modes["schema_constraint_violation"] += 1
                elif not getattr(r, "schema_pattern_valid", True):
                    failure_modes["schema_constraint_violation"] += 1
                elif not getattr(r, "schema_nested_valid", True):
                    failure_modes["schema_constraint_violation"] += 1

            # Error recovery failure modes.
            if r.category == "error_recovery":
                if isinstance(r, MultiTurnTaskResult):
                    if not r.turns_correct:
                        failure_modes["error_no_retry"] += 1
                    elif not getattr(r, "no_hallucinated_params", True):
                        failure_modes["error_retry_invalid"] += 1

    _OVERLAY_FIELDS: dict[str, list[tuple[str, str, Any]]] = {
        "ambiguous": [
            ("no_hallucinated_params", "no_hallucinated_params", True),
            ("ambig_strategy", "ambig_strategy", "assumption"),
            ("ambig_details", "ambig_details", ""),
        ],
        "schema_compliance": [
            ("no_hallucinated_params", "no_hallucinated_params", True),
            ("ambig_strategy", "ambig_strategy", "schema_compliance"),
            ("ambig_details", "ambig_details", ""),
            ("schema_required_present", "schema_required_present", None),
            ("schema_length_valid", "schema_length_valid", None),
            ("schema_enum_valid", "schema_enum_valid", None),
            ("schema_pattern_valid", "schema_pattern_valid", None),
            ("schema_nested_valid", "schema_nested_valid", None),
        ],
        "error_recovery": [
            ("no_hallucinated_params", "no_hallucinated_params", True),
            ("ambig_strategy", "ambig_strategy", "error_recovery"),
            ("ambig_details", "ambig_details", ""),
        ],
    }

    per_task_serialized: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "task_id": r.task_id,
            "category": r.category,
            "prompt": r.prompt,
            "expected_tool": r.expected_tool,
            "actual_tool": r.actual_tool,
            "tool_call_id": r.tool_call_id,
            "actual_params": r.actual_params,
            "correct": r.correct,
            "tool_correct": r.tool_correct,
            "params_complete": r.params_complete,
            "params_valid": r.params_valid,
            "composite_score": r.composite_score,
            "details": r.details,
            "finish_reason": r.finish_reason,
            "completion_tokens": r.completion_tokens,
        }

        # Multi-turn extras.
        if isinstance(r, MultiTurnTaskResult):
            entry.update({
                "orchestration_correct": r.orchestration_correct,
                "data_flow_correct": r.data_flow_correct,
                "turns_correct": r.turns_correct,
                "parallel_detected": r.parallel_detected,
                "multi_score": r.multi_score,
                "chain_length": len(r.trace),
                "tool_sequence": [t.tool_name for t in r.trace if t.tool_name],
                "turn_finish_reasons": r.turn_finish_reasons,
            })

        # Overlay-specific extras.
        for out_key, in_key, default in _OVERLAY_FIELDS.get(r.category, []):
            entry[out_key] = getattr(r, in_key, default)

        per_task_serialized.append(entry)

    agg = BenchmarkResult(
        config=config,
        total_tasks=len(results),
        total_correct=total_correct,
        composite_score=composite,
        per_task=per_task_serialized,
        category_scores=category_scores,
        failure_modes=failure_modes,
    )
    # Attach composite components so run_tool_calling_benchmark() can include
    # them in the returned dict.
    agg.composite_components = _weighted_components  # type: ignore[attr-defined]
    return agg


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_tool_calling_benchmark(
    client: ModelClient,
    tasks_path: str | Path = "datasets/tool_calling_tasks.yaml",
    task_set: str = "full",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    chat_template_kwargs: dict[str, Any] | None = None,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    repeats: int = 1,
) -> dict[str, Any]:
    """Run tool-calling benchmark suite.

    Args:
        client: ModelClient connected to a running vLLM endpoint.
        tasks_path: Path to the YAML task definition file.
        task_set: ``"lite"`` (subset) or ``"full"`` (all).
        max_tokens: Maximum output tokens per request. A thinking model
            spends hundreds of tokens before its first tool call, so keep
            this high unless thinking is off.
        temperature: Sampling temperature (default 0 for deterministic eval).
        chat_template_kwargs: Optional ``chat_template_kwargs`` for every
            request, for example ``{"enable_thinking": False}``.
        request_timeout_s: HTTP read timeout in seconds for each request.
            A timeout counts as a ``request_timeout`` failure.
        repeats: Number of passes over the task set. The top-level fields
            hold the first pass.

    Returns:
        Dict with ``config``, ``total_tasks``, ``composite_score``,
        ``composite_components``, ``category_scores``, ``per_task``, and
        ``failure_modes``. With ``repeats`` > 1, also ``repeats``: the
        values, mean, stdev, min, and max of ``composite_score`` and
        ``total_correct``.
    """

    _, tasks = load_tasks(resolve_tasks_path(tasks_path))

    # Lite mode: take the first task of each category, at most 8 tasks
    if task_set == "lite":
        categories_seen = set()
        lite_tasks: list[dict] = []
        for t in tasks:
            cat = t.get("category", "single_tool")
            if cat not in categories_seen:
                lite_tasks.append(t)
                categories_seen.add(cat)
                if len(lite_tasks) >= 8:
                    break
        tasks = lite_tasks

    config = {
        "benchmark_version": "1.0",
        "task_file": str(tasks_path),
        "task_set": task_set,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": chat_template_kwargs,
        "request_timeout_s": request_timeout_s,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    # Send the union of the tools of all tasks with every request, so the
    # model always sees the full tool list.
    all_tools: list[dict] = []
    tools_seen_names: set[str] = set()
    for task in tasks:
        for tool_spec in task.get("tools", []):
            name = tool_spec["name"]
            if name not in tools_seen_names:
                all_tools.append(tool_spec)
                tools_seen_names.add(name)
    tools_definition = _build_tools_definition(all_tools)
    config["tool_definitions_count"] = len(tools_definition)

    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    config["repeats"] = repeats
    runs: list[BenchmarkResult] = []
    for rep in range(1, repeats + 1):
        if repeats > 1:
            print(f"[tool_calling] repeat {rep}/{repeats}")
        results = _run_task_list(
            client, tasks, tools_definition, max_tokens=max_tokens,
            chat_template_kwargs=chat_template_kwargs, temperature=temperature,
            request_timeout_s=request_timeout_s,
        )
        runs.append(_aggregate(results, config))

    # The top-level fields hold the first repeat, so one repeat gives the
    # same output as before. ``repeats`` holds the spread over all repeats.
    aggregated = runs[0]
    aggregated.config["end_time"] = datetime.now(timezone.utc).isoformat()

    output = {
        "config": aggregated.config,
        "total_tasks": aggregated.total_tasks,
        "total_correct": aggregated.total_correct,
        "composite_score": aggregated.composite_score,
        "composite_components": getattr(
            aggregated, "composite_components", {}
        ),
        "category_scores": aggregated.category_scores,
        "per_task": aggregated.per_task,
        "failure_modes": aggregated.failure_modes,
    }
    if repeats > 1:
        output["repeats"] = _repeat_summary(runs)
    return output


def _run_task_list(
    client: ModelClient,
    tasks: list[dict],
    tools_definition: list[dict],
    max_tokens: int,
    chat_template_kwargs: dict[str, Any] | None,
    temperature: float,
    request_timeout_s: float,
) -> list[TaskResult]:
    """Run each task once, in file order, and return the task results."""
    # Run tasks sequentially. Tasks of category multi_tool and tasks with an
    # expected_sequence (the error-recovery tasks) use the multi-turn loop.
    # All other tasks use the one-shot runner.
    results: list[TaskResult] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, 1):
        is_multi = task.get("category") == "multi_tool" or "expected_sequence" in task
        desc = task["expected"].get("tool") or "no tool" if not is_multi else (
            f"{task.get('expected_turns', '?')} turns"
        )
        print(
            f"[tool_calling] {idx}/{total}: {task['id']} "
            f"(expected: {desc})"
        )
        exc_details = None
        try:
            if is_multi:
                r = _run_multi_turn_task(
                    client, task, tools_definition, max_tokens=max_tokens,
                    chat_template_kwargs=chat_template_kwargs,
                    temperature=temperature, timeout_s=request_timeout_s,
                )
            else:
                r = _run_task(
                    client, task, tools_definition, max_tokens=max_tokens,
                    chat_template_kwargs=chat_template_kwargs,
                    temperature=temperature, timeout_s=request_timeout_s,
                )
        except Exception as exc:
            exc_details = f"exception: {exc}"
            r = TaskResult(
                task_id=task["id"],
                category=task.get("category", "unknown"),
                prompt=task["prompt"],
                expected_tool=task.get("expected", {}).get("tool"),
                actual_tool=None,
                tool_call_id=None,
                actual_params=None,
                correct=False,
                tool_correct=False,
                params_complete=0.0,
                params_valid=False,
                composite_score=0.0,
                details=f"Exception: {exc}",
            )
            _attach_overlay_defaults(r, task, exc_details)

        results.append(r)
        if idx < total:
            time.sleep(0.5)  # small stabilization gap

    return results


def _repeat_summary(runs: list[BenchmarkResult]) -> dict[str, Any]:
    """Give the mean and the spread of the composite and the pass count."""
    summary: dict[str, Any] = {"n": len(runs)}
    for key in ("composite_score", "total_correct"):
        values = [getattr(r, key) for r in runs]
        summary[key] = {
            "values": values,
            "mean": round(statistics.mean(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Standalone entry point for running the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Tool-calling benchmark")
    parser.add_argument(
        "--tasks", default="datasets/tool_calling_tasks.yaml",
        help="Path to the YAML task file",
    )
    parser.add_argument(
        "--base-url", required=True,
        help="vLLM base URL (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name to use for requests",
    )
    parser.add_argument(
        "--task-set", choices=["lite", "full"], default="full",
        help="Task subset to run",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Maximum output tokens per request (thinking models need room)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP read timeout in seconds for each request",
    )
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="Send chat_template_kwargs={'enable_thinking': false} with every request",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Run the task set N times and report the mean and the spread",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: results/tool_calling.json in the repo)",
    )
    args = parser.parse_args()

    client = ModelClient(
        base_url=args.base_url,
        model_name=args.model,
        chat=True,
    )

    result = run_tool_calling_benchmark(
        client,
        tasks_path=args.tasks,
        task_set=args.task_set,
        max_tokens=args.max_tokens,
        chat_template_kwargs={"enable_thinking": False} if args.disable_thinking else None,
        request_timeout_s=args.request_timeout,
        repeats=args.repeats,
    )

    output_path = Path(args.output) if args.output else ROOT / "results" / "tool_calling.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[tool_calling] results written to {output_path}")
    print(
        f"[tool_calling] composite_score={result['composite_score']}  "
        f"{result['total_correct']}/{result['total_tasks']} correct"
    )
    if "repeats" in result:
        spread = result["repeats"]["composite_score"]
        print(
            f"[tool_calling] {args.repeats} repeats: composite mean={spread['mean']} "
            f"stdev={spread['stdev']} range={spread['min']}..{spread['max']}"
        )


if __name__ == "__main__":
    main()
