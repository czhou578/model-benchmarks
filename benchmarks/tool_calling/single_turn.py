"""Single-turn tasks: the category overlay scoring and the one-shot runner."""

from __future__ import annotations

import uuid
from typing import Any

import requests

from benchmarks.tool_calling.protocol import (
    DEFAULT_REQUEST_TIMEOUT_S,
    _build_completion_url,
    _get_tool_schema,
    _parse_tool_calls,
    _request_error_details,
    _response_metadata,
)
from benchmarks.tool_calling.results import TaskResult
from benchmarks.tool_calling.scoring import (
    _score_ambiguous_task,
    _score_params,
    _score_schema_compliance_task,
    _tools_match,
)
from harness.client import ModelClient

# --------------------------------------------------------------------------- #
# Category overlay scoring — dispatch helper
# --------------------------------------------------------------------------- #


def _apply_overlay_scoring(
    result: TaskResult,
    task: dict,
    tools_def: list[dict],
    actual_tool: str | None,
    actual_params: dict[str, Any] | None,
    response_content: str | None,
) -> tuple[bool, str]:
    """Run category-specific overlay scoring and attach fields to *result*.

    For ambiguous / schema-compliance / error-recovery tasks this recomputes
    correctness and appends extra detail text.

    Returns:
        ``(final_correct, final_details)`` so the caller can use them in the
        TaskResult constructor.
    """
    category = task.get("category", "unknown")
    scoring = task.get("scoring", {})
    details = result.details

    # ── Default overlay fields (always set) ─────────────────────────────────
    result.no_hallucinated_params = True
    result.ambig_strategy = task.get("scoring", {}).get(
        "strategy", task.get("strategy", "assumption"),
    )
    result.ambig_details = ""

    # ── Ambiguous tasks ─────────────────────────────────────────────────────
    if category == "ambiguous":
        score_correct, score_no_halluc, score_details = _score_ambiguous_task(
            actual_tool, actual_params, response_content, task,
        )
        result.no_hallucinated_params = score_no_halluc
        result.ambig_strategy = task.get("scoring", {}).get(
            "strategy", task.get("strategy", "assumption"),
        )
        result.ambig_details = score_details

        # For ambiguous tasks, correct requires BOTH standard AND ambiguous scoring
        if not score_correct:
            result.correct = False

        if score_details:
            details = f"{details}; {score_details}" if details else score_details

    # ── Schema-compliance tasks ─────────────────────────────────────────────
    if category == "schema_compliance":
        actual_schema = _get_tool_schema(tools_def, actual_tool or "")
        schema_ok, schema_constraint_results, schema_details = (
            _score_schema_compliance_task(
                actual_params, actual_schema, task.get("scoring", {}),
            )
        )
        result.no_hallucinated_params = schema_ok
        result.ambig_strategy = "schema_compliance"
        result.ambig_details = schema_details
        for key, val in schema_constraint_results.items():
            setattr(result, f"schema_{key}", val)

        if not schema_ok:
            result.correct = False
            details = f"Schema violations: {schema_details}"

    # ── Default details for non-ambiguous tasks ─────────────────────────────
    if category not in ("ambiguous", "schema_compliance"):
        expected_tool = task["expected"].get("tool")
        required_params = _get_tool_schema(tools_def, expected_tool or "").get("required", [])
        if not result.tool_correct:
            expected_display = expected_tool or "no tool"
            actual_display = actual_tool or "no tool"
            details = f"Expected tool={expected_display}, got {actual_display}"
            if result.finish_reason == "length":
                details += " (output hit max_tokens before a tool call)"
        elif result.params_complete < scoring.get("param_completeness", 1.0):
            missing = [p for p in required_params if p not in (actual_params or {})]
            details = f"Missing params: {missing}"
        elif not result.params_valid:
            details = "Parameter value(s) failed schema validation"

    result.details = details
    return result.correct, result.details


def _attach_overlay_defaults(result: TaskResult, task: dict, reason: str) -> None:
    """Set overlay fields on an exception result.

    This is called for tasks that failed before normal scoring could
    populate the dynamic attributes.
    """
    category = task.get("category", "unknown")
    if category == "ambiguous":
        result.no_hallucinated_params = True
        result.ambig_strategy = task.get("scoring", {}).get(
            "strategy", task.get("strategy", "assumption"),
        )
        result.ambig_details = reason
    elif category == "schema_compliance":
        result.no_hallucinated_params = True
        result.ambig_strategy = "schema_compliance"
        result.ambig_details = reason
    elif category == "error_recovery":
        result.no_hallucinated_params = False
        result.ambig_strategy = "error_recovery"
        result.ambig_details = reason


# --------------------------------------------------------------------------- #
# Run a single task
# --------------------------------------------------------------------------- #


def _run_task(
    client: ModelClient,
    task: dict,
    tools_def: list[dict],
    max_tokens: int = 4096,
    chat_template_kwargs: dict[str, Any] | None = None,
    temperature: float = 0.0,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> TaskResult:
    """Send one task to the model and score the response."""

    expected_tool = task["expected"].get("tool")       # null → refusal
    scoring = task.get("scoring", {})

    url = _build_completion_url(client)

    payload = {
        "model": client.model_name,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "tools": tools_def,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    cache_salt = uuid.uuid4().hex
    payload["cache_salt"] = cache_salt
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)

    try:
        resp = requests.post(
            url, headers=client.headers, json=payload, timeout=timeout_s
        )
        resp.raise_for_status()
        response = resp.json()
    except Exception as exc:
        error_details = _request_error_details(exc, timeout_s)
        result = TaskResult(
            task_id=task["id"],
            category=task.get("category", "unknown"),
            prompt=task["prompt"],
            expected_tool=expected_tool,
            actual_tool=None,
            tool_call_id=None,
            actual_params=None,
            correct=False,
            tool_correct=False,
            params_complete=0.0,
            params_valid=False,
            composite_score=0.0,
            details=error_details,
            raw_response={"error": error_details},
        )
        _attach_overlay_defaults(result, task, error_details.lower())
        return result

    finish_reason, completion_tokens = _response_metadata(response)
    actual_tool, actual_params = _parse_tool_calls(response)
    call_id = None
    response_content = None  # captured for ambiguous-task scoring
    if actual_tool:
        tc = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls", [])
        if tc:
            call_id = tc[0].get("id")
    else:
        # No tool call — capture any content text (for clarification check).
        message = (response.get("choices") or [{}])[0].get("message", {})
        response_content = message.get("content")

    # --- Scoring ---
    expected_schema = _get_tool_schema(tools_def, expected_tool or "")

    tool_correct = _tools_match(expected_tool, actual_tool)

    if actual_params is None or not actual_params:
        params_complete = 1.0 if expected_tool is None else 0.0
        params_valid = expected_tool is None
    else:
        params_complete, params_valid = _score_params(actual_params, expected_schema)

    # Composite = tool_correct × completeness × validity
    composite = int(tool_correct) * params_complete * int(params_valid)

    correct = tool_correct and params_complete >= scoring.get("param_completeness", 1.0)

    # Build standard result with overlay scoring applied in-place.
    result = TaskResult(
        task_id=task["id"],
        category=task.get("category", "unknown"),
        prompt=task["prompt"],
        expected_tool=expected_tool,
        actual_tool=actual_tool,
        tool_call_id=call_id,
        actual_params=actual_params,
        correct=correct,
        tool_correct=tool_correct,
        params_complete=round(params_complete, 4),
        params_valid=params_valid,
        composite_score=round(composite, 4),
        details="",
        raw_response=response,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
    )
    _apply_overlay_scoring(result, task, tools_def, actual_tool, actual_params, response_content)

    return result
