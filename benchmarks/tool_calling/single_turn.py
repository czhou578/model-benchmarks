"""Single-turn tasks: the category overlay scoring and the one-shot runner."""

from __future__ import annotations

from typing import Any

from benchmarks.tool_calling.protocol import _get_tool_schema
from benchmarks.tool_calling.results import TaskResult
from benchmarks.tool_calling.scoring import _score_ambiguous_task, _score_schema_compliance_task


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
