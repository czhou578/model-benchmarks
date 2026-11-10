"""Aggregation of the tool-calling task results.

The module computes the category scores, the failure modes, and the
composite score. It also holds ``run_tool_calling_benchmark``.
"""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.tool_calling.multi_turn import _run_multi_turn_task
from benchmarks.tool_calling.protocol import DEFAULT_REQUEST_TIMEOUT_S, _build_tools_definition
from benchmarks.tool_calling.results import BenchmarkResult, MultiTurnTaskResult, TaskResult
from benchmarks.tool_calling.single_turn import _attach_overlay_defaults, _run_task
from benchmarks.tool_calling.tasks import load_tasks, resolve_tasks_path
from harness.client import ModelClient


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
