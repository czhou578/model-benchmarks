"""The tool-calling benchmark run: task selection, the task loop, and the output."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.tool_calling.aggregate import _aggregate, _repeat_summary
from benchmarks.tool_calling.multi_turn import _run_multi_turn_task
from benchmarks.tool_calling.protocol import (
    DEFAULT_REQUEST_TIMEOUT_S,
    _build_tools_definition,
)
from benchmarks.tool_calling.results import BenchmarkResult, TaskResult
from benchmarks.tool_calling.single_turn import _attach_overlay_defaults, _run_task
from benchmarks.tool_calling.tasks import load_tasks, resolve_tasks_path
from harness.client import ModelClient

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
        r: TaskResult
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
