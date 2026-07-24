"""Tool-calling benchmark — Phase 1: Single-Tool Invocation.

Runs a model through single-tool prompts, verifies that the model:
  1. Selected the correct tool (or correctly chose no tool).
  2. Populated required parameters.
  3. Used valid values (types, enums, ranges).

Input : ``datasets/tool_calling_tasks.yaml``
Output: ``tool_calling.json`` inside the run directory.

Usage::

    python -m benchmarks.tool_calling --tasks datasets/tool_calling_tasks.yaml --base-url http://localhost:8000/v1 --model <model-name>

Or import ``run_tool_calling_benchmark(client, tasks)`` from core_runner benchmarks.

Colin is 
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from core_runner import ModelClient


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class TaskResult:
    """Result for a single benchmark task."""

    task_id: str
    category: str
    prompt: str
    expected_tool: str | None       # ``null`` for refusal tasks
    actual_tool: str | None         # ``null`` if model called no tool
    tool_call_id: str | None        # vLLM call ID (for tracing)
    actual_params: dict[str, Any] | None
    correct: bool
    tool_correct: bool
    params_complete: float          # fraction of required params present
    params_valid: bool              # types + enums + ranges all OK
    composite_score: float          # tool_correct × params_complete × params_valid
    details: str = ""               # human-readable reason for failure
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark output."""

    config: dict[str, Any]
    total_tasks: int
    total_correct: int
    composite_score: float
    per_task: list[dict[str, Any]]
    category_scores: dict[str, dict[str, Any]]
    failure_modes: dict[str, int]


# --------------------------------------------------------------------------- #
# Helpers — response parsing & scoring
# --------------------------------------------------------------------------- #


def _parse_tool_calls(response: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Extract the first tool call from a vLLM OpenAI response.

    Returns:
        (tool_name_or_None, arguments_dict_or_None)
    """
    choices = response.get("choices")
    if not choices:
        return None, None

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None, None

    first = tool_calls[0]
    name = first.get("function", {}).get("name")
    try:
        args = json.loads(first.get("function", {}).get("arguments", "{}"))
    except json.JSONDecodeError:
        args = {}
    return name, args


def _validate_param_type(value: Any, expected_type: str) -> bool:
    """Check whether *value* matches the expected JSON Schema type string."""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True  # unknown type → accept
    # bool is a subclass of int in Python — special-case it
    if expected_type == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _validate_param(value: Any, schema: dict) -> bool:
    """Validate a single parameter against its JSON Schema fragment."""
    if "type" in schema:
        if not _validate_param_type(value, schema["type"]):
            return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "minimum" in schema and isinstance(value, (int, float)):
        if value < schema["minimum"]:
            return False
    if "maximum" in schema and isinstance(value, (int, float)):
        if value > schema["maximum"]:
            return False
    if "minLength" in schema and isinstance(value, str):
        if len(value) < schema["minLength"]:
            return False
    if "maxLength" in schema and isinstance(value, str):
        if len(value) > schema["maxLength"]:
            return False
    return True


def _score_params(
    actual: dict[str, Any] | None,
    schema: dict,
) -> tuple[float, bool]:
    """Score parameter completeness and correctness.

    Returns:
        (completeness_ratio, all_valid)
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    if not properties:
        return 1.0, True

    # --- Completeness ---
    present_count = sum(1 for p in required if p in (actual or {}))
    completeness = present_count / len(required) if required else 1.0

    # --- Correctness (values present) ---
    all_valid = True
    if actual:
        for param_name, param_value in actual.items():
            param_schema = properties.get(param_name, {})
            if not _validate_param(param_value, param_schema):
                all_valid = False
                break

    return round(completeness, 4), all_valid


# --------------------------------------------------------------------------- #
# Build tool definitions from YAML task schema
# --------------------------------------------------------------------------- #


def _build_tools_definition(tools_yaml: list[dict]) -> list[dict]:
    """Convert YAML tool specs to OpenAI ``tools`` format.

    Each entry in *tools_yaml* is:
        {name, description, parameters: {type, properties, required, ...}}
    """
    tools = []
    for t in tools_yaml:
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        })
    return tools


# --------------------------------------------------------------------------- #
# Load tasks
# --------------------------------------------------------------------------- #


def load_tasks(path: str | Path) -> list[dict]:
    """Load the task YAML and return (config, tasks) tuple."""
    with open(path) as f:
        data = yaml.safe_load(f)

    config = data.get("version", "1.0")
    tasks = data.get("tasks", [])
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return config, tasks


# --------------------------------------------------------------------------- #
# Run a single task
# --------------------------------------------------------------------------- #


def _run_task(
    client: ModelClient,
    task: dict,
    tools_def: list[dict],
    max_tokens: int = 256,
) -> TaskResult:
    """Send one task to the model and score the response."""

    expected_tool = task["expected"].get("tool")       # null → refusal
    scoring = task.get("scoring", {})

    # vLLM supports tool calling via the OpenAI-compatible /v1/chat/completions
    # endpoint.  The base_url may or may not already include /v1 — handle both.
    base = client.base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    payload = {
        "model": client.model_name,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "tools": tools_def,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    cache_salt = uuid.uuid4().hex
    payload["extra_body"] = {"cache_salt": cache_salt}

    try:
        resp = requests.post(
            url, headers=client.headers, json=payload, timeout=120
        )
        resp.raise_for_status()
        response = resp.json()
    except Exception as exc:
        return TaskResult(
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
            details=f"Request failed: {exc}",
            raw_response={"error": str(exc)},
        )

    actual_tool, actual_params = _parse_tool_calls(response)
    call_id = None
    if actual_tool:
        tc = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls", [])
        if tc:
            call_id = tc[0].get("id")

    # --- Scoring ---
    # Determine required params from the expected tool schema
    expected_schema = {}
    for t in tools_def:
        if t["function"]["name"] == (expected_tool or ""):
            expected_schema = t["function"].get("parameters", {})
            break
    required_params = expected_schema.get("required", [])

    tool_correct = (actual_tool == expected_tool) or (
        expected_tool is None and actual_tool is None
    )

    if actual_params is None or not actual_params:
        params_complete = 1.0 if expected_tool is None else 0.0
        params_valid = expected_tool is None
    else:
        params_complete, params_valid = _score_params(actual_params, expected_schema)

    # Composite = tool_correct × completeness × validity
    composite = (1.0 if tool_correct else 0.0) * params_complete * (
        1.0 if params_valid else 0.0
    )

    correct = tool_correct and params_complete >= scoring.get("param_completeness", 1.0)
    details = ""
    if not tool_correct:
        expected_display = expected_tool or "no tool"
        actual_display = actual_tool or "no tool"
        details = f"Expected tool={expected_display}, got {actual_display}"
    elif params_complete < scoring.get("param_completeness", 1.0):
        missing = [p for p in required_params if p not in (actual_params or {})]
        details = f"Missing params: {missing}"
    elif not params_valid:
        details = "Parameter value(s) failed schema validation"

    return TaskResult(
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
        details=details,
        raw_response=response,
    )


# --------------------------------------------------------------------------- #
# Aggregate scoring
# --------------------------------------------------------------------------- #


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

        category_scores[cat] = {
            "pass_count": correct,
            "total": total,
            "pass_rate": round(correct / total, 4) if total else None,
            "tool_accuracy": round(tool_correct_count / total, 4) if total else None,
            "param_completeness_avg": avg_completeness,
            "param_correctness_avg": round(avg_validity, 4),
            "composite_avg": avg_composite,
        }

    total_correct = sum(1 for r in results if r.correct)
    composite = round(
        statistics.mean(r.composite_score for r in results), 4
    ) if results else 0.0

    # Failure modes
    failure_modes: dict[str, int] = {
        "wrong_tool_selected": 0,
        "missing_required_param": 0,
        "invalid_enum_value": 0,
        "wrong_param_type": 0,
        "incorrect_refusal": 0,
        "unnecessary_tool_call": 0,
        "request_failed": 0,
    }
    for r in results:
        if not r.correct:
            if r.expected_tool is None and r.actual_tool is not None:
                failure_modes["unnecessary_tool_call"] += 1
            elif r.expected_tool is not None and r.actual_tool is not None and r.actual_tool != r.expected_tool:
                failure_modes["wrong_tool_selected"] += 1
            elif r.expected_tool is not None and r.actual_tool is None:
                failure_modes["missing_required_param"] += 1
            elif "failed:" in r.details:
                failure_modes["request_failed"] += 1

    per_task_serialized = [
        {
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
        }
        for r in results
    ]

    return BenchmarkResult(
        config=config,
        total_tasks=len(results),
        total_correct=total_correct,
        composite_score=composite,
        per_task=per_task_serialized,
        category_scores=category_scores,
        failure_modes=failure_modes,
    )


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_tool_calling_benchmark(
    client: ModelClient,
    tasks_path: str | Path = "datasets/tool_calling_tasks.yaml",
    task_set: str = "full",
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run tool-calling benchmark suite.

    Args:
        client: ModelClient connected to a running vLLM endpoint.
        tasks_path: Path to the YAML task definition file.
        task_set: ``"lite"`` (subset) or ``"full"`` (all).
        max_tokens: Maximum output tokens per request.
        temperature: Sampling temperature (default 0 for deterministic eval).

    Returns:
        Dict with ``config``, ``total_tasks``, ``composite_score``,
        ``category_scores``, ``per_task``, and ``failure_modes``.
    """

    _, tasks = load_tasks(tasks_path)

    # Lite mode: take first 8 tasks (2 per category)
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
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    # Resolve tools from the first task (all Phase-1 tasks share the same set).
    # If tasks define different tool subsets, we pick the union so the model
    # can always see the full tool list.
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

    # Run tasks sequentially
    results: list[TaskResult] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, 1):
        print(
            f"[tool_calling] {idx}/{total}: {task['id']} "
            f"(expected: {task['expected'].get('tool') or 'no tool'})"
        )
        try:
            r = _run_task(client, task, tools_definition, max_tokens=max_tokens)
        except Exception as exc:
            r = TaskResult(
                task_id=task["id"],
                category=task.get("category", "unknown"),
                prompt=task["prompt"],
                expected_tool=task["expected"].get("tool"),
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
        results.append(r)
        if idx < total:
            time.sleep(0.5)  # small stabilization gap

    aggregated = _aggregate(results, config)
    aggregated.config["end_time"] = datetime.now(timezone.utc).isoformat()

    return {
        "config": aggregated.config,
        "total_tasks": aggregated.total_tasks,
        "total_correct": aggregated.total_correct,
        "composite_score": aggregated.composite_score,
        "category_scores": aggregated.category_scores,
        "per_task": aggregated.per_task,
        "failure_modes": aggregated.failure_modes,
    }


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Standalone entry point for running the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Tool-calling benchmark (Phase 1)")
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
        "--max-tokens", type=int, default=256,
        help="Maximum output tokens per request",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: tool_calling.json)",
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
    )

    output_path = args.output or "tool_calling.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[tool_calling] results written to {output_path}")
    print(
        f"[tool_calling] composite_score={result['composite_score']}  "
        f"{result['total_correct']}/{result['total_tasks']} correct"
    )


if __name__ == "__main__":
    main()