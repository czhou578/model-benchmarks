"""Tool-calling benchmark.

Phase 1 — Single-Tool Invocation
  Simple one-shot tool calls. Tests tool selection and parameter validation.

Phase 2 — Multi-Tool Chaining
  Multi-turn chains where the model must orchestrate multiple tools.
  The output of one tool feeds into the next.

Phase 3 — Schema Compliance & Error Recovery
  Schema compliance: strict JSON Schema constraint validation (minLength,
  maxLength, pattern, enum, required, nested objects).
  Error recovery: model self-corrects when a tool returns an error.

Input : ``datasets/tool_calling_tasks.yaml``
Output: ``tool_calling.json`` inside the run directory.

Usage::

    python -m benchmarks.tool_calling --tasks datasets/tool_calling_tasks.yaml --base-url http://localhost:8000/v1 --model <model-name>

Or import ``run_tool_calling_benchmark(client, tasks)`` from core_runner benchmarks.
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
import re

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
    # Dynamic attributes for ambiguous-task scoring (set after construction).
    no_hallucinated_params: bool = True
    ambig_strategy: str = "assumption"
    ambig_details: str = ""


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
# Multi-turn result data structures
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallTrace:
    """One turn in the multi-turn chain."""

    step: int
    tool_name: str | None
    params: dict[str, Any] | None
    tool_response: dict[str, Any] | None  # simulated response


@dataclass
class MultiTurnTaskResult(TaskResult):
    """Extended result for a multi-tool chain task."""

    trace: list[ToolCallTrace] = field(default_factory=list)
    orchestration_correct: bool = False    # right tools in right order
    data_flow_correct: bool = False        # used previous output as input
    turns_correct: bool = False            # stopped at expected turn
    parallel_detected: bool = False        # called two tools in one turn
    multi_score: float = 0.0              # orchestration * data_flow * turns
    details: str = ""


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


def _tools_match(expected: str | None, actual: str | None) -> bool:
    """Return True if actual matches expected (or both are None / no tool)."""
    return actual == expected or (expected is None and actual is None)


# --------------------------------------------------------------------------- #
# Ambiguous-task scoring
# --------------------------------------------------------------------------- #


_CLARIFICATION_KEYWORDS = [
    "which", "what is", "could you", "would you", "can you", "do you",
    "please clarify", "more information", "specifically", "help me",
    "i'm not sure", "i don't know", "what would", "i can", "which one",
    "could you tell", "are you asking", "let me know",
]


def _has_clarification(text: str) -> bool:
    """Check whether *text* contains clarification language (a question or
    request for more detail) rather than a definitive answer."""
    if not text:
        return False
    # Ask a question
    if "?" in text:
        return True
    # Ask for more info
    return any(
        keyword in text.lower()
        for keyword in _CLARIFICATION_KEYWORDS[:5]
    )


def _score_ambiguous_task(
    actual_tool: str | None,
    actual_params: dict[str, Any] | None,
    response_content: str | None,
    task: dict,
) -> tuple[bool, bool, str]:
    """Score an ambiguous / underspecified task.

    Returns:
        (correct, no_hallucinated_params, details)
    """

    strategy = task.get("scoring", {}).get(
        "strategy", task.get("strategy", "assumption"),
    )
    expected_tool = task.get("expected", {}).get("tool")

    # --- Case 1: no_tool — model should not call any tool ---
    if strategy == "no_tool":
        return (actual_tool is None, True, "")

    # --- Case 2: underspecified — model should clarify, not call a tool.
    # However, if the model calls a tool with reasonable params (no
    # hallucination), accept it as a transparent default rather than
    # penalising for calling a tool at all.
    if strategy == "underspecified":
        if actual_tool is None:
            has_clar = _has_clarification(response_content or "")
            return (has_clar, True, "" if has_clar else "no clarification")
        # Model called a tool. Check params for hallucination.
        expected_params = task.get("expected", {}).get("params", {})
        no_halluc = True
        if expected_params and actual_params:
            for key in actual_params:
                if key not in expected_params:
                    no_halluc = False
                    break
                if expected_params[key] is not None and actual_params[key] != expected_params[key]:
                    no_halluc = False
                    break
        return (no_halluc, no_halluc, "reasonable default" if no_halluc else "hallucinated params")

    # --- Case 3: assumption — model may make a reasonable default call ---
    if expected_tool is None:
        return (actual_tool is None, True, "unexpected tool call")

    expected_params = task.get("expected", {}).get("params", {})
    no_halluc = True
    if expected_params and actual_params:
        for key, value in (actual_params or {}).items():
            if key not in expected_params:
                no_halluc = False
                break
            if expected_params[key] is not None and actual_params[key] != expected_params[key]:
                no_halluc = False
                break

    return (no_halluc, no_halluc, "" if no_halluc else "hallucinated params")


def _score_schema_compliance_task(
    actual_params: dict[str, Any] | None,
    tool_params_schema: dict,
    scoring: dict,
) -> tuple[bool, dict[str, bool], str]:
    """Score a schema compliance task.

    Checks that the model's params satisfy all schema constraints.

    Returns:
        (correct, constraint_results, details)
    """
    properties = tool_params_schema.get("properties", {})
    required = tool_params_schema.get("required", [])

    if not actual_params:
        if required:
            return False, {"required_present": False}, "no params provided"
        return True, {}, "no params, no required fields"

    results: dict[str, bool] = {
        "required_present": True,
        "length_valid": True,
        "enum_valid": True,
        "pattern_valid": True,
        "nested_valid": True,
    }

    details_parts: list[str] = []

    # Check required fields
    for req in required:
        if req not in actual_params:
            results["required_present"] = False
            details_parts.append(f"missing required: {req}")

    # Check each provided param against its schema
    for param_name, param_value in actual_params.items():
        prop_schema = properties.get(param_name, {})

        # String constraints
        if isinstance(param_value, str):
            if "minLength" in prop_schema:
                if len(param_value) < prop_schema["minLength"]:
                    results["length_valid"] = False
                    details_parts.append(
                        f"{param_name}: too short (len={len(param_value)}, min={prop_schema['minLength']})"
                    )
            if "maxLength" in prop_schema:
                if len(param_value) > prop_schema["maxLength"]:
                    results["length_valid"] = False
                    details_parts.append(
                        f"{param_name}: too long (len={len(param_value)}, max={prop_schema['maxLength']})"
                    )
            if "pattern" in prop_schema:
                if not re.match(prop_schema["pattern"], param_value):
                    results["pattern_valid"] = False
                    details_parts.append(
                        f"{param_name}: violates pattern {prop_schema['pattern']}"
                    )
            if "enum" in prop_schema and param_value not in prop_schema["enum"]:
                results["enum_valid"] = False
                details_parts.append(
                    f"{param_name}: {param_value!r} not in {prop_schema['enum']}"
                )

        # Array constraints
        if isinstance(param_value, list):
            item_schema = prop_schema.get("items", {})
            for item in param_value:
                if "enum" in item_schema and isinstance(item, str):
                    if item not in item_schema["enum"]:
                        results["enum_valid"] = False
                        details_parts.append(f"label: {item!r} not in enum")

        # Nested object (assignees)
        if isinstance(param_value, list) and prop_schema.get("items", {}).get("type") == "object":
            items_schema = prop_schema["items"]
            nested_props = items_schema.get("properties", {})
            nested_required = items_schema.get("required", [])
            for idx, entry in enumerate(param_value):
                if isinstance(entry, dict):
                    # Check required nested fields
                    for nr in nested_required:
                        if nr not in entry:
                            results["nested_valid"] = False
                            details_parts.append(f"assignee[{idx}]: missing {nr}")
                    # Check nested string constraints
                    for ne, nv in entry.items():
                        np_ = nested_props.get(ne, {})
                        if isinstance(nv, str):
                            if "pattern" in np_ and not re.match(np_["pattern"], nv):
                                results["pattern_valid"] = False
                                details_parts.append(f"assignee[{idx}].{ne}: pattern violation")
                            if "enum" in np_ and nv not in np_["enum"]:
                                results["enum_valid"] = False
                                details_parts.append(f"assignee[{idx}].{ne}: {nv!r} not in enum")

    all_ok = all(results.values())
    if not results["required_present"]:
        all_ok = False
    if not results["length_valid"]:
        all_ok = False
    if not results["enum_valid"]:
        all_ok = False
    if not results["pattern_valid"]:
        all_ok = False
    if not results["nested_valid"]:
        all_ok = False

    details = "; ".join(details_parts) if details_parts else ""
    return all_ok, results, details


# --------------------------------------------------------------------------- #
# Mock tool executor — deterministic simulated responses
# --------------------------------------------------------------------------- #

# Deterministic responses that the benchmark uses to simulate tool outputs.
# The multi-turn loop calls these instead of hitting a real API.
_MOCK_TOOL_RESPONSES: dict[str, Any] = {
    "get_stock_price": {
        "NVDA": {"price": 150.0, "symbol": "NVDA", "change_pct": 2.3},
        "AAPL": {"price": 190.0, "symbol": "AAPL", "change_pct": -0.5},
        "MSFT": {"price": 420.0, "symbol": "MSFT", "change_pct": 1.1},
        "TSLA": {"price": 250.0, "symbol": "TSLA", "change_pct": -1.8},
    },
    "get_weather": {
        "Tokyo": {"temp": 22, "condition": "sunny", "location": "Tokyo"},
        "London": {"temp": 15, "condition": "cloudy", "location": "London"},
        "Sydney": {"temp": 28, "condition": "clear", "location": "Sydney"},
        "Canberra": {"temp": 18, "condition": "partly cloudy", "location": "Canberra"},
        "Ottawa": {"temp": -2, "condition": "snowy", "location": "Ottawa"},
        "Rome": {"temp": 25, "condition": "sunny", "location": "Rome"},
    },
    "convert_currency": {
        ("USD", "JPY"): {"amount": 100, "rate": 149.50, "result": 14950.0},
        ("EUR", "GBP"): {"amount": 50, "rate": 0.85, "result": 42.5},
        ("GBP", "EUR"): {"amount": 250, "rate": 1.18, "result": 295.0},
        ("GBP", "USD"): {"amount": 50, "rate": 1.27, "result": 63.5},
        ("USD", "EUR"): {"amount": 100, "rate": 0.92, "result": 92.0},
        ("USD", "JPY"): {"amount": 230, "rate": 149.50, "result": 34385.0},
    },
    "calculate": {
        "150 * 1.2": {"expression": "150 * 1.2", "result": 180.0},
        "190 * 0.9": {"expression": "190 * 0.9", "result": 171.0},
        "250 * 1.15": {"expression": "250 * 1.15", "result": 287.5},
        "250 * 0.90": {"expression": "250 * 0.90", "result": 225.0},
        "200 * 1.15": {"expression": "200 * 1.15", "result": 230.0},
        "1000 / 8": {"expression": "1000 / 8", "result": 125.0},
        "1000 * 0.20": {"expression": "1000 * 0.20", "result": 200.0},
        "250 * 0.20": {"expression": "250 * 0.20", "result": 50.0},
        "14000000 * 0.10": {"expression": "14000000 * 0.10", "result": 1400000.0},
        "50 * 0.25": {"expression": "50 * 0.25", "result": 12.5},
        "50 + 75": {"expression": "50 + 75", "result": 125.0},
        "1000 * 0.30": {"expression": "1000 * 0.30", "result": 300.0},
        "(100 + 200) * 3 / 4": {"expression": "(100 + 200) * 3 / 4", "result": 225.0},
        "250 * 0.2": {"expression": "250 * 0.2", "result": 50.0},
        "150 * 0.2": {"expression": "150 * 0.2", "result": 30.0},
    },
    "search": {
        "default": {"results_count": 5, "snippet": "Search results available"},
    },
    "send_email": {
        "default": {"message_id": "msg_12345", "status": "sent"},
    },
    "schedule_meeting": {
        "default": {"meeting_id": "mtg_67890", "confirmed": True},
    },
}


def _mock_tool_response(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic simulated response for a tool call.

    This is used by the multi-turn loop so the benchmark can run
    without an external tool server.
    """
    if tool_name not in _MOCK_TOOL_RESPONSES:
        return {"error": f"Unknown tool: {tool_name}"}

    registry = _MOCK_TOOL_RESPONSES[tool_name]

    if "symbol" in params and tool_name == "get_stock_price":
        return registry.get(params["symbol"], {"error": f"Unknown symbol: {params.get('symbol')}"})
    if "location" in params and tool_name == "get_weather":
        return registry.get(params["location"], {"error": f"Unknown location: {params.get('location')}"})
    if tool_name == "convert_currency":
        key = (params.get("from_currency", ""), params.get("to_currency", ""))
        return registry.get(key, {"error": f"Unknown conversion: {key}"})
    if tool_name == "calculate":
        expr = params.get("expression", "")
        return registry.get(expr, {"expression": expr, "result": "unknown", "error": "Expression not mocked"})
    if tool_name == "search":
        query = params.get("query", "")
        # Return a synthetic number for queries about quantities
        if "state" in query.lower() and "how many" in query.lower():
            return {"results_count": 1, "snippet": "The United States has 50 states"}
        if "population" in query.lower():
            return {"results_count": 1, "snippet": "Tokyo population is approximately 14 million"}
        return {"results_count": 1, "snippet": f"Results for: {query}"}
    return registry.get("default", {"error": "No mock response"})


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


def load_tasks(path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
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
            details=f"Request failed: {exc}",
            raw_response={"error": str(exc)},
        )
        # For ambiguous tasks, request failure means no params were produced.
        if task.get("category") == "ambiguous":
            result.no_hallucinated_params = True
            result.ambig_strategy = task.get("scoring", {}).get(
                "strategy", task.get("strategy", "assumption"),
            )
            result.ambig_details = f"request failed: {exc}"
        return result

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
    # Determine required params from the expected tool schema
    expected_schema = {}
    for t in tools_def:
        if t["function"]["name"] == (expected_tool or ""):
            expected_schema = t["function"].get("parameters", {})
            break
    required_params = expected_schema.get("required", [])

    tool_correct = _tools_match(expected_tool, actual_tool)

    if actual_params is None or not actual_params:
        params_complete = 1.0 if expected_tool is None else 0.0
        params_valid = expected_tool is None
    else:
        params_complete, params_valid = _score_params(actual_params, expected_schema)

    # Composite = tool_correct × completeness × validity
    composite = int(tool_correct) * params_complete * int(params_valid)

    correct = tool_correct and params_complete >= scoring.get("param_completeness", 1.0)
    details = ""

    # Default values for non-ambiguous tasks.
    ambig_correct = correct
    ambig_no_halluc = True
    ambig_details = ""

    # ── Ambiguous-task overlay scoring ────────────────────────────────────
    if task.get("category") == "ambiguous":
        score_correct, score_no_halluc, score_details = _score_ambiguous_task(
            actual_tool, actual_params, response_content, task,
        )
        ambig_correct = score_correct
        ambig_no_halluc = score_no_halluc
        ambig_details = score_details

        # For ambiguous tasks, correct requires BOTH standard AND ambiguous scoring
        if not ambig_correct:
            correct = False

    # Default values for schema-compliance scoring (overridden if category matches).
    schema_ok = True
    schema_constraint_results: dict[str, bool] = {}
    schema_details = ""

    # ── Schema-compliance overlay scoring ───────────────────────────────────
    if task.get("category") == "schema_compliance":
        # Get the tool schema for the expected tool
        actual_schema = {}
        for t in tools_def:
            if t["function"]["name"] == actual_tool:
                actual_schema = t["function"].get("parameters", {})
                break

        schema_ok, schema_constraint_results, schema_details = _score_schema_compliance_task(
            actual_params, actual_schema, task.get("scoring", {}),
        )
        if not schema_ok:
            correct = False
            details = f"Schema violations: {schema_details}"

    # Standard details (tool/param mismatch)
    if not tool_correct:
        expected_display = expected_tool or "no tool"
        actual_display = actual_tool or "no tool"
        details = f"Expected tool={expected_display}, got {actual_display}"
    elif params_complete < scoring.get("param_completeness", 1.0):
        missing = [p for p in required_params if p not in (actual_params or {})]
        details = f"Missing params: {missing}"
    elif not params_valid:
        details = "Parameter value(s) failed schema validation"

    # Append ambiguity-specific details (for ambiguous tasks)
    if task.get("category") == "ambiguous" and ambig_details:
        if details:
            details += f"; {ambig_details}"
        else:
            details = ambig_details

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
        details=details,
        raw_response=response,
    )

    # Attach ambiguity-specific fields for ambiguous tasks.
    if task.get("category") == "ambiguous":
        result.no_hallucinated_params = ambig_no_halluc
        result.ambig_strategy = task.get("scoring", {}).get(
            "strategy", task.get("strategy", "assumption"),
        )
        result.ambig_details = ambig_details

    # Attach schema-compliance fields for schema_compliance tasks.
    if task.get("category") == "schema_compliance":
        result.no_hallucinated_params = schema_ok
        result.ambig_strategy = "schema_compliance"
        result.ambig_details = schema_details
        for key, val in schema_constraint_results.items():
            setattr(result, f"schema_{key}", val)

    return result


# --------------------------------------------------------------------------- #
# Multi-turn helpers
# --------------------------------------------------------------------------- #


def _parse_all_tool_calls(response: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Extract all tool calls from a vLLM response.

    Returns a list of (tool_name, arguments_dict) tuples.
    """
    choices = response.get("choices")
    if not choices:
        return []

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return []

    calls = []
    for tc in tool_calls:
        name = tc.get("function", {}).get("name")
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        if name:
            calls.append((name, args))
    return calls


def _get_tool_schema(
    tools_def: list[dict],
    tool_name: str,
) -> dict:
    """Get the JSON Schema for a given tool name from the definitions."""
    for t in tools_def:
        if t["function"]["name"] == tool_name:
            return t["function"].get("parameters", {})
    return {}


def _run_multi_turn_task(
    client: ModelClient,
    task: dict,
    tools_def: list[dict],
    max_tokens: int = 256,
) -> MultiTurnTaskResult:
    """Execute a multi-tool chain task.

    The loop runs up to ``max_expected_turns`` rounds:
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
    base = client.base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    # Initialise the conversation with just the user prompt.
    messages = [{"role": "user", "content": prompt}]
    trace: list[ToolCallTrace] = []
    actual_sequence: list[dict[str, Any]] = []
    response: dict[str, Any] = {}
    tool_calls: list[tuple[str, dict[str, Any]]] = []

    # Track intermediate values that flow between turns.
    # e.g. {"step_1_result": 150, "step_2_input_from_step_1": 150}
    intermediate_results: dict[str, Any] = {}

    for turn_idx in range(1, max_turns + 1):
        cache_salt = uuid.uuid4().hex
        payload = {
            "model": client.model_name,
            "messages": messages,
            "tools": tools_def,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        payload["extra_body"] = {"cache_salt": cache_salt}

        try:
            resp = requests.post(
                url, headers=client.headers, json=payload, timeout=120
            )
            resp.raise_for_status()
            response = resp.json()
        except Exception:
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

        # For each tool call in this turn, execute and record.
        for call_name, call_params in tool_calls:
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

            # Feed the tool response back to the model for the next turn.
            tool_call_id = f"call_{turn_idx}_{len(actual_sequence)}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(mock_resp),
            })

        # Add the assistant's tool-call message (OpenAI format).
        tool_call_entries = []
        for i, (name, params) in enumerate(tool_calls):
            tool_call_entries.append({
                "id": f"call_{turn_idx}_{i + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(params or {}),
                },
            })
        messages.append({
            "role": "assistant",
            "tool_calls": tool_call_entries,
        })

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
        trace=trace,
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

# Default weights per the plan.  Each key maps to the ``category_scores``
# field name that provides the component value.
_WEIGHTS: dict[str, tuple[str, str]] = {
    "tool_accuracy": ("single_tool", "tool_accuracy"),
    "param_completeness": ("single_tool", "param_completeness_avg"),
    "param_correctness": ("single_tool", "param_correctness_avg"),
    "multi_tool": ("multi_tool", "multi_tool_score_avg"),
    "schema_compliance": ("schema_compliance", "strict_compliance"),
    "refusal": ("refusal", "correct_refusal_or_default"),
}


def _compute_weighted_composite(
    category_scores: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Compute the weighted composite score across categories.

    Returns:
        ``(composite, components)`` where *components* is a dict of the
        individual (unweighted) component values used.
    """
    components: dict[str, float] = {}
    composite = 0.0
    for key, (cat, field) in _WEIGHTS.items():
        score = category_scores.get(cat, {}).get(field)
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

    total_correct = sum(1 for r in results if r.correct)
    # Compute the weighted composite score from category scores.
    composite, _weighted_components = (
        _compute_weighted_composite(category_scores)
    ) if category_scores else (0.0, {})

    # Failure modes
    failure_modes: dict[str, int] = {
        "wrong_tool_selected": 0,
        "missing_required_param": 0,
        "invalid_enum_value": 0,
        "wrong_param_type": 0,
        "incorrect_refusal": 0,
        "unnecessary_tool_call": 0,
        "request_failed": 0,
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
            if r.expected_tool is None and r.actual_tool is not None:
                failure_modes["unnecessary_tool_call"] += 1
            elif r.expected_tool is not None and r.actual_tool is not None and r.actual_tool != r.expected_tool:
                failure_modes["wrong_tool_selected"] += 1
            elif r.expected_tool is not None and r.actual_tool is None:
                failure_modes["missing_required_param"] += 1
            elif "failed:" in r.details:
                failure_modes["request_failed"] += 1

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

    per_task_serialized = []
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
        }
        if isinstance(r, MultiTurnTaskResult):
            entry.update({
                "orchestration_correct": r.orchestration_correct,
                "data_flow_correct": r.data_flow_correct,
                "turns_correct": r.turns_correct,
                "parallel_detected": r.parallel_detected,
                "multi_score": r.multi_score,
                "chain_length": len(r.trace),
                "tool_sequence": [t.tool_name for t in r.trace if t.tool_name],
            })

        # Ambiguous-task-specific fields.
        if r.category == "ambiguous":
            entry.update({
                "no_hallucinated_params": getattr(r, "no_hallucinated_params", True),
                "ambig_strategy": getattr(r, "ambig_strategy", "assumption"),
                "ambig_details": getattr(r, "ambig_details", ""),
            })

        # Schema-compliance-specific fields.
        if r.category == "schema_compliance":
            entry.update({
                "no_hallucinated_params": getattr(r, "no_hallucinated_params", True),
                "ambig_strategy": getattr(r, "ambig_strategy", "schema_compliance"),
                "ambig_details": getattr(r, "ambig_details", ""),
                "schema_required_present": getattr(r, "schema_required_present", None),
                "schema_length_valid": getattr(r, "schema_length_valid", None),
                "schema_enum_valid": getattr(r, "schema_enum_valid", None),
                "schema_pattern_valid": getattr(r, "schema_pattern_valid", None),
                "schema_nested_valid": getattr(r, "schema_nested_valid", None),
            })

        # Error-recovery-specific fields.
        if r.category == "error_recovery":
            entry.update({
                "no_hallucinated_params": getattr(r, "no_hallucinated_params", True),
                "ambig_strategy": getattr(r, "ambig_strategy", "error_recovery"),
                "ambig_details": getattr(r, "ambig_details", ""),
            })

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
        ``composite_components``, ``category_scores``, ``per_task``, and
        ``failure_modes``.
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

    # Run tasks sequentially.  Multi-tool tasks use the multi-turn loop;
    # single-tool tasks use the original one-shot runner.
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
                )
            else:
                r = _run_task(client, task, tools_definition, max_tokens=max_tokens)
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
        # Attach ambiguity fields even for exceptions.
        if task.get("category") == "ambiguous":
            r.no_hallucinated_params = True
            r.ambig_strategy = task.get("scoring", {}).get(
                "strategy", task.get("strategy", "assumption"),
            )
            r.ambig_details = exc_details or r.details

        # Attach schema-compliance fields even for exceptions.
        if task.get("category") == "schema_compliance":
            r.no_hallucinated_params = True
            r.ambig_strategy = "schema_compliance"
            r.ambig_details = f"exception: {exc_details or r.details}"

        # Attach error-recovery fields even for exceptions.
        if task.get("category") == "error_recovery":
            r.no_hallucinated_params = False
            r.ambig_strategy = "error_recovery"
            r.ambig_details = f"exception: {exc_details or r.details}"

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
        "composite_components": getattr(
            aggregated, "composite_components", {}
        ),
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

    parser = argparse.ArgumentParser(description="Tool-calling benchmark (Phase 1–3)")
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