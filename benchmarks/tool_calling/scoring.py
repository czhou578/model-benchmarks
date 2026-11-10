"""Scoring helpers: parameter checks, ambiguous tasks, and schema compliance."""

from __future__ import annotations

from typing import Any


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
