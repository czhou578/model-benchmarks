"""Chat-completions helpers: the request URL, the timeout, and the tool-call parsing."""

from __future__ import annotations

import json
from typing import Any

import requests

from harness.client import ModelClient

# Seconds to wait for one chat completion. The default 4096-token answer
# takes about 330 s at 12.3 tok/s (Qwen3.8-27B), so 120 s cut off real answers.
DEFAULT_REQUEST_TIMEOUT_S = 600.0


def _build_completion_url(client: ModelClient) -> str:
    """Return the ``chat/completions`` URL for the given client.

    Handles both ``http://host/v1`` and ``http://host`` base URLs.
    """
    base = client.base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _response_metadata(response: dict[str, Any]) -> tuple[str | None, int | None]:
    """Return ``(finish_reason, completion_tokens)`` from a chat response.

    ``finish_reason == "length"`` means the output hit ``max_tokens``. For a
    thinking model this usually means the model never left ``<think>``, so
    the harness saw no tool call. Without this field a truncated run looks
    like a refusal.
    """
    choices = response.get("choices") or [{}]
    finish_reason = choices[0].get("finish_reason")
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return finish_reason, completion_tokens


def _request_error_details(exc: Exception, timeout_s: float) -> str:
    """Describe a failed request, with a timeout kept distinct from other errors."""
    if isinstance(exc, requests.exceptions.Timeout):
        return f"Request timed out after {timeout_s:g} s: {exc}"
    return f"Request failed: {exc}"


def _parse_all_tool_calls(response: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    """Extract all tool calls from a vLLM response.

    Returns:
        A list of ``(tool_name, arguments_dict)`` tuples.
    """
    choices = response.get("choices")
    if not choices:
        return []

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return []

    calls: list[tuple[str, dict[str, Any] | None]] = []
    for tc in tool_calls:
        name = tc.get("function", {}).get("name")
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            args = None
        if name:
            calls.append((name, args))
    return calls


def _parse_tool_calls(
    response: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    """Extract the first tool call from a vLLM OpenAI response.

    Returns:
        (tool_name_or_None, arguments_dict_or_None)
    """
    all_calls = _parse_all_tool_calls(response)
    if all_calls:
        return all_calls[0]
    return None, None


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


def _get_tool_schema(
    tools_def: list[dict],
    tool_name: str,
) -> dict:
    """Get the JSON Schema for a given tool name from the definitions."""
    for t in tools_def:
        if t["function"]["name"] == tool_name:
            return t["function"].get("parameters", {})
    return {}
