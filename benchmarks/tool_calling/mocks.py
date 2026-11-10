"""Mock tool executor: deterministic tool results for the multi-turn loop."""

from __future__ import annotations

from typing import Any

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
    # Rates keyed by (from, to). The mock computes amount * rate from the
    # call arguments, so every amount gets a consistent result.
    "convert_currency": {
        ("USD", "JPY"): 149.50,
        ("EUR", "GBP"): 0.85,
        ("GBP", "EUR"): 1.18,
        ("GBP", "USD"): 1.27,
        ("USD", "EUR"): 0.92,
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
        if key not in registry:
            return {"error": f"Unknown conversion: {key}"}
        try:
            raw_amount: Any = params.get("amount")
            amount = float(raw_amount)
        except (TypeError, ValueError):
            return {"error": f"Invalid amount: {params.get('amount')!r}"}
        rate = registry[key]
        return {"amount": amount, "rate": rate, "result": round(amount * rate, 2)}
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
