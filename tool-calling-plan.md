# Tool Calling Effectiveness — Benchmark Plan

**Phase:** Roadmap item #19 (Capability dimension — structured output reliability)
**Type:** Capability benchmark (not system performance)
**Output:** `tool_calling.json`

---

## What We Want to Measure

| Metric | Question it answers |
|---|---|
| **Tool invocation accuracy** | Does the model call the right tool(s) at all? |
| **Parameter correctness** | Are the arguments valid — types, required fields, enums, ranges? |
| **Multi-tool orchestration** | Can the model chain multiple tool calls in one turn? |
| **Schema adherence** | Does the model follow strict JSON schemas under pressure? |
| **Robustness to ambiguity** | Does the model handle underspecified inputs, or does it make up parameters? |
| **Failure recovery** | When a tool call returns an error, does the model self-correct? |

---

## Why This Is Hard To Measure

The difficulty isn't running the LLM — it's judging the output. A tool-calling response is JSON with a `tool_calls` array. We can't just check `"passed"` or `"failed"`; we need to verify:

1. **Correctness:** The right tool was selected.
2. **Completeness:** All required parameters are present and well-typed.
3. **Validity:** Values fall in accepted ranges / enums.
4. **Intent alignment:** The tool choice matches the user's intent.

A model can produce perfectly valid JSON that calls the wrong tool, or call the right tool with hallucinated parameter values. Both are failures, but different kinds.

---

## Task Categories

### Category 1 — Single-Tool Invocation (Foundations)

Simple one-shot tool calls. Tests whether the model can:
- Identify which tool to call from a list.
- Populate required parameters.
- Omit or default optional parameters correctly.

**Example task set (5–10 per tool):**

```
Tools available:
  - search(query: str, max_results: int = 5)
  - get_weather(location: str, unit: "celsius" | "fahrenheit")
  - convert_currency(amount: float, from: str, to: str)

Prompt: "How hot is Tokyo right now?"
Expected: get_weather(location="Tokyo", unit="celsius")
```

**What to verify:**
- Correct tool selected (accuracy)
- Required params present (completeness)
- Optional params omitted or defaulted correctly (precision)
- Enum values from the accepted set (schema validity)

### Category 2 — Multi-Tool Chaining

Models must orchestrate multiple tools across one or more turns.

```
Tools available:
  - search(query: str) -> [results: [{title, url, snippet}]]
  - get_stock_price(symbol: str) -> {price, change}
  - calculate(expression: str) -> {result}

Prompt: "Look up the current price of NVDA and tell me if a 20% increase
would put it above $200."

Expected flow:
  Turn 1: get_stock_price(symbol="NVDA") → returns price=$150
  Turn 2: calculate(expression="150 * 1.2") → returns result=180
  Final answer: "No, a 20% increase would put it at $180, below $200."
```

**What to verify:**
- Correct sequence of tools
- Output from one tool used as input to the next (data flow)
- Number of turns to reach answer

### Category 3 — Ambiguous / Underspecified Input

Tests robustness when the user's intent isn't precise.

```
Tools available:
  - search(query: str, max_results: int = 5)
  - get_weather(location: str, unit: "celsius" | "fahrenheit")
  - convert_currency(amount: float, from: str, to: str)

Prompt: "Find me something about Japan."

Expected behavior:
  - The model should NOT hallucinate a specific tool call without clarifying
    which tool is relevant.
  - It should either ask a clarifying question or make a reasonable default
    choice (search with a broad query) and note the assumption.
```

**What to verify:**
- No hallucinated tool calls with fabricated parameters
- Appropriate clarification or transparent assumption
- Graceful handling of underspecified requests

### Category 4 — Error Recovery

Tests whether the model self-corrects when a tool returns an error.

```
Tools available:
  - search(query: str) -> [{title, url, snippet}]
  - get_weather(location: str, unit: "celsius" | "fahrenheit")

Round 1:
  Tool call: get_weather(location="NotARealPlaceXX", unit="celsius")
  Tool response: {"error": "Location not found: NotARealPlaceXX"}
  Expected model response: Retry with a different location or clarify.
```

**What to verify:**
- Does the model retry with different inputs?
- Does it ask the user for clarification?
- Does it hallucinate a fake result for the failed call?

### Category 5 — Complex Schema Compliance

Tests strict adherence to complex nested schemas (JSON Schema / function calling).

```json
{
  "name": "create_ticket",
  "parameters": {
    "type": "object",
    "properties": {
      "title": {"type": "string", "minLength": 5, "maxLength": 200},
      "description": {"type": "string", "maxLength": 2000},
      "labels": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["bug", "feature", "enhancement", "documentation", "security"]
        }
      },
      "priority": {
        "type": "string",
        "enum": ["p0", "p1", "p2", "p3", "p4"]
      },
      "assignees": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "username": {"type": "string", "pattern": "^[a-zA-Z0-9._-]+$"},
            "role": {"type": "string", "enum": ["primary", "secondary"]}
          },
          "required": ["username"]
        }
      }
    },
    "required": ["title", "priority"]
  }
}
```

**What to verify:**
- Nested objects are properly structured
- Enum values are from the accepted set
- Array items match type constraints
- String constraints (minLength, maxLength, pattern) are respected
- Required fields are all present

### Category 6 — Refusal / No-Tool Calls

Tests whether the model correctly refrains from calling tools when none are relevant.

```
Tools available:
  - search(query: str)
  - get_weather(location: str)

Prompt: "Tell me a joke."
Expected: No tool calls. A direct answer.

Prompt: "What is your name?"
Expected: No tool calls. A direct answer.
```

**What to verify:**
- No unnecessary tool calls
- Direct response when appropriate

---

## Scoring Methodology

Each task produces a structured verdict per dimension:

| Dimension | Score | Method |
|---|---|---|
| **Tool accuracy** | Pass / Fail | Is the correct tool invoked? (LLM judge or rule-based) |
| **Parameter completeness** | 0–100% | Fraction of required params present |
| **Parameter correctness** | 0–100% | Fraction of params with valid values (type, enum, range) |
| **Multi-tool score** | 0–100% | Product of tool-accuracy × parameter-correctness across all tools in the chain |
| **Recovery success** | Pass / Fail | Did the model correct after an error response? |
| **Schema compliance** | Pass / Fail (strict) | All constraints satisfied? (binary) |
| **Refusal correctness** | Pass / Fail | No tool calls when none are relevant? |

**Overall composite score** (per model):

```
tool_accuracy_weight  ×  tool_accuracy
+ param_completeness_weight  ×  param_completeness
+ param_correctness_weight  ×  param_correctness
+ multi_tool_weight  ×  multi_tool_score
+ schema_compliance_weight  ×  schema_compliance
+ refusal_weight  ×  refusal_correctness
```

Default weights (tune empirically):
- tool_accuracy: 0.25
- param_completeness: 0.15
- param_correctness: 0.20
- multi_tool: 0.20
- schema_compliance: 0.10
- refusal: 0.10

---

## Tool Definitions (Mock Server)

We need a small mock tool server that:
1. Defines 6–8 tools with realistic signatures.
2. Returns deterministic, useful outputs (hardcoded or simple logic).
3. Returns error responses when called with bad params (to test recovery).

**Proposed tool set:**

| Tool | Purpose | Returns |
|---|---|---|
| `search` | Web search | [{title, url, snippet}] |
| `get_weather` | Current weather | {temp, condition, location} |
| `convert_currency` | FX conversion | {amount, rate, result} |
| `get_stock_price` | Stock quote | {price, symbol, change_pct} |
| `calculate` | Math expression | {result} |
| `create_ticket` | Ticket creation | {ticket_id, status} — errors on bad input |
| `send_email` | Email | {message_id, status} — errors on invalid address |
| `schedule_meeting` | Calendar | {meeting_id, confirmed} — errors on conflicts |

The mock server returns static data (e.g., USD→JPY rate = 149.50) so results are reproducible.

---

## How It Fits Into the Existing Suite

### Integration point

Add as a new benchmark module in `benchmarks/tool_calling.py`, following the existing pattern:

```
benchmarks/
├── tool_calling.py       ← new: run_tool_calling_benchmark(client, tasks, ...)
```

The function signature mirrors the existing benchmarks:

```python
def run_tool_calling_benchmark(
    client: ModelClient,
    task_set: str = "full",  # "lite" | "full"
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run tool-calling benchmark suite.
    
    Returns a dict with per-category scores and the composite score.
    """
```

### Task definition format

Tasks are defined as a YAML file in `datasets/tool_calling_tasks.yaml`:

```yaml
version: "1.0"

tasks:
  - id: weather_tokyo
    category: single_tool
    tools: [get_weather, search, convert_currency]
    prompt: "How hot is Tokyo right now?"
    expected:
      tool: get_weather
      params:
        location: "Tokyo"
        unit: "celsius"
    scoring:
      tool_accuracy: true       # correct tool
      param_completeness: 1.0   # all required params present
      param_correctness: 1.0    # all values valid
```

### Runner flow

1. `core_runner.py` loads the task YAML.
2. Starts the mock tool server (lightweight Python HTTP server).
3. Configures the model's `tools` / `functions` parameter with the tool definitions.
4. For each task:
   - Sends the prompt with tools available.
   - Parses the model's tool_call response.
   - Verifies against the expected verdict.
   - Records results.
5. Writes `tool_calling.json` with per-category and composite scores.

### Mock tool server

A minimal server (`mock_tool_server.py`) that:
- Runs on a random port (same pattern as vLLM managed mode).
- Accepts tool invocations via HTTP.
- Returns deterministic responses.
- Supports both sync (one call at a time) and async (batch) modes.

```python
# Simple approach: one server, tool calls come via the vLLM tools API
# vLLM natively supports function calling when tools are passed in the request.
# We don't need a separate tool server — we define tools in the OpenAI API
# request and let vLLM call them. vLLM supports a `tool_choice` parameter
# and returns structured tool_call responses.
```

Actually, vLLM does **not** execute tool calls server-side — it returns the tool call as part of the generation response (as a `tool_calls` field in the `choices[].message` object). So we don't need a mock tool server; we need a **verifier** that:

1. Defines the tool schema in the request.
2. Gets back the model's `tool_calls` JSON.
3. Validates the tool name, parameter names/types/values against expected.

This is simpler — no mock server needed. The tools are declarative schemas passed in the request, and the model does the "calling." We verify the output.

### Revised runner flow (no mock server)

1. For each task, build a request with:
   - The prompt message.
   - The relevant tool definitions (as OpenAI `tools` parameter).
   - `tool_choice: "auto"` (let the model decide).
2. Send to vLLM (which supports function calling via the OpenAI API).
3. Parse the `tool_calls` array from the response.
4. Score each call against the expected verdict.

---

## Task Set Sizes

| Set | Description | Use case |
|---|---|---|
| **Lite** | ~20 tasks (2 per category) | Quick sanity check, CI gate |
| **Full** | ~60–80 tasks (6–10 per category) | Comprehensive benchmark |
| **Stress** | ~200 tasks + adversarial prompts | Edge case discovery |

---

## Expected Output Format

```json
{
  "model": "qwen3.6-35b-a3b-nvfp4",
  "timestamp": "2026-07-22T...",
  "tool_definitions_count": 8,
  "total_tasks": 64,
  "composite_score": 0.73,
  "scores": {
    "single_tool": {
      "accuracy": 0.92,
      "param_completeness": 0.88,
      "param_correctness": 0.85,
      "pass_count": 44,
      "total": 48
    },
    "multi_tool": {
      "orchestration_accuracy": 0.65,
      "avg_chain_length": 2.1,
      "pass_count": 10,
      "total": 16
    },
    "ambiguous": {
      "correct_refusal_or_default": 0.70,
      "no_hallucinated_params": 0.85,
      "pass_count": 14,
      "total": 20
    },
    "error_recovery": {
      "self_corrected": 0.60,
      "hallucinated_fix": 0.10,
      "asked_for_clarification": 0.20,
      "gave_up": 0.10,
      "pass_count": 8,
      "total": 10
    },
    "schema_compliance": {
      "strict_compliance": 0.55,
      "pass_count": 11,
      "total": 20
    },
    "refusal": {
      "correct_no_tool": 0.95,
      "unnecessary_tool_call": 0.05,
      "pass_count": 19,
      "total": 20
    }
  },
  "failure_modes": {
    "wrong_tool_selected": 8,
    "missing_required_param": 12,
    "invalid_enum_value": 5,
    "wrong_param_type": 3,
    "hallucinated_param_value": 7,
    "incorrect_refusal": 2,
    "failed_self_correction": 5
  }
}
```

---

## Implementation Phases

### Phase 1 — Single-Tool Benchmark (Baseline) DONE

- Define 20 tasks in `datasets/tool_calling_tasks.yaml`.
- Implement `run_tool_calling_benchmark()` in `benchmarks/tool_calling.py`.
- Score: tool accuracy, param completeness, param correctness.
- Estimated time: 1–2 days.

### Phase 2 — Multi-Tool Chaining

- Add 15–20 multi-turn tasks.
- Implement multi-turn loop: send tool output back as user message.
- Score: orchestration accuracy, data flow correctness.
- Estimated time: 2–3 days.

### Phase 3 — Schema Compliance & Robustness

- Add complex nested schema tasks (60+ property tasks).
- Add ambiguity tests, error recovery tests, refusal tests.
- Implement scoring for all 6 categories.
- Estimated time: 2–3 days.

### Phase 4 — Integration & Reporting

- Wire into `core_runner.py`.
- Add composite score calculation.
- Add failure mode reporting.
- Write results to `tool_calling.json` alongside other benchmark outputs.
- Estimated time: 1 day.

---

## Comparison Targets

Once implemented, this benchmark lets you answer:

| Comparison | What it reveals |
|---|---|
| Qwen 3.5 vs Qwen 3.6 | Tool calling improves with scale? |
| Full vs sparse (35B vs 3.5B active) | Does active parameter count drive tool accuracy? |
| FP4 vs FP8 vs FP16 | Quantization impact on structured output? |
| Different models, same tools | Which models are best at agentic workflows? |

---

## Future Enhancements

- **Tool-use latency:** Time from prompt → tool call selection → parameter generation.
- **Streaming tool calls:** Can the model stream partial tool calls correctly?
- **Tool documentation quality:** Measure whether adding tool descriptions / docstrings improves performance.
- **Adversarial tool calls:** Prompts designed to trick the model into calling tools with wrong args (e.g., SQL injection in parameters, prompt injection via tool output).
- **Cross-modal tool calls:** Image generation tools, audio tools, code execution — does tool calling hold up beyond text params?