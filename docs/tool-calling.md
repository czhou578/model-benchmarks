# Tool-calling benchmark

The tool-calling benchmark sends tasks with tool definitions to the model and scores the tool calls. The code is in the `benchmarks/tool_calling/` package. The tasks are in `datasets/tool_calling_tasks.yaml`.

The package has these modules:

| Module | Contents |
|---|---|
| `runner.py` | `run_tool_calling_benchmark`: the task selection, the task loop, and the output dict |
| `single_turn.py` | `_run_task` and the category overlay scoring |
| `multi_turn.py` | `_run_multi_turn_task`: the multi-turn loop and its scoring |
| `scoring.py` | The parameter checks, `_score_ambiguous_task`, and `_score_schema_compliance_task` |
| `aggregate.py` | `_aggregate`, `_WEIGHTS`, `_compute_weighted_composite`, and the repeat summary |
| `mocks.py` | `_MOCK_TOOL_RESPONSES` and `_mock_tool_response` |
| `protocol.py` | The request URL, `DEFAULT_REQUEST_TIMEOUT_S`, and the tool-call parsing |
| `tasks.py` | `load_tasks` and `resolve_tasks_path` |
| `results.py` | The result data classes |
| `cli.py` | `main`, the command line |

`benchmarks/tool_calling/__init__.py` re-exports these names, so `from benchmarks.tool_calling import run_tool_calling_benchmark` works.

## Run the benchmark

`core_runner.py` runs it after the FLOP analysis. `--skip-tool-calling` skips it. When the benchmark raises an error, the runner prints a warning and continues. `summary.json` then records `tool_calling.status: failed` and the `error`.

To run it alone against a running server:

```bash
python -m benchmarks.tool_calling --base-url http://127.0.0.1:8000 \
    --model nvidia/Qwen3.8-27B-NVFP4 --task-set lite
```

The other options are `--tasks`, `--max-tokens` (default 4096), `--request-timeout` (default 600 s), `--disable-thinking`, `--repeats` (default 1), and `--output` (default `results/tool_calling.json` in the repo root).

## Requests

- Each request goes to `/v1/chat/completions`, also when `endpoint.chat` is `false`.
- The request holds one user message with the task prompt, `tool_choice: auto`, the `temperature` argument (default 0), `stream: false`, and a new `cache_salt`.
- The `tools` list is the union of the `tools` of all selected tasks. Every task sees all tools, not only its own. When two tasks define a tool with the same name, the first definition wins.
- The read timeout of each request is `tool_calling_request_timeout_s` (default 600 s). The benchmark waits 0.5 s between tasks.
- A relative task path that does not exist in the current directory resolves against the repo root.

## Config keys

| Key | Default | Meaning |
|---|---|---|
| `tool_calling_tasks` | `datasets/tool_calling_tasks.yaml` | Task file. |
| `tool_calling_task_set` | `full` | `full` runs all tasks. `lite` runs a subset. |
| `tool_calling_max_tokens` | 4096 | `max_tokens` of each request. In a multi-turn task, it applies to each turn. |
| `tool_calling_chat_template_kwargs` | none | Copied into each request as `chat_template_kwargs`. `config.chat_template_kwargs` in `tool_calling.json` records it. |
| `tool_calling_request_timeout_s` | 600 | Read timeout of each request, in seconds. In a multi-turn task, it applies to each turn. `config.request_timeout_s` records it. |
| `tool_calling_repeats` | 1 | Number of passes over the task set. `config.repeats` records it. |

## Repeats

With `repeats` above 1, the benchmark runs the full task set that number of times. The top-level fields hold the first pass only. The `repeats` field holds `n` and two entries: `composite_score` and `total_correct`. Each entry has the `values` of all passes, the `mean`, the `stdev`, the `min`, and the `max`. With 1 repeat, the output has no `repeats` field. The `summary.json` entry copies the `repeats` field.

A thinking model writes its reasoning before the first tool call. vLLM parses tool calls only after the reasoning ends. A low `tool_calling_max_tokens` cuts the output before any tool call. The task then records `finish_reason: length`, and `failure_modes.truncated_output` counts it.

A full 4096-token answer takes about 330 s at 12.3 tok/s (Qwen3.8-27B). The old 120 s timeout cut off such answers. The 600 s default covers 4096 tokens down to about 7 tok/s. Increase the timeout for a slower model or a higher `tool_calling_max_tokens`.

When a request times out, the task fails. Its `details` starts with `Request timed out after <N> s`, and `failure_modes.request_timeout` counts it. Other request errors give `Request failed: <error>` and count in `failure_modes.request_failed`. When a task raises an exception, its `details` is `Exception: <error>`. From E5, `failure_modes.request_failed` also counts it. A multi-turn task stops at the failed turn and records the same `details`.

`{enable_thinking: false}` in `tool_calling_chat_template_kwargs` turns off thinking in chat templates that read this flag, for example Qwen3.

## Task sets

- `full` runs all 85 tasks in the file.
- `lite` runs the first task of each category, up to 8 tasks. With the current file, `lite` holds 8 single-tool tasks only. Three composite components are then 0, so the composite is at most 3.

| Category | Tasks | Runner | Scored as |
|---|---|---|---|
| `search_basic`, `search_params`, `weather_basic`, `weather_enum`, `currency_basic`, `stock_basic`, `calculate_basic`, `ticket_basic`, `ticket_complex`, `email_basic`, `meeting_basic` | 21 | single turn | `single_tool` roll-up |
| `refusal` | 2 | single turn | refusal |
| `ambiguous` | 9 | single turn | ambiguous check |
| `schema_compliance` | 20 | single turn | schema check |
| `multi_tool` | 16 | multi-turn | multi-tool check |
| `error_recovery` | 17 | multi-turn | error-recovery check |

A task runs multi-turn when its category is `multi_tool` or when it has `expected_sequence`.

## Task YAML fields

The file has a top-level `version` and a `tasks` list. The code reads `version` but does not use it.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Required. Task ID in `per_task`. |
| `category` | string | Selects the runner and the scoring. Default `unknown`. |
| `prompt` | string | Required. The user message. |
| `tools` | list | Tool specs: `name`, `description`, `parameters` (JSON Schema). |
| `expected.tool` | string or null | Required for single-turn tasks. `null` means that no tool call is expected. |
| `expected.params` | map | Read only for `ambiguous` tasks. A `null` value accepts any value. |
| `expected_sequence` | list of `step`, `tool`, `params` | Expected tool order of a multi-turn task. The code compares only `tool`. |
| `expected_turns` | int | Expected number of tool calls. Default: the length of `expected_sequence`. |
| `expected_intermediate_values` | map | Keys `step_<N>_result` and `step_<N+1>_input_from_step_<N>` for the data-flow check. |
| `strategy`, `scoring.strategy` | string | `ambiguous` tasks only: `no_tool`, `underspecified`, or `assumption` (default). `scoring.strategy` wins. |
| `scoring.param_completeness` | float | Minimum parameter completeness for a single-turn pass. Default 1.0. |

The code does not read the other `scoring` keys, for example `tool_accuracy`, `strict_compliance`, `constraint_detail`, and `recovery_type`. They document the intent of the task only.

For single-turn tasks other than `ambiguous`, the code does not compare parameter values with `expected.params`. It checks only the required parameters and the schema.

## Scoring

### Single-turn task (`_run_task`)

| Score | Rule |
|---|---|
| `tool_correct` | The called tool name equals `expected.tool`. No call and `null` also match. |
| `params_complete` | Fraction of the `required` parameters of the expected tool that the call contains. No parameters: 1.0 when no tool is expected, else 0.0. |
| `params_valid` | Each parameter passes `type`, `enum`, `minimum`, `maximum`, `minLength`, and `maxLength` of the expected tool schema. No parameters: `true` only when no tool is expected. |
| `composite_score` | `tool_correct` × `params_complete` × `params_valid`. |
| `correct` | `tool_correct` and `params_complete` ≥ `scoring.param_completeness`. `params_valid` has no effect. |

Two categories add a check after the base check. `correct` then needs both checks to pass.

- `ambiguous` (`_score_ambiguous_task`):
  - `no_tool` passes when the model calls no tool.
  - `underspecified` without a tool call passes when the reply asks for detail. The reply must contain `?` or one of these phrases, in any case: `which`, `what is`, `could you`, `would you`, `can you`, `do you`, `please clarify`, `more information`, `what would`, `which one`, `could you tell`, `are you asking`, `let me know`. Before E5, the code checked only `?` and the first five phrases.
  - `assumption` with `expected.tool: null` passes when the model calls no tool.
  - `underspecified` with a tool call, and `assumption` with an expected tool, check the parameters against `expected.params`. They pass when each parameter name is in `expected.params` and no value differs from a non-null expected value. An empty `expected.params` accepts any parameters.
- `schema_compliance` (`_score_schema_compliance_task`): the parameters must pass `required`, `minLength`, `maxLength`, `pattern`, and `enum`. The check also applies `enum` to array items, and `required`, `pattern`, and `enum` to nested objects. It uses the schema of the tool that the model called.

### Multi-turn task (`_run_multi_turn_task`)

The loop runs at most `max(expected_turns, 5)` turns. Each turn sends the full history. `_mock_tool_response` returns a fixed response for each tool call. The loop stops when a turn has no tool call or a request fails.

| Score | Rule |
|---|---|
| `orchestration_correct` | The first tool names equal the `expected_sequence` names. Extra later calls are allowed. |
| `data_flow_correct` | For each `step_<N>_result` key, the `expression` parameter of the first call in turn N+1 contains the expected input value. When turn N+1 has no call, the check passes. |
| `turns_correct` | The total number of tool calls equals `expected_turns`. Parallel calls in one turn count one by one. |
| `multi_score` | `orchestration_correct` × `data_flow_correct` × `turns_correct`. |
| `correct` | `orchestration_correct` and `data_flow_correct`. `turns_correct` has no effect for `multi_tool`. |

For `error_recovery`, `correct` has two more conditions. The call count must equal `expected_turns`. The last call must have no blank string parameter.

The mock returns an error only for an unknown tool or an unknown input. An unknown input is a symbol, location, currency pair, or expression that is not in its table. For `convert_currency`, the table holds one rate for each currency pair. The mock returns `amount` × rate, with the `amount` from the call arguments. A non-numeric `amount` gives an error.

### Category scores and composite

Each category gets `pass_count`, `total`, `pass_rate`, `tool_accuracy`, `param_completeness_avg`, `param_correctness_avg`, and `composite_avg`, plus fields of its own. The `single_tool` entry rolls up all categories that are not `multi_tool`, `error_recovery`, `schema_compliance`, `ambiguous`, or `refusal`.

`composite_score` is the sum of six components with weight 1 each, so its range is 0 to 6 (`_compute_weighted_composite`). The name `_WEIGHTS` holds no numbers. It maps each component to a category field:

| Component | Category | Field |
|---|---|---|
| `tool_accuracy` | `single_tool` | `tool_accuracy` |
| `param_completeness` | `single_tool` | `param_completeness_avg` |
| `param_correctness` | `single_tool` | `param_correctness_avg` |
| `multi_tool` | `multi_tool` | `multi_tool_score_avg` |
| `schema_compliance` | `schema_compliance` | `strict_compliance` |
| `refusal` | `refusal` | `correct_refusal` |

- When the field is not in the category, the component uses the category `pass_rate`. When the category is not in the run, the component is 0.
- `refusal.correct_refusal` is the fraction of refusal tasks with no tool call, a successful request, and a `finish_reason` other than `length`. A truncated reply passes the task, but it is not a refusal.
- In runs before this field, the `refusal` component read the missing `correct_refusal_or_default` field. It was the `pass_rate` of the 2 refusal tasks, so a truncated reply counted as a refusal.
- `ambiguous` and `error_recovery` tasks do not enter the composite. They count in `total_correct` and `category_scores`.
- `error_recovery.self_corrected` counts tasks with `no_hallucinated_params`. The multi-turn runner never sets that field, so the value is 1.0 unless a task raised an exception.

### Failure modes

`failure_modes` counts failed tasks only. One task can add to several modes. The code never increments `invalid_enum_value`, `wrong_param_type`, or `incorrect_refusal`, so these stay 0.

## Versions and old runs

- `config.benchmark_version` is the constant `"1.0"` in `tool_calling/runner.py`, `ttft_breakdown.py`, and `prefill.py`. No code reads it. It does not tell old scoring from new scoring.
- The result files hold no scoring version. [comparability.md](comparability.md) tells which tool-calling results you can compare, and how large the noise is.

## Example task

```yaml
- id: chain_stock_calc
  category: multi_tool
  tools:
    - name: get_stock_price
      description: "Get the current stock price for a symbol"
      parameters:
        type: object
        properties:
          symbol: {type: string, description: "Stock ticker symbol (e.g. AAPL, NVDA, MSFT)"}
        required: [symbol]
    - name: calculate
      description: "Evaluate a mathematical expression"
      parameters:
        type: object
        properties:
          expression: {type: string, description: "A valid mathematical expression (e.g. '2 + 3 * 4')"}
        required: [expression]
  prompt: "Look up the current price of NVDA and tell me if a 20% increase would put it above $200."
  expected_turns: 2
  expected_sequence:
    - {step: 1, tool: get_stock_price, params: {symbol: "NVDA"}}
    - {step: 2, tool: calculate, params: {expression: "150 * 1.2"}}
  expected_intermediate_values:
    step_1_result: 150
    step_2_input_from_step_1: 150
  scoring:
    orchestration_accuracy: true
    data_flow_correct: true
    turns_correct: true
```

The mock returns a price of 150.0 for `NVDA`. The task passes when the model calls `get_stock_price`, then calls `calculate` with an expression that contains `150`.

## Add a task

1. Add the task to `datasets/tool_calling_tasks.yaml`.
2. For a multi-turn task, add each new tool to `_MOCK_TOOL_RESPONSES` in `tool_calling/mocks.py`. Also add each new input of `get_stock_price`, `get_weather`, `convert_currency`, or `calculate`. Without the entry, the mock returns an error.
3. Run `pytest -q`.
