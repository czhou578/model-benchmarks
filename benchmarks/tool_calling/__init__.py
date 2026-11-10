"""Tool-calling benchmark.

The default task file has 85 tasks in these groups:

Single-tool invocation
  One-shot tool calls. Tests tool selection and parameter validation.
  The group also holds the ``ambiguous`` and ``refusal`` tasks, where the
  correct answer can be a question or no tool call.

Multi-tool chaining (``multi_tool``)
  Multi-turn chains where the model must orchestrate multiple tools.
  The output of one tool feeds into the next.

Schema compliance (``schema_compliance``)
  Strict JSON Schema constraint validation (minLength, maxLength, pattern,
  enum, required, nested objects).

Error recovery (``error_recovery``)
  Multi-turn tasks. The model must correct its call after a tool error.

The composite score is the plain sum of six components, from 0 to 6
(``_compute_weighted_composite``). docs/tool-calling.md describes the scoring.

Input : ``datasets/tool_calling_tasks.yaml``
Output: ``tool_calling.json`` inside the run directory. The standalone CLI
writes ``results/tool_calling.json`` unless --output names another file.

Usage::

    python -m benchmarks.tool_calling --tasks datasets/tool_calling_tasks.yaml --base-url http://localhost:8000/v1 --model <model-name>

Or import ``run_tool_calling_benchmark(client, tasks_path=...)`` from benchmarks.tool_calling.

The package keeps the names of the former module
``benchmarks/tool_calling.py``. Modules:

- ``results``: the result data classes.
- ``protocol``: the request URL, the request timeout, and the tool-call parsing.
- ``tasks``: the task file loading.
- ``scoring``: the parameter, ambiguous-task, and schema-compliance scoring.
- ``mocks``: the mock tool results of the multi-turn loop.
- ``single_turn``: the category overlay scoring and the one-shot runner.
- ``multi_turn``: the multi-turn loop and its scoring.
- ``aggregate``: the category scores, the failure modes, the composite,
  and the summary of the repeats.
- ``runner``: ``run_tool_calling_benchmark`` and the task loop.
- ``cli``: the command-line entry point.
"""

import time

import requests

from benchmarks.tool_calling.aggregate import (
    _WEIGHTS,
    _aggregate,
    _compute_weighted_composite,
)
from benchmarks.tool_calling.cli import main
from benchmarks.tool_calling.mocks import _MOCK_TOOL_RESPONSES, _mock_tool_response
from benchmarks.tool_calling.multi_turn import _run_multi_turn_task
from benchmarks.tool_calling.protocol import DEFAULT_REQUEST_TIMEOUT_S, _build_tools_definition
from benchmarks.tool_calling.results import (
    BenchmarkResult,
    MultiTurnTaskResult,
    TaskResult,
    ToolCallTrace,
)
from benchmarks.tool_calling.scoring import _score_ambiguous_task, _score_schema_compliance_task
from benchmarks.tool_calling.runner import _run_task_list, run_tool_calling_benchmark
from benchmarks.tool_calling.single_turn import _run_task
from benchmarks.tool_calling.tasks import load_tasks, resolve_tasks_path

# ``requests`` and ``time`` stay package attributes: tests patch
# ``tool_calling.requests.post`` and ``tool_calling.time.sleep``.
__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_S",
    "BenchmarkResult",
    "MultiTurnTaskResult",
    "TaskResult",
    "ToolCallTrace",
    "_MOCK_TOOL_RESPONSES",
    "_WEIGHTS",
    "_aggregate",
    "_build_tools_definition",
    "_compute_weighted_composite",
    "_mock_tool_response",
    "_run_multi_turn_task",
    "_run_task",
    "_run_task_list",
    "_score_ambiguous_task",
    "_score_schema_compliance_task",
    "load_tasks",
    "main",
    "requests",
    "resolve_tasks_path",
    "run_tool_calling_benchmark",
    "time",
]
