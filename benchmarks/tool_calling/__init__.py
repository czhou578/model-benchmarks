"""Tool-calling benchmark package.

The package keeps the names that ``benchmarks.tool_calling`` had as a
single module. ``python -m benchmarks.tool_calling`` runs the CLI.
docs/tool-calling.md describes the tasks and the scoring.
"""

import time

import requests

from benchmarks.tool_calling.aggregate import (
    _WEIGHTS,
    _aggregate,
    _compute_weighted_composite,
    _run_multi_turn_task,
    _run_task,
    _run_task_list,
    _score_ambiguous_task,
    _score_schema_compliance_task,
    main,
    run_tool_calling_benchmark,
)
from benchmarks.tool_calling.mocks import _MOCK_TOOL_RESPONSES, _mock_tool_response
from benchmarks.tool_calling.protocol import DEFAULT_REQUEST_TIMEOUT_S, _build_tools_definition
from benchmarks.tool_calling.results import (
    BenchmarkResult,
    MultiTurnTaskResult,
    TaskResult,
    ToolCallTrace,
)
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
