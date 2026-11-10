"""Result data classes of the tool-calling benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    # Generation metadata, used to tell truncation from refusal.
    finish_reason: str | None = None
    completion_tokens: int | None = None


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
    turn_finish_reasons: list[str | None] = field(default_factory=list)
