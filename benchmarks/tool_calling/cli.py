"""Command-line entry point of the tool-calling benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tool_calling.protocol import DEFAULT_REQUEST_TIMEOUT_S
from benchmarks.tool_calling.runner import run_tool_calling_benchmark
from harness.client import ModelClient
from harness.paths import ROOT


def main() -> None:
    """Standalone entry point for running the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Tool-calling benchmark")
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
        "--max-tokens", type=int, default=4096,
        help="Maximum output tokens per request (thinking models need room)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP read timeout in seconds for each request",
    )
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="Send chat_template_kwargs={'enable_thinking': false} with every request",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Run the task set N times and report the mean and the spread",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: results/tool_calling.json in the repo)",
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
        chat_template_kwargs={"enable_thinking": False} if args.disable_thinking else None,
        request_timeout_s=args.request_timeout,
        repeats=args.repeats,
    )

    output_path = Path(args.output) if args.output else ROOT / "results" / "tool_calling.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[tool_calling] results written to {output_path}")
    print(
        f"[tool_calling] composite_score={result['composite_score']}  "
        f"{result['total_correct']}/{result['total_tasks']} correct"
    )
    if "repeats" in result:
        spread = result["repeats"]["composite_score"]
        print(
            f"[tool_calling] {args.repeats} repeats: composite mean={spread['mean']} "
            f"stdev={spread['stdev']} range={spread['min']}..{spread['max']}"
        )


if __name__ == "__main__":
    main()
