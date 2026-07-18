"""
concurrency.py — Measure throughput and latency degradation as request
concurrency increases.

How does aggregate throughput change when I fire 1, 4, 16, or 32 requests
simultaneously? At what point does the GPU bottleneck show up?
"""

from __future__ import annotations

import statistics
import time
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from core_runner import ModelClient, build_prompt_of_length, count_tokens, _percentile


# --------------------------------------------------------------------------- #
# Test prompt — reuse the creative-writing prompt, deterministic 256 tokens
# --------------------------------------------------------------------------- #

CONCURRENCY_PROMPT = build_prompt_of_length(256)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class ConcurrentRequestResult:
    """Result from a single request within a concurrency batch."""

    success: bool
    index: int
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    total_time_s: float = 0.0
    error: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _stat_summary(values: list[float]) -> dict[str, Any]:
    """Compute avg / median / p95 / min / max."""
    if not values:
        return {"avg_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None}
    return {
        "avg_s": round(statistics.mean(values), 4),
        "median_s": round(statistics.median(values), 4),
        "p95_s": round(_percentile(values, 95), 4),
        "min_s": round(min(values), 4),
        "max_s": round(max(values), 4),
    }


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #

def run_concurrency_test(
    client: ModelClient,
    concurrency_levels: list[int] | None = None,
    requests_per_level: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Fire concurrent requests and measure throughput / latency scaling.

    Args:
        client: ModelClient instance connected to a vLLM endpoint.
        concurrency_levels: List of concurrency levels to test (default 1,2,4,8,16).
        requests_per_level: How many requests to fire at each concurrency level.
        max_tokens: Max output tokens per request.
        temperature: Generation temperature.

    Returns:
        Dict with per-concurrency-level results.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]

    results: dict[str, Any] = {
        "config": {
            "concurrency_levels": concurrency_levels,
            "requests_per_level": requests_per_level,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_tokens": count_tokens(CONCURRENCY_PROMPT),
        },
        "per_concurrency_level": {},
    }

    for level in concurrency_levels:
        request_results: list[ConcurrentRequestResult] = []

        def _run_one(idx: int) -> ConcurrentRequestResult:
            try:
                gen = client.generate(CONCURRENCY_PROMPT, max_tokens=max_tokens, temperature=temperature)
                return ConcurrentRequestResult(
                    success=True,
                    index=idx,
                    prompt_tokens=gen.prompt_tokens,
                    output_tokens=gen.output_tokens,
                    ttft_s=gen.ttft_s,
                    total_time_s=gen.total_time_s,
                )
            except Exception as e:
                return ConcurrentRequestResult(
                    success=False,
                    index=idx,
                    error=str(e),
                )

        # Fire all requests simultaneously via thread pool
        start_wall = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(_run_one, i) for i in range(requests_per_level)]
            for f in concurrent.futures.as_completed(futures):
                request_results.append(f.result())
        wall_time = time.time() - start_wall

        # Aggregate
        successes = [r for r in request_results if r.success]
        failures = [r for r in request_results if not r.success]
        ttfts = [r.ttft_s for r in successes]
        latencies = [r.total_time_s for r in successes]
        total_output = sum(r.output_tokens for r in successes)

        level_key = str(level)
        results["per_concurrency_level"][level_key] = {
            "wall_time_s": round(wall_time, 3),
            "total_output_tokens": total_output,
            "aggregate_throughput_tok_s": round(total_output / wall_time, 1) if wall_time > 0 else None,
            "n_requests": len(request_results),
            "n_success": len(successes),
            "n_failed": len(failures),
            "success_rate": round(len(successes) / len(request_results), 3) if request_results else 0.0,
            "ttft": _stat_summary(ttfts),
            "total_time_s": _stat_summary(latencies),
            "individual_requests": [
                {
                    "index": r.index,
                    "success": r.success,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "ttft_s": round(r.ttft_s, 4),
                    "total_time_s": round(r.total_time_s, 4),
                    "error": r.error,
                }
                for r in request_results
            ],
        }

    return results


if __name__ == "__main__":
    import json
    import sys

    # Quick smoke test: python -m benchmarks.concurrency <model_config>
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/qwen3.6_35b_redhat_nvfp4.yml")
    from core_runner import load_model_config, make_run_dir

    cfg = load_model_config(cfg_path)
    client = ModelClient(
        cfg["endpoint"]["base_url"],
        cfg["endpoint"]["model_name"],
        cfg["endpoint"].get("api_key"),
        cfg["endpoint"].get("chat", True),
    )
    results = run_concurrency_test(client, concurrency_levels=[1, 2, 4], requests_per_level=3)
    print(json.dumps(results, indent=2))