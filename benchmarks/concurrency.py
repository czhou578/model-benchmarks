"""
concurrency.py — Measure throughput and latency degradation as request
concurrency increases.

How does aggregate throughput change when I fire 1, 4, 16, or 32 requests
simultaneously? At what point does the GPU bottleneck show up?
"""

from __future__ import annotations

import time
import concurrent.futures
from typing import Any

from core_runner import (
    ModelClient,
    _stat_summary,
    build_prompt_of_length,
    count_tokens,
)


# --------------------------------------------------------------------------- #
# Test prompt — reuse the creative-writing prompt, deterministic 256 tokens
# --------------------------------------------------------------------------- #

CONCURRENCY_PROMPT = build_prompt_of_length(256)


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
        request_results: list[dict[str, Any]] = []

        def _run_one(idx: int) -> dict[str, Any]:
            try:
                gen = client.generate(CONCURRENCY_PROMPT, max_tokens=max_tokens, temperature=temperature)
                return {
                    "success": True,
                    "index": idx,
                    "prompt_tokens": gen.prompt_tokens,
                    "output_tokens": gen.output_tokens,
                    "ttft_s": gen.ttft_s,
                    "total_time_s": gen.total_time_s,
                }
            except Exception as e:
                return {"success": False, "index": idx, "error": str(e)}

        start_wall = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(_run_one, i) for i in range(requests_per_level)]
            for f in concurrent.futures.as_completed(futures):
                request_results.append(f.result())
        wall_time = time.time() - start_wall

        successes = [r for r in request_results if r["success"]]
        failures = [r for r in request_results if not r["success"]]
        ttfts = [r["ttft_s"] for r in successes]
        latencies = [r["total_time_s"] for r in successes]
        total_output = sum(r["output_tokens"] for r in successes)

        results["per_concurrency_level"][str(level)] = {
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
                    "index": r["index"],
                    "success": r["success"],
                    "prompt_tokens": r["prompt_tokens"],
                    "output_tokens": r["output_tokens"],
                    "ttft_s": round(r["ttft_s"], 4),
                    "total_time_s": round(r["total_time_s"], 4),
                    "error": r.get("error", ""),
                }
                for r in request_results
            ],
        }

    return results