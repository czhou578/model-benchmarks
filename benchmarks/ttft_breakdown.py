"""TTFT Breakdown benchmark — Phase 1.3 of the upgrade roadmap.

Splits client-side TTFT (wall-clock time to first token) into:
  1. Scheduler / queue delay        (server-side queue_time_s)
  2. Prefill time                   (server-side prompt_time_s)
  3. First decode + scheduling      (TTFT − queue − prefill)

The gap in (3) is the time from the last prefill token to the first output
token — it includes the first decode kernel launch + scheduler overhead.

vLLM already emits request_metrics with these fields per stream chunk, but
they live only inside ModelClient._execute_request() and the GenerationResult
dataclass. This module exposes them as a structured benchmark.

Output file: ttft_breakdown.json
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from core_runner import ModelClient, GpuMonitor


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TtftRequestResult:
    """Single request within the TTFT breakdown benchmark."""

    index: int
    success: bool
    # Token counts
    prompt_tokens: int
    prompt_tokens_exact: bool
    output_tokens: int
    # Client-side wall-clock (ms)
    ttft_ms: float
    total_time_ms: float
    # Server-side metrics (s) — may be None on older vLLM versions
    queue_time_s: float | None = None
    prefill_time_s: float | None = None
    server_ttft_s: float | None = None
    # Derived breakdown components (ms)
    scheduler_delay_ms: float | None = None
    prefill_ms: float | None = None
    first_decode_ms: float | None = None
    # Cache
    cached_tokens: int = 0
    # Error
    error: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ms(v: float | None) -> float | None:
    """Convert seconds to ms, preserving None."""
    return round(v * 1000, 4) if v is not None else None


def _stat_summary(values: list[float]) -> dict[str, Any]:
    """Compute avg, median, p95, min, max for a list of floats."""
    if not values:
        return {
            "avg_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    s = sorted(values)
    k = (len(s) - 1) * 0.95
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    p95 = s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]
    return {
        "avg_ms": round(statistics.mean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "p95_ms": round(p95, 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
    }


def _probe_server_metrics(client: ModelClient) -> bool:
    """Send a probe request to check if the server emits request_metrics.

    Returns True if request_metrics is found in a streamed response chunk,
    False otherwise (or on connection failure).
    """
    try:
        probe_payload: dict[str, Any] = {
            "model": client.model_name,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
        }
        if client.chat:
            probe_payload["messages"] = [{"role": "user", "content": "hello"}]
        else:
            probe_payload["prompt"] = "hello"

        url = (
            f"{client.base_url}/v1/chat/completions"
            if client.chat
            else f"{client.base_url}/v1/completions"
        )
        resp = requests.post(
            url, headers=client.headers, json=probe_payload, stream=True, timeout=30
        )
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            line = raw_line.decode("utf-8")
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return False  # never saw request_metrics
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("request_metrics"):
                return True
        return False
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_ttft_breakdown(
    client: ModelClient,
    prompt_lengths: list[int] | None = None,
    repetitions: int = 10,
    gpu_monitor: GpuMonitor | None = None,
) -> dict[str, Any]:
    """Measure TTFT breakdown across a range of prompt lengths.

    For each length, sends ``repetitions`` cold (no-cache) requests and
    records the server-side timing metrics that vLLM emits in
    ``request_metrics`` per streamed chunk.

    Args:
        client: ModelClient connected to a running vLLM endpoint.
        prompt_lengths: Prompt lengths in tokens (default 128, 512, 2K, 8K, 32K).
        repetitions: Requests per length.
        gpu_monitor: Optional GPU telemetry collector.

    Returns:
        Dict with ``config`` metadata and per-length ``per_length`` results.
    """
    if prompt_lengths is None:
        prompt_lengths = [128, 512, 2048, 8192, 32768]

    # Detect cache isolation
    is_header = client.preflight_cache_salt()

    # Probe server-side metrics availability
    server_metrics_supported = _probe_server_metrics(client)
    if not server_metrics_supported:
        print(
            "[ttft_breakdown] WARNING: server does not emit request_metrics. "
            "Sub-components will be null. "
            "Ensure vLLM >= 0.6.0 with --disable-log-requests off."
        )

    # Build a simple text-based prompt for each length.
    # We use a simple repetition so the prompt is guaranteed to be at least
    # the target token count.  The prefill.py module uses a proper tokenizer
    # to hit exact lengths — that would be a future enhancement.
    prompts: dict[str, str] = {}
    for length in prompt_lengths:
        # "hello world" ≈ 2 tokens; pad to reach target
        words = "hello world " * max(1, length // 2)
        prompts[str(length)] = words

    # Struct result
    result: dict[str, Any] = {
        "config": {
            "benchmark_version": "1.0",
            "definition": "TTFT decomposition into scheduler delay, prefill time, and first decode overhead",
            "prompt_lengths": prompt_lengths,
            "repetitions": repetitions,
            "cache_isolation_method": "cache_salt" if is_header else "text_salt",
            "server_metrics_supported": server_metrics_supported,
            "max_tokens": 1,
            "temperature": 0.0,
            "start_time": datetime.now(datetime.timezone.utc).isoformat(),
        },
        "per_length": {},
    }

    stopped = False
    for length in prompt_lengths:
        length_key = str(length)

        if stopped:
            result["per_length"][length_key] = {"status": "skipped_after_oom"}
            continue

        # GPU telemetry window for this length
        if gpu_monitor is not None:
            gpu_monitor.start_window(f"ttft_length_{length_key}")

        prompt_text = prompts.get(length_key, "x " * length)

        length_results: list[TtftRequestResult] = []
        length_status = "success"

        for req_idx in range(repetitions):
            req_salt = uuid.uuid4().hex

            # Send generation with cache isolation
            try:
                gen = client.generate(prompt_text, max_tokens=1, cache_salt=req_salt)
            except Exception as exc:
                msg = str(exc).lower()
                if "memory" in msg or "oom" in msg:
                    length_status = "oom"
                    stopped = True

                length_results.append(
                    TtftRequestResult(
                        index=req_idx,
                        success=False,
                        prompt_tokens=0,
                        prompt_tokens_exact=False,
                        output_tokens=0,
                        ttft_ms=0.0,
                        total_time_ms=0.0,
                        error=str(exc),
                    )
                )
                continue

            # Extract server-side metrics (may be None)
            queue_s = gen.queue_time_s
            prefill_s = gen.prefill_time_s
            server_ttft = gen.time_to_first_token_s

            # Compute derived breakdown (ms)
            sched_delay_s = None
            pref_s = None
            first_dec_s = None

            if server_metrics_supported and queue_s is not None:
                sched_delay_s = queue_s
            if server_metrics_supported and prefill_s is not None:
                pref_s = prefill_s

            # first_decode = TTFT − queue − prefill
            if (
                server_metrics_supported
                and queue_s is not None
                and prefill_s is not None
                and server_ttft is not None
            ):
                first_dec_s = max(0.0, server_ttft - queue_s - prefill_s)
            elif server_ttft is not None:
                first_dec_s = server_ttft

            client_ttft_ms = _ms(gen.ttft_s) if gen.ttft_s is not None else None
            total_ms = _ms(gen.total_time_s) if gen.total_time_s is not None else None

            length_results.append(
                TtftRequestResult(
                    index=req_idx,
                    success=True,
                    prompt_tokens=gen.prompt_tokens,
                    prompt_tokens_exact=gen.prompt_tokens_exact,
                    output_tokens=gen.output_tokens,
                    ttft_ms=client_ttft_ms,
                    total_time_ms=total_ms,
                    queue_time_s=queue_s,
                    prefill_time_s=prefill_s,
                    server_ttft_s=server_ttft,
                    scheduler_delay_ms=_ms(sched_delay_s),
                    prefill_ms=_ms(pref_s),
                    first_decode_ms=_ms(first_dec_s),
                    cached_tokens=gen.cached_tokens,
                    error="",
                )
            )

            if req_idx < repetitions - 1:
                time.sleep(0.5)  # stabilization gap

        # Stop GPU window for this length
        gpu_summary = None
        if gpu_monitor is not None:
            gpu_summary = gpu_monitor.stop_window(f"ttft_length_{length_key}")

        # Aggregate
        successes = [r for r in length_results if r.success]
        n_success = len(successes)
        n_failed = len(length_results) - n_success

        # Collect breakdown values in ms
        ttfts = [r.ttft_ms for r in successes if r.ttft_ms is not None]
        sched_delays = [
            r.scheduler_delay_ms for r in successes if r.scheduler_delay_ms is not None
        ]
        prefills = [r.prefill_ms for r in successes if r.prefill_ms is not None]
        first_decs = [
            r.first_decode_ms for r in successes if r.first_decode_ms is not None
        ]

        # Cache hit rate among successes
        cache_hits = sum(1 for r in successes if r.cached_tokens > 0)

        # Serialize per-request data for JSON output
        per_request_serialized = [
            {
                "index": r.index,
                "success": r.success,
                "prompt_tokens": r.prompt_tokens,
                "prompt_tokens_exact": r.prompt_tokens_exact,
                "output_tokens": r.output_tokens,
                "ttft_ms": r.ttft_ms,
                "total_time_ms": r.total_time_ms,
                "queue_time_s": r.queue_time_s,
                "prefill_time_s": r.prefill_time_s,
                "server_ttft_s": r.server_ttft_s,
                "scheduler_delay_ms": r.scheduler_delay_ms,
                "prefill_ms": r.prefill_ms,
                "first_decode_ms": r.first_decode_ms,
                "cached_tokens": r.cached_tokens,
                "error": r.error,
            }
            for r in length_results
        ]

        # Use first success's prompt_tokens as actual_tokens
        actual_tokens = successes[0].prompt_tokens if successes else length

        result["per_length"][length_key] = {
            "status": length_status,
            "requested_tokens": length,
            "actual_tokens": actual_tokens,
            "n_requests": len(length_results),
            "n_success": n_success,
            "n_failed": n_failed,
            "cache_hit_rate": round(cache_hits / n_success, 3) if n_success else None,
            "per_request": per_request_serialized,
            "aggregated": {
                "ttft": _stat_summary(ttfts),
                "scheduler_delay": _stat_summary(sched_delays),
                "prefill_time": _stat_summary(prefills),
                "first_decode": _stat_summary(first_decs),
                "gpu": gpu_summary or {},
            },
        }

    result["config"]["end_time"] = datetime.now(datetime.timezone.utc).isoformat()
    return result