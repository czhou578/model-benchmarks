"""TTFT and prefill throughput: the latency sweep and the deep context benchmark."""

from __future__ import annotations

import statistics
import uuid
from typing import Any

import requests

from harness.client import ModelClient
from harness.prompts import build_prompt_of_length
from harness.stats import percentile


def run_latency_sweep(client: ModelClient, prompt_lengths: list[int],
                       repeats: int, decode_tokens_for_ttft: int = 8) -> dict[str, Any]:
    results = {}
    for plen in prompt_lengths:
        prompt = build_prompt_of_length(plen)
        ttfts = []
        prefill_tps = []
        exact = None
        for _ in range(repeats):
            # A fresh salt per request keeps the prefix cache from serving
            # repeats 2..N, which would collapse TTFT to the cache-hit time.
            gen = client.generate(
                prompt, max_tokens=decode_tokens_for_ttft, temperature=0.0,
                cache_salt=uuid.uuid4().hex,
            )
            exact = gen.prompt_tokens_exact
            if gen.ttft_s == gen.ttft_s:
                ttfts.append(gen.ttft_s)
                if gen.ttft_s > 0:
                    prefill_tps.append(gen.prompt_tokens / gen.ttft_s)
        results[str(plen)] = {
            "requested_prompt_tokens": plen,
            "prompt_tokens_exact": exact,
            "n": len(ttfts),
            "ttft_avg_s": round(statistics.mean(ttfts), 4) if ttfts else None,
            "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
            "ttft_p95_s": round(percentile(ttfts, 95), 4) if ttfts else None,
            "ttft_p99_s": round(percentile(ttfts, 99), 4) if ttfts else None,
            "prefill_tps_avg": round(statistics.mean(prefill_tps), 1) if prefill_tps else None,
        }
    return results


def run_deep_context(client: ModelClient, context_lengths: list[int],
                      output_tokens: int = 64, repeats: int = 5) -> dict[str, Any]:
    """Measure TTFT and prefill throughput at extended context lengths.

    For each context length, builds a prompt of that length and generates
    a small decode (``output_tokens``). Captures TTFT and prefill TPS.
    Each request carries a fresh cache salt, so repeats never hit the
    prefix cache. On OOM it records ``status: "oom"``. On another request
    failure it records ``status: "request_error"``. Both keep the error text
    in ``error`` and go on to the next length.
    """
    results = {}
    for ctx_len in context_lengths:
        label = str(ctx_len)
        prompt = build_prompt_of_length(ctx_len)
        ttfts: list[float] = []
        prefill_tps: list[float] = []
        exact = None
        status = "success"
        error: str | None = None
        for _i in range(repeats):
            try:
                gen = client.generate(
                    prompt, max_tokens=output_tokens, temperature=0.0,
                    cache_salt=uuid.uuid4().hex,
                )
                exact = gen.prompt_tokens_exact
                if gen.ttft_s == gen.ttft_s and gen.ttft_s > 0:
                    ttfts.append(gen.ttft_s)
                    prefill_tps.append(gen.prompt_tokens / gen.ttft_s)
            except Exception as e:  # Record the error and go to the next length.
                error = f"{type(e).__name__}: {e}"
                text = str(e).lower()
                if isinstance(e, requests.exceptions.ConnectionError) or (
                    isinstance(e, requests.exceptions.HTTPError)
                    and ("memory" in text or "oom" in text)
                ):
                    status = "oom"
                else:
                    status = "request_error"
                print(f"  deep context {label}: {status}: {error}")
                break
        results[label] = {
            "requested_context_tokens": ctx_len,
            "prompt_tokens_exact": exact,
            "n": len(ttfts),
            "ttft_avg_s": round(statistics.mean(ttfts), 4) if ttfts else None,
            "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
            "ttft_p95_s": round(percentile(ttfts, 95), 4) if ttfts else None,
            "prefill_tps_avg": round(statistics.mean(prefill_tps), 1) if prefill_tps else None,
            "status": status,
        }
        if error is not None:
            results[label]["error"] = error
    return results

