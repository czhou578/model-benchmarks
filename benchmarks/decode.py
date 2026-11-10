"""Decode speed at fixed output lengths, and the speculative decoding comparison."""

from __future__ import annotations

import statistics
import uuid
from typing import Any

from benchmarks.reasoning import analyze_reasoning_tokens
from harness.client import ModelClient
from harness.prompts import build_prompt_of_length, count_tokens


def event_rates(gaps: list[float], token_deltas: list[int] | None) -> tuple[list[float], str]:
    """Return the rate of each gap between stream events, and the method.

    ``token_delta``: tokens that arrived in the gap / gap. The server sent
    per-chunk usage, so a multi-token event (speculative decoding) counts
    all its tokens. ``event_gap``: 1 / gap, in events per second. The
    fallback when the server sends no per-chunk usage.
    """
    if token_deltas is not None and len(token_deltas) == len(gaps):
        return [d / g for d, g in zip(token_deltas, gaps, strict=True) if g > 0], "token_delta"
    return [1.0 / g for g in gaps if g > 0], "event_gap"


def run_decode_speed(client: ModelClient, output_lengths: list[int],
                      fixed_prompt_tokens: int = 128) -> dict[str, Any]:
    prompt = build_prompt_of_length(fixed_prompt_tokens)
    results = {}
    for n_out in output_lengths:
        # A new salt per request keeps a cached prefill out of the TTFT.
        gen = client.generate(prompt, max_tokens=n_out, temperature=0.0,
                              cache_salt=uuid.uuid4().hex, continuous_usage=True)
        decode_time = gen.total_time_s - gen.ttft_s if gen.ttft_s == gen.ttft_s else gen.total_time_s
        tok_per_sec_overall = gen.output_tokens / decode_time if decode_time > 0 else None
        instant_rates, rate_method = event_rates(gen.per_token_times, gen.per_event_token_deltas)

        # Analyze reasoning tokens (thinking vs answer)
        # Use the already-split reasoning and answer text from the model client
        reasoning_tokens_count: int = 0
        answer_tokens_count: int = 0
        reasoning_ratio: float | None = None
        if gen.reasoning_text:
            reasoning_tokens_count = count_tokens(gen.reasoning_text)
            answer_tokens_count = count_tokens(gen.answer_text)
            if answer_tokens_count:
                reasoning_ratio = round(reasoning_tokens_count / answer_tokens_count, 3)
        elif gen.output_text:
            # Fall back to analyzing combined output_text for models without split
            reasoning_stats = analyze_reasoning_tokens(gen.output_text)
            reasoning_tokens_count = reasoning_stats["thinking_tokens"]
            answer_tokens_count = reasoning_stats["answer_tokens"]
            reasoning_ratio = reasoning_stats.get("ratio_thinking_to_answer")

        text_preview = gen.output_text if len(gen.output_text) <= 512 else (
            gen.output_text[:512] + f"\n  [truncated, total length: {len(gen.output_text)} chars]"
        )
        results[str(n_out)] = {
            "requested_output_tokens": n_out,
            "actual_output_tokens": gen.output_tokens,
            "output_tokens_exact": gen.output_tokens_exact,
            "stopped_early": gen.output_tokens < n_out,
            "finish_reason": gen.finish_reason,
            "rate_method": rate_method,
            "ttft_s": round(gen.ttft_s, 4) if gen.ttft_s == gen.ttft_s else None,
            "decode_time_s": round(decode_time, 3),
            "tok_per_sec_avg": round(tok_per_sec_overall, 2) if tok_per_sec_overall else None,
            "tok_per_sec_peak": round(max(instant_rates), 2) if instant_rates else None,
            "tok_per_sec_min": round(min(instant_rates), 2) if instant_rates else None,
            "tok_per_sec_median": round(statistics.median(instant_rates), 2) if instant_rates else None,
            "output_text_preview": text_preview,
            "reasoning_tokens": reasoning_tokens_count,
            "answer_tokens": answer_tokens_count,
            "reasoning_ratio": reasoning_ratio,
        }
    return results


def compare_spec_decode_results(spec_results: dict[str, Any],
                                no_spec_results: dict[str, Any],
                                output_lengths: list[int]) -> dict[str, Any]:
    """Compare previously collected speculative and baseline decode results."""

    delta: dict[str, Any] = {}
    for key in spec_results:
        spec_entry = spec_results[key]
        no_spec_entry = no_spec_results.get(key, {})
        spec_tps = spec_entry.get("tok_per_sec_avg")
        no_spec_tps = no_spec_entry.get("tok_per_sec_avg")
        spec_ttft = spec_entry.get("ttft_s")
        no_spec_ttft = no_spec_entry.get("ttft_s")

        tps_pct = None
        if no_spec_tps and no_spec_tps > 0:
            tps_pct = round((spec_tps - no_spec_tps) / no_spec_tps * 100, 2) if spec_tps else None

        ttft_pct = None
        if no_spec_ttft and no_spec_ttft > 0:
            ttft_pct = round((spec_ttft - no_spec_ttft) / no_spec_ttft * 100, 2) if spec_ttft else None

        delta[key] = {
            "tok_s_improvement_pct": tps_pct,
            "ttft_change_pct": ttft_pct,
        }

    return {
        "config": {
            "output_lengths": output_lengths,
        },
        "spec_enabled": spec_results,
        "spec_disabled": no_spec_results,
        "delta": delta,
    }

