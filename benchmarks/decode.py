"""Decode speed at fixed output lengths, and the speculative decoding comparison."""

from __future__ import annotations

import statistics
from typing import Any, Optional

from benchmarks.reasoning import analyze_reasoning_tokens
from harness.client import ModelClient
from harness.prompts import build_prompt_of_length, count_tokens


def run_decode_speed(client: ModelClient, output_lengths: list[int],
                      fixed_prompt_tokens: int = 128) -> dict[str, Any]:
    prompt = build_prompt_of_length(fixed_prompt_tokens)
    results = {}
    for n_out in output_lengths:
        gen = client.generate(prompt, max_tokens=n_out, temperature=0.0)
        decode_time = gen.total_time_s - gen.ttft_s if gen.ttft_s == gen.ttft_s else gen.total_time_s
        tok_per_sec_overall = gen.output_tokens / decode_time if decode_time > 0 else None
        instant_rates = [1.0 / g for g in gen.per_token_times if g > 0]
        output_truncated = gen.output_tokens != n_out

        # Analyze reasoning tokens (thinking vs answer)
        # Use the already-split reasoning and answer text from the model client
        reasoning_tokens_count: int = 0
        answer_tokens_count: int = 0
        reasoning_ratio: Optional[float] = None
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
            "output_truncated": output_truncated,
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

