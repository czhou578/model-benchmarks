"""Reasoning benchmark: thinking tokens and answer tokens per prompt."""

from __future__ import annotations

import statistics
from typing import Any

from harness.client import ModelClient
from harness.prompts import count_tokens


def analyze_reasoning_tokens(text: str) -> dict[str, Any]:
    """Split a model output into thinking tokens and answer tokens.

    The function checks three formats in this order:

    1. ``<antThinking>...</antThinking>`` then the answer.
    2. ``<thinking>...</thinking>`` then the answer.
    3. Plain text: a reasoning marker phrase, such as "let me think", in
       the first 500 characters, with list or step markup after it. The text
       before the marker counts as thinking, the rest as the answer.
       Without such a marker, all text is the answer.

    The function does not parse ``<think>``, the tag that Qwen3 writes. The
    vLLM qwen3 reasoning parser moves that text to ``delta.reasoning``, so
    the benchmarks call this function only when the stream has no reasoning
    text. Counts come from ``count_tokens``.
    """
    # Format 1: <antThinking> tags (checked first)
    think_open, think_close = "<antThinking>", "</antThinking>"
    if think_open in text and think_close in text:
        start = text.index(think_open) + len(think_open)
        end = text.index(think_close)
        thinking = text[start:end]
        answer = text[end + len(think_close):]
        think_tokens = count_tokens(thinking)
        answer_tokens = count_tokens(answer)
        return {
            "thinking_tokens": think_tokens,
            "answer_tokens": answer_tokens,
            "ratio_thinking_to_answer": round(think_tokens / answer_tokens, 3) if answer_tokens else None,
        }

    # Format 2: <thinking> tags
    think_open, think_close = "<thinking>", "</thinking>"
    if think_open in text and think_close in text:
        start = text.index(think_open) + len(think_open)
        end = text.index(think_close)
        thinking = text[start:end]
        answer = text[end + len(think_close):]
        think_tokens = count_tokens(thinking)
        answer_tokens = count_tokens(answer)
        return {
            "thinking_tokens": think_tokens,
            "answer_tokens": answer_tokens,
            "ratio_thinking_to_answer": round(think_tokens / answer_tokens, 3) if answer_tokens else None,
        }

    # Format 3: Plain-text reasoning markers
    reasoning_markers = [
        "here's a thinking",
        "here is a thinking",
        "let me think",
        "let me analyze",
        "let's think",
        "step by step",
        "first,",
        "firstly,",
        "to solve",
        "to analyze",
        "break this down",
        "breaking this down",
    ]
    split_at = None
    for marker in reasoning_markers:
        idx = text.find(marker, 0, min(500, len(text)))  # search first 500 chars
        if idx > 0:
            split_at = idx
            break

    if split_at:
        # Check if this looks like reasoning (has structured elements)
        after = text[split_at:split_at + 300]
        has_structure = any(x in after for x in ["\n1.", "**", "* **", "•", "-", "Step", "Phase"])
        if has_structure and len(text) > split_at + 200:
            thinking = text[:split_at]
            answer = text[split_at:]
        else:
            thinking = ""
            answer = text
    else:
        thinking = ""
        answer = text

    think_tokens = count_tokens(thinking)
    answer_tokens = count_tokens(answer)
    return {
        "thinking_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "ratio_thinking_to_answer": round(think_tokens / answer_tokens, 3) if answer_tokens else None,
    }


def run_reasoning_benchmark(client: ModelClient, prompts: list[str], max_tokens: int = 1024) -> dict[str, Any]:
    per_prompt = []
    for p in prompts:
        gen = client.generate(p, max_tokens=max_tokens, temperature=0.0)

        # The stream splits the text: delta.reasoning is the thinking text,
        # and delta.content (or the completions text) is the answer. The
        # fallback runs only when both are empty, so its input is empty too.
        if gen.reasoning_text or gen.answer_text:
            stats = {
                "thinking_tokens": count_tokens(gen.reasoning_text),
                "answer_tokens": count_tokens(gen.answer_text),
                "reasoning_text_preview": gen.reasoning_text[:128] if gen.reasoning_text else "",
                "answer_text_preview": gen.answer_text[:128] if gen.answer_text else "",
                "ratio_thinking_to_answer": round(
                    count_tokens(gen.reasoning_text) / count_tokens(gen.answer_text), 3
                ) if count_tokens(gen.answer_text) else None,
            }
        else:
            stats = analyze_reasoning_tokens(gen.output_text)
            stats["reasoning_text_preview"] = ""
            stats["answer_text_preview"] = ""

        stats["prompt_preview"] = p[:80]
        per_prompt.append(stats)

    think_lens = [r["thinking_tokens"] for r in per_prompt]
    answer_lens = [r["answer_tokens"] for r in per_prompt]
    return {
        "per_prompt": per_prompt,
        "thinking_tokens_avg": round(statistics.mean(think_lens), 1) if think_lens else None,
        "thinking_tokens_max": max(think_lens) if think_lens else None,
        "thinking_tokens_median": statistics.median(think_lens) if think_lens else None,
        "answer_tokens_avg": round(statistics.mean(answer_lens), 1) if answer_lens else None,
        "answer_tokens_max": max(answer_lens) if answer_lens else None,
        "answer_tokens_median": statistics.median(answer_lens) if answer_lens else None,
    }

