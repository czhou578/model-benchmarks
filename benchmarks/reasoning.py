"""Reasoning benchmark: thinking tokens and answer tokens per prompt."""

from __future__ import annotations

import statistics
import uuid
from typing import Any

from harness.client import ModelClient
from harness.prompts import count_tokens


def analyze_reasoning_tokens(text: str) -> dict[str, Any]:
    """Split a model output into thinking tokens and answer tokens.

    The function checks four formats in this order:

    1. ``<antThinking>...</antThinking>`` then the answer.
    2. ``<thinking>...</thinking>`` then the answer.
    3. Qwen3 ``<think>...</think>`` then the answer. The chat template can
       put ``<think>`` in the prompt, so the output can start without it.
       All text before ``</think>`` is then thinking. A ``<think>`` without
       ``</think>`` (the output hit ``max_tokens``) is all thinking.
    4. Plain text: a reasoning marker phrase, such as "let me think", in
       the first 500 characters, with list or step markup after it. The text
       before the marker counts as thinking, the rest as the answer.
       Without such a marker, all text is the answer.

    The vLLM qwen3 reasoning parser moves the ``<think>`` text to
    ``delta.reasoning``, so the benchmarks call this function only when the
    stream has no reasoning text. Counts come from ``count_tokens``.
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

    # Format 3: Qwen3 <think> tags. The opening tag is optional.
    if "</think>" in text or text.lstrip().startswith("<think>"):
        head, closed, tail = text.partition("</think>")
        thinking = head.replace("<think>", "", 1)
        answer = tail if closed else ""
        return _split_counts(thinking, answer)

    # Format 4: Plain-text reasoning markers
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


def _split_counts(thinking: str, answer: str) -> dict[str, Any]:
    think_tokens = count_tokens(thinking)
    answer_tokens = count_tokens(answer)
    return {
        "thinking_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "ratio_thinking_to_answer": round(think_tokens / answer_tokens, 3) if answer_tokens else None,
    }


THINK_TAGS = ("<think>", "</think>", "<thinking>", "<antThinking>")


def run_reasoning_benchmark(client: ModelClient, prompts: list[str], max_tokens: int = 4096) -> dict[str, Any]:
    per_prompt = []
    for p in prompts:
        # A new salt per request keeps a cached prefill out of the run.
        gen = client.generate(p, max_tokens=max_tokens, temperature=0.0, cache_salt=uuid.uuid4().hex)

        # First source: the reasoning parser. delta.reasoning is the thinking
        # text, and delta.content (or the completions text) is the answer.
        # Without a parser, the tags in the output text split it.
        if gen.reasoning_text:
            stats = _split_counts(gen.reasoning_text, gen.answer_text)
            stats["split_method"] = "reasoning_parser"
        elif any(tag in gen.output_text for tag in THINK_TAGS):
            stats = analyze_reasoning_tokens(gen.output_text)
            stats["split_method"] = "tags"
        else:
            stats = _split_counts("", gen.output_text)
            stats["split_method"] = "none"
        stats["reasoning_text_preview"] = gen.reasoning_text[:128]
        stats["answer_text_preview"] = gen.answer_text[:128]
        stats["output_tokens"] = gen.output_tokens
        stats["finish_reason"] = gen.finish_reason
        # True when max_tokens cut the output. None when the server sent no finish_reason.
        stats["truncated"] = None if gen.finish_reason is None else gen.finish_reason == "length"
        stats["prompt_preview"] = p[:80]
        per_prompt.append(stats)

    think_lens = [r["thinking_tokens"] for r in per_prompt]
    answer_lens = [r["answer_tokens"] for r in per_prompt]
    return {
        "max_tokens": max_tokens,
        "n_truncated": sum(1 for r in per_prompt if r["truncated"]),
        "per_prompt": per_prompt,
        "thinking_tokens_avg": round(statistics.mean(think_lens), 1) if think_lens else None,
        "thinking_tokens_max": max(think_lens) if think_lens else None,
        "thinking_tokens_median": statistics.median(think_lens) if think_lens else None,
        "answer_tokens_avg": round(statistics.mean(answer_lens), 1) if answer_lens else None,
        "answer_tokens_max": max(answer_lens) if answer_lens else None,
        "answer_tokens_median": statistics.median(answer_lens) if answer_lens else None,
    }

