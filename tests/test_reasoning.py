"""Tests for the reasoning benchmark: <think> parsing, truncation, and the salt."""

from __future__ import annotations

from typing import Any

from benchmarks import reasoning
from harness.client import GenerationResult
from harness.prompts import count_tokens


def test_think_tags_split_the_output():
    stats = reasoning.analyze_reasoning_tokens("<think>a b c</think>The answer is 80.")
    assert stats["thinking_tokens"] == count_tokens("a b c")
    assert stats["answer_tokens"] == count_tokens("The answer is 80.")


def test_think_close_without_open_tag():
    # The chat template can put <think> in the prompt.
    stats = reasoning.analyze_reasoning_tokens("a b c</think>80 mph")
    assert stats["thinking_tokens"] == count_tokens("a b c")
    assert stats["answer_tokens"] == count_tokens("80 mph")


def test_unclosed_think_is_all_thinking():
    stats = reasoning.analyze_reasoning_tokens("<think>still thinking")
    assert stats["thinking_tokens"] == count_tokens("still thinking")
    assert stats["answer_tokens"] == 0


class _Client:
    def __init__(self, gens: list[GenerationResult]) -> None:
        self.gens = gens
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        self.calls.append(kwargs)
        return self.gens[len(self.calls) - 1]


def _gen(reasoning_text: str, answer_text: str, finish: str | None) -> GenerationResult:
    return GenerationResult(
        prompt_tokens=10, ttft_s=0.1, total_time_s=1.0, output_text=reasoning_text + answer_text,
        output_tokens=50, reasoning_text=reasoning_text, answer_text=answer_text, finish_reason=finish,
    )


def test_benchmark_records_truncation_split_method_and_salt():
    client = _Client([
        _gen("think hard", "80 mph", "stop"),
        _gen("", "<think>loop forever", "length"),
        _gen("", "8 sheep", None),
    ])
    result = reasoning.run_reasoning_benchmark(client, ["p1", "p2", "p3"], max_tokens=4096)

    assert all(c["max_tokens"] == 4096 for c in client.calls)
    salts = [c["cache_salt"] for c in client.calls]
    assert all(salts) and len(set(salts)) == 3
    first, second, third = result["per_prompt"]
    assert first["split_method"] == "reasoning_parser" and first["truncated"] is False
    assert second["split_method"] == "tags" and second["truncated"] is True
    assert second["answer_tokens"] == 0 and second["thinking_tokens"] > 0
    assert third["split_method"] == "none" and third["truncated"] is None
    assert third["answer_tokens"] == count_tokens("8 sheep")
    assert result["max_tokens"] == 4096 and result["n_truncated"] == 1
