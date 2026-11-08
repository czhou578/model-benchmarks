"""Tests for the distinct prompts of the concurrency benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from benchmarks import concurrency
from core_runner import count_tokens

TEMPLATE_TOKENS = 12


@dataclass
class _Count:
    count: int


class _FakeTokenizer:
    """The chat template adds 12 tokens. The text costs 4 characters per token."""

    def tokenize_prompt(self, prompt: str) -> _Count:
        return _Count(TEMPLATE_TOKENS + len(prompt) // 4)


class _BrokenTokenizer:
    def tokenize_prompt(self, prompt: str) -> _Count:
        raise requests.HTTPError("404 Not Found")


@dataclass
class _Gen:
    prompt_tokens: int
    output_tokens: int = 8
    ttft_s: float = 0.01
    total_time_s: float = 0.05


class _FakeClient(_FakeTokenizer):
    def __init__(self, salt_supported: bool = True) -> None:
        self.salt_supported = salt_supported
        self.calls: list[tuple[str, str | None]] = []

    def preflight_cache_salt(self) -> bool:
        return self.salt_supported

    def generate(self, prompt, max_tokens=256, temperature=0.0, cache_salt=None):
        self.calls.append((prompt, cache_salt))
        return _Gen(prompt_tokens=self.tokenize_prompt(prompt).count)


def test_prompts_are_distinct_from_the_first_words():
    prompts, _ = concurrency.build_distinct_prompts(80, 256, _FakeTokenizer())

    assert len(set(prompts)) == 80
    # The opening differs, not only the end of the text.
    assert len({p[:64] for p in prompts}) == 80
    assert len({p.split()[0] for p in prompts[:16]}) == 16


def test_prompts_are_deterministic():
    first, _ = concurrency.build_distinct_prompts(20, 256, _FakeTokenizer())
    second, _ = concurrency.build_distinct_prompts(20, 256, _FakeTokenizer())

    assert first == second


def test_server_tokenizer_gives_the_exact_length():
    prompts, info = concurrency.build_distinct_prompts(40, 256, _FakeTokenizer())

    assert all(_FakeTokenizer().tokenize_prompt(p).count == 256 for p in prompts)
    assert info == {
        "prompt_length_method": "server_tokenize",
        "prompt_tokens": 256,
        "prompt_tokens_exact": True,
        "prompt_tokens_min": 256,
        "prompt_tokens_max": 256,
    }


def test_client_count_sets_the_length_when_the_tokenizer_fails():
    prompts, info = concurrency.build_distinct_prompts(10, 256, _BrokenTokenizer())

    assert info["prompt_length_method"] == "client_estimate"
    assert len(set(prompts)) == 10
    assert {count_tokens(p) for p in prompts} == {256}
    assert info["prompt_tokens_min"] == info["prompt_tokens_max"] == 256


def test_run_sends_a_distinct_prompt_and_salt_per_request():
    client = _FakeClient()

    result = concurrency.run_concurrency_test(client, [1, 4], requests_per_level=4, max_tokens=8)

    prompts = [p for p, _ in client.calls]
    salts = [s for _, s in client.calls]
    assert len(set(prompts)) == 8
    assert None not in salts and len(set(salts)) == 8
    config = result["config"]
    assert config["prompt_mode"] == "distinct"
    assert config["prompt_tokens"] == 256 and config["prompt_tokens_exact"] is True
    assert config["cache_isolation_method"] == "cache_salt"
    level_4 = result["per_concurrency_level"]["4"]["individual_requests"]
    assert sorted(r["prompt_index"] for r in level_4) == [4, 5, 6, 7]
    assert {r["prompt_tokens"] for r in level_4} == {256}


def test_run_skips_the_salt_when_the_server_rejects_it():
    client = _FakeClient(salt_supported=False)

    result = concurrency.run_concurrency_test(client, [2], requests_per_level=2)

    assert [s for _, s in client.calls] == [None, None]
    assert result["config"]["cache_isolation_method"] == "distinct_text"
