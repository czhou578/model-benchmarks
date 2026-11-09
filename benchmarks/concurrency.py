"""
concurrency.py — Measure throughput and latency degradation as request
concurrency increases.

How does aggregate throughput change when I fire 1, 4, 16, or 32 requests
simultaneously? At what point does the GPU bottleneck show up?

Each request sends its own prompt. All prompts have the same token count.
Their text differs from the first words, so the prefix cache cannot share
blocks between requests, and greedy outputs differ.
"""

from __future__ import annotations

import concurrent.futures
import random
import time
import uuid
from typing import Any

import requests

from benchmarks.prefill import PromptTokenizer, calibrate_prompt
from core_runner import ModelClient
from harness import prompts
from harness.prompts import count_tokens
from harness.stats import stat_summary

PROMPT_TOKENS = 256

# The opening of prompt k takes a subject, a setting, and a style from these
# lists. The subject changes with each k, so the first word changes between
# consecutive prompts. 16 x 16 x 8 = 2048 distinct openings.
_SUBJECTS = (
    "Lighthouse keepers", "Desert caravans", "Lunar miners", "River ferry pilots",
    "Orbital gardeners", "Glacier surveyors", "Night-market cooks", "Deep-sea archivists",
    "Mountain rail engineers", "Coral reef doctors", "Storm-chasing pilots", "Forest cartographers",
    "Clockmaker apprentices", "Volcano watchers", "Polar radio operators", "Canal city mayors",
)
_SETTINGS = (
    "the year 1820", "a flooded future city", "a drifting ice shelf", "a generation starship",
    "a medieval harbor", "a Martian colony", "an island monastery", "a cloud-top research station",
    "a sunken subway", "a border mountain pass", "a salt-flat town", "a terraced rice valley",
    "an abandoned observatory", "a floating market", "a tundra outpost", "a canyon library",
)
_STYLES = (
    "as a field diary", "as a folk tale", "as a mystery", "as a series of letters",
    "as an adventure", "as a quiet drama", "as a ship's log", "as an oral history",
)
MAX_DISTINCT_PROMPTS = len(_SUBJECTS) * len(_SETTINGS) * len(_STYLES)

_DIRECTIONS = (
    "Describe the place in detail: its weather, its sounds, and its light.",
    "Introduce three characters with different goals and give each a clear voice.",
    "Explain the daily work of the characters and the tools that they use.",
    "Add a conflict that grows slowly and forces a difficult choice.",
    "Include a short scene of dialogue that shows a disagreement.",
    "Describe a discovery that changes how the characters see their world.",
    "Show how the community trades, celebrates, and settles disputes.",
    "Add a journey through dangerous terrain and describe each stage of it.",
    "Describe an old object and the history that it carries.",
    "End with a reflective passage about what the characters learned.",
)


def build_source_text(prompt_index: int, min_chars: int) -> str:
    """Build the deterministic source text of concurrency prompt *prompt_index*.

    The text starts with a distinct opening. A shuffle with a fixed seed
    orders the body sentences. The text has at least *min_chars* characters.
    """
    if not 0 <= prompt_index < MAX_DISTINCT_PROMPTS:
        raise ValueError(f"prompt_index must be in [0, {MAX_DISTINCT_PROMPTS})")
    subject = _SUBJECTS[prompt_index % len(_SUBJECTS)]
    setting = _SETTINGS[(prompt_index // len(_SUBJECTS)) % len(_SETTINGS)]
    style = _STYLES[prompt_index // (len(_SUBJECTS) * len(_SETTINGS))]
    parts = [f"{subject} in {setting}, told {style}: write a detailed, multi-paragraph story."]
    # A string seed gives the same shuffle in every Python process.
    rng = random.Random(f"concurrency-prompt-{prompt_index}")
    size = len(parts[0])
    while size < min_chars:
        order = list(_DIRECTIONS)
        rng.shuffle(order)
        parts.extend(order)
        size += sum(len(s) + 1 for s in order)
    return " ".join(parts)


def _truncate_client(text: str, target_tokens: int) -> str:
    """Cut *text* to *target_tokens* tokens of the client count (``count_tokens``)."""
    enc = prompts._ENC
    if enc is not None:
        return enc.decode(enc.encode(text)[:target_tokens])
    # Without tiktoken, count_tokens estimates from the word count.
    # Keep the longest word prefix whose estimate is at most the target.
    words = text.split()
    low, high = 1, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(" ".join(words[:middle])) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low])


def build_distinct_prompts(
    count: int,
    target_tokens: int,
    tokenizer: PromptTokenizer | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build *count* distinct, deterministic prompts of *target_tokens* tokens.

    With a *tokenizer*, ``calibrate_prompt`` cuts each text to the exact
    server token count, chat template included. When the tokenizer is absent
    or fails, the client count (``count_tokens``) sets the length.

    Returns the prompts and the length fields for the results ``config``.
    """
    if count > MAX_DISTINCT_PROMPTS:
        raise ValueError(f"at most {MAX_DISTINCT_PROMPTS} distinct prompts, got {count}")
    min_chars = max(4096, target_tokens * 8)
    sources = [build_source_text(i, min_chars) for i in range(count)]

    method = "client_estimate"
    prompts: list[str] | None = None
    counts: list[int] = []
    if tokenizer is not None:
        try:
            calibrated = [calibrate_prompt(tokenizer, s, target_tokens) for s in sources]
        except (requests.RequestException, ValueError) as exc:
            print(f"[concurrency] server tokenizer failed ({exc}); using the client token count")
        else:
            method = "server_tokenize"
            prompts = [c.text for c in calibrated]
            counts = [c.actual_tokens for c in calibrated]
    if prompts is None:
        prompts = [_truncate_client(s, target_tokens) for s in sources]
        counts = [count_tokens(p) for p in prompts]

    return prompts, {
        "prompt_length_method": method,
        "prompt_tokens": target_tokens,
        "prompt_tokens_exact": all(c == target_tokens for c in counts),
        "prompt_tokens_min": min(counts, default=None),
        "prompt_tokens_max": max(counts, default=None),
    }


def run_concurrency_test(
    client: ModelClient,
    concurrency_levels: list[int] | None = None,
    requests_per_level: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
    prompt_tokens: int = PROMPT_TOKENS,
) -> dict[str, Any]:
    """Fire concurrent requests and measure throughput / latency scaling.

    Every request of the run gets its own prompt from
    ``build_distinct_prompts``. Prompt k goes to request ``k % requests_per_level``
    of level ``k // requests_per_level``, so a level never repeats a prompt
    of an earlier level. When the server accepts ``cache_salt``, each request
    also carries a new salt.

    Args:
        client: ModelClient instance connected to a vLLM endpoint.
        concurrency_levels: List of concurrency levels to test (default 1,2,4,8,16).
        requests_per_level: How many requests to fire at each concurrency level.
        max_tokens: Max output tokens per request.
        temperature: Generation temperature.
        prompt_tokens: Target token count of each prompt.

    Returns:
        Dict with per-concurrency-level results.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]

    n_prompts = len(concurrency_levels) * requests_per_level
    prompts, length_info = build_distinct_prompts(n_prompts, prompt_tokens, client)
    # Without server support, skip the salt. The text salt fallback of
    # generate() would change the prompt length. Distinct text already
    # keeps the prefix cache from sharing blocks.
    use_salt = client.preflight_cache_salt() is True

    results: dict[str, Any] = {
        "config": {
            "concurrency_levels": concurrency_levels,
            "requests_per_level": requests_per_level,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_mode": "distinct",
            "n_distinct_prompts": n_prompts,
            **length_info,
            "cache_isolation_method": "cache_salt" if use_salt else "distinct_text",
        },
        "per_concurrency_level": {},
    }

    def _run_one(idx: int, prompt_index: int) -> dict[str, Any]:
        try:
            gen = client.generate(
                prompts[prompt_index], max_tokens=max_tokens, temperature=temperature,
                cache_salt=uuid.uuid4().hex if use_salt else None,
            )
            return {
                "success": True,
                "index": idx,
                "prompt_index": prompt_index,
                "prompt_tokens": gen.prompt_tokens,
                "output_tokens": gen.output_tokens,
                "ttft_s": gen.ttft_s,
                "total_time_s": gen.total_time_s,
            }
        except Exception as e:
            return {"success": False, "index": idx, "prompt_index": prompt_index, "error": str(e)}

    for level_idx, level in enumerate(concurrency_levels):
        request_results: list[dict[str, Any]] = []
        offset = level_idx * requests_per_level

        start_wall = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(_run_one, i, offset + i) for i in range(requests_per_level)]
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
            "ttft": stat_summary(ttfts),
            "total_time_s": stat_summary(latencies),
            "individual_requests": [
                {
                    "index": r["index"],
                    "prompt_index": r["prompt_index"],
                    "success": r["success"],
                    # A failed request has no token counts or times.
                    "prompt_tokens": r.get("prompt_tokens"),
                    "output_tokens": r.get("output_tokens"),
                    "ttft_s": round(r["ttft_s"], 4) if "ttft_s" in r else None,
                    "total_time_s": round(r["total_time_s"], 4) if "total_time_s" in r else None,
                    "error": r.get("error", ""),
                }
                for r in request_results
            ],
        }

    return results
