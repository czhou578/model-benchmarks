"""Model-specific prompt preparation for the prefill-scaling benchmark.

The approximate tokenizer in core_runner is useful for fallback accounting, but
benchmark lengths must use the tokenizer and chat template of the running model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenCount(Protocol):
    count: int


class PromptTokenizer(Protocol):
    """The small portion of ModelClient needed during calibration."""

    def tokenize_prompt(self, prompt: str) -> TokenCount: ...


@dataclass(frozen=True)
class CalibratedPrompt:
    text: str
    requested_tokens: int
    actual_tokens: int

    @property
    def exact(self) -> bool:
        return self.actual_tokens == self.requested_tokens


_PASSAGES = (
    "The survey vessel crossed the continental shelf before dawn. Its sonar "
    "mapped ridges, sediment fans, and narrow channels while the navigation "
    "team compared each return with observations from earlier expeditions.",
    "Geologists catalogued alternating layers of basalt, clay, and carbonate. "
    "Each sample was photographed, weighed, and sealed before its position was "
    "added to the expedition's chronological field record.",
    "Engineers monitored battery temperature, hydraulic pressure, and acoustic "
    "telemetry throughout the descent. Small control adjustments kept the "
    "vehicle stable despite an unpredictable cross-current.",
    "Biologists recorded translucent fish, drifting colonies, and microbial "
    "mats near the vents. They described behavior and habitat separately so "
    "later reviewers could distinguish observation from interpretation.",
    "Historians reconstructed the coast's trade routes from harbor inventories, "
    "weather logs, and letters between merchants. Conflicting dates were kept "
    "in the archive instead of being silently reconciled.",
    "The final report connected the physical evidence to several competing "
    "hypotheses. It identified unanswered questions and proposed measurements "
    "that a future expedition could use to discriminate among them.",
)


def build_candidate_document(minimum_chars: int) -> str:
    """Build deterministic, document-like ASCII text of at least this size."""
    if minimum_chars <= 0:
        raise ValueError("minimum_chars must be positive")

    sections: list[str] = []
    size = 0
    index = 0
    while size < minimum_chars:
        passage = _PASSAGES[index % len(_PASSAGES)]
        section = f"\n\nField record {index + 1}\n{passage}"
        sections.append(section)
        size += len(section)
        index += 1
    return "".join(sections)


def calibrate_prompt(
    client: PromptTokenizer,
    source: str,
    target_tokens: int,
    *,
    boundary_scan_chars: int = 64,
) -> CalibratedPrompt:
    """Find a source prefix with exactly target_tokens rendered model tokens.

    Binary search locates the relevant character boundary efficiently. A local
    scan handles flat or irregular subword-tokenizer boundaries. If no exact
    boundary exists, the closest under-target prompt is returned with exact
    set to False, so callers cannot accidentally claim an exact length.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if boundary_scan_chars < 0:
        raise ValueError("boundary_scan_chars must not be negative")

    cache: dict[int, int] = {}

    def count_at(end: int) -> int:
        if end not in cache:
            cache[end] = client.tokenize_prompt(source[:end]).count
        return cache[end]

    empty_count = count_at(0)
    if empty_count > target_tokens:
        raise ValueError(
            f"target {target_tokens} is smaller than the rendered empty prompt "
            f"({empty_count} tokens)"
        )

    full_count = count_at(len(source))
    if full_count < target_tokens:
        raise ValueError(
            f"candidate document is too short: {full_count} < {target_tokens} tokens"
        )

    low = 0
    high = len(source)
    best_end = 0
    best_count = empty_count
    while low <= high:
        middle = (low + high) // 2
        count = count_at(middle)
        if count == target_tokens:
            return CalibratedPrompt(source[:middle], target_tokens, count)
        if count < target_tokens:
            if count > best_count or (count == best_count and middle > best_end):
                best_end = middle
                best_count = count
            low = middle + 1
        else:
            high = middle - 1

    scan_start = max(0, best_end - boundary_scan_chars)
    scan_end = min(len(source), max(best_end, low) + boundary_scan_chars)
    for end in range(scan_start, scan_end + 1):
        count = count_at(end)
        if count == target_tokens:
            return CalibratedPrompt(source[:end], target_tokens, count)
        if count < target_tokens and (
            count > best_count or (count == best_count and end > best_end)
        ):
            best_end = end
            best_count = count

    return CalibratedPrompt(source[:best_end], target_tokens, best_count)


def prepare_exact_prompt(
    client: PromptTokenizer,
    target_tokens: int,
    *,
    initial_chars_per_token: int = 5,
    max_growth_attempts: int = 8,
) -> CalibratedPrompt:
    """Generate and calibrate a model-specific prompt of the requested length."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if initial_chars_per_token <= 0:
        raise ValueError("initial_chars_per_token must be positive")
    if max_growth_attempts <= 0:
        raise ValueError("max_growth_attempts must be positive")

    candidate_chars = max(4096, target_tokens * initial_chars_per_token)
    last_count = 0
    for _ in range(max_growth_attempts):
        source = build_candidate_document(candidate_chars)
        last_count = client.tokenize_prompt(source).count
        if last_count >= target_tokens:
            return calibrate_prompt(client, source, target_tokens)
        candidate_chars *= 2

    raise ValueError(
        f"could not build a document of {target_tokens} tokens after "
        f"{max_growth_attempts} attempts; largest candidate was {last_count} tokens"
    )
