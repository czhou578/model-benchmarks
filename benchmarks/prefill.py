"""Model-specific prompt preparation for the prefill-scaling benchmark.

The approximate tokenizer in core_runner is useful for fallback accounting, but
benchmark lengths must use the tokenizer and chat template of the running model.
"""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from core_runner import ModelClient


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


# --------------------------------------------------------------------------- #
# Benchmark helpers
# --------------------------------------------------------------------------- #


def _stat_summary(values: list[float]) -> dict[str, Any]:
    """Compute avg, median, p95, min, max for a list of floats."""
    if not values:
        return {"avg_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None}
    s = sorted(values)
    k = (len(s) - 1) * 0.95
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    p95 = s[f] + (s[c] - s[f]) * (k - f) if f != c else s[f]
    return {
        "avg_s": round(statistics.mean(values), 4),
        "median_s": round(statistics.median(values), 4),
        "p95_s": round(p95, 4),
        "min_s": round(min(values), 4),
        "max_s": round(max(values), 4),
    }


@dataclass(frozen=True)
class PrefillRequestResult:
    """Result from a single prefill request."""

    index: int
    success: bool
    prompt_tokens: int
    prompt_tokens_exact: bool
    client_ttft_s: float
    total_time_s: float
    effective_prefill_tps: float | None
    cache_isolation_method: str
    error: str = ""
    start_time: float | None = None
    end_time: float | None = None
    # vLLM server-side metrics
    cached_tokens: int = 0
    server_ttft_s: float | None = None
    queue_time_s: float | None = None
    prefill_time_s: float | None = None
    engine_prefill_tps: float | None = None


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #

def run_prefill_scaling(
    client: ModelClient,
    target_lengths: list[int] = None,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Measure cold-prefill throughput across a range of prompt lengths.

    Each request uses ``max_tokens=1`` to minimize decode contamination.
    A unique ``cache_salt`` per request guarantees no prefix-cache reuse.

    Args:
        client: ModelClient connected to a running vLLM endpoint.
        target_lengths: Prompt lengths to benchmark (default 512, 2K, 8K, 32K, 64K).
        repetitions: Measured requests per length.

    Returns:
        Dict with ``config`` metadata and per-length ``per_length`` results.
    """
    if target_lengths is None:
        target_lengths = [512, 2048, 8192, 32768, 65536]

    # Detect cache isolation method
    is_header = client.preflight_cache_salt()
    cache_isolation_method = "cache_salt" if is_header else "text_salt"

    results: dict[str, Any] = {
        "config": {
            "benchmark_version": "1.0",
            "target_lengths": target_lengths,
            "cache_isolation_method": cache_isolation_method,
            "repetitions": repetitions,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        "per_length": {},
    }

    stopped = False
    for length in target_lengths:
        length_key = str(length)

        if stopped:
            results["per_length"][length_key] = {"status": "skipped_after_oom"}
            continue

        # Calibrate prompt to exact length
        try:
            calibrated = prepare_exact_prompt(client, length)
        except Exception as exc:
            results["per_length"][length_key] = {
                "status": "request_error",
                "error": str(exc),
            }
            continue

        # Warmup
        warmup_salt = f"warmup_{length}"
        try:
            client.generate(calibrated.text, max_tokens=1, cache_salt=warmup_salt)
        except Exception:
            pass  # warmup errors are excluded from results

        time.sleep(0.5)

        # Measured requests
        length_results: list[PrefillRequestResult] = []
        length_status = "success"

        for req_idx in range(repetitions):
            req_salt = uuid.uuid4().hex
            req_start = time.monotonic()
            req_result: PrefillRequestResult | None = None

            try:
                gen = client.generate(
                    calibrated.text, max_tokens=1, cache_salt=req_salt
                )
                tps = (gen.prompt_tokens / gen.ttft_s) if gen.ttft_s > 0 else None
                engine_tps = None
                if gen.prefill_time_s is not None and gen.prefill_time_s > 0:
                    engine_tps = round(gen.prompt_tokens / gen.prefill_time_s, 2)
                req_result = PrefillRequestResult(
                    index=req_idx,
                    success=True,
                    prompt_tokens=gen.prompt_tokens,
                    prompt_tokens_exact=gen.prompt_tokens_exact,
                    client_ttft_s=gen.ttft_s,
                    total_time_s=gen.total_time_s,
                    effective_prefill_tps=round(tps, 2) if tps else None,
                    cache_isolation_method=cache_isolation_method,
                    start_time=req_start,
                    end_time=time.monotonic(),
                    cached_tokens=gen.cached_tokens,
                    server_ttft_s=gen.time_to_first_token_s,
                    engine_prefill_tps=engine_tps,
                    queue_time_s=gen.queue_time_s,
                    prefill_time_s=gen.prefill_time_s,
                )
            except requests.exceptions.ConnectionError:
                length_status = "server_unavailable"
                stopped = True
                req_result = PrefillRequestResult(
                    index=req_idx,
                    success=False,
                    prompt_tokens=0,
                    prompt_tokens_exact=False,
                    client_ttft_s=0.0,
                    total_time_s=0.0,
                    effective_prefill_tps=None,
                    cache_isolation_method=cache_isolation_method,
                    error="server_unreachable",
                    start_time=req_start,
                    end_time=time.monotonic(),
                    cached_tokens=0,
                    server_ttft_s=None,
                    queue_time_s=None,
                    prefill_time_s=None,
                )
            except requests.exceptions.HTTPError as exc:
                msg = str(exc).lower()
                if "memory" in msg or "oom" in msg:
                    length_status = "oom"
                    stopped = True
                req_result = PrefillRequestResult(
                    index=req_idx,
                    success=False,
                    prompt_tokens=0,
                    prompt_tokens_exact=False,
                    client_ttft_s=0.0,
                    total_time_s=0.0,
                    effective_prefill_tps=None,
                    cache_isolation_method=cache_isolation_method,
                    error=str(exc),
                    start_time=req_start,
                    end_time=time.monotonic(),
                    cached_tokens=0,
                    server_ttft_s=None,
                    queue_time_s=None,
                    prefill_time_s=None,
                )
            except Exception as exc:
                req_result = PrefillRequestResult(
                    index=req_idx,
                    success=False,
                    prompt_tokens=0,
                    prompt_tokens_exact=False,
                    client_ttft_s=0.0,
                    total_time_s=0.0,
                    effective_prefill_tps=None,
                    cache_isolation_method=cache_isolation_method,
                    error=str(exc),
                    start_time=req_start,
                    end_time=time.monotonic(),
                    cached_tokens=0,
                    server_ttft_s=None,
                    queue_time_s=None,
                    prefill_time_s=None,
                )

            if req_result is not None:
                length_results.append(req_result)

            # Stabilization gap between requests (not after last)
            if req_idx < repetitions - 1:
                time.sleep(0.5)

        # Aggregate
        successes = [r for r in length_results if r.success]
        ttfts = [r.client_ttft_s for r in successes]
        tps = [r.effective_prefill_tps for r in successes if r.effective_prefill_tps is not None]
        engine_tps = [r.engine_prefill_tps for r in successes if r.engine_prefill_tps is not None]

        results["per_length"][length_key] = {
            "status": length_status,
            "requested_tokens": length,
            "actual_tokens": calibrated.actual_tokens,
            "n_requests": len(length_results),
            "n_success": len(successes),
            "per_request": [
                {
                    "index": r.index,
                    "success": r.success,
                    "prompt_tokens": r.prompt_tokens,
                    "prompt_tokens_exact": r.prompt_tokens_exact,
                    "client_ttft_s": round(r.client_ttft_s, 4),
                    "total_time_s": round(r.total_time_s, 4),
                    "effective_prefill_tps": r.effective_prefill_tps,
                    "engine_prefill_tps": r.engine_prefill_tps,
                    "cached_tokens": r.cached_tokens,
                    "server_ttft_s": r.server_ttft_s,
                    "queue_time_s": r.queue_time_s,
                    "prefill_time_s": r.prefill_time_s,
                    "cache_isolation_method": r.cache_isolation_method,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "error": r.error,
                }
                for r in length_results
            ],
            "aggregated": {
                "ttft": _stat_summary(ttfts),
                "effective_prefill_tps": _stat_summary(tps) if tps else {"avg_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None},
                "engine_prefill_tps": _stat_summary(engine_tps) if engine_tps else {"avg_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None},
            },
        }

    return results
