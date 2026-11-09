"""HTTP client for an OpenAI-compatible server, such as vLLM.

The client sends streaming generation requests and records the client-side
timing (TTFT, gaps between stream events) and the vLLM per-request metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class TokenizedPrompt:
    count: int
    tokens: list[int]


@dataclass
class GenerationResult:
    prompt_tokens: int
    ttft_s: float
    total_time_s: float
    output_text: str
    output_tokens: int
    reasoning_text: str = ""          # thinking tokens (Qwen3 plain-text or <thinking> tags)
    answer_text: str = ""            # post-reasoning answer
    per_token_times: list[float] = field(default_factory=list)
    prompt_tokens_exact: bool = False
    output_tokens_exact: bool = False
    # vLLM server-side metrics. The server adds a ``metrics`` object to the
    # final usage chunk when it runs with ``--enable-per-request-metrics``.
    # vLLM reports milliseconds; this client stores seconds.
    cached_tokens: int = 0
    queue_time_s: float | None = None
    time_to_first_token_s: float | None = None
    # vLLM measures TTFT from scheduling to the first token, so the server
    # TTFT does not include the queue time. ``post_queue_time_s`` is the
    # time after the queue. It covers prefill plus the first decode step.
    # vLLM does not report prefill time. The field stays None unless a
    # server adds one.
    prefill_time_s: float | None = None
    post_queue_time_s: float | None = None
    generation_time_s: float | None = None
    server_metrics: dict[str, Any] | None = None


def _parse_server_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a vLLM ``metrics`` object to seconds.

    vLLM reports ``time_to_first_token_ms``, ``queue_time_ms`` and
    ``generation_time_ms``. It does not report prefill time. vLLM 0.25.1
    and 0.33.0 compute the two timings from engine timestamps:

    - ``queue_time_ms`` is ``scheduled_ts - queued_ts``.
    - ``time_to_first_token_ms`` is ``first_token_ts - scheduled_ts``.

    The server TTFT thus starts after the queue. The returned
    ``post_queue_time_s`` is equal to the server TTFT. Do not subtract the
    queue time from it. Second-based keys are accepted as a fallback for
    servers that already report seconds.
    """
    out: dict[str, Any] = {
        "time_to_first_token_s": None,
        "queue_time_s": None,
        "prefill_time_s": None,
        "post_queue_time_s": None,
        "generation_time_s": None,
    }
    if not metrics:
        return out

    def _seconds(key_ms: str, key_s: str) -> float | None:
        value = metrics.get(key_ms)
        if value is not None:
            return float(value) / 1000.0
        value = metrics.get(key_s)
        return float(value) if value is not None else None

    out["time_to_first_token_s"] = _seconds("time_to_first_token_ms", "time_to_first_token_s")
    out["queue_time_s"] = _seconds("queue_time_ms", "queue_time_s")
    out["prefill_time_s"] = _seconds("prefill_time_ms", "prompt_time_s")
    out["generation_time_s"] = _seconds("generation_time_ms", "generation_time_s")
    out["post_queue_time_s"] = out["time_to_first_token_s"]
    return out


MAX_ERROR_BODY_CHARS = 1000


class ServerHTTPError(requests.HTTPError):
    """An HTTP error response. The message holds the status and the server body."""

    def __init__(self, response: Any, url: str) -> None:
        self.status_code = int(response.status_code)
        try:
            body = response.text or ""
        except Exception:  # noqa: BLE001 - the body is only for the message
            body = ""
        self.body = body[:MAX_ERROR_BODY_CHARS]
        super().__init__(f"HTTP {self.status_code} from {url}: {self.body}", response=response)

