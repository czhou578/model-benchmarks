"""HTTP client for an OpenAI-compatible server, such as vLLM.

The client sends streaming generation requests and records the client-side
timing (TTFT, gaps between stream events) and the vLLM per-request metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from harness.prompts import count_tokens


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


def stream_generation(
    url: str,
    headers: dict[str, str],
    payload: dict,
    chat: bool,
) -> GenerationResult:
    """Send one streaming request and return the result.

    *chat* selects the stream format: ``delta`` fields for chat completions,
    ``text`` for completions. Raises ``ServerHTTPError`` on an HTTP error status.
    """
    start = time.time()
    first_token_time = None
    last_event_time = start
    per_token_gaps: list[float] = []
    chunks: list[str] = []
    usage: Optional[dict] = None
    server_metrics: dict[str, Any] | None = None
    cached_tokens: int = 0

    reasoning_chunks: list[str] = []
    answer_chunks: list[str] = []
    with requests.post(
        url, headers=headers, json=payload, stream=True, timeout=300
    ) as resp:
        if resp.status_code >= 400:
            raise ServerHTTPError(resp, url)
        for raw_line in resp.iter_lines():
            line: str = raw_line.decode("utf-8")
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            if obj.get("usage"):
                usage = obj["usage"]

            choice = (obj.get("choices") or [{}])[0]

            # vLLM per-request timing metrics. The server puts a
            # ``metrics`` object on the final usage chunk (or on the
            # choice) when it runs with --enable-per-request-metrics.
            metrics = obj.get("metrics") or choice.get("metrics")
            if metrics:
                server_metrics = dict(metrics)
            delta = choice.get("delta") or {}

            if chat:
                reasoning_text: Optional[str] = delta.get("reasoning")
                content_text: Optional[str] = delta.get("content")
            else:
                reasoning_text = None
                content_text = choice.get("text")

            has_text = False
            for text_parts in (reasoning_text, content_text):
                if text_parts:
                    has_text = True
                    break

            if has_text:
                now = time.time()
                if first_token_time is None:
                    first_token_time = now
                else:
                    per_token_gaps.append(now - last_event_time)
                last_event_time = now

            if reasoning_text:
                reasoning_chunks.append(reasoning_text)
                chunks.append(reasoning_text)
            if content_text:
                answer_chunks.append(content_text)
                chunks.append(content_text)

    end = time.time()
    output_text = "".join(chunks)
    parsed_metrics = _parse_server_metrics(server_metrics)
    # vLLM reports prefix-cache hits under usage.prompt_tokens_details.
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    if prompt_details.get("cached_tokens") is not None:
        cached_tokens = int(prompt_details["cached_tokens"])

    exact_prompt = usage.get("prompt_tokens") if usage else None
    exact_output = usage.get("completion_tokens") if usage else None

    return GenerationResult(
        prompt_tokens=exact_prompt if exact_prompt is not None else count_tokens(payload.get("prompt", "")),
        ttft_s=(first_token_time - start) if first_token_time else float("nan"),
        total_time_s=end - start,
        output_text=output_text,
        output_tokens=exact_output if exact_output is not None else count_tokens(output_text),
        reasoning_text="".join(reasoning_chunks),
        answer_text="".join(answer_chunks),
        per_token_times=per_token_gaps,
        prompt_tokens_exact=exact_prompt is not None,
        output_tokens_exact=exact_output is not None,
        cached_tokens=cached_tokens,
        queue_time_s=parsed_metrics["queue_time_s"],
        time_to_first_token_s=parsed_metrics["time_to_first_token_s"],
        prefill_time_s=parsed_metrics["prefill_time_s"],
        post_queue_time_s=parsed_metrics["post_queue_time_s"],
        generation_time_s=parsed_metrics["generation_time_s"],
        server_metrics=server_metrics,
    )
