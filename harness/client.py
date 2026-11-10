"""HTTP client for an OpenAI-compatible server, such as vLLM.

The client sends streaming generation requests and records the client-side
timing (TTFT, gaps between stream events) and the vLLM per-request metrics.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    reasoning_text: str = ""          # text of delta.reasoning (the server reasoning parser)
    answer_text: str = ""            # post-reasoning answer
    per_token_times: list[float] = field(default_factory=list)
    # Tokens that arrived in each gap of ``per_token_times``. Only set when
    # every text event carries per-chunk usage (``continuous_usage=True``).
    per_event_token_deltas: list[int] | None = None
    finish_reason: str | None = None
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
        except Exception:  # The body is only for the message.
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
    usage: dict | None = None
    server_metrics: dict[str, Any] | None = None
    cached_tokens: int = 0
    finish_reason: str | None = None
    # Cumulative ``usage.completion_tokens`` at each text event, from
    # per-chunk usage. A None entry means that the event had no usage.
    event_token_counts: list[int | None] = []

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
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            # vLLM per-request timing metrics. The server puts a
            # ``metrics`` object on the final usage chunk (or on the
            # choice) when it runs with --enable-per-request-metrics.
            metrics = obj.get("metrics") or choice.get("metrics")
            if metrics:
                server_metrics = dict(metrics)
            delta = choice.get("delta") or {}

            if chat:
                reasoning_text: str | None = delta.get("reasoning")
                content_text: str | None = delta.get("content")
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
                event_token_counts.append((obj.get("usage") or {}).get("completion_tokens"))

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
    token_deltas: list[int] | None = None
    if event_token_counts and None not in event_token_counts:
        token_deltas = [int(b) - int(a) for a, b in itertools.pairwise(event_token_counts)]

    return GenerationResult(
        prompt_tokens=exact_prompt if exact_prompt is not None else count_tokens(payload.get("prompt", "")),
        ttft_s=(first_token_time - start) if first_token_time else float("nan"),
        total_time_s=end - start,
        output_text=output_text,
        output_tokens=exact_output if exact_output is not None else count_tokens(output_text),
        reasoning_text="".join(reasoning_chunks),
        answer_text="".join(answer_chunks),
        per_token_times=per_token_gaps,
        per_event_token_deltas=token_deltas,
        finish_reason=finish_reason,
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


class ModelClient:
    """Client for one model on an OpenAI-compatible server (chat or completions)."""

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None,
                 chat: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat = chat
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        # None: not known yet. The first salted request or the preflight
        # sets it. After that the client does not test the field again.
        self.cache_salt_supported: bool | None = None

    def _request_target(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Return the URL and the payload base for *prompt* in the endpoint mode."""
        if self.chat:
            return (f"{self.base_url}/v1/chat/completions",
                    {"model": self.model_name, "messages": [{"role": "user", "content": prompt}]})
        return f"{self.base_url}/v1/completions", {"model": self.model_name, "prompt": prompt}

    def wait_until_ready(
        self,
        timeout_s: int = 600,
        poll_s: float = 3.0,
        process_check: Callable[[], None] | None = None,
    ) -> None:
        """Wait until the vLLM engine is loaded and can actually process a request.

        The /v1/models HTTP 200 check only verifies the server is listening,
        not that the model weights have been loaded.  A dummy generation
        request (chat or completions, as the client mode sets) forces the
        engine to respond, which catches the loading period and
        prevents the latency sweep (which fires immediately after this call)
        from hitting a still-initialising model — that was why every latency
        entry showed n=0 (TTFT was NaN).
        """
        health_url = f"{self.base_url}/v1/models"
        deadline = time.time() + timeout_s
        last_err = None

        # Step 1: wait for the HTTP server to accept connections.
        while time.time() < deadline:
            if process_check:
                process_check()
            try:
                r = requests.get(health_url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    break
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll_s)
        else:
            raise TimeoutError(f"HTTP server unreachable after {timeout_s}s (last error: {last_err})")

        # Step 2: drive a real test request to confirm the model engine
        # is actually able to produce output.  This is what /v1/models
        # alone cannot tell us.
        warmup_done = False
        while time.time() < deadline:
            if process_check:
                process_check()
            try:
                url, payload = self._request_target("hi")
                payload.update({
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                })
                resp = requests.post(url, headers=self.headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    warmup_done = True
                    break
                last_err = f"HTTP {resp.status_code}: {resp.text[:MAX_ERROR_BODY_CHARS]}"
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll_s)

        if not warmup_done:
            raise TimeoutError(
                f"Model engine not ready after {timeout_s}s — "
                f"/v1/models returned 200 but a real generation request failed "
                f"(last error: {last_err}). "
                f"The model weights are likely still being loaded from disk."
            )

    def tokenize_prompt(self, prompt: str) -> TokenizedPrompt:
        """Count the tokens the server sees for *prompt* through ``/tokenize``.

        In chat mode the request carries the same single user message as
        ``generate`` and ``add_generation_prompt``, so the count includes the
        chat template. The prefill benchmark uses this count to build prompts
        of an exact token length.
        """
        if self.chat:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "add_generation_prompt": True,
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "add_special_tokens": True,
            }

        response = requests.post(
            f"{self.base_url}/tokenize",
            headers=self.headers,
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        tokens = data.get("tokens") or []
        count = data.get("count")
        return TokenizedPrompt(count=int(count) if count is not None else len(tokens), tokens=tokens)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_salt: str | None = None,
        continuous_usage: bool = False,
    ) -> GenerationResult:
        """Send a generation request, optionally isolating it from prefix cache.

        With *continuous_usage*, the request also sets
        ``stream_options.continuous_usage_stats``. vLLM then adds the
        cumulative ``usage`` to each chunk, and the result holds
        ``per_event_token_deltas``. Other requests stay the same.

        If *cache_salt* is given the request carries it as a top-level
        ``cache_salt`` field so vLLM's prefix cache cannot reuse KV blocks.
        The client decides the salt mode once (``cache_salt_supported``).
        While the mode is unknown, an HTTP 400 whose body names ``cache_salt``
        sets the text mode: the client then appends `` salt=<value>`` to the
        prompt text for this and all later requests. Any other HTTP error
        raises ``ServerHTTPError`` with the status and the server body.
        """
        options = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if continuous_usage:
            options["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}

        # vLLM ignores unknown fields such as ``extra_body``, so the salt
        # must sit at the top level of the JSON body.
        mode = self.cache_salt_supported
        if cache_salt is not None and mode is not False:
            url, payload = self._request_target(prompt)
            try:
                result = self._execute_request(url, {**payload, **options, "cache_salt": cache_salt})
            except ServerHTTPError as e:
                if mode is True or e.status_code != 400 or "cache_salt" not in e.body:
                    raise
                self.cache_salt_supported = False
            else:
                self.cache_salt_supported = True
                return result

        salt = f" salt={cache_salt}" if cache_salt is not None else ""
        url, payload = self._request_target(prompt + salt)
        return self._execute_request(url, {**payload, **options})

    def preflight_cache_salt(self) -> bool | None:
        """Test whether the server supports the ``cache_salt`` field.

        Sends one minimal request with a top-level ``cache_salt`` field.
        Returns ``True`` if the server accepts it, ``False`` on another status,
        ``None`` on connection failure. A 200 or a 400 also sets
        ``cache_salt_supported``, which ``generate`` uses.
        """
        try:
            url, payload = self._request_target("hi")
            payload.update({
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
                "cache_salt": "preflight",
            })
            resp = requests.post(
                url, headers=self.headers, json=payload, timeout=30
            )
        except (requests.RequestException, ValueError):
            return None
        if resp.status_code in (200, 400):
            self.cache_salt_supported = resp.status_code == 200
        return resp.status_code == 200

    def _execute_request(self, url: str, payload: dict) -> GenerationResult:
        """Send one streaming request. Raises ``ServerHTTPError`` on an HTTP error status."""
        return stream_generation(url, self.headers, payload, chat=self.chat)

