#!/usr/bin/env python3
"""
core_runner.py — Phase 1 core benchmark runner (simplified).

Trimmed to the metrics that actually matter for GPU-bound single-node
inference tuning (e.g. vLLM on a DGX Spark). Dropped relative to the full
version: host CPU%/RAM/swap tracking, GPU clock sampling, and verbose
platform/kernel metadata. Kept: GPU identity, TTFT, prefill throughput,
decode speed, GPU memory + power + energy/token, and reasoning-token ratio.

Captures:
  - environment: GPU name/driver/CUDA/torch/vLLM versions (just enough to
    know what you ran against)
  - deep context (32K/64K): TTFT and prefill throughput at extended contexts
  - first-token latency + prefill throughput, swept across prompt lengths
  - decode speed (avg/peak/min/median tok/sec) at several output lengths
  - GPU memory + power sampled at 1 Hz -> avg/peak + energy (Wh) + energy/token
  - reasoning token count (<think>...</think> vs answer) for Qwen3-style
    reasoning-parser output

Not implemented here (plug in later via register_benchmark()):
  code correctness (HumanEval/MBPP), Deep-SWE, HLE.

Usage:
    python core_runner.py --model models/qwen3_35b.yaml

Every run gets its own timestamped directory under results/<model>/<ts>/.

Dependencies:
    pip install requests pyyaml
Optional (more accurate token counting):
    pip install tiktoken
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
import requests
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from vllm_server import VllmServer

try:
    import tiktoken
except ImportError:
    _ENC = None
else:
    _ENC = tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------- #
# Token counting helper (used only as a fallback when the server doesn't
# return usage.prompt_tokens / usage.completion_tokens)
# --------------------------------------------------------------------------- #

def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, int(len(text.split()) * 0.75))


def build_prompt_of_length(target_tokens: int) -> str:
    """Build a substantive, non-repetitive prompt that forces creative generation.

    The old approach (repeat a pangram) caused the model to comment on the
    repetitive structure rather than generating independent content, hitting
    a ~500-token wall of meta-commentary before EOS killed the stream.
    """
    # A single rich prompt — long enough to cover most test cases.
    # When target is larger, pad with additional creative-direction blocks
    # that are distinct enough to prevent repetition loops.
    blocks = [
        "You are a creative writer. Write a detailed, multi-paragraph story about "
        "the discovery of an ancient underwater civilization. Describe the ocean "
        "environment, the architecture of their cities, their technology, their "
        "beliefs and customs, and what happened to them.",

        "Now add more details about this civilization. Describe how they interacted "
        "with marine life, their conflicts with surface dwellers, and their "
        "relationship with deep sea creatures. Be creative and specific.",

        "Continue with more details about this civilization. Describe their "
        "underground cities, their trade routes through the ocean trenches, "
        "and how they communicated across vast distances using bioluminescent "
        "signals and whale songs.",

        "Describe the eventual fall of this civilization. What natural disasters "
        "or environmental changes led to their decline? How did their final "
        "generations respond? What artifacts or evidence survived for modern "
        "archaeologists to find?",

        "Finally, describe the modern scientists who first encountered these "
        "artifacts. What puzzles remain unsolved? What mysteries does this "
        "civilization still hold? End with a reflective note about what their "
        "existence means for our understanding of human history.",
    ]

    text = blocks[0]
    idx = 1
    while count_tokens(text) < target_tokens:
        text += " " + blocks[idx % len(blocks)]
        idx += 1

    if _ENC is not None:
        ids = _ENC.encode(text)[:target_tokens]
        text = _ENC.decode(ids)
    return text


# --------------------------------------------------------------------------- #
# Environment fingerprint — GPU + framework versions only
# --------------------------------------------------------------------------- #

def _run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": datetime.now(datetime.timezone.utc).isoformat()
    }

    gpu_query = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    if gpu_query:
        fields = [f.strip() for f in gpu_query.splitlines()[0].split(",")]
        env["gpu_name"] = fields[0] if len(fields) > 0 else None
        env["driver_version"] = fields[1] if len(fields) > 1 else None
        env["gpu_memory_total"] = fields[2] if len(fields) > 2 else None
    else:
        env["gpu_name"] = None

    nvcc = _run(["nvcc", "--version"])
    env["cuda_version"] = nvcc.splitlines()[-1] if nvcc else None

    for pkg in ("torch", "vllm"):
        env[f"{pkg}_version"] = _run(
            [sys.executable, "-c", f"import {pkg}; print({pkg}.__version__)"]
        )

    return env


# --------------------------------------------------------------------------- #
# GPU resource monitor — memory + power only, sampled at 1 Hz
# --------------------------------------------------------------------------- #


def _window_summary(
    samples: list[dict[str, Any]],
    interval_s: float,
    idle_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-window GPU telemetry.

    Args:
        samples: list of samples collected during the window.
        interval_s: seconds between samples.
        idle_samples: baseline samples recorded before the benchmark started.

    Returns a dict with avg/peak/min/max power, utilization, memory,
    energy (Wh), energy-per-token (Wh/token), and incremental energy
    above idle.
    """
    if not samples:
        return {}

    def _vals(key: str) -> list[float]:
        return [s[key] for s in samples if s.get(key) is not None]

    powers = _vals("gpu_power_w")
    utils = _vals("gpu_util_pct")
    mem_utils = _vals("gpu_mem_util_pct")
    mems = _vals("gpu_mem_used_mib")

    idle_powers = _vals("gpu_power_w") if idle_samples else []
    idle_base = round(statistics.mean(idle_powers), 2) if idle_powers else None

    duration_h = (len(samples) * interval_s) / 3600.0
    energy_wh = round(statistics.mean(powers) * duration_h, 6) if powers else None
    energy_above_wh = None
    if idle_base is not None and idle_base < max(powers):
        energy_above_wh = round(
            (statistics.mean(powers) - idle_base) * duration_h, 6
        )

    result: dict[str, Any] = {
        "num_samples": len(samples),
        "duration_s": round(len(samples) * interval_s, 2),
        "gpu_util_avg_pct": round(statistics.mean(utils), 2) if utils else None,
        "gpu_util_peak_pct": round(max(utils), 2) if utils else None,
        "gpu_power_avg_w": round(statistics.mean(powers), 2) if powers else None,
        "gpu_power_peak_w": round(max(powers), 2) if powers else None,
        "gpu_mem_util_avg_pct": round(statistics.mean(mem_utils), 2) if mem_utils else None,
        "gpu_mem_util_peak_pct": round(max(mem_utils), 2) if mem_utils else None,
        "gpu_mem_used_avg_mib": round(statistics.mean(mems), 1) if mems else None,
        "gpu_mem_used_peak_mib": round(max(mems), 1) if mems else None,
        "energy_wh": energy_wh,
    }
    if idle_base is not None:
        result["idle_base_power_w"] = idle_base
        result["energy_above_idle_wh"] = energy_above_wh
    return result


# --------------------------------------------------------------------------- #
# GPU resource monitor — memory + power only, sampled at 1 Hz
# --------------------------------------------------------------------------- #

class GpuMonitor:
    """Background thread sampling GPU memory + power draw once per second."""

    FIELDS = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    # Keys that map to the CSV fields above
    KEYS = ("gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_power_w")

    def __init__(self, out_dir: Path, interval_s: float = 1.0):
        self.out_dir = out_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[dict[str, Any]] = []
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        # Per-length window tracking
        self._windows: dict[str, list[dict[str, Any]]] = {}
        self._idle_samples: list[dict[str, Any]] = []

    def _parse_row(self, out: str) -> dict[str, Any]:
        """Parse one nvidia-smi CSV line into a row dict."""
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        row: dict[str, Any] = {}
        for k, v in zip(self.KEYS, parts):
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                row[k] = None
        return row

    def _sample_once(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": time.time()}
        if self.has_nvidia_smi:
            out = _run([
                "nvidia-smi",
                f"--query-gpu={self.FIELDS}",
                "--format=csv,noheader,nounits",
            ])
            if out:
                parsed = self._parse_row(out)
                row.update(parsed)
        return row

    def _loop(self):
        while not self._stop.is_set():
            sample = self._sample_once()
            self.samples.append(sample)
            # Also record in every active window
            for window in self._windows.values():
                window.append(sample)
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Window management for per-length telemetry
    # ------------------------------------------------------------------ #

    def start_idle(self) -> None:
        """Begin recording an idle baseline before benchmark work."""
        self._idle_samples = []

    def record_idle(self) -> None:
        """Capture one sample into the idle baseline (caller must hold lock or schedule)."""
        sample = self._sample_once()
        self._idle_samples.append(sample)

    def start_window(self, name: str) -> None:
        """Start a new telemetry window by name."""
        self._windows[name] = []

    def stop_window(self, name: str) -> dict[str, Any] | None:
        """Stop the named window and return a summary.  Returns ``None`` if the
        window is empty (e.g. the benchmark loop was short and no sample was
        taken)."""
        window = self._windows.pop(name, None)
        if not window:
            return None
        return _window_summary(window, self.interval_s, self._idle_samples)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

        csv_path = self.out_dir / "gpu_samples.csv"
        if self.samples:
            keys = sorted({k for row in self.samples for k in row.keys()})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.samples)

        powers = [s["gpu_power_w"] for s in self.samples if s.get("gpu_power_w") is not None]
        mem = [s["gpu_mem_used_mib"] for s in self.samples if s.get("gpu_mem_used_mib") is not None]
        util = [s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None]
        mem_util = [s["gpu_mem_util_pct"] for s in self.samples if s.get("gpu_mem_util_pct") is not None]
        summary: dict[str, Any] = {
            "samples_csv": str(csv_path) if self.samples else None,
            "num_samples": len(self.samples),
            "gpu_util_avg_pct": round(statistics.mean(util), 2) if util else None,
            "gpu_util_peak_pct": round(max(util), 2) if util else None,
            "gpu_power_avg_w": round(statistics.mean(powers), 2) if powers else None,
            "gpu_power_peak_w": round(max(powers), 2) if powers else None,
            "gpu_mem_used_avg_mib": round(statistics.mean(mem), 1) if mem else None,
            "gpu_mem_used_peak_mib": round(max(mem), 1) if mem else None,
            "gpu_mem_util_avg_pct": round(statistics.mean(mem_util), 2) if mem_util else None,
            "gpu_mem_util_peak_pct": round(max(mem_util), 2) if mem_util else None,
        }
        if powers:
            duration_h = (len(powers) * self.interval_s) / 3600.0
            summary["energy_wh"] = round(statistics.mean(powers) * duration_h, 4)
        return summary


# --------------------------------------------------------------------------- #
# Model client — OpenAI-compatible streaming completions
# --------------------------------------------------------------------------- #

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
    # vLLM server-side metrics (available when the response includes them)
    cached_tokens: int = 0
    queue_time_s: float | None = None
    time_to_first_token_s: float | None = None
    prefill_time_s: float | None = None


class ModelClient:
    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = None,
                 chat: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat = chat
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def wait_until_ready(
        self,
        timeout_s: int = 600,
        poll_s: float = 3.0,
        process_check: Optional[Callable[[], None]] = None,
    ) -> None:
        """Wait until the vLLM engine is loaded and can actually process a request.

        The /v1/models HTTP 200 check only verifies the server is listening,
        not that the model weights have been loaded.  A dummy chat completion
        forces the engine to respond, which catches the loading period and
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
                test_prompt = {"role": "user", "content": "hi"}
                url = (f"{self.base_url}/v1/chat/completions"
                       if self.chat else f"{self.base_url}/v1/completions")
                payload = {
                    "model": self.model_name,
                    "messages": [test_prompt],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                }
                resp = requests.post(url, headers=self.headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    warmup_done = True
                    break
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll_s)

        if not warmup_done:
            raise TimeoutError(
                f"Model engine not ready after {timeout_s}s — "
                f"/v1/models returned 200 but a real chat completion failed "
                f"(last error: {last_err}). "
                f"The model weights are likely still being loaded from disk."
            )

    def tokenize_prompt(self, prompt: str) -> TokenizedPrompt:
        if self.chat:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                # Match normal chat-completion rendering.
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

        tokens = data.get("tokens", [])
        count = data.get("count", len(tokens))

        return TokenizedPrompt(count=count, tokens=tokens)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_salt: str | None = None,
    ) -> GenerationResult:
        """Send a generation request, optionally isolating it from prefix cache.

        If *cache_salt* is given the request carries it via ``extra_body`` so
        vLLM's prefix cache is prevented from reusing KV blocks.  On a 400
        response from the server the request is retried with a text-based salt
        appended to the prompt (ensures the content itself is unique).
        """
        base_payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if self.chat:
            url = f"{self.base_url}/v1/chat/completions"
            base_payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            url = f"{self.base_url}/v1/completions"
            base_payload["prompt"] = prompt

        # Try 1: cache_salt via extra_body (preferred — content unchanged)
        if cache_salt is not None:
            result = self._execute_request(url, {
                **base_payload,
                "extra_body": {"cache_salt": cache_salt},
            })
            if result is not None:
                return result

        # Fallback: server rejected cache_salt → append salt to prompt text
        salt = f" salt={cache_salt}" if cache_salt is not None else ""
        if self.chat:
            base_payload["messages"] = [{"role": "user", "content": prompt + salt}]
        else:
            base_payload["prompt"] = prompt + salt
        return self._execute_request(url, base_payload)

    def preflight_cache_salt(self) -> bool | None:
        """Test whether the server supports the cache_salt header.

        Sends one minimal request with ``extra_body={"cache_salt": "preflight"}``.
        Returns ``True`` if the server accepts it, ``False`` on 400, ``None`` on
        connection failure.
        """
        try:
            url = (f"{self.base_url}/v1/chat/completions"
                   if self.chat else f"{self.base_url}/v1/completions")
            payload = {
                "model": self.model_name,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
                "extra_body": {"cache_salt": "preflight"},
            }
            if self.chat:
                payload["messages"] = [{"role": "user", "content": "hi"}]
            else:
                payload["prompt"] = "hi"

            resp = requests.post(
                url, headers=self.headers, json=payload, timeout=30
            )
            return resp.status_code == 200
        except (requests.RequestException, ValueError):
            return None

    def _execute_request(
        self,
        url: str,
        payload: dict,
    ) -> GenerationResult | None:
        """Send one streaming request and return the result, or None on 400."""
        start = time.time()
        first_token_time = None
        last_event_time = start
        per_token_gaps: list[float] = []
        chunks: list[str] = []
        usage: Optional[dict] = None
        server_ttft: float | None = None
        queue_time: float | None = None
        prefill_time: float | None = None
        cached_tokens: int = 0

        reasoning_chunks: list[str] = []
        answer_chunks: list[str] = []
        with requests.post(
            url, headers=self.headers, json=payload, stream=True, timeout=300
        ) as resp:
                # If the server rejects the cache_salt param, bail so the
                # caller can retry with the text-based fallback.
                if resp.status_code == 400:
                    return None
                resp.raise_for_status()
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

                    # vLLM server-side timing metrics (on the first chunk that carries them)
                    if first_token_time is None:
                        metrics = obj.get("request_metrics")
                        if metrics:
                            ttf = metrics.get("time_to_first_token_s")
                            if ttf is not None:
                                server_ttft = float(ttf)
                            qt = metrics.get("queue_time_s")
                            if qt is not None:
                                queue_time = float(qt)
                            pt = metrics.get("prompt_time_s")
                            if pt is not None:
                                prefill_time = float(pt)
                    # vLLM cached tokens (prefill cache hits)
                    details = obj.get("completion_tokens_details")
                    if details:
                        ct = details.get("cached_tokens")
                        if ct is not None:
                            cached_tokens = int(ct)

                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}

                    if self.chat:
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
            queue_time_s=queue_time,
            time_to_first_token_s=server_ttft,
            prefill_time_s=prefill_time,
        )


# --------------------------------------------------------------------------- #
# Deep context benchmark (A1) — TTFT and prefill at extended contexts
# --------------------------------------------------------------------------- #

def run_deep_context(client: ModelClient, context_lengths: list[int],
                      output_tokens: int = 64, repeats: int = 5) -> dict[str, Any]:
    """Measure TTFT and prefill throughput at extended context lengths.

    For each context length, builds a prompt of that length and generates
    a small decode (``output_tokens``). Captures TTFT and prefill TPS.
    On OOM: records ``status: "oom"`` and continues to next length.
    """
    results = {}
    for ctx_len in context_lengths:
        label = str(ctx_len)
        prompt = build_prompt_of_length(ctx_len)
        ttfts: list[float] = []
        prefill_tps: list[float] = []
        exact = None
        oom_detected = False
        for _ in range(repeats):
            try:
                gen = client.generate(prompt, max_tokens=output_tokens, temperature=0.0)
                exact = gen.prompt_tokens_exact
                if gen.ttft_s == gen.ttft_s and gen.ttft_s > 0:
                    ttfts.append(gen.ttft_s)
                    prefill_tps.append(gen.prompt_tokens / gen.ttft_s)
            except requests.exceptions.ConnectionError:
                oom_detected = True
                break
            except requests.exceptions.HTTPError as e:
                if "memory" in str(e).lower() or "oom" in str(e).lower():
                    oom_detected = True
                    break
                raise
        results[label] = {
            "requested_context_tokens": ctx_len,
            "prompt_tokens_exact": exact,
            "n": len(ttfts),
            "ttft_avg_s": round(statistics.mean(ttfts), 4) if ttfts else None,
            "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
            "ttft_p95_s": round(_percentile(ttfts, 95), 4) if ttfts else None,
            "prefill_tps_avg": round(statistics.mean(prefill_tps), 1) if prefill_tps else None,
            "status": "oom" if oom_detected else "success",
        }
    return results


# --------------------------------------------------------------------------- #
# Latency + prefill throughput sweep
# --------------------------------------------------------------------------- #

def run_latency_sweep(client: ModelClient, prompt_lengths: list[int],
                       repeats: int, decode_tokens_for_ttft: int = 8) -> dict[str, Any]:
    results = {}
    for plen in prompt_lengths:
        prompt = build_prompt_of_length(plen)
        ttfts = []
        prefill_tps = []
        exact = None
        for _ in range(repeats):
            gen = client.generate(prompt, max_tokens=decode_tokens_for_ttft, temperature=0.0)
            exact = gen.prompt_tokens_exact
            if gen.ttft_s == gen.ttft_s:
                ttfts.append(gen.ttft_s)
                if gen.ttft_s > 0:
                    prefill_tps.append(gen.prompt_tokens / gen.ttft_s)
        results[str(plen)] = {
            "requested_prompt_tokens": plen,
            "prompt_tokens_exact": exact,
            "n": len(ttfts),
            "ttft_avg_s": round(statistics.mean(ttfts), 4) if ttfts else None,
            "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
            "ttft_p95_s": round(_percentile(ttfts, 95), 4) if ttfts else None,
            "ttft_p99_s": round(_percentile(ttfts, 99), 4) if ttfts else None,
            "prefill_tps_avg": round(statistics.mean(prefill_tps), 1) if prefill_tps else None,
        }
    return results


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# --------------------------------------------------------------------------- #
# Decode speed
# --------------------------------------------------------------------------- #

def run_decode_speed(client: ModelClient, output_lengths: list[int],
                      fixed_prompt_tokens: int = 128) -> dict[str, Any]:
    prompt = build_prompt_of_length(fixed_prompt_tokens)
    results = {}
    for n_out in output_lengths:
        gen = client.generate(prompt, max_tokens=n_out, temperature=0.0)
        decode_time = gen.total_time_s - gen.ttft_s if gen.ttft_s == gen.ttft_s else gen.total_time_s
        tok_per_sec_overall = gen.output_tokens / decode_time if decode_time > 0 else None
        instant_rates = [1.0 / g for g in gen.per_token_times if g > 0]
        output_truncated = gen.output_tokens != n_out

        # Analyze reasoning tokens (thinking vs answer)
        # Use the already-split reasoning and answer text from the model client
        reasoning_tokens_count: int = 0
        answer_tokens_count: int = 0
        reasoning_ratio: Optional[float] = None
        if gen.reasoning_text:
            reasoning_tokens_count = count_tokens(gen.reasoning_text)
            answer_tokens_count = count_tokens(gen.answer_text)
            if answer_tokens_count:
                reasoning_ratio = round(reasoning_tokens_count / answer_tokens_count, 3)
        elif gen.output_text:
            # Fall back to analyzing combined output_text for models without split
            reasoning_stats = analyze_reasoning_tokens(gen.output_text)
            reasoning_tokens_count = reasoning_stats["thinking_tokens"]
            answer_tokens_count = reasoning_stats["answer_tokens"]
            reasoning_ratio = reasoning_stats.get("ratio_thinking_to_answer")

        text_preview = gen.output_text if len(gen.output_text) <= 512 else (
            gen.output_text[:512] + f"\n  [truncated, total length: {len(gen.output_text)} chars]"
        )
        results[str(n_out)] = {
            "requested_output_tokens": n_out,
            "actual_output_tokens": gen.output_tokens,
            "output_tokens_exact": gen.output_tokens_exact,
            "output_truncated": output_truncated,
            "ttft_s": round(gen.ttft_s, 4) if gen.ttft_s == gen.ttft_s else None,
            "decode_time_s": round(decode_time, 3),
            "tok_per_sec_avg": round(tok_per_sec_overall, 2) if tok_per_sec_overall else None,
            "tok_per_sec_peak": round(max(instant_rates), 2) if instant_rates else None,
            "tok_per_sec_min": round(min(instant_rates), 2) if instant_rates else None,
            "tok_per_sec_median": round(statistics.median(instant_rates), 2) if instant_rates else None,
            "output_text_preview": text_preview,
            "reasoning_tokens": reasoning_tokens_count,
            "answer_tokens": answer_tokens_count,
            "reasoning_ratio": reasoning_ratio,
        }
    return results


# --------------------------------------------------------------------------- #
# Reasoning token count (Qwen3-style <think>...</think>)
# --------------------------------------------------------------------------- #

def analyze_reasoning_tokens(text: str) -> dict[str, Any]:
    """Separate thinking tokens from answer tokens in a model's output.

    Handles three formats:
    1. Qwen3 XML tags:  <antThinking>...thinking...</antThinking>answer...
    2. Anthropic XML tags: <thinking>...thinking...</thinking>answer...
    3. Plain-text reasoning: common reasoning markers followed by structured answer.
    """
    # Format 1: Qwen3 XML tags (check first — more specific)
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

    # Format 2: Anthropic XML tags
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

        # Prefer the client's pre-split reasoning/answer text (for Qwen3-style
        # reasoning mode where the server returns delta.reasoning separately).
        # Fall back to re-analyzing the combined output_text for non-chat mode.
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


# --------------------------------------------------------------------------- #
# Speculative decoding comparison (A2)
# --------------------------------------------------------------------------- #

def compare_spec_decode_results(spec_results: dict[str, Any],
                                no_spec_results: dict[str, Any],
                                output_lengths: list[int]) -> dict[str, Any]:
    """Compare previously collected speculative and baseline decode results."""

    delta: dict[str, Any] = {}
    for key in spec_results:
        spec_entry = spec_results[key]
        no_spec_entry = no_spec_results.get(key, {})
        spec_tps = spec_entry.get("tok_per_sec_avg")
        no_spec_tps = no_spec_entry.get("tok_per_sec_avg")
        spec_ttft = spec_entry.get("ttft_s")
        no_spec_ttft = no_spec_entry.get("ttft_s")

        tps_pct = None
        if no_spec_tps and no_spec_tps > 0:
            tps_pct = round((spec_tps - no_spec_tps) / no_spec_tps * 100, 2) if spec_tps else None

        ttft_pct = None
        if no_spec_ttft and no_spec_ttft > 0:
            ttft_pct = round((spec_ttft - no_spec_ttft) / no_spec_ttft * 100, 2) if spec_ttft else None

        delta[key] = {
            "tok_s_improvement_pct": tps_pct,
            "ttft_change_pct": ttft_pct,
        }

    return {
        "config": {
            "output_lengths": output_lengths,
        },
        "spec_enabled": spec_results,
        "spec_disabled": no_spec_results,
        "delta": delta,
    }


# --------------------------------------------------------------------------- #
# Plugin hook for future phases (code correctness, Deep-SWE, HLE, ...)
# --------------------------------------------------------------------------- #

_REGISTERED_BENCHMARKS: dict[str, Callable[[ModelClient, dict], dict]] = {}


def register_benchmark(name: str):
    def _wrap(fn: Callable[[ModelClient, dict], dict]):
        _REGISTERED_BENCHMARKS[name] = fn
        return fn
    return _wrap


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def load_model_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_run_dir(model_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / model_name / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def make_client(cfg: dict) -> ModelClient:
    endpoint = cfg["endpoint"]
    return ModelClient(
        base_url=endpoint["base_url"],
        model_name=endpoint.get("model_name", cfg["name"]),
        api_key=endpoint.get("api_key"),
        chat=endpoint.get("chat", True),
    )


def make_managed_server(
    cfg: dict,
    run_dir: Path,
    log_name: str = "vllm.log",
    command: list[str] | None = None,
) -> VllmServer:
    server_cfg = cfg.get("server")
    if not isinstance(server_cfg, dict):
        raise ValueError("managed server mode requires a server section in the model config")
    resolved_command = command or server_cfg.get("command")
    return VllmServer(
        command=resolved_command,
        base_url=cfg["endpoint"]["base_url"],
        log_path=run_dir / log_name,
        environment=server_cfg.get("environment", {}),
        shutdown_timeout_s=server_cfg.get("shutdown_timeout_s", 30),
    )


def wait_for_endpoint(client: ModelClient, cfg: dict,
                      server: VllmServer | None = None) -> None:
    client.wait_until_ready(
        timeout_s=cfg.get("server", {}).get(
            "startup_timeout_s", cfg.get("ready_timeout_s", 600)
        ),
        poll_s=cfg.get("ready_poll_s", 3.0),
        process_check=server.check_running if server else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 1 core benchmark runner (simplified)")
    parser.add_argument("--model", required=True, help="Path to model config YAML")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip attention backend sweep benchmark")
    parser.add_argument("--skip-ttft", action="store_true")
    parser.add_argument("--compare-spec", action="store_true",
                        help="Run decode benchmark with spec-dec enabled and disabled, then compare")
    parser.add_argument(
        "--server-mode",
        choices=("managed", "external"),
        default=None,
        help="Override the server mode from the model config",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model)
    cfg = load_model_config(model_config_path)
    model_name = cfg["name"]
    run_dir = make_run_dir(model_name)
    print(f"[core_runner] writing results to {run_dir}")
    shutil.copy2(model_config_path, run_dir / "model_config.yml")

    env = collect_environment()
    save_json(run_dir / "environment.json", env)
    print(f"[core_runner] environment: {env.get('gpu_name')}, torch={env.get('torch_version')}, vllm={env.get('vllm_version')}")

    server_cfg = cfg.get("server", {})
    server_mode = args.server_mode or server_cfg.get("mode", "external")
    if server_mode not in ("managed", "external"):
        raise ValueError(f"unsupported server mode: {server_mode}")

    concurrency_levels = cfg.get("concurrency_levels", [1, 2, 4, 8, 16])
    requests_per_level = cfg.get("concurrency_requests_per_level", 5)
    if concurrency_levels and requests_per_level < max(concurrency_levels):
        print(
            "[core_runner] warning: concurrency_requests_per_level is below the "
            "largest concurrency level; high levels will submit fewer requests than workers"
        )

    server: VllmServer | None = None
    monitor: GpuMonitor | None = None
    summary: dict[str, Any] = {
        "model": model_name,
        "run_dir": str(run_dir),
        "server_mode": server_mode,
        "status": "running",
    }

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, _frame) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        if server_mode == "managed":
            server = make_managed_server(cfg, run_dir)
            print(f"[core_runner] starting managed vLLM: {' '.join(server.command)}")
            server.start()
            server.save_metadata(run_dir / "resolved_server.json")

        client = make_client(cfg)
        print("[core_runner] waiting for model endpoint to be ready...")
        wait_for_endpoint(client, cfg, server)
        print("[core_runner] endpoint ready")

        monitor = GpuMonitor(run_dir, interval_s=cfg.get("monitor_interval_s", 1.0))
        monitor.start()

        prompt_lengths = cfg.get("prompt_lengths", [32, 128, 512, 2048, 8192, 16384])

        if not args.skip_latency:
            repeats = cfg.get("latency_repeats", 10)
            print(f"[core_runner] latency sweep over {prompt_lengths} ({repeats} reps each)")
            latency_results = run_latency_sweep(client, prompt_lengths, repeats)
            save_json(run_dir / "latency.json", latency_results)
            summary["latency"] = latency_results

        # A1: Deep context benchmark
        context_lengths = cfg.get("context_lengths")
        if context_lengths:
            context_repeats = cfg.get("context_repeats", 5)
            print(f"[core_runner] deep context benchmark over {context_lengths} ({context_repeats} reps each)")
            deep_context_results = run_deep_context(client, context_lengths, output_tokens=64, repeats=context_repeats)
            save_json(run_dir / "deep_context.json", deep_context_results)
            summary["deep_context"] = deep_context_results

        if not args.skip_ttft:
            from benchmarks.ttft_breakdown import run_ttft_breakdown

            ttft_lengths = cfg.get(
                "ttft_prompt_lengths", [128, 512, 2048, 8192, 32768]
            )
            ttft_reps = cfg.get("ttft_repetitions", 10)
            print(
                f"[core_runner] TTFT breakdown over {ttft_lengths} "
                f"({ttft_reps} reps each)"
            )
            ttft_results = run_ttft_breakdown(
                client, prompt_lengths=ttft_lengths, repetitions=ttft_reps, gpu_monitor=monitor
            )
            save_json(run_dir / "ttft_breakdown.json", ttft_results)
            summary["ttft_breakdown"] = ttft_results

        if not args.skip_decode:
            output_lengths = cfg.get("decode_lengths", [512, 1024, 2048])
            print(f"[core_runner] decode speed benchmark over {output_lengths}")
            decode_results = run_decode_speed(client, output_lengths)
            save_json(run_dir / "decode.json", decode_results)
            summary["decode"] = decode_results

        if not args.skip_reasoning:
            reasoning_prompts = cfg.get("reasoning_prompts", [
                "Solve: if a train travels 60 miles in 45 minutes, what is its speed in mph? Show your reasoning.",
                "A farmer has 17 sheep, all but 9 die. How many are left? Explain your reasoning step by step.",
            ])
            print(f"[core_runner] reasoning-token benchmark ({len(reasoning_prompts)} prompts)")
            reasoning_results = run_reasoning_benchmark(client, reasoning_prompts)
            save_json(run_dir / "reasoning.json", reasoning_results)
            summary["reasoning"] = reasoning_results

        if not args.skip_concurrency:
            # Lazy import to avoid circular import (core_runner <-> benchmarks.concurrency)
            from benchmarks.concurrency import run_concurrency_test

            max_tokens = cfg.get("concurrency_max_tokens", 256)
            temperature = cfg.get("concurrency_temperature", 0.0)
            print(
                f"[core_runner] concurrency test: levels={concurrency_levels} "
                f"({requests_per_level} reqs/level, {max_tokens} max_tokens, "
                f"t={temperature})"
            )
            concurrency_results = run_concurrency_test(
                client, concurrency_levels, requests_per_level,
                max_tokens=max_tokens, temperature=temperature,
            )
            save_json(run_dir / "concurrency.json", concurrency_results)
            summary["concurrency"] = concurrency_results

        if not args.skip_prefill:
            from benchmarks.prefill import run_prefill_scaling

            target_lengths = cfg.get(
                "prefill_target_lengths",
                [512, 2048, 8192],
            )
            repetitions = cfg.get("prefill_repetitions", 5)
            # Start idle baseline for GPU telemetry
            gpu_interval = cfg.get("monitor_interval_s", 1.0)
            if gpu_interval > 0.2:
                print("[core_runner] capturing GPU idle baseline...")
            monitor.start_idle()
            for _ in range(int(5.0 / gpu_interval)):
                monitor.record_idle()
                time.sleep(gpu_interval)
            print("[core_runner] idle baseline captured, starting prefill benchmark")

            print(f"[core_runner] prefill scaling benchmark over {target_lengths} ({repetitions} reps)")
            prefill_results = run_prefill_scaling(client, target_lengths, repetitions=repetitions, gpu_monitor=monitor)
            save_json(run_dir / "prefill_scaling.json", prefill_results)
            summary["prefill_scaling"] = {
                k: {"status": v.get("status"), "n_success": v.get("n_success")}
                for k, v in prefill_results.get("per_length", {}).items()
            }

        # Attention backend sweep
        if not args.skip_attention:
            if server_mode != "managed" or server is None:
                print("[core_runner] skipping attention backend sweep: managed server mode required")
            else:
                from benchmarks.attention_backend import run_attention_backend_sweep

                backends = cfg.get("attention_backends", None)
                print(f"[core_runner] attention backend sweep: backends={backends}")
                attention_results = run_attention_backend_sweep(
                    client, server, cfg, run_dir, gpu_monitor=monitor,
                    backends=backends,
                    prompt_lengths=prompt_lengths,
                    decode_lengths=cfg.get("decode_lengths", [512, 1024, 2048]),
                    repetitions=cfg.get("attention_repetitions", 5),
                )
                save_json(run_dir / "attention_backend.json", attention_results)
                summary["attention_backend"] = attention_results

        for name, fn in _REGISTERED_BENCHMARKS.items():
            print(f"[core_runner] running registered benchmark: {name}")
            plugin_results = fn(client, cfg)
            save_json(run_dir / f"{name}.json", plugin_results)
            summary[name] = plugin_results

        # A2: Speculative decoding comparison
        if args.compare_spec:
            if server_mode != "managed" or server is None:
                raise ValueError("--compare-spec requires managed server mode")
            spec_config = cfg.get("speculative_config")
            if not spec_config:
                raise ValueError("--compare-spec requires speculative_config in the model YAML")

            output_lengths = cfg.get("decode_lengths", [512, 1024, 2048])
            base_command = list(server.command)
            server.stop()
            server.save_metadata(run_dir / "resolved_server.json")
            server = None

            def run_variant(label: str, command: list[str]) -> dict[str, Any]:
                variant_server = make_managed_server(
                    cfg, run_dir, log_name=f"vllm_{label}.log", command=command
                )
                try:
                    print(f"[core_runner] starting vLLM variant: {label}")
                    variant_server.start()
                    variant_client = make_client(cfg)
                    wait_for_endpoint(variant_client, cfg, variant_server)
                    return run_decode_speed(variant_client, output_lengths)
                finally:
                    variant_server.stop()
                    variant_server.save_metadata(
                        run_dir / f"resolved_server_{label}.json"
                    )

            spec_results = run_variant(
                "spec_enabled",
                base_command + ["--speculative-config", str(spec_config)],
            )
            save_json(run_dir / "spec_enabled.json", spec_results)
            non_spec_results = run_variant("spec_disabled", base_command)
            save_json(run_dir / "spec_disabled.json", non_spec_results)

            comparison = compare_spec_decode_results(
                spec_results, non_spec_results, output_lengths
            )
            save_json(run_dir / "spec_comparison.json", comparison)
            summary["spec_comparison"] = comparison

        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if monitor is not None:
            print("[core_runner] stopping GPU monitor")
            gpu_summary = monitor.stop()
            summary["gpu"] = gpu_summary

            total_output_tokens = 0
            if "decode" in summary:
                total_output_tokens += sum(
                    v["actual_output_tokens"] for v in summary["decode"].values()
                    if v.get("actual_output_tokens")
                )
            if total_output_tokens and gpu_summary.get("energy_wh") is not None:
                summary["energy_per_token_wh"] = round(
                    gpu_summary["energy_wh"] / total_output_tokens, 6
                )

        if server is not None:
            print("[core_runner] stopping managed vLLM")
            server.stop()
            server.save_metadata(run_dir / "resolved_server.json")

        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        save_json(run_dir / "summary.json", summary)
        print(f"[core_runner] summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
