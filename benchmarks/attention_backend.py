"""Attention backend sweep benchmark.

Compares vLLM attention backends by running a standard latency + decode
workload under each backend, with GPU telemetry per backend.

Output: attention_backend.json
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_runner import ModelClient, GpuMonitor, build_prompt_of_length
from vllm_server import VllmServer


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LatencyRequestResult:
    """Single request within a latency sweep."""
    index: int
    success: bool
    prompt_tokens: int
    prompt_tokens_exact: bool
    ttft_s: float
    total_time_s: float
    error: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stat_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "avg_s": None,
            "median_s": None,
            "p95_s": None,
            "min_s": None,
            "max_s": None,
        }
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


def _detect_available_backends() -> list[str]:
    """Return a list of backend names available on this hardware.

    Filters out ROCM-only backends when running on NVIDIA.
    """
    import torch
    has_rocm = torch.version.hip is not None
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return []

    backends: list[str] = []
    for backend in AttentionBackendEnum:
        name = backend.name.lower()
        # Skip ROCM-only backends on NVIDIA
        if not has_rocm and ("rocm" in name or "aiter" in name or "xpu" in name):
            continue
        # Skip ViT-only
        if name == "torch_sdpa":
            continue
        backends.append(name)
    return backends


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_attention_backend_sweep(
    client: ModelClient,
    server: VllmServer,
    cfg: dict,
    run_dir: Path,
    gpu_monitor: GpuMonitor | None = None,
    backends: list[str] | None = None,
    prompt_lengths: list[int] | None = None,
    decode_lengths: list[int] | None = None,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Sweep vLLM attention backends and measure TTFT + decode throughput.

    For each backend, restarts the vLLM server with the appropriate
    ``--attention-backend`` flag, runs the workload, then proceeds to the
    next backend.

    Args:
        client: ModelClient for the vLLM endpoint.
        server: Managed VllmServer (will be restarted per backend).
        cfg: Model config dict.
        run_dir: Results directory (for per-backend log files).
        gpu_monitor: Optional GPU telemetry collector.
        backends: List of backend names to sweep (default: auto-detect).
        prompt_lengths: Prompt lengths to test.
        decode_lengths: Output lengths to test.
        repetitions: Requests per length per backend.

    Returns:
        Dict with per-backend results.
    """
    if backends is None:
        backends = _detect_available_backends()
    if prompt_lengths is None:
        prompt_lengths = [32, 128, 512, 2048]
    if decode_lengths is None:
        decode_lengths = [512, 1024, 2048]

    base_command = list(server.command)

    # Find the index of --attention-backend in the base command so we can
    # replace its value without duplicating the entire command list.
    attn_idx = None
    for i, arg in enumerate(base_command):
        if arg == "--attention-backend":
            attn_idx = i
            break

    results: dict[str, Any] = {
        "config": {
            "benchmark_version": "1.0",
            "definition": "sweep vLLM attention backends and measure TTFT + decode throughput",
            "backends_swept": backends,
            "prompt_lengths": prompt_lengths,
            "decode_lengths": decode_lengths,
            "repetitions": repetitions,
            "start_time": datetime.now(timezone.utc).isoformat(),
        },
        "per_backend": {},
    }

    # Capture GPU idle baseline before the first backend run
    if gpu_monitor is not None:
        gpu_monitor.start_idle()
        idle_interval = cfg.get("monitor_interval_s", 1.0)
        for _ in range(int(5.0 / idle_interval)):
            gpu_monitor.record_idle()
            time.sleep(idle_interval)

    for backend in backends:
        backend_key = backend
        label = f"backend_{backend}"

        print(f"[attention_backend] starting vLLM with --attention-backend {backend}")

        # Build command for this backend
        if attn_idx is not None:
            command = list(base_command)
            command[attn_idx + 1] = backend
        else:
            command = base_command + ["--attention-backend", backend]

        # Start fresh server for this backend
        backend_server = VllmServer(
            command=command,
            base_url=cfg["endpoint"]["base_url"],
            log_path=run_dir / f"vllm_{label}.log",
            environment=server.environment,
            shutdown_timeout_s=server.shutdown_timeout_s,
        )

        try:
            backend_server.start()
        except Exception as exc:
            results["per_backend"][backend_key] = {
                "status": "server_start_failed",
                "error": str(exc),
            }
            print(f"[attention_backend] WARNING: could not start backend {backend}: {exc}")
            continue

        # Reuse the client already passed in (avoids per-backend socket churn)
        backend_client = client

        try:
            backend_client.wait_until_ready(
                timeout_s=cfg.get("server", {}).get("startup_timeout_s", 600),
                poll_s=cfg.get("ready_poll_s", 3.0),
                process_check=backend_server.check_running,
            )
        except BaseException as exc:
            results["per_backend"][backend_key] = {
                "status": "server_not_ready",
                "error": str(exc),
            }
            print(f"[attention_backend] WARNING: backend {backend} did not become ready: {exc}")
            backend_server.stop()
            backend_server.save_metadata(
                run_dir / f"resolved_server_{label}.json"
            )
            if not isinstance(exc, Exception):
                raise
            continue

        gpu_window_started = False
        try:
            # GPU telemetry window for this backend
            if gpu_monitor is not None:
                gpu_monitor.start_window(f"attention_{backend_key}")
                gpu_window_started = True

            # --- Latency sweep ---
            latency_results: dict[str, Any] = {}
            for plen in prompt_lengths:
                prompt = build_prompt_of_length(plen)
                ttfts: list[float] = []
                for i in range(repetitions):
                    try:
                        gen = backend_client.generate(
                            prompt, max_tokens=8, temperature=0.0
                        )
                        if gen.ttft_s > 0:
                            ttfts.append(gen.ttft_s)
                    except Exception:
                        pass
                    if i < repetitions - 1:
                        time.sleep(0.25)

                latency_results[str(plen)] = {
                    "requested_tokens": plen,
                    "n_success": len(ttfts),
                    "ttft": (
                        _stat_summary(ttfts)
                        if ttfts
                        else {
                            "avg_s": None,
                            "median_s": None,
                            "p95_s": None,
                            "min_s": None,
                            "max_s": None,
                        }
                    ),
                }

            # --- Decode sweep ---
            decode_results: dict[str, Any] = {}
            fixed_prompt = build_prompt_of_length(128)
            for n_out in decode_lengths:
                try:
                    gen = backend_client.generate(
                        fixed_prompt, max_tokens=n_out, temperature=0.0
                    )
                    decode_time = (
                        gen.total_time_s - gen.ttft_s
                        if gen.ttft_s == gen.ttft_s
                        else gen.total_time_s
                    )
                    tok_per_sec = (
                        gen.output_tokens / decode_time if decode_time > 0 else None
                    )
                    instant_rates = [
                        1.0 / g for g in gen.per_token_times if g > 0
                    ]
                    decode_results[str(n_out)] = {
                        "requested_output_tokens": n_out,
                        "actual_output_tokens": gen.output_tokens,
                        "tok_per_sec_avg": (
                            round(tok_per_sec, 2) if tok_per_sec else None
                        ),
                        "tok_per_sec_peak": (
                            round(max(instant_rates), 2) if instant_rates else None
                        ),
                        "n": 1,
                    }
                except Exception as exc:
                    decode_results[str(n_out)] = {
                        "status": "error",
                        "error": str(exc),
                    }

            # Stop GPU window
            gpu_summary = None
            if gpu_monitor is not None:
                gpu_summary = gpu_monitor.stop_window(f"attention_{backend_key}")
                gpu_window_started = False

            results["per_backend"][backend_key] = {
                "command": command,
                "status": "success",
                "gpu": gpu_summary or {},
                "latency": latency_results,
                "decode": decode_results,
            }

        finally:
            if gpu_window_started and gpu_monitor is not None:
                gpu_monitor.stop_window(f"attention_{backend_key}")
            backend_server.stop()
            backend_server.save_metadata(
                run_dir / f"resolved_server_{label}.json"
            )

    results["config"]["end_time"] = datetime.now(timezone.utc).isoformat()
    return results