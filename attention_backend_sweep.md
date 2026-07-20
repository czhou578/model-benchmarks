# Attention Backend Sweep — Implementation Plan

## Overview

Add a new benchmark `benchmarks/attention_backend.py:run_attention_backend_sweep()` that sweeps vLLM's supported attention backends (e.g. `flashinfer`, `flash_attn`, `triton_attn`) and reports TTFT, decode throughput, GPU utilization, and memory per backend.

The benchmark uses **managed server mode** with `VllmServer` lifecycle management — the runner starts vLLM with each backend flag, runs the workload, then restarts with the next backend. This mirrors the existing `--compare-spec` pattern in `core_runner.py`.

---

## Files to Create / Modify

| Action | File |
|--------|------|
| **Create** | `benchmarks/attention_backend.py` — new benchmark module |
| **Modify** | `core_runner.py` — add `--skip-attention` flag and orchestration block |

No YAML changes required — backends are hardcoded in the benchmark.

---

## Design Decisions

### 1. Workload: reuse latency + decode (not prefill)
- Attention backends primarily affect **decode** and **TTFT** (prefill is less sensitive for most models at moderate lengths).
- Reuse the existing latency sweep (e.g. 32, 128, 512, 2048) + decode (512, 1024, 2048 tokens).
- Rationale: the existing latency benchmark already has calibrated prompts, warmup, and OOM handling.

### 2. Server lifecycle: stop → restart per backend
- Each backend is a different `--attention-backend` flag value on the vLLM CLI.
- The existing model YAML already has a default `--attention-backend` value (e.g. `flashinfer`).
- For each sweep value, start a fresh `VllmServer`, run the workload, stop it, then proceed to the next.
- This is identical to the `--compare-spec` pattern at `core_runner.py:1237-1278`.

### 3. GPU telemetry: shared GpuMonitor
- A single `GpuMonitor` instance runs throughout the entire sweep.
- Each backend run gets its own `start_window` / `stop_window` for per-backend GPU stats.
- The `start_idle()` / `record_idle()` pattern is only called once before the first backend.

### 4. Backends to sweep
Default: **all available backends** detected at runtime.

The benchmark will detect which backends are supported by checking `vllm.v1.attention.backends.registry.AttentionBackendEnum` and filtering out backends that require unsupported hardware (e.g. ROCM_ATTN on NVIDIA).

Default sweep set for NVIDIA GPUs:
- `FLASH_ATTN` — default, uses flash-attn kernels
- `FLASH_ATTN_DIFFKV` — flash-attn with different prefill/decode kernels
- `TRITON_ATTN` — triton-based attention
- `TRITON_ATTN_DIFFKV` — triton with split prefill/decode

Configurable via `attention_backends: [flash_attn, triton_attn]` in the YAML.

### 5. Output file: `attention_backend.json`

```json
{
  "config": {
    "benchmark_version": "1.0",
    "definition": "sweep vLLM attention backends and measure TTFT + decode throughput",
    "backends_swept": ["flash_attn", "flash_attn_diffkv", "triton_attn"],
    "prompt_lengths": [32, 128, 512, 2048],
    "decode_lengths": [512, 1024, 2024],
    "repetitions": 5,
    "start_time": "2026-07-20T...",
    "end_time": "2026-07-20T..."
  },
  "per_backend": {
    "flash_attn": {
      "gpu": { "gpu_util_avg_pct": 85.2, "gpu_power_avg_w": 280.5, ... },
      "latency": {
        "32": { "ttft_avg_s": 0.0012, "n": 5 },
        "128": { "ttft_avg_s": 0.0034, "n": 5 },
        ...
      },
      "decode": {
        "512": { "tok_per_sec_avg": 245.3, "n": 1 },
        ...
      }
    },
    "triton_attn": {
      ...
    }
  }
}
```

---

## Implementation Details

### `benchmarks/attention_backend.py`

```python
"""Attention backend sweep benchmark.

Compares vLLM attention backends by running a standard latency + decode
workload under each backend, with GPU telemetry per backend.

Output: attention_backend.json
"""

from __future__ import annotations

import json
import time
import uuid
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_runner import ModelClient, GpuMonitor, build_prompt_of_length, count_tokens
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

        backend_client = ModelClient(
            base_url=cfg["endpoint"]["base_url"],
            model_name=cfg["endpoint"].get("model_name", cfg["name"]),
            api_key=cfg["endpoint"].get("api_key"),
            chat=cfg["endpoint"].get("chat", True),
        )

        try:
            backend_client.wait_until_ready(
                timeout_s=cfg.get("server", {}).get("startup_timeout_s", 600),
                poll_s=cfg.get("ready_poll_s", 3.0),
            )
        except TimeoutError as exc:
            results["per_backend"][backend_key] = {
                "status": "server_not_ready",
                "error": str(exc),
            }
            continue

        # GPU telemetry window for this backend
        if gpu_monitor is not None:
            gpu_monitor.start_window(f"attention_{backend_key}")

        # --- Latency sweep ---
        latency_results: dict[str, Any] = {}
        for plen in prompt_lengths:
            prompt = build_prompt_of_length(plen)
            ttfts: list[float] = []
            for i in range(repetitions):
                try:
                    gen = backend_client.generate(prompt, max_tokens=8, temperature=0.0)
                    if gen.ttft_s > 0:
                        ttfts.append(gen.ttft_s)
                except Exception:
                    pass
                if i < repetitions - 1:
                    time.sleep(0.25)

            latency_results[str(plen)] = {
                "requested_tokens": plen,
                "n_success": len(ttfts),
                "ttft": _stat_summary(ttfts) if ttfts else {"avg_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None},
            }

        # --- Decode sweep ---
        decode_results: dict[str, Any] = {}
        fixed_prompt = build_prompt_of_length(128)
        for n_out in decode_lengths:
            try:
                gen = backend_client.generate(fixed_prompt, max_tokens=n_out, temperature=0.0)
                decode_time = gen.total_time_s - gen.ttft_s if gen.ttft_s == gen.ttft_s else gen.total_time_s
                tok_per_sec = gen.output_tokens / decode_time if decode_time > 0 else None
                instant_rates = [1.0 / g for g in gen.per_token_times if g > 0]
                decode_results[str(n_out)] = {
                    "requested_output_tokens": n_out,
                    "actual_output_tokens": gen.output_tokens,
                    "tok_per_sec_avg": round(tok_per_sec, 2) if tok_per_sec else None,
                    "tok_per_sec_peak": round(max(instant_rates), 2) if instant_rates else None,
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

        results["per_backend"][backend_key] = {
            "command": command,
            "status": "success",
            "gpu": gpu_summary or {},
            "latency": latency_results,
            "decode": decode_results,
        }

        backend_server.stop()
        backend_server.save_metadata(run_dir / f"resolved_server_{label}.json")

    results["config"]["end_time"] = datetime.now(timezone.utc).isoformat()
    return results

```

### `core_runner.py` modifications

Add at the argparse section (around line 1068):
```python
parser.add_argument("--skip-attention", action="store_true",
                    help="Skip attention backend sweep benchmark")
```

Add after the `--skip-prefill` block (around line 1228):
```python
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
```

---

## Testing

1. **Smoke test** (single backend, no restart needed — just verify import):
   ```bash
   python -c "from benchmarks.attention_backend import _detect_available_backends; print(_detect_available_backends())"
   ```

2. **Full sweep** (managed server mode):
   ```bash
   python core_runner.py --model models/qwen3.6_35b_nvidia_nvfp4.yml --skip-latency --skip-decode --skip-reasoning --skip-concurrency --skip-prefill --skip-ttft --skip-compare-spec
   ```
   This exercises the new benchmark in isolation.

3. **Verify output**: check `results/<model>/<ts>/attention_backend.json` for per-backend `latency` and `decode` sections plus `gpu` telemetry.

---

## Complexity Estimate

- **New file**: `benchmarks/attention_backend.py` (~150 lines)
- **Modified file**: `core_runner.py` (~15 lines added)
- **Total**: ~165 lines, two files changed, zero tests added (benchmarks follow the existing pattern — tested by running).