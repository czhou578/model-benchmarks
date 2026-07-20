# MoE Backend Sweep — Implementation Plan

## Overview

Add a new benchmark `benchmarks/moe_backend.py:run_moe_backend_sweep()` that sweeps vLLM's supported MoE expert-computation kernels and reports throughput, latency, and GPU telemetry per backend.

The benchmark uses **managed server mode** with `VllmServer` lifecycle management — the runner starts vLLM with each `--moe-backend` flag value, runs the workload, then restarts with the next backend. This mirrors the existing `--compare-spec` and `attention_backend` patterns.

---

## Context: What is a MoE Backend?

MoE (Mixture of Experts) models route each token through a small subset of expert networks. The "MoE backend" controls which low-level kernel performs the expert computation (matrix multiply of the routed tokens). For the Qwen3.6-35B-A3B-NVFP4 (35B total / 3B active per token), the choice of MoE backend significantly affects throughput.

**Available `--moe-backend` options** (from `vllm.config.kernel.MoEBackend`):

| Backend | Description | FP4 Support | Relevance |
|---------|-------------|-------------|-----------|
| `auto` | vLLM auto-selects | — | Baseline (implicit) |
| `triton` | Generic Triton kernels | Limited | Reference/compatibility |
| `cutlass` | vLLM CUTLASS kernels | Yes | CUDA 12+ |
| `flashinfer_trtllm` | FlashInfer + TRTLLM-GEN | Yes | High perf |
| `flashinfer_cutlass` | FlashInfer + CUTLASS | Yes | High perf |
| `flashinfer_cutedsl` | FlashInfer + CuteDSL (FP4 only) | Yes | FP4 models |
| `flashinfer_b12x` | FlashInfer b12x batch kernels | Yes | FP4 (current unsloth config) |
| `marlin` | Marlin INT4/FP4 quantized kernel | Yes | Current default (nvidia config) |

Excluded on H100 Hopper (SM 90): `deep_gemm` and `deep_gemm_mega_moe` require Blackwell (SM 100+).

**Default sweep for FP4 NVIDIA**: `[marlin, triton, flashinfer_b12x, flashinfer_cutlass, cutlass, flashinfer_cutedsl]` (skip backends incompatible with hardware via detection).

---

## Files to Create / Modify

| Action | File |
|--------|------|
| **Create** | `benchmarks/moe_backend.py` — new benchmark module |
| **Modify** | `core_runner.py` — add `--skip-moe` flag and orchestration block |

No YAML changes required — backends are defined in the benchmark.

---

## Workload Design

MoE backends primarily affect **decode throughput** (expert routing is the bottleneck during generation). Pre-fill is less affected because token routing is the same regardless of backend — only the compute changes.

**Workload selection:**

1. **Decode sweep** — the primary metric. Run 512, 1024, 2048 token outputs with a fixed short prompt. This measures expert dispatch + computation throughput.
2. **TTFT / latency at medium prompt** — measures routing overhead at moderate concurrency. 512-token prompt, short output (8 tokens). This captures any MoE-backend-induced latency in token selection.
3. **Concurrent requests** — one concurrency level (8, matching the default) to measure expert load balancing across backends.

Rationale: MoE backends change **compute characteristics**, not memory layout, so decode throughput is the signal. TTFT adds routing overhead measurement.

---

## Server Lifecycle

Each backend is a different `--moe-backend` flag value. The runner:

1. Reads the base command from the model config.
2. Finds the `--moe-backend` position in the command list (or appends it).
3. For each backend: builds a new command, starts `VllmServer`, runs workload, stops server, saves metadata.

This is identical to the `--compare-spec` pattern (core_runner.py:1237-1278).

---

## GPU Telemetry

A single `GpuMonitor` runs throughout the sweep. Each backend gets its own `start_window` / `stop_window` for per-backend GPU stats. The idle baseline is captured once before the first backend.

---

## Output File: `moe_backend.json`

```json
{
  "config": {
    "benchmark_version": "1.0",
    "definition": "sweep vLLM MoE expert-computation kernels",
    "backends_swept": ["marlin", "triton", "flashinfer_b12x", "cutlass"],
    "decode_lengths": [512, 1024, 2048],
    "prompt_length": 512,
    "concurrency": 8,
    "repetitions": 5,
    "start_time": "...",
    "end_time": "..."
  },
  "per_backend": {
    "marlin": {
      "status": "success",
      "gpu": {
        "gpu_util_avg_pct": 78.3,
        "gpu_power_avg_w": 290.5,
        "gpu_mem_used_avg_mib": 28100.0,
        "energy_wh": 0.123
      },
      "decode": {
        "512": { "tok_per_sec_avg": 312.4, "tok_per_sec_peak": 345.1, "n": 1 },
        "1024": { ... },
        "2048": { ... }
      },
      "latency": {
        "512": {
          "ttft_avg_s": 0.0234,
          "ttft_p95_s": 0.0289,
          "prefill_tps_avg": 1850.3,
          "n": 5
        }
      },
      "concurrency": {
        "8": {
          "aggregate_throughput_tok_s": 456.7,
          "ttft_avg_s": 0.0567,
          "n_requests": 16,
          "n_success": 16
        }
      }
    },
    "triton": { ... }
  }
}
```

---

## Implementation Details

### `benchmarks/moe_backend.py`

```python
"""MoE backend sweep benchmark.

Compares vLLM MoE expert-computation kernels by running a standard decode,
latency, and concurrency workload under each kernel, with GPU telemetry per
backend.

Output: moe_backend.json
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_runner import ModelClient, GpuMonitor, build_prompt_of_length, count_tokens
from vllm_server import VllmServer


# --------------------------------------------------------------------------- #
# Data structures
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


def _detect_available_backends() -> list[str]:
    """Return a list of MoE backends available on this hardware.

    Filters out backends that require unsupported GPU architectures.
    deep_gemm backends require Blackwell (SM 100+) — skipped on Hopper (SM 90).
    """
    import torch
    major = torch.cuda.get_device_properties(0).major
    # SM 90 = Hopper (H100), SM 100 = Blackwell (B200), SM 80 = Ampere (A100)

    backends = [
        "marlin",
        "triton",
        "cutlass",
        "flashinfer_cutlass",
        "flashinfer_cutedsl",
        "flashinfer_b12x",
        "flashinfer_trtllm",
    ]

    if major >= 100:
        backends.extend(["deep_gemm", "deep_gemm_mega_moe"])

    return backends


# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_moe_backend_sweep(
    client: ModelClient,
    server: VllmServer,
    cfg: dict,
    run_dir: Path,
    gpu_monitor: GpuMonitor | None = None,
    backends: list[str] | None = None,
    decode_lengths: list[int] | None = None,
    prompt_length: int = 512,
    concurrency_level: int = 8,
    requests_per_concurrent: int = 16,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Sweep vLLM MoE backends and measure throughput + latency per backend.

    Args:
        client: ModelClient for the vLLM endpoint.
        server: Managed VllmServer (will be restarted per backend).
        cfg: Model config dict.
        run_dir: Results directory (for per-backend log files).
        gpu_monitor: Optional GPU telemetry collector.
        backends: List of backend names to sweep (default: auto-detect).
        decode_lengths: Output token lengths for decode sweep.
        prompt_length: Prompt length for latency/ttft measurements.
        concurrency_level: Concurrency level for concurrency test.
        requests_per_concurrent: Requests per concurrency level.
        repetitions: Requests per length per backend.

    Returns:
        Dict with per-backend results.
    """
    if backends is None:
        backends = _detect_available_backends()
    if decode_lengths is None:
        decode_lengths = [512, 1024, 2048]

    base_command = list(server.command)

    # Find the index of --moe-backend in the base command so we can
    # replace its value without duplicating the entire command list.
    moe_idx = None
    for i, arg in enumerate(base_command):
        if arg == "--moe-backend":
            moe_idx = i
            break

    results: dict[str, Any] = {
        "config": {
            "benchmark_version": "1.0",
            "definition": "sweep vLLM MoE expert-computation kernels",
            "backends_swept": backends,
            "decode_lengths": decode_lengths,
            "prompt_length": prompt_length,
            "concurrency_level": concurrency_level,
            "requests_per_concurrent": requests_per_concurrent,
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
        label = f"backend_{backend}"
        print(f"[moe_backend] starting vLLM with --moe-backend {backend}")

        # Build command for this backend
        if moe_idx is not None:
            command = list(base_command)
            command[moe_idx + 1] = backend
        else:
            command = base_command + ["--moe-backend", backend]

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
            results["per_backend"][backend] = {
                "status": "server_start_failed",
                "error": str(exc),
                "gpu": {},
                "decode": {},
                "latency": {},
                "concurrency": {},
            }
            print(f"[moe_backend] WARNING: could not start backend {backend}: {exc}")
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
            results["per_backend"][backend] = {
                "status": "server_not_ready",
                "error": str(exc),
                "gpu": {},
                "decode": {},
                "latency": {},
                "concurrency": {},
            }
            continue

        # GPU telemetry window for this backend
        if gpu_monitor is not None:
            gpu_monitor.start_window(f"moe_{backend}")

        # 1. Decode sweep
        decode_results: dict[str, Any] = {}
        fixed_prompt = build_prompt_of_length(128)
        for n_out in decode_lengths:
            try:
                gen = backend_client.generate(fixed_prompt, max_tokens=n_out, temperature=0.0)
                decode_time = (gen.total_time_s - gen.ttft_s
                               if gen.ttft_s == gen.ttft_s else gen.total_time_s)
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

        # 2. Latency / TTFT sweep
        latency_results: dict[str, Any] = {}
        prompt = build_prompt_of_length(prompt_length)
        ttfts: list[float] = []
        prefill_tps: list[float] = []
        for i in range(repetitions):
            try:
                gen = backend_client.generate(prompt, max_tokens=8, temperature=0.0)
                if gen.ttft_s > 0:
                    ttfts.append(gen.ttft_s)
                if gen.ttft_s > 0:
                    prefill_tps.append(gen.prompt_tokens / gen.ttft_s)
            except Exception:
                pass
            if i < repetitions - 1:
                time.sleep(0.25)

        latency_results["aggregated"] = {
            "requested_tokens": prompt_length,
            "n_success": len(ttfts),
            "ttft": _stat_summary(ttfts),
            "prefill_tps": _stat_summary(prefill_tps) if prefill_tps else {"avg_s": None, "median_s": None, "p95_s": None, "min_s": None, "max_s": None},
        }

        # 3. Concurrency test
        concurrency_results: dict[str, Any] = {}
        try:
            import concurrent.futures
            conc_level = [concurrency_level]
            conc_requests = requests_per_concurrent

            def _run_one(idx: int):
                try:
                    gen = backend_client.generate(
                        build_prompt_of_length(256), max_tokens=256, temperature=0.0
                    )
                    return {
                        "index": idx,
                        "success": True,
                        "ttft_s": round(gen.ttft_s, 4),
                        "total_time_s": round(gen.total_time_s, 4),
                        "output_tokens": gen.output_tokens,
                    }
                except Exception as e:
                    return {"index": idx, "success": False, "error": str(e)}

            start_wall = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
                futures = [executor.submit(_run_one, i) for i in range(conc_requests)]
                conc_reqs = [f.result() for f in concurrent.futures.as_completed(futures)]
            wall_time = time.time() - start_wall

            successes = [r for r in conc_reqs if r["success"]]
            total_output = sum(r["output_tokens"] for r in successes if r.get("success"))
            conc_key = str(concurrency_level)
            concurrency_results[conc_key] = {
                "wall_time_s": round(wall_time, 3),
                "total_output_tokens": total_output,
                "aggregate_throughput_tok_s": round(total_output / wall_time, 1) if wall_time > 0 else None,
                "n_requests": len(conc_reqs),
                "n_success": len(successes),
                "ttft": _stat_summary([r["ttft_s"] for r in successes]),
                "total_time_s": _stat_summary([r["total_time_s"] for r in successes]),
            }
        except Exception as exc:
            concurrency_results["error"] = str(exc)

        # Stop GPU window
        gpu_summary = None
        if gpu_monitor is not None:
            gpu_summary = gpu_monitor.stop_window(f"moe_{backend}")

        results["per_backend"][backend] = {
            "command": command,
            "status": "success",
            "gpu": gpu_summary or {},
            "decode": decode_results,
            "latency": latency_results,
            "concurrency": concurrency_results,
        }

        backend_server.stop()
        backend_server.save_metadata(run_dir / f"resolved_server_{label}.json")

    results["config"]["end_time"] = datetime.now(timezone.utc).isoformat()
    return results

```

### `core_runner.py` modifications

**Argparse** (add near line 1069):
```python
parser.add_argument("--skip-moe", action="store_true",
                    help="Skip MoE backend sweep benchmark")
```

**Orchestration** (add after `--skip-attention` block):
```python
if not args.skip_moe:
    if server_mode != "managed" or server is None:
        print("[core_runner] skipping MoE backend sweep: managed server mode required")
    else:
        from benchmarks.moe_backend import run_moe_backend_sweep

        print("[core_runner] MoE backend sweep")
        moe_results = run_moe_backend_sweep(
            client, server, cfg, run_dir, gpu_monitor=monitor,
            decode_lengths=cfg.get("decode_lengths", [512, 1024, 2048]),
            repetitions=cfg.get("moe_repetitions", 5),
        )
        save_json(run_dir / "moe_backend.json", moe_results)
        summary["moe_backend"] = moe_results
```

---

## Key Differences from Attention Backend Sweep

| Aspect | Attention Backend | MoE Backend |
|--------|-------------------|-------------|
| CLI flag | `--attention-backend` | `--moe-backend` |
| Primary signal | Decode + TTFT (attention kernels) | Decode (expert routing/compute) |
| Backends | 4-6 attention kernels | 6-7 MoE kernels |
| Model dependency | GPU arch (RTX vs A100) | GPU arch + model quant format |
| FP4 relevance | N/A (attention is generic) | Critical (marlin/flashinfer_b12x) |

The MoE benchmark is **simpler** than attention because:
- The workload is primarily decode (attention is the bottleneck; MoE changes the expert path)
- No need for per-length prefill calibration
- Same prompt/workload for all backends (just swap the kernel flag)

---

## Testing

1. **Smoke test**:
   ```bash
   python -c "from benchmarks.moe_backend import _detect_available_backends; print(_detect_available_backends())"
   ```

2. **Full sweep** (managed server mode):
   ```bash
   python core_runner.py --model models/qwen3.6_35b_nvidia_nvfp4.yml \
       --skip-latency --skip-decode --skip-reasoning --skip-concurrency \
       --skip-prefill --skip-ttft --skip-compare-spec --skip-attention
   ```

3. **Verify output**: check `results/<model>/<ts>/moe_backend.json` for per-backend `decode`, `latency`, `concurrency`, and `gpu` sections.

---

## Complexity Estimate

- **New file**: `benchmarks/moe_backend.py` (~180 lines)
- **Modified file**: `core_runner.py` (~12 lines added)
- **Total**: ~192 lines, two files changed