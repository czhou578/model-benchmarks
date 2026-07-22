"""Roofline analysis benchmark — Phase 7 (Systems Analysis).

Estimates whether workloads are limited by:
  - Compute (peak FLOPS / arithmetic intensity)
  - Memory bandwidth (HBM bandwidth / arithmetic intensity)
  - Scheduler overhead (queue + kernel launch time)

The module computes:
  1. FLOP/token and memory-traffic/token estimates from model architecture
  2. Arithmetic intensity = FLOP / byte (per workload type)
  3. Theoretical throughput bounds from GPU specs (roofline peak)
  4. Comparison of measured vs theoretical throughput
  5. Workload classification: compute-bound, memory-bound, or mixed

Output file: ``roofline.json``

"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

from core_runner import ModelClient, GpuMonitor, _run

from benchmarks.roofline_spec_db import (
    get_roofline_threshold,
    lookup_gpu_specs,
)
from benchmarks.architecture_flops import (
    ModelFlopsEstimator,
    load_architecture_config,
)


# --------------------------------------------------------------------------- #
# Model architecture introspection
# --------------------------------------------------------------------------- #


def _get_model_config(client: ModelClient) -> dict[str, Any] | None:
    """Fetch the unmodified model config exposed by the vLLM server."""
    try:
        resp = client.base_url
        import requests
        r = requests.get(f"{resp}/v1/models", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        model_data = data.get("data", [{}])[0] if data.get("data") else {}
        config = model_data.get("config", {})
        if isinstance(config, dict) and config:
            return config
    except Exception:
        pass
    return None




# --------------------------------------------------------------------------- #
# Roofline analysis — compute bound, memory bound, classification
# --------------------------------------------------------------------------- #


def compute_arithmetic_intensity(flops: float, bytes_val: float) -> float:
    """Compute arithmetic intensity (FLOP / byte)."""
    return flops / bytes_val if bytes_val > 0 else float("inf")


def classify_workload(arithmetic_intensity: float,
                      gpu_threshold: float) -> str:
    """Classify the workload bound type.

    Args:
        arithmetic_intensity: FLOP / byte ratio.
        gpu_threshold: GPU-specific bytes/FLOP threshold
            (peak_bandwidth / peak_flops).

    Returns:
        One of "compute", "memory", "mixed".
    """
    if gpu_threshold <= 0:
        return "unknown"

    ratio = arithmetic_intensity / gpu_threshold

    if ratio < 0.5:
        return "memory"
    elif ratio > 1.5:
        return "compute"
    else:
        return "mixed"


def _compute_bounds_from_estimator(
    flops_result: dict[str, Any],
    gpu_specs: dict[str, float],
    weight_bits: int,
    *,
    flops_estimator: "ModelFlopsEstimator",
    sequence_length: int,
) -> dict[str, Any]:
    """Compute roofline bounds using component-estimator results.

    Replaces the old ``compute_theoretical_bounds`` for models that use
    component estimators (MoE, linear attention, etc.).
    """
    total_flops = flops_result.get("total", 0)
    # For prefill, total_flops is the *total* work for S tokens.
    # Divide by S to get per-token-equivalent for the roofline calculation.
    n_tokens = sequence_length if flops_result.get("mode") == "prefill" else 1
    flops_per_token = total_flops / n_tokens

    threshold = get_roofline_threshold(gpu_specs)

    # Peak throughput (compute-bound)
    peak_tflops = gpu_specs.get("peak_tflops_tf32") or gpu_specs.get("peak_tflops_fp8", 0)
    peak_flops = peak_tflops * 1e12
    compute_bound_tps = peak_flops / flops_per_token if flops_per_token > 0 else float("inf")

    # Memory bound: estimate bytes per token from weights + KV cache
    dtype_bytes_int = int(weight_bits / 8)
    weight_bytes = flops_estimator.weight_bytes(dtype_bytes_int)
    total_weight_bytes = sum(
        v for v in weight_bytes.values() if isinstance(v, (int, float))
    )

    # KV cache: only for full-attention layers
    kv_bytes = flops_estimator.kv_state_bytes(sequence_length, dtype_bytes_int)
    kv_read = kv_bytes.get("full_attention_kv", 0)  # per-sequence, read every token

    # Estimate traffic: per-token traffic is dominated by weights + KV read
    # For prefill, KV read is zero (nothing cached yet), but KV write matters
    # For decode, KV read is the full cache per token
    bytes_per_token = total_weight_bytes + kv_read

    if bytes_per_token > 0:
        memory_bound_tps = (gpu_specs.get("hbm_bandwidth_gbs", 0) * 1e9) / bytes_per_token
    else:
        memory_bound_tps = float("inf")

    ai = compute_arithmetic_intensity(flops_per_token, bytes_per_token)
    bound_type = classify_workload(ai, threshold) if threshold else "unknown"

    effective_tps = min(compute_bound_tps, memory_bound_tps)

    # Build per-component breakdown
    flops_breakdown = {k: round(v, 1) for k, v in flops_result.items()
                       if isinstance(v, (int, float)) and k not in ("total", "num_layers", "sequence_length")}

    return {
        "flops_per_token": round(flops_per_token, 1),
        "flops_per_token_breakdown": flops_breakdown,
        "bytes_per_token": round(bytes_per_token, 1),
        "bytes_per_token_breakdown": {
            "weights": round(total_weight_bytes, 1),
            "kv_cache_read": round(kv_read, 1),
        },
        "arithmetic_intensity": round(ai, 4),
        "roofline_threshold": threshold,
        "bound": bound_type,
        "compute_bound_tps": round(compute_bound_tps, 1),
        "memory_bound_tps": round(memory_bound_tps, 1),
        "theoretical_tps": round(effective_tps, 1),
    }




# --------------------------------------------------------------------------- #
# Main benchmark function
# --------------------------------------------------------------------------- #


def run_roofline_analysis(
    model_config: dict[str, Any],
    client: ModelClient,
    gpu_monitor: GpuMonitor | None = None,
    prefill_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Run roofline analysis on a model and GPU.

    Computes theoretical throughput bounds for prefill workloads at various
    sequence lengths, classifies each as compute-bound or memory-bound,
    and compares against any available measured performance data.

    Args:
        model_config: the full model YAML config dict.
        client: ModelClient connected to the running vLLM endpoint.
        gpu_monitor: optional GPU telemetry monitor.
        prefill_lengths: sequence lengths to analyze (default: [512, 2K, 8K, 32K]).

    Returns:
        Dict with config metadata, GPU specs, per-length analysis,
        and utilization score.
    """
    if prefill_lengths is None:
        prefill_lengths = [512, 2048, 8192, 32768]

    # Load in precedence order: local path, Hugging Face, complete server config.
    config_result = load_architecture_config(
        model_config,
        server_config=_get_model_config(client),
    )
    if config_result.status == "unsupported":
        return {
            "config": {
                "benchmark_version": "1.1",
                "definition": "Roofline analysis: architecture config unavailable",
                "model_name": model_config.get("name", "unknown"),
                "start_time": datetime.utcnow().isoformat() + "Z",
                "end_time": datetime.utcnow().isoformat() + "Z",
            },
            "estimate_status": "unsupported",
            "architecture": None,
            "warnings": list(config_result.warnings),
            "per_length": {},
            "decode": {},
        }

    assert config_result.normalized_config is not None
    assert config_result.features is not None
    normalized = config_result.normalized_config
    features = config_result.features

    # Build the component-based flops estimator
    try:
        flops_estimator = ModelFlopsEstimator(normalized, features)
        estimate_status = flops_estimator.estimate_status
    except Exception as exc:
        # Fall back: report unsupported rather than crashing
        return {
            "config": {
                "benchmark_version": "1.1",
                "definition": "Roofline analysis: component estimators failed",
                "model_name": model_config.get("name", "unknown"),
                "model_config": normalized.to_dict(),
                "start_time": datetime.utcnow().isoformat() + "Z",
                "end_time": datetime.utcnow().isoformat() + "Z",
            },
            "estimate_status": "partial",
            "architecture": features.to_dict(),
            "warnings": [*config_result.warnings, str(exc)],
            "per_length": {},
            "decode": {},
        }

    # Detect weight precision from the model name or config
    weight_bits = 16  # default
    model_name = str(model_config.get("name", "")).lower()
    if "nf4" in model_name or "nvfp4" in model_name:
        weight_bits = 4
    elif "nf8" in model_name or "fp8" in model_name:
        weight_bits = 8
    elif "int8" in model_name:
        weight_bits = 8
    elif "int4" in model_name:
        weight_bits = 4

    # Identify GPU
    gpu_name: str | None = None
    gpu_specs: dict[str, float] | None = None
    try:
        gpu_out = _run([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
        ])
        if gpu_out:
            gpu_name = gpu_out.strip().splitlines()[0].strip()
            gpu_specs = lookup_gpu_specs(gpu_name)
    except Exception:
        pass

    # Collect GPU telemetry summary if available
    gpu_util = {}
    if gpu_monitor and gpu_monitor.samples:
        util_samples = [s for s in gpu_monitor.samples if s.get("gpu_util_pct") is not None]
        if util_samples:
            utils = [s["gpu_util_pct"] for s in util_samples]
            mem_utils = [s["gpu_mem_util_pct"] for s in util_samples if s.get("gpu_mem_util_pct") is not None]
            gpu_util = {
                "gpu_utilization_avg": round(statistics.mean(utils), 1),
                "gpu_utilization_peak": round(max(utils), 1),
                "gpu_memory_utilization_avg": round(statistics.mean(mem_utils), 1) if mem_utils else None,
                "gpu_memory_utilization_peak": round(max(mem_utils), 1) if mem_utils else None,
                "num_samples": len(util_samples),
            }

    # Build per-length analysis using component estimators
    per_length: dict[str, Any] = {}
    max_seq = max(prefill_lengths) if prefill_lengths else 8192
    for length in prefill_lengths:
        key = str(length)
        pf = flops_estimator.prefill_flops(length)

        # Compute roofline bounds from estimator results
        pf_analysis = _compute_bounds_from_estimator(
            pf, gpu_specs or {}, weight_bits,
            flops_estimator=flops_estimator, sequence_length=length,
        )
        pf_analysis["mode"] = "prefill"
        pf_analysis["sequence_length"] = length
        per_length[key] = pf_analysis

    # Decode analysis (one new token, context = max_seq)
    decode_result = flops_estimator.decode_model_flops(context_length=max_seq)
    decode_analysis = _compute_bounds_from_estimator(
        decode_result, gpu_specs or {}, weight_bits,
        flops_estimator=flops_estimator, sequence_length=max_seq,
    )
    decode_analysis["mode"] = "decode"
    decode_analysis["sequence_length"] = max_seq

    # Build result
    result: dict[str, Any] = {
        "config": {
            "benchmark_version": "1.1",
            "definition": "Roofline analysis: theoretical throughput bounds and workload classification",
            "model_name": model_config.get("name", "unknown"),
            "model_config": normalized.to_dict(),
            "weight_bits": weight_bits,
            "gpu_name": gpu_name,
            "prefill_lengths_analyzed": prefill_lengths,
            "start_time": datetime.utcnow().isoformat() + "Z",
        },
        "gpu": {
            "name": gpu_name,
            "specs": gpu_specs if gpu_specs else {},
            "spec_source": "known_database" if gpu_specs else "unknown",
            "telemetry": gpu_util,
        },
        "per_length": per_length,
        "decode": decode_analysis,
        "estimate_status": estimate_status,
        "architecture": features.to_dict(),
        "assumptions": list(flops_estimator.assumptions),
        "warnings": list(config_result.warnings),
    }

    result["config"]["end_time"] = datetime.utcnow().isoformat() + "Z"
    return result