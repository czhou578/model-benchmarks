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
from benchmarks.architecture_flops import load_architecture_config


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
# FLOP estimation — from model architecture
# --------------------------------------------------------------------------- #


def estimate_flops_per_token(model_config: dict[str, Any],
                              sequence_length: int) -> dict[str, float]:
    """Estimate FLOPs per token for inference (decode or prefill).

    Args:
        model_config: dict with keys hidden_size, num_hidden_layers,
            intermediate_size, num_attention_heads, vocab_size.
        sequence_length: context length in tokens.

    Returns:
        Dict with ``flops_per_token`` and per-component breakdown:
            - ``flops_per_token``     total FLOPs per output token
            - ``ffn``                 feed-forward network (dominant)
            - ``attn_proj``           Q/K/V projection matrices
            - ``attn_score``          QK^T + softmax + V multiply
            - ``attn_out_proj``       attention output projection
            - ``lm_head``             language model head (negligible)
    """
    d = model_config.get("hidden_size", 4096)
    l = model_config.get("num_hidden_layers", 32)
    m = model_config.get("intermediate_size", 11008)
    v = model_config.get("vocab_size", 32000)
    s = sequence_length

    # Per-layer FLOPs (2 FLOPs per multiply-add)
    ffn = 4 * l * d * m              # GELU/activation: ~2× more
    attn_proj = 2 * l * 3 * d * d    # Q, K, V projections
    attn_score = l * 2 * d * s + 2 * d * s     # QK^T (softmax is negligible)
    attn_out = 2 * l * d * d         # output projection

    total = ffn + attn_proj + attn_score + attn_out + 2 * l * d * v

    return {
        "flops_per_token": total,
        "ffn": ffn,
        "attn_proj": attn_proj,
        "attn_score": attn_score,
        "attn_out_proj": attn_out,
        "lm_head": 2 * d * v,
    }


# --------------------------------------------------------------------------- #
# Memory-traffic estimation
# --------------------------------------------------------------------------- #


def estimate_bytes_per_token(model_config: dict[str, Any],
                              sequence_length: int,
                              weight_bits: int = 16) -> dict[str, float]:
    """Estimate bytes transferred per token (GPU memory traffic).

    Args:
        model_config: same keys as estimate_flops_per_token.
        sequence_length: context length in tokens.
        weight_bits: bits per weight parameter (16 for FP16/BF16, 8 for FP8).

    Returns:
        Dict with ``bytes_per_token`` and breakdown:
            - ``bytes_per_token``  total bytes / token
            - ``weights``          model weights (moved every token)
            - ``kv_cache``         KV cache for the current sequence
            - ``activations``      temporary activations (Q, K, V, output)
    """
    d = model_config.get("hidden_size", 4096)
    l = model_config.get("num_hidden_layers", 32)
    m = model_config.get("intermediate_size", 11008)
    v = model_config.get("vocab_size", 32000)
    s = sequence_length

    wb = weight_bits / 8  # bytes per weight param

    # Model weights: total parameters × bytes-per-param
    # Approximate: hidden×intermediate×2 (FFN) + hidden²×3 (QKV) + hidden² (out) + hidden×vocab
    total_params = l * (2 * d * m + 3 * d * d + d * d) + l * d * v
    weights_bytes = total_params * wb

    # KV cache: 2 × layers × seq × head_dim × bytes (per token's access during this step)
    # For decode, we access the full KV cache: 2 × l × s × (d/n_kv) × 2 (FP16)
    kv_bytes = 2 * l * s * (d // max(1, (d // 128))) * 2

    # Activations: Q output (d×seq), V output (d×1), etc.
    # Per token: Q output matrix multiply ~ d × 1, plus intermediate activations
    activations_bytes = 3 * l * d * 4  # Q/K/V projections (d×d) output in FP32 for precision

    total = weights_bytes + kv_bytes + activations_bytes

    return {
        "bytes_per_token": total,
        "weights": weights_bytes,
        "kv_cache": kv_bytes,
        "activations": activations_bytes,
    }


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


def compute_theoretical_bounds(
    model_config: dict[str, Any],
    gpu_specs: dict[str, float],
    sequence_length: int,
    weight_bits: int = 16,
) -> dict[str, Any]:
    """Compute theoretical maximum throughput from roofline analysis.

    Args:
        model_config: model architecture.
        gpu_specs: GPU specifications (from roofline_spec_db).
        sequence_length: context length.
        weight_bits: bits per weight.

    Returns:
        Dict with:
            - ``flops_per_token``
            - ``bytes_per_token``
            - ``arithmetic_intensity``
            - ``roofline_threshold``
            - ``bound``: compute / memory / mixed
            - ``compute_bound_tps``: theoretical max tokens/s if compute-limited
            - ``memory_bound_tps``: theoretical max tokens/s if memory-limited
            - ``theoretical_tps``: effective bound (min of compute and memory bounds)
    """
    flops = estimate_flops_per_token(model_config, sequence_length)
    bytes_info = estimate_bytes_per_token(model_config, sequence_length, weight_bits)

    flops_per_tok = flops["flops_per_token"]
    bytes_per_tok = bytes_info["bytes_per_token"]
    ai = compute_arithmetic_intensity(flops_per_tok, bytes_per_tok)
    threshold = get_roofline_threshold(gpu_specs)

    # Peak throughput (compute-bound): peak_FLOPS / FLOP_per_token
    peak_tflops = gpu_specs.get("peak_tflops_tf32") or gpu_specs.get("peak_tflops_fp8", 0)
    peak_flops = peak_tflops * 1e12
    compute_bound_tps = peak_flops / flops_per_tok if flops_per_tok > 0 else float("inf")

    # Peak throughput (memory-bound): bandwidth / bytes_per_token
    bandwidth_gbs = gpu_specs.get("hbm_bandwidth_gbs", 0)
    bandwidth_bps = bandwidth_gbs * 1e9
    memory_bound_tps = bandwidth_bps / bytes_per_tok if bytes_per_tok > 0 else float("inf")

    bound_type = classify_workload(ai, threshold) if threshold else "unknown"

    effective_tps = min(compute_bound_tps, memory_bound_tps)

    return {
        "flops_per_token": flops_per_tok,
        "flops_per_token_breakdown": {
            k: v for k, v in flops.items() if k != "flops_per_token"
        },
        "bytes_per_token": bytes_per_tok,
        "bytes_per_token_breakdown": {
            k: round(v, 1) for k, v in bytes_info.items() if k != "bytes_per_token"
        },
        "arithmetic_intensity": ai,
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
    arch = normalized.raw_text_config

    # Phase 1 detects these components but never routes them through the old
    # dense/full-attention estimator. Their estimators are Phase 2 work.
    unsupported_components: list[str] = []
    if features.ffn == "moe":
        unsupported_components.append("moe_ffn")
    unsupported_components.extend(
        mixer for mixer in features.token_mixers if mixer != "full_attention"
    )
    unsupported_components.extend(features.unsupported_layer_types)
    if unsupported_components:
        return {
            "config": {
                "benchmark_version": "1.1",
                "definition": "Roofline analysis: component estimators pending",
                "model_name": model_config.get("name", "unknown"),
                "model_config": normalized.to_dict(),
                "start_time": datetime.utcnow().isoformat() + "Z",
                "end_time": datetime.utcnow().isoformat() + "Z",
            },
            "estimate_status": "partial",
            "architecture": features.to_dict(),
            "unsupported_components": sorted(set(unsupported_components)),
            "warnings": list(config_result.warnings),
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

    # Build per-length analysis
    per_length: dict[str, Any] = {}
    for length in prefill_lengths:
        key = str(length)
        analysis = compute_theoretical_bounds(arch, gpu_specs or {}, length, weight_bits)
        per_length[key] = analysis

    # Decode analysis (typically compute-bound due to minimal memory traffic)
    decode_analysis = compute_theoretical_bounds(
        arch, gpu_specs or {}, sequence_length=max(prefill_lengths) if prefill_lengths else 8192,
        weight_bits=weight_bits
    )

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
        "estimate_status": config_result.status,
        "architecture": features.to_dict(),
        "warnings": list(config_result.warnings),
    }

    result["config"]["end_time"] = datetime.utcnow().isoformat() + "Z"
    return result