"""GPU specifications database for roofline analysis.

Provides known peak FLOPS, HBM bandwidth, and memory capacity for common
GPUs used in LLM inference. Used by the roofline benchmark to compute
theoretical throughput bounds and classify workloads as compute-bound vs
memory-bound.

To add a new GPU, extend the GPU_SPECS dict.  All values are sourced from
official NVIDIA data sheets or white papers.
"""

from __future__ import annotations

from typing import Any

GPU_SPECS: dict[str, dict[str, float]] = {
    # ------------------------------------------------------------------ #
    # NVIDIA H100 family
    # ------------------------------------------------------------------ #
    "NVIDIA H100 80GB HBM3": {
        "peak_tflops_tf32": 1.979,       # Tensor Core, SXM5
        "peak_tflops_fp8": 3.958,         # Tensor Core, SXM5 (sparsity)
        "hbm_bandwidth_gbs": 3350,        # GB/s
        "hbm_bytes": 80e9,
        "sm_count": 132,
        "sm_clock_mhz": 1980,
        "architecture": "Hopper",
        "chip_name": "GH100",
    },
    "NVIDIA H100 80GB PCIe": {
        "peak_tflops_tf32": 1.513,       # PCIe variant
        "peak_tflops_fp8": 3.026,
        "hbm_bandwidth_gbs": 2000,        # PCIe version has lower BW
        "hbm_bytes": 80e9,
        "sm_count": 132,
        "sm_clock_mhz": 1770,
        "architecture": "Hopper",
        "chip_name": "GH100",
    },
    "NVIDIA H100 80GB SXM": {
        "peak_tflops_tf32": 1.979,
        "peak_tflops_fp8": 3.958,
        "hbm_bandwidth_gbs": 3350,
        "hbm_bytes": 80e9,
        "sm_count": 132,
        "sm_clock_mhz": 1980,
        "architecture": "Hopper",
        "chip_name": "GH100",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA H200
    # ------------------------------------------------------------------ #
    "NVIDIA H200 141GB HBM3e": {
        "peak_tflops_tf32": 1.979,
        "peak_tflops_fp8": 3.958,
        "hbm_bandwidth_gbs": 4800,
        "hbm_bytes": 141e9,
        "sm_count": 132,
        "sm_clock_mhz": 1980,
        "architecture": "Hopper",
        "chip_name": "GH200",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA A100 family
    # ------------------------------------------------------------------ #
    "NVIDIA A100 80GB SXM4": {
        "peak_tflops_tf32": 1.565,       # Tensor Core
        "peak_tflops_fp16": 0.313,
        "hbm_bandwidth_gbs": 2000,
        "hbm_bytes": 80e9,
        "sm_count": 108,
        "sm_clock_mhz": 1410,
        "architecture": "Ampere",
        "chip_name": "GA100",
    },
    "NVIDIA A100 40GB": {
        "peak_tflops_tf32": 1.565,
        "hbm_bandwidth_gbs": 2000,
        "hbm_bytes": 40e9,
        "sm_count": 108,
        "sm_clock_mhz": 1410,
        "architecture": "Ampere",
        "chip_name": "GA100",
    },
    "NVIDIA A100 80GB PCIe": {
        "peak_tflops_tf32": 1.565,
        "hbm_bandwidth_gbs": 1555,
        "hbm_bytes": 80e9,
        "sm_count": 108,
        "sm_clock_mhz": 1410,
        "architecture": "Ampere",
        "chip_name": "GA100",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA L40 / L40S
    # ------------------------------------------------------------------ #
    "NVIDIA L40": {
        "peak_tflops_tf32": 1.810,
        "hbm_bandwidth_gbs": 504,
        "hbm_bytes": 48e9,
        "sm_count": 150,
        "sm_clock_mhz": 1740,
        "architecture": "Ada Lovelace",
        "chip_name": "AD107",
    },
    "NVIDIA L40S": {
        "peak_tflops_tf32": 1.810,
        "hbm_bandwidth_gbs": 504,
        "hbm_bytes": 48e9,
        "sm_count": 150,
        "sm_clock_mhz": 1740,
        "architecture": "Ada Lovelace",
        "chip_name": "AD107",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA L-series (LPDDR5X)
    # ------------------------------------------------------------------ #
    "NVIDIA L20": {
        "peak_tflops_tf32": 1.195,
        "hbm_bandwidth_gbs": 640,
        "hbm_bytes": 48e9,
        "sm_count": 96,
        "sm_clock_mhz": 1815,
        "architecture": "Ada Lovelace",
        "chip_name": "AD102",
    },
    "NVIDIA L4": {
        "peak_tflops_tf32": 0.189,
        "hbm_bandwidth_gbs": 150,
        "hbm_bytes": 24e9,
        "sm_count": 40,
        "sm_clock_mhz": 1950,
        "architecture": "Ada Lovelace",
        "chip_name": "L24",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA Grace Hopper (GB200 / GH200)
    # ------------------------------------------------------------------ #
    "NVIDIA Grace Hopper GH200": {
        "peak_tflops_tf32": 1.979,
        "peak_tflops_fp8": 3.958,
        "hbm_bandwidth_gbs": 4800,
        "hbm_bytes": 96e9,
        "sm_count": 132,
        "sm_clock_mhz": 1980,
        "architecture": "Hopper",
        "chip_name": "GH200",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA Blackwell family (GB100 / GB200)
    # ------------------------------------------------------------------ #
    "NVIDIA GB100 80GB HBM3e": {
        "peak_tflops_tf32": 3.140,       # Tensor Core estimate
        "peak_tflops_fp8": 6.280,
        "hbm_bandwidth_gbs": 4800,
        "hbm_bytes": 80e9,
        "sm_count": 196,
        "sm_clock_mhz": 2000,
        "architecture": "Blackwell",
        "chip_name": "GB100",
    },
    "NVIDIA GB200": {
        "peak_tflops_tf32": 3.140,
        "peak_tflops_fp8": 6.280,
        "hbm_bandwidth_gbs": 4800,
        "hbm_bytes": 100e9,
        "sm_count": 196,
        "sm_clock_mhz": 2000,
        "architecture": "Blackwell",
        "chip_name": "GB200",
    },
    "NVIDIA GB10": {
        "peak_tflops_tf32": 2.200,
        "hbm_bandwidth_gbs": 3800,
        "hbm_bytes": 32e9,
        "sm_count": 126,
        "sm_clock_mhz": 2000,
        "architecture": "Blackwell",
        "chip_name": "GB10",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA V100 / T4
    # ------------------------------------------------------------------ #
    "NVIDIA V100 32GB SXM2": {
        "peak_tflops_tf32": 0.749,
        "peak_tflops_fp16": 1.503,
        "hbm_bandwidth_gbs": 900,
        "hbm_bytes": 32e9,
        "sm_count": 64,
        "sm_clock_mhz": 1530,
        "architecture": "Volta",
        "chip_name": "GV100",
    },
    "NVIDIA V100 16GB PCIe": {
        "peak_tflops_tf32": 0.749,
        "hbm_bandwidth_gbs": 700,
        "hbm_bytes": 16e9,
        "sm_count": 64,
        "sm_clock_mhz": 1530,
        "architecture": "Volta",
        "chip_name": "GV100",
    },
    "NVIDIA T4": {
        "peak_tflops_tf32": 0.084,
        "peak_tflops_fp16": 0.168,
        "hbm_bandwidth_gbs": 320,
        "hbm_bytes": 16e9,
        "sm_count": 40,
        "sm_clock_mhz": 1590,
        "architecture": "Turing",
        "chip_name": "TU104",
    },

    # ------------------------------------------------------------------ #
    # NVIDIA H800 (China variant)
    # # ------------------------------------------------------------------ #
    "NVIDIA H800 80GB": {
        "peak_tflops_tf32": 1.979,
        "peak_tflops_fp8": 3.958,
        "hbm_bandwidth_gbs": 3000,
        "hbm_bytes": 80e9,
        "sm_count": 132,
        "sm_clock_mhz": 1980,
        "architecture": "Hopper",
        "chip_name": "GH100",
    },

    # ------------------------------------------------------------------ #
    # AMD (add as needed)
    # ------------------------------------------------------------------ #
    "AMD MI300X": {
        "peak_tflops_fp16": 212,         # MI300X has huge FP16 throughput
        "hbm_bandwidth_gbs": 1630,
        "hbm_bytes": 192e9,
        "compute_units": 150,
        "cu_clock_mhz": 1900,
        "architecture": "CDNA 3",
        "chip_name": "MI300X",
    },
}


def lookup_gpu_specs(gpu_name: str) -> dict[str, float] | None:
    """Fuzzy-match a GPU name against known specifications.

    Returns the spec dict for the best matching GPU, or ``None`` when
    the GPU is not in the database.
    """
    if not gpu_name:
        return None

    # Exact match first
    if gpu_name in GPU_SPECS:
        return GPU_SPECS[gpu_name]

    # Case-insensitive exact match
    name_lower = gpu_name.lower()
    for key, specs in GPU_SPECS.items():
        if key.lower() == name_lower:
            return specs

    # Substring / contains match
    best_match: dict[str, float] | None = None
    best_len = 0
    for key, specs in GPU_SPECS.items():
        key_lower = key.lower()
        if name_lower in key_lower or key_lower in name_lower:
            # Prefer the longer (more specific) match
            if len(key) > best_len:
                best_match = specs
                best_len = len(key)

    return best_match


def get_roofline_threshold(specs: dict[str, float]) -> float | None:
    """Compute the arithmetic intensity threshold (bytes / FLOP) where the
    GPU transitions from memory-bound to compute-bound.

    Threshold = peak_bandwidth (bytes/s) / peak_flops (FLOP/s) = bytes/FLOP.

    Returns the threshold or None if the GPU is not known.
    """
    bandwidth_gbs = specs.get("hbm_bandwidth_gbs")
    if bandwidth_gbs is None:
        return None

    # Prefer TF32/FP8 peak FLOPS; fall back to FP16
    peak_tflops = specs.get("peak_tflops_tf32") or specs.get("peak_tflops_fp8")
    if peak_tflops is None:
        return None

    bandwidth_bytes_s = bandwidth_gbs * 1e9        # GB/s → B/s
    peak_flops = peak_tflops * 1e12                 # TFLOPS → FLOP/s

    return bandwidth_bytes_s / peak_flops            # bytes / FLOP