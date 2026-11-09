"""The environment record of a run: GPU, driver, CUDA, and package versions."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def run_command(cmd: list[str]) -> Optional[str]:
    """Run *cmd* and return its stripped stdout, or None when it fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    gpu_query = run_command([
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

    nvcc = run_command(["nvcc", "--version"])
    env["cuda_version"] = nvcc.splitlines()[-1] if nvcc else None

    for pkg in ("torch", "vllm"):
        env[f"{pkg}_version"] = run_command(
            [sys.executable, "-c", f"import {pkg}; print({pkg}.__version__)"]
        )

    return env

