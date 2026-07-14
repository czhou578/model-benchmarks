"""
core_runner.py — Phase 1 core benchmark runner.

Orchestrates a full benchmark pass against an OpenAI-compatible inference
endpoint (vLLM, SGLang, llama.cpp server, etc.) and captures:

  Phase 2  — environment fingerprint (GPU/driver/CUDA/torch/vLLM/CPU/RAM/kernel/clocks)
  Phase 3  — first-token latency sweep (avg/median/p95/p99 across prompt lengths)
  Phase 4  — prefill throughput (tokens/sec vs prompt length)
  Phase 5  — decode speed (avg/peak/min/median tok/sec at several output lengths)
  Phase 6  — memory usage sampled every second (GPU + CPU/RAM), written to CSV
  Phase 7  — power draw sampled every second, avg/peak watts + energy (Wh) + energy/token
  Phase 8  — reasoning token count (<think>...</think> vs answer tokens, ratios)

Phases 9-11 (HumanEval/MBPP, Deep-SWE, HLE) are NOT implemented here — they need
dataset harnesses and sandboxed execution. This runner exposes a
`register_benchmark()` plugin hook so those can be dropped in later without
touching this file's core loop.

Usage:
    python core_runner.py --model models/qwen3_35b.yaml

Every run gets its own timestamped directory under results/<model>/<ts>/ so
nothing is ever overwritten.

Dependencies:
    pip install requests pyyaml psutil
Optional (more accurate token counting):
    pip install tiktoken
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import requests
import psutil
import yaml
import tiktoken
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

_ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Rough heuristic: ~0.75 tokens per word for English text.
    return max(1, int(len(text.split()) * 0.75))

_BASE_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "storm clouds gather over the distant mountains, and engineers debate "
    "the tradeoffs between throughput and latency in modern inference "
    "systems. "
)

def build_prompt_of_length(target_tokens: int) -> str:
    """Build a prompt approximately target_tokens long by repeating filler text."""
    text = _BASE_PARAGRAPH
    while count_tokens(text) < target_tokens:
        text += _BASE_PARAGRAPH
    if _ENC is not None:
        ids = _ENC.encode(text)[:target_tokens]
        text = _ENC.decode(ids)
    return text

# --------------------------------------------------------------------------- #
# Phase 2 — Environment fingerprint
# --------------------------------------------------------------------------- #

def _run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "kernel": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.uname().processor,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }

    gpu_query = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,power.max_limit,"
        "clocks.max.sm,clocks.max.mem",
        "--format=csv,noheader",
    ])    

    if gpu_query:
        fields = [f.strip() for f in gpu_query.splitlines()[0].split(",")]
        env["gpu"] = {
            "name": fields[0] if len(fields) > 0 else None,
            "driver_version": fields[1] if len(fields) > 1 else None,
            "memory_total": fields[2] if len(fields) > 2 else None,
            "power_max_limit": fields[3] if len(fields) > 3 else None,
            "sm_clock_max": fields[4] if len(fields) > 4 else None,
            "mem_clock_max": fields[5] if len(fields) > 5 else None,
        }
    else:
        env["gpu"] = None    

    nvcc = _run(["nvcc", "--version"])
    env["cuda_version"] = nvcc.splitlines()[-1] if nvcc else _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )

    power_mode = _run(["nvidia-smi", "-q", "-d", "POWER"])
    env["power_mode_raw"] = power_mode.splitlines()[0] if power_mode else None

    return env

# --------------------------------------------------------------------------- #
# Phase 6/7 — Resource monitor (GPU memory/util/temp/clocks + power), 1 Hz
# --------------------------------------------------------------------------- #

class ResourceMonitor:
    """Background thread sampling GPU + host resources once per second."""

    NVIDIA_SMI_FIELDS = (
        "utilization.gpu,memory.used,memory.total,temperature.gpu,"
        "power.draw,clocks.sm,clocks.mem"
    )

    def __init__(self, out_dir: Path, interval_s: float = 1.0):
        self.out_dir = out_dir
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[dict[str, Any]] = []
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None  

    def _sample_once(self) -> dict[str, Any]:
        row: dict[str, Any] = {"t": time.time()}
        row["cpu_percent"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        row["ram_used_gb"] = round(vm.used / 1e9, 3)
        sw = psutil.swap_memory()
        row["swap_used_gb"] = round(sw.used / 1e9, 3)

        if self.has_nvidia_smi:
            out = _run([
                "nvidia-smi",
                f"--query-gpu={self.NVIDIA_SMI_FIELDS}",
                "--format=csv,noheader,nounits",
            ])
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                keys = ["gpu_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib",
                        "gpu_temp_c", "gpu_power_w", "sm_clock_mhz", "mem_clock_mhz"]
                for k, v in zip(keys, parts):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = None
        return row
    
    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._sample_once())
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

        csv_path = self.out_dir / "resource_samples.csv"
        if self.samples:
            keys = sorted({k for row in self.samples for k in row.keys()})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.samples)  

        powers = [s["gpu_power_w"] for s in self.samples if s.get("gpu_power_w") is not None]
        gpu_mem = [s["gpu_mem_used_mib"] for s in self.samples if s.get("gpu_mem_used_mib") is not None]
        summary = {
            "samples_csv": str(csv_path) if self.samples else None,
            "num_samples": len(self.samples),
            "gpu_power_avg_w": round(statistics.mean(powers), 2) if powers else None,
            "gpu_power_peak_w": round(max(powers), 2) if powers else None,
            "gpu_mem_used_avg_mib": round(statistics.mean(gpu_mem), 1) if gpu_mem else None,
            "gpu_mem_used_peak_mib": round(max(gpu_mem), 1) if gpu_mem else None,
        }
        if powers:
            duration_h = (len(powers) * self.interval_s) / 3600.0
            summary["energy_wh"] = round(statistics.mean(powers) * duration_h, 4)
        return summary

# --------------------------------------------------------------------------- #
# Model client — OpenAI-compatible streaming completions
# --------------------------------------------------------------------------- #

@dataclass
class GenerationResult:
    prompt_tokens: int
    ttft_s: float                 # time to first token
    total_time_s: float
    output_text: str
    output_tokens: int
    per_token_times: list[float] = field(default_factory=list)  # inter-token gaps  

class ModelClient:
    def __init__(self, base_url: str, model_name: str, api_key: Optional[str] = None,
                 chat: bool = True, extra_headers: Optional[dict] = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat = chat
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            self.headers.update(extra_headers)

    def wait_until_ready(self, timeout_s: int = 600, poll_s: float = 3.0) -> None:
        health_url = f"{self.base_url}/v1/models"
        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(health_url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(poll_s)
        raise TimeoutError(f"Model endpoint not ready after {timeout_s}s (last error: {last_err})")   

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        if self.chat:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
        else:
            url = f"{self.base_url}/v1/completions"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }

        start = time.time()
        first_token_time = None
        last_event_time = start
        per_token_gaps = []
        chunks: list[str] = []

        with requests.post(url, headers=self.headers, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta_text = None
                choice = (obj.get("choices") or [{}])[0]
                if self.chat:
                    delta_text = (choice.get("delta") or {}).get("content")
                else:
                    delta_text = choice.get("text")

                if delta_text:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        per_token_gaps.append(now - last_event_time)
                    last_event_time = now
                    chunks.append(delta_text)                                    

        end = time.time()
        output_text = "".join(chunks)
        return GenerationResult(
            prompt_tokens=count_tokens(prompt),
            ttft_s=(first_token_time - start) if first_token_time else float("nan"),
            total_time_s=end - start,
            output_text=output_text,
            output_tokens=count_tokens(output_text),
            per_token_times=per_token_gaps,
        )

# --------------------------------------------------------------------------- #
# Phase 3/4 — Latency + prefill throughput sweep
# --------------------------------------------------------------------------- #

def run_latency_sweep(client: ModelClient, prompt_lengths: list[int],
                       repeats: int, decode_tokens_for_ttft: int = 8) -> dict[str, Any]:
    """For each prompt length, run `repeats` generations with a short output
    (so we're really measuring prefill/TTFT, not decode) and report TTFT and
    prefill-throughput statistics."""
    results = {}
    for plen in prompt_lengths:
        prompt = build_prompt_of_length(plen)
        ttfts = []
        prefill_tps = []
        for _ in range(repeats):
            gen = client.generate(prompt, max_tokens=decode_tokens_for_ttft, temperature=0.0)
            if gen.ttft_s == gen.ttft_s:  # not NaN
                ttfts.append(gen.ttft_s)
                if gen.ttft_s > 0:
                    prefill_tps.append(gen.prompt_tokens / gen.ttft_s)
        results[str(plen)] = {
            "requested_prompt_tokens": plen,
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
# Phase 5 — Decode speed
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
        results[str(n_out)] = {
            "requested_output_tokens": n_out,
            "actual_output_tokens": gen.output_tokens,
            "decode_time_s": round(decode_time, 3),
            "tok_per_sec_avg": round(tok_per_sec_overall, 2) if tok_per_sec_overall else None,
            "tok_per_sec_peak": round(max(instant_rates), 2) if instant_rates else None,
            "tok_per_sec_min": round(min(instant_rates), 2) if instant_rates else None,
            "tok_per_sec_median": round(statistics.median(instant_rates), 2) if instant_rates else None,
        }
    return results


# --------------------------------------------------------------------------- #
# Phase 8 — Reasoning token count
# --------------------------------------------------------------------------- #

def analyze_reasoning_tokens(text: str) -> dict[str, Any]:
    think_open, think_close = "<think>", "</think>"
    if think_open in text and think_close in text:
        start = text.index(think_open) + len(think_open)
        end = text.index(think_close)
        thinking = text[start:end]
        answer = text[end + len(think_close):]
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
        stats = analyze_reasoning_tokens(gen.output_text)
        stats["prompt_preview"] = p[:80]
        per_prompt.append(stats)

    think_lens = [r["thinking_tokens"] for r in per_prompt]
    return {
        "per_prompt": per_prompt,
        "thinking_tokens_avg": round(statistics.mean(think_lens), 1) if think_lens else None,
        "thinking_tokens_max": max(think_lens) if think_lens else None,
        "thinking_tokens_median": statistics.median(think_lens) if think_lens else None,
    }


# --------------------------------------------------------------------------- #
# Plugin hook for future phases (code correctness, Deep-SWE, HLE, ...)
# --------------------------------------------------------------------------- #

_REGISTERED_BENCHMARKS: dict[str, Callable[[ModelClient, dict], dict]] = {}


def register_benchmark(name: str):
    """Decorator: register an additional benchmark plugin.

    def my_bench(client: ModelClient, config: dict) -> dict: ...
    register_benchmark("humaneval")(my_bench)
    """
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
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite
    return run_dir


def save_json(path: Path, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 core benchmark runner")
    parser.add_argument("--model", required=True, help="Path to model config YAML")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    args = parser.parse_args()

    cfg = load_model_config(Path(args.model))
    model_name = cfg["name"]
    run_dir = make_run_dir(model_name)
    print(f"[core_runner] writing results to {run_dir}")

    # Phase 2 — environment
    env = collect_environment()
    save_json(run_dir / "environment.json", env)
    print("[core_runner] environment captured")

    # Connect + wait ready
    client = ModelClient(
        base_url=cfg["endpoint"]["base_url"],
        model_name=cfg["endpoint"].get("model_name", model_name),
        api_key=cfg["endpoint"].get("api_key"),
        chat=cfg["endpoint"].get("chat", True),
    )
    print("[core_runner] waiting for model endpoint to be ready...")
    client.wait_until_ready(timeout_s=cfg.get("ready_timeout_s", 600))
    print("[core_runner] endpoint ready")

    # Phase 6/7 — start resource monitor before any load-generating work
    monitor = ResourceMonitor(run_dir, interval_s=cfg.get("monitor_interval_s", 1.0))
    monitor.start()

    summary: dict[str, Any] = {"model": model_name, "run_dir": str(run_dir)}

    try:
        if not args.skip_latency:
            prompt_lengths = cfg.get("prompt_lengths", [32, 128, 512, 2048, 8192, 16384])
            repeats = cfg.get("latency_repeats", 10)  # doc suggests 100; default lower for sane runtime
            print(f"[core_runner] running latency sweep over {prompt_lengths} ({repeats} reps each)")
            latency_results = run_latency_sweep(client, prompt_lengths, repeats)
            save_json(run_dir / "latency.json", latency_results)
            summary["latency"] = latency_results

        if not args.skip_decode:
            output_lengths = cfg.get("decode_lengths", [512, 1024, 2048])
            print(f"[core_runner] running decode speed benchmark over {output_lengths}")
            decode_results = run_decode_speed(client, output_lengths)
            save_json(run_dir / "decode.json", decode_results)
            summary["decode"] = decode_results

        if not args.skip_reasoning:
            reasoning_prompts = cfg.get("reasoning_prompts", [
                "Solve: if a train travels 60 miles in 45 minutes, what is its speed in mph? Show your reasoning.",
                "A farmer has 17 sheep, all but 9 die. How many are left? Explain your reasoning step by step.",
            ])
            print(f"[core_runner] running reasoning-token benchmark ({len(reasoning_prompts)} prompts)")
            reasoning_results = run_reasoning_benchmark(client, reasoning_prompts)
            save_json(run_dir / "reasoning.json", reasoning_results)
            summary["reasoning"] = reasoning_results

        for name, fn in _REGISTERED_BENCHMARKS.items():
            print(f"[core_runner] running registered benchmark: {name}")
            plugin_results = fn(client, cfg)
            save_json(run_dir / f"{name}.json", plugin_results)
            summary[name] = plugin_results

    finally:
        print("[core_runner] stopping resource monitor")
        resource_summary = monitor.stop()
        summary["resources"] = resource_summary

        total_output_tokens = 0
        if "decode" in summary:
            total_output_tokens += sum(
                v["actual_output_tokens"] for v in summary["decode"].values()
                if v.get("actual_output_tokens")
            )
        if total_output_tokens and resource_summary.get("energy_wh") is not None:
            summary["energy_per_token_wh"] = round(
                resource_summary["energy_wh"] / total_output_tokens, 6
            )

    save_json(run_dir / "summary.json", summary)
    print(f"[core_runner] done. Summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()    