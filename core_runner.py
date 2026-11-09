#!/usr/bin/env python3
"""Run the benchmark suite against one model and write one run dir.

Usage:
    python core_runner.py --model models/qwen3.8_27b_nvidia_nvfp4.yml

A run does these steps. docs/methodology.md gives the details.

1. Load the model config. Make the run dir results/<name>/<timestamp>/.
   --results-dir changes the results/ folder.
2. Write environment.json.
3. In managed mode, start vLLM (vllm_server.py). In both modes, wait
   until the endpoint answers a real generation request. Add the vLLM
   version from GET /version to environment.json.
4. Start the GPU monitor (harness/telemetry.py).
5. Run the benchmark steps in the order of run_benchmarks().
6. Stop the monitor and the server. Write summary.json.

benchmarks/ holds one module per benchmark. harness/ holds the shared
client, prompts, statistics, telemetry, and environment code. This file
holds the command line, the run state (Run), and the step order.
"""

from __future__ import annotations

import argparse
import json
import signal
import shutil
import time
import requests
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.architecture_flops import run_flops_analysis
from benchmarks.concurrency import run_concurrency_test
from benchmarks.decode import compare_spec_decode_results, run_decode_speed
from benchmarks.latency import run_deep_context, run_latency_sweep
from benchmarks.prefill import run_prefill_scaling
from benchmarks.reasoning import run_reasoning_benchmark
from benchmarks.tool_calling import run_tool_calling_benchmark
from benchmarks.ttft_breakdown import run_ttft_breakdown
from harness.client import ModelClient
from harness.environment import collect_environment
from harness.paths import ROOT, repo_relative
from harness.provenance import fetch_server_version, harness_provenance
from harness.telemetry import GpuMonitor, decode_energy_summary
from vllm_server import VllmServer


def _get_server_model_config(client: ModelClient) -> dict[str, Any]:
    """Fetch the unmodified model config from the running vLLM server."""
    try:
        r = requests.get(f"{client.base_url}/v1/models", timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        model_data = data.get("data", [{}])[0] if data.get("data") else {}
        config = model_data.get("config", {})
        if isinstance(config, dict) and config:
            return config
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def load_model_config(path: Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"model config must be a YAML mapping: {path}")
    # Used to resolve an architecture_config_path relative to this YAML file.
    config.setdefault("_config_dir", str(path.resolve().parent))
    return config


def make_run_dir(model_name: str, results_dir: Path = ROOT / "results") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / model_name / ts
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
    resolved_command = command or server_cfg.get("command") or []
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


@dataclass
class Run:
    """The state of one run. The benchmark steps read it and add to it."""

    cfg: dict[str, Any]
    args: argparse.Namespace
    run_dir: Path
    summary: dict[str, Any]
    client: ModelClient | None = None
    monitor: GpuMonitor | None = None
    server: VllmServer | None = None
    # Start and stop time of the decode benchmark, for its energy summary.
    decode_window: tuple[float, float] | None = None

    def record(self, key: str, result: Any, summary_entry: Any = None) -> None:
        """Write ``<key>.json`` to the run dir and add *key* to the summary.

        Without *summary_entry*, the summary holds a full copy of *result*.
        """
        save_json(self.run_dir / f"{key}.json", result)
        self.summary[key] = result if summary_entry is None else summary_entry


# --------------------------------------------------------------------------- #
# Benchmark steps. Each step reads its keys from run.cfg, writes one JSON
# file to the run dir, and adds one entry to the summary.
# docs/model-config.md lists the keys and their defaults.
# --------------------------------------------------------------------------- #

DEFAULT_REASONING_PROMPTS = [
    "Solve: if a train travels 60 miles in 45 minutes, what is its speed in mph? Show your reasoning.",
    "A farmer has 17 sheep, all but 9 die. How many are left? Explain your reasoning step by step.",
]


def step_latency(run: Run) -> None:
    prompt_lengths = run.cfg.get("prompt_lengths", [32, 128, 512, 2048, 8192, 16384])
    repeats = run.cfg.get("latency_repeats", 10)
    print(f"[core_runner] latency sweep over {prompt_lengths} ({repeats} reps each)")
    run.record("latency", run_latency_sweep(run.client, prompt_lengths, repeats))


def step_deep_context(run: Run) -> None:
    context_lengths = run.cfg["context_lengths"]
    context_repeats = run.cfg.get("context_repeats", 5)
    output_tokens = int(run.cfg.get("deep_context_output_tokens", 64))
    print(f"[core_runner] deep context benchmark over {context_lengths} ({context_repeats} reps each)")
    results = run_deep_context(
        run.client, context_lengths, output_tokens=output_tokens, repeats=context_repeats
    )
    run.record("deep_context", results)


def step_ttft_breakdown(run: Run) -> None:
    ttft_lengths = run.cfg.get("ttft_prompt_lengths", [128, 512, 2048, 8192, 32768])
    ttft_reps = run.cfg.get("ttft_repetitions", 10)
    print(f"[core_runner] TTFT breakdown over {ttft_lengths} ({ttft_reps} reps each)")
    results = run_ttft_breakdown(
        run.client, prompt_lengths=ttft_lengths, repetitions=ttft_reps, gpu_monitor=run.monitor
    )
    run.record("ttft_breakdown", results)


def step_decode(run: Run) -> None:
    output_lengths = run.cfg.get("decode_lengths", [512, 1024, 2048])
    print(f"[core_runner] decode speed benchmark over {output_lengths}")
    decode_start = time.time()
    results = run_decode_speed(run.client, output_lengths)
    run.decode_window = (decode_start, time.time())
    run.record("decode", results)


def step_reasoning(run: Run) -> None:
    prompts = run.cfg.get("reasoning_prompts", DEFAULT_REASONING_PROMPTS)
    # 1024 tokens cut the Qwen3.8 palindrome prompt in every run (0 answer
    # tokens), and two prompts in each Qwen3.6 run.
    max_tokens = run.cfg.get("reasoning_max_tokens", 4096)
    print(f"[core_runner] reasoning-token benchmark ({len(prompts)} prompts, max_tokens={max_tokens})")
    run.record("reasoning", run_reasoning_benchmark(run.client, prompts, max_tokens=max_tokens))


def step_concurrency(run: Run) -> None:
    levels = run.cfg.get("concurrency_levels", [1, 2, 4, 8, 16])
    requests_per_level = run.cfg.get("concurrency_requests_per_level", 5)
    max_tokens = run.cfg.get("concurrency_max_tokens", 256)
    temperature = run.cfg.get("concurrency_temperature", 0.0)
    print(
        f"[core_runner] concurrency test: levels={levels} "
        f"({requests_per_level} reqs/level, {max_tokens} max_tokens, t={temperature})"
    )
    results = run_concurrency_test(
        run.client, levels, requests_per_level, max_tokens=max_tokens, temperature=temperature,
    )
    run.record("concurrency", results)


def step_prefill(run: Run) -> None:
    target_lengths = run.cfg.get("prefill_target_lengths", [512, 2048, 8192])
    repetitions = run.cfg.get("prefill_repetitions", 5)
    # Record a GPU idle baseline (default 5 s) before the prefill requests.
    gpu_interval = float(run.cfg.get("monitor_interval_s", 1.0))
    if gpu_interval <= 0:
        raise ValueError(f"monitor_interval_s must be above 0, got {gpu_interval}")
    idle_baseline_s = float(run.cfg.get("idle_baseline_s", 5.0))
    if gpu_interval > 0.2:
        print("[core_runner] capturing GPU idle baseline...")
    run.monitor.start_idle()
    for _ in range(int(idle_baseline_s / gpu_interval)):
        run.monitor.record_idle()
        time.sleep(gpu_interval)
    print("[core_runner] idle baseline captured, starting prefill benchmark")

    print(f"[core_runner] prefill scaling benchmark over {target_lengths} ({repetitions} reps)")
    results = run_prefill_scaling(run.client, target_lengths, repetitions=repetitions, gpu_monitor=run.monitor)
    run.record("prefill_scaling", results, summary_entry={
        k: {"status": v.get("status"), "n_success": v.get("n_success")}
        for k, v in results.get("per_length", {}).items()
    })


def step_flops(run: Run) -> None:
    print("[core_runner] running FLOP analysis")
    flop_lengths = run.cfg.get("flop_analysis_lengths", [512, 2048, 8192, 32768])
    server_config: dict[str, Any] | None = None
    try:
        server_config = _get_server_model_config(run.client)
    except Exception as exc:
        print(f"[core_runner] warning: could not fetch server config: {exc}")
    result = run_flops_analysis(
        run.cfg, server_config=server_config,
        prefill_lengths=flop_lengths, context_lengths=flop_lengths,
    )
    run.record("flops_analysis", result, summary_entry={
        "status": result.get("estimate_status"),
        "model": result.get("architecture", {}).get("model", "unknown"),
    })


def step_tool_calling(run: Run) -> None:
    """Run the tool-calling tasks. A failure goes to the summary, and the run continues."""
    cfg = run.cfg
    try:
        task_set = cfg.get("tool_calling_task_set", "full")
        # A thinking model spends hundreds of tokens before its first
        # tool call, and vLLM parses tool calls only after </think>.
        # 256 tokens truncated most ticket tasks in the July 2026 runs.
        # 4096 tokens at ~12 tok/s take ~330 s, so the timeout is 600 s.
        print(f"[core_runner] running tool-calling benchmark (task_set={task_set})")
        result = run_tool_calling_benchmark(
            run.client,
            tasks_path=cfg.get("tool_calling_tasks", "datasets/tool_calling_tasks.yaml"),
            task_set=task_set,
            max_tokens=cfg.get("tool_calling_max_tokens", 4096),
            chat_template_kwargs=cfg.get("tool_calling_chat_template_kwargs"),
            request_timeout_s=float(cfg.get("tool_calling_request_timeout_s", 600)),
            repeats=int(cfg.get("tool_calling_repeats", 1)),
        )
        summary_entry = {
            "status": "completed",
            "composite_score": result.get("composite_score"),
            "composite_components": result.get("composite_components"),
            "total_tasks": result.get("total_tasks"),
            "total_correct": result.get("total_correct"),
            "categories": list(result.get("category_scores", {}).keys()),
            "failure_modes": result.get("failure_modes"),
        }
        if "repeats" in result:
            summary_entry["repeats"] = result["repeats"]
        run.record("tool_calling", result, summary_entry=summary_entry)
        print(
            f"[core_runner] tool_calling: score={result['composite_score']} "
            f"{result['total_correct']}/{result['total_tasks']}  "
            f"components={result.get('composite_components', {})}"
        )
    except Exception as exc:
        run.summary["tool_calling"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[core_runner] warning: tool-calling benchmark failed: {exc}")


# vLLM 0.33.0 flags that turn on speculative decoding. Each takes one value,
# as the next argument or after "=". "--speculative-config.<key>" also counts.
SPEC_FLAGS = ("--speculative-config", "-sc", "--spec-method", "--spec-model", "--spec-tokens")


def strip_spec_flags(command: list[str]) -> tuple[list[str], list[str]]:
    """Return *command* without the speculative decoding flags, and the removed arguments."""
    kept: list[str] = []
    removed: list[str] = []
    i = 0
    while i < len(command):
        arg = command[i]
        name = arg.split("=", 1)[0]
        if name in SPEC_FLAGS or name.startswith("--speculative-config."):
            n = 1 if "=" in arg else 2
            removed.extend(command[i:i + n])
            i += n
        else:
            kept.append(arg)
            i += 1
    return kept, removed


def speculative_config_arg(value: Any) -> str:
    """Return the ``--speculative-config`` value: JSON for a mapping, a string as is."""
    return json.dumps(value) if isinstance(value, dict) else str(value)


def step_spec_comparison(run: Run) -> None:
    """Restart vLLM with and without speculative decoding, and compare decode speed.

    The runner removes the speculative decoding flags of ``server.command``
    from both variants. The enabled variant then adds ``speculative_config``.
    """
    if run.summary["server_mode"] != "managed" or run.server is None:
        raise ValueError("--compare-spec requires managed server mode")
    spec_config = run.cfg.get("speculative_config")
    if not spec_config:
        raise ValueError("--compare-spec requires speculative_config in the model YAML")

    output_lengths = run.cfg.get("decode_lengths", [512, 1024, 2048])
    base_command, removed_flags = strip_spec_flags(list(run.server.command))
    if removed_flags:
        print(f"[core_runner] warning: removed {removed_flags} from server.command for the spec comparison")
    run.server.stop()
    run.server.save_metadata(run.run_dir / "resolved_server.json")
    run.server = None

    def run_variant(label: str, command: list[str]) -> dict[str, Any]:
        variant_server = make_managed_server(
            run.cfg, run.run_dir, log_name=f"vllm_{label}.log", command=command
        )
        try:
            print(f"[core_runner] starting vLLM variant: {label}")
            variant_server.start()
            variant_client = make_client(run.cfg)
            wait_for_endpoint(variant_client, run.cfg, variant_server)
            return run_decode_speed(variant_client, output_lengths)
        finally:
            variant_server.stop()
            variant_server.save_metadata(run.run_dir / f"resolved_server_{label}.json")

    spec_results = run_variant(
        "spec_enabled", base_command + ["--speculative-config", speculative_config_arg(spec_config)]
    )
    save_json(run.run_dir / "spec_enabled.json", spec_results)
    non_spec_results = run_variant("spec_disabled", base_command)
    save_json(run.run_dir / "spec_disabled.json", non_spec_results)
    comparison = compare_spec_decode_results(spec_results, non_spec_results, output_lengths)
    comparison["config"]["removed_server_flags"] = removed_flags
    run.record("spec_comparison", comparison)


def run_benchmarks(run: Run) -> None:
    """Run the benchmark steps in their fixed order, without the skipped ones."""
    args, cfg = run.args, run.cfg
    if not args.skip_latency:
        step_latency(run)
    if not args.skip_deep_context and cfg.get("context_lengths"):
        step_deep_context(run)
    if not args.skip_ttft:
        step_ttft_breakdown(run)
    if not args.skip_decode:
        step_decode(run)
    if not args.skip_reasoning:
        step_reasoning(run)
    if not args.skip_concurrency:
        step_concurrency(run)
    if not args.skip_prefill:
        step_prefill(run)
    if not args.skip_flops:
        step_flops(run)
    if not args.skip_tool_calling:
        step_tool_calling(run)
    if args.compare_spec:
        step_spec_comparison(run)


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark suite against one model config.")
    parser.add_argument("--model", required=True, help="Path to model config YAML")
    parser.add_argument("--skip-latency", action="store_true", help="Skip the latency sweep")
    parser.add_argument("--skip-decode", action="store_true", help="Skip the decode speed benchmark")
    parser.add_argument("--skip-reasoning", action="store_true", help="Skip the reasoning benchmark")
    parser.add_argument("--skip-concurrency", action="store_true", help="Skip the concurrency benchmark")
    parser.add_argument("--skip-prefill", action="store_true", help="Skip the prefill scaling benchmark")
    parser.add_argument("--skip-ttft", action="store_true", help="Skip the TTFT breakdown")
    parser.add_argument("--skip-deep-context", action="store_true", help="Skip the deep context benchmark")
    parser.add_argument("--skip-flops", action="store_true", help="Skip the FLOP analysis")
    parser.add_argument("--skip-tool-calling", action="store_true", help="Skip the tool-calling benchmark")
    parser.add_argument("--compare-spec", action="store_true",
                        help="Run decode benchmark with spec-dec enabled and disabled, then compare")
    parser.add_argument(
        "--server-mode",
        choices=("managed", "external"),
        default=None,
        help="Override the server mode from the model config",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory that holds the run directories (default: results/ next to this script)",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model)
    cfg = load_model_config(model_config_path)
    model_name = cfg["name"]
    run_dir = make_run_dir(model_name, args.results_dir)
    print(f"[core_runner] writing results to {run_dir}")
    shutil.copy2(model_config_path, run_dir / "model_config.yml")

    env = collect_environment()
    # Set after readiness: the vLLM version that the server reports.
    env["vllm_server_version"] = None
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

    summary: dict[str, Any] = {
        "model": model_name,
        "run_dir": repo_relative(run_dir),
        "server_mode": server_mode,
        "status": "running",
        "harness": harness_provenance(),
    }
    if cfg.get("run_label"):
        summary["run_label"] = str(cfg["run_label"])
    run = Run(cfg=cfg, args=args, run_dir=run_dir, summary=summary)

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, frame) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        if server_mode == "managed":
            run.server = make_managed_server(cfg, run_dir)
            print(f"[core_runner] starting managed vLLM: {' '.join(run.server.command)}")
            run.server.start()
            run.server.save_metadata(run_dir / "resolved_server.json")

        run.client = make_client(cfg)
        print("[core_runner] waiting for model endpoint to be ready...")
        wait_for_endpoint(run.client, cfg, run.server)
        print("[core_runner] endpoint ready")
        env["vllm_server_version"] = fetch_server_version(
            run.client.base_url, getattr(run.client, "headers", None)
        )
        save_json(run_dir / "environment.json", env)
        print(f"[core_runner] server reports vllm={env['vllm_server_version']}")

        run.monitor = GpuMonitor(run_dir, interval_s=cfg.get("monitor_interval_s", 1.0))
        run.monitor.start()

        run_benchmarks(run)
        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if run.monitor is not None:
            print("[core_runner] stopping GPU monitor")
            summary["gpu"] = run.monitor.stop()
            if run.decode_window is not None and "decode" in summary:
                summary["decode_energy"] = decode_energy_summary(
                    run.monitor.samples, summary["decode"], *run.decode_window
                )

        if run.server is not None:
            print("[core_runner] stopping managed vLLM")
            run.server.stop()
            run.server.save_metadata(run_dir / "resolved_server.json")

        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        save_json(run_dir / "summary.json", summary)
        print(f"[core_runner] summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
