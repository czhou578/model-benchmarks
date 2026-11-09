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
  - GPU memory + power sampled at 1 Hz -> avg/peak + energy (Wh) + decode
    energy per output token
  - reasoning token count (<think>...</think> vs answer) for Qwen3-style
    reasoning-parser output

Not implemented here (plug in later via register_benchmark()):
  code correctness (HumanEval/MBPP), Deep-SWE, HLE.

Usage:
    python core_runner.py --model models/qwen3.8_27b_nvidia_nvfp4.yml

Every run gets its own timestamped directory under results/<model>/<ts>/.
The results/ directory sits next to this script. --results-dir changes it.

Dependencies:
    pip install -r requirements.txt    # requests, PyYAML, tiktoken
"""

from __future__ import annotations

import argparse
import json
import signal
import shutil
import statistics
import time
import requests
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from benchmarks.latency import run_deep_context, run_latency_sweep
from harness.client import ModelClient
from harness.environment import collect_environment
from harness.paths import ROOT, repo_relative
from harness.prompts import build_prompt_of_length, count_tokens
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


def main():
    parser = argparse.ArgumentParser(description="Phase 1 core benchmark runner (simplified)")
    parser.add_argument("--model", required=True, help="Path to model config YAML")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-reasoning", action="store_true")
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-ttft", action="store_true")
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
    decode_window: tuple[float, float] | None = None
    summary: dict[str, Any] = {
        "model": model_name,
        "run_dir": repo_relative(run_dir),
        "server_mode": server_mode,
        "status": "running",
    }

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, frame) -> None:
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

        if not args.skip_latency:
            prompt_lengths = cfg.get("prompt_lengths", [32, 128, 512, 2048, 8192, 16384])
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
            decode_start = time.time()
            decode_results = run_decode_speed(client, output_lengths)
            decode_window = (decode_start, time.time())
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

        # Architecture-based FLOP estimation
        print("[core_runner] running FLOP analysis")
        from benchmarks.architecture_flops import run_flops_analysis

        flop_lengths = cfg.get("flop_analysis_lengths", [512, 2048, 8192, 32768])
        # Fetch the model config from the running vLLM server
        server_config: dict[str, Any] | None = None
        try:
            server_config = _get_server_model_config(client)
        except Exception as exc:
            print(f"[core_runner] warning: could not fetch server config: {exc}")
        flops_result = run_flops_analysis(
            cfg, server_config=server_config,
            prefill_lengths=flop_lengths, context_lengths=flop_lengths,
        )
        save_json(run_dir / "flops_analysis.json", flops_result)
        summary["flops_analysis"] = {
            "status": flops_result.get("estimate_status"),
            "model": flops_result.get("architecture", {}).get("model", "unknown"),
        }

        # Tool-calling benchmark (Phase 1–3)
        try:
            from benchmarks.tool_calling import run_tool_calling_benchmark

            tc_tasks = cfg.get("tool_calling_tasks", "datasets/tool_calling_tasks.yaml")
            tc_task_set = cfg.get("tool_calling_task_set", "full")
            # A thinking model spends hundreds of tokens before its first
            # tool call, and vLLM parses tool calls only after </think>.
            # 256 tokens truncated most ticket tasks in the July 2026 runs.
            tc_max_tokens = cfg.get("tool_calling_max_tokens", 4096)
            tc_template_kwargs = cfg.get("tool_calling_chat_template_kwargs")
            # 4096 tokens at ~12 tok/s take ~330 s; 120 s cut answers off.
            tc_timeout_s = float(cfg.get("tool_calling_request_timeout_s", 600))
            print(f"[core_runner] running tool-calling benchmark (task_set={tc_task_set})")
            tc_result = run_tool_calling_benchmark(
                client,
                tasks_path=tc_tasks,
                task_set=tc_task_set,
                max_tokens=tc_max_tokens,
                chat_template_kwargs=tc_template_kwargs,
                request_timeout_s=tc_timeout_s,
            )
            save_json(run_dir / "tool_calling.json", tc_result)
            summary["tool_calling"] = {
                "status": "completed",
                "composite_score": tc_result.get("composite_score"),
                "composite_components": tc_result.get("composite_components"),
                "total_tasks": tc_result.get("total_tasks"),
                "total_correct": tc_result.get("total_correct"),
                "categories": list(tc_result.get("category_scores", {}).keys()),
                "failure_modes": tc_result.get("failure_modes"),
            }
            components = tc_result.get("composite_components", {})
            print(
                f"[core_runner] tool_calling: "
                f"score={tc_result['composite_score']} "
                f"{tc_result['total_correct']}/{tc_result['total_tasks']}  "
                f"components={components}"
            )
        except Exception as exc:
            # Record the failure in summary.json; continue with the run.
            summary["tool_calling"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[core_runner] warning: tool-calling benchmark failed: {exc}")

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
            if decode_window is not None and "decode" in summary:
                summary["decode_energy"] = decode_energy_summary(
                    monitor.samples, summary["decode"], *decode_window
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
