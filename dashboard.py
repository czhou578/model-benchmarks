#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the runs under results/.

Usage:
    python3 dashboard.py                      # writes reports/dashboard.html
    python3 dashboard.py --out /tmp/d.html    # writes another path
    python3 dashboard.py --fragment           # page body only, for embedding
    python3 dashboard.py --json data.json     # also write the collected data

The script reads results/<model>/<YYYYMMDD_HHMMSS>/ and needs only the
standard library. It joins the CSS and JS modules in reports/dashboard/
into reports/dashboard/page.html and embeds the data, so the page opens
offline from the file system.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PAGE_DIR = ROOT / "reports" / "dashboard"
DATA_MARKER = "/*__BENCH_DATA__*/null"
CSS_MARKER = "/*__BENCH_CSS__*/"
JS_MARKER = "/*__BENCH_JS__*/"
# Joined in this order. The JS modules share one function scope; core.js
# defines start(), which runs after the last module.
CSS_MODULES = [
    "tokens.css",
    "layout.css",
    "frame.css",
    "runs.css",
    "heat.css",
]
JS_MODULES = [
    "core.js",
    "frame.js",
    "runs.js",
    "bars.js",
    "lines.js",
    "scores.js",
]

# PR #42 and PR #44 (merged 2026-11-07) fixed the cache-salt placement and
# the tool-calling scorer. Latency, deep-context, and tool-calling numbers
# from earlier runs are not comparable with later runs.
SCORING_FIX_DATE = datetime(2026, 11, 7)
MAX_MEMORY_POINTS = 500


def load_json(path: Path) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def option_value(command: list[str], flag: str) -> str | None:
    if flag in command:
        index = command.index(flag)
        if index + 1 < len(command):
            return command[index + 1]
    return None


def by_int_key(mapping: dict[str, Any] | None) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for key, value in (mapping or {}).items():
        try:
            rows.append((int(key), value if isinstance(value, dict) else {}))
        except (TypeError, ValueError):
            continue
    return sorted(rows)


def ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


def server_info(run_dir: Path) -> dict[str, Any]:
    server = load_json(run_dir / "resolved_server.json") or {}
    command = [str(arg) for arg in server.get("command") or []]
    served = command[2] if len(command) > 2 and command[1] == "serve" else None
    spec = option_value(command, "--speculative-config")
    return {
        "served_model": served,
        "speculative": spec is not None,
        "speculative_config": spec,
        "prefix_caching": "--enable-prefix-caching" in command,
        "gpu_memory_utilization": option_value(command, "--gpu-memory-utilization"),
        "kv_cache_dtype": option_value(command, "--kv-cache-dtype"),
        "max_model_len": option_value(command, "--max-model-len"),
        "tool_call_parser": option_value(command, "--tool-call-parser"),
        "command": " ".join(command) if command else None,
    }


def latency_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "x": length,
            "ttft_median_ms": ms(v.get("ttft_median_s")),
            "ttft_p95_ms": ms(v.get("ttft_p95_s")),
            "ttft_avg_ms": ms(v.get("ttft_avg_s")),
            "prefill_tps": v.get("prefill_tps_avg"),
            "n": v.get("n"),
        }
        for length, v in by_int_key(summary.get("latency"))
    ]


def deep_context_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "x": length,
            "ttft_median_ms": ms(v.get("ttft_median_s")),
            "ttft_p95_ms": ms(v.get("ttft_p95_s")),
            "prefill_tps": v.get("prefill_tps_avg"),
            "status": v.get("status"),
        }
        for length, v in by_int_key(summary.get("deep_context"))
    ]


def decode_rows(decode: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        {
            "x": length,
            "avg": v.get("tok_per_sec_avg"),
            "peak": v.get("tok_per_sec_peak"),
            "min": v.get("tok_per_sec_min"),
            "median": v.get("tok_per_sec_median"),
            "ttft_ms": ms(v.get("ttft_s")),
            "output_tokens": v.get("actual_output_tokens"),
        }
        for length, v in by_int_key(decode)
    ]


def concurrency_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    levels = (summary.get("concurrency") or {}).get("per_concurrency_level")
    return [
        {
            "x": level,
            "throughput": v.get("aggregate_throughput_tok_s"),
            "ttft_median_ms": ms((v.get("ttft") or {}).get("median_s")),
            "latency_median_s": (v.get("total_time_s") or {}).get("median_s"),
            "success_rate": v.get("success_rate"),
        }
        for level, v in by_int_key(levels)
    ]


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def ttft_breakdown_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    # vLLM reports TTFT and queue time per request, but no prefill time.
    # The client records post_queue_ms (TTFT minus queue) per request only,
    # so the median comes from the per-request rows.
    per_length = (summary.get("ttft_breakdown") or {}).get("per_length")
    rows = []
    for length, v in by_int_key(per_length):
        agg = v.get("aggregated") or {}
        requests = [r for r in v.get("per_request") or [] if r.get("success")]
        post_queue = median([r.get("post_queue_ms") for r in requests])
        rows.append({
            "x": length,
            "ttft_median_ms": (agg.get("ttft") or {}).get("median_ms"),
            "queue_median_ms": (agg.get("scheduler_delay") or {}).get("median_ms"),
            "post_queue_median_ms": None if post_queue is None else round(post_queue, 1),
        })
    return rows


def reasoning_info(summary: dict[str, Any]) -> dict[str, Any] | None:
    r = summary.get("reasoning")
    if not isinstance(r, dict):
        return None
    return {
        "thinking_avg": r.get("thinking_tokens_avg"),
        "thinking_median": r.get("thinking_tokens_median"),
        "thinking_max": r.get("thinking_tokens_max"),
        "answer_avg": r.get("answer_tokens_avg"),
        "answer_median": r.get("answer_tokens_median"),
        "prompts": len(r.get("per_prompt") or []),
    }


def tool_calling_info(summary: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    tc = summary.get("tool_calling")
    if not isinstance(tc, dict):
        return None
    info = {
        "composite": tc.get("composite_score"),
        "components": tc.get("composite_components") or {},
        "total": tc.get("total_tasks"),
        "correct": tc.get("total_correct"),
        "failure_modes": tc.get("failure_modes"),
        "categories": None,
        # "lite" runs at most 8 single-tool tasks; its composite is not
        # comparable with "full". Runs without tool_calling.json are full.
        "task_set": "full",
    }
    detail = load_json(run_dir / "tool_calling.json")
    if not isinstance(detail, dict):
        return info
    config = detail.get("config")
    if isinstance(config, dict) and config.get("task_set"):
        info["task_set"] = config["task_set"]
    if isinstance(detail.get("category_scores"), dict):
        info["categories"] = {
            name: {"pass_rate": score.get("pass_rate"), "total": score.get("total")}
            for name, score in detail["category_scores"].items()
            if isinstance(score, dict)
        }
        info["failure_modes"] = info["failure_modes"] or detail.get("failure_modes")
    return info


def spec_info(summary: dict[str, Any]) -> list[dict[str, Any]] | None:
    comparison = summary.get("spec_comparison")
    if not isinstance(comparison, dict):
        return None
    enabled = dict(by_int_key(comparison.get("spec_enabled")))
    disabled = dict(by_int_key(comparison.get("spec_disabled")))
    rows = []
    for length in sorted(set(enabled) | set(disabled)):
        rows.append({
            "x": length,
            "spec": enabled.get(length, {}).get("tok_per_sec_avg"),
            "baseline": disabled.get(length, {}).get("tok_per_sec_avg"),
        })
    return rows


def memory_info(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "memory_samples.csv"
    if not path.exists():
        return None
    samples = []
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                samples.append((
                    float(row["elapsed_s"]),
                    float(row["mem_available_gib"]),
                    float(row["mem_total_gib"]),
                ))
    except (OSError, KeyError, ValueError):
        return None
    if not samples:
        return None
    step = max(1, len(samples) // MAX_MEMORY_POINTS)
    points = samples[::step]
    if points[-1] is not samples[-1]:
        points.append(samples[-1])
    guard = load_json(run_dir / "memory_guard.json") or {}
    return {
        "total_gib": samples[0][2],
        "min_available_gib": min(s[1] for s in samples),
        "floor_gib": guard.get("floor_gib"),
        "warn_gib": guard.get("warn_gib"),
        "stopped_for_memory": guard.get("stopped_for_memory"),
        "duration_min": round(samples[-1][0] / 60, 1),
        "points": [[round(t / 60, 2), avail] for t, avail, _ in points],
    }


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    summary = load_json(run_dir / "summary.json")
    if not isinstance(summary, dict):
        return None
    try:
        started = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    env = load_json(run_dir / "environment.json") or {}
    gpu = summary.get("gpu") or {}
    return {
        "id": f"{run_dir.parent.name}/{run_dir.name}",
        "model": summary.get("model") or run_dir.parent.name,
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
        "started": started.isoformat(timespec="minutes"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "pre_fix": started < SCORING_FIX_DATE,
        "env": {
            "gpu": env.get("gpu_name"),
            "vllm": env.get("vllm_version"),
            "torch": env.get("torch_version"),
            "driver": env.get("driver_version"),
        },
        "server": server_info(run_dir),
        "latency": latency_rows(summary),
        "deep_context": deep_context_rows(summary),
        "ttft_breakdown": ttft_breakdown_rows(summary),
        "decode": decode_rows(summary.get("decode")),
        "concurrency": concurrency_rows(summary),
        "reasoning": reasoning_info(summary),
        "tool_calling": tool_calling_info(summary, run_dir),
        "spec": spec_info(summary),
        "gpu": {
            "power_avg_w": gpu.get("gpu_power_avg_w"),
            "power_peak_w": gpu.get("gpu_power_peak_w"),
            "util_avg_pct": gpu.get("gpu_util_avg_pct"),
            "energy_wh": gpu.get("energy_wh"),
            "energy_per_token_wh": summary.get("energy_per_token_wh"),
        },
        "memory": memory_info(run_dir),
    }


def collect_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(results_dir.glob("*/*/summary.json")):
        run = collect_run(summary_path.parent)
        if run is not None:
            runs.append(run)
    # Oldest first, so a new run takes the next color and old runs keep theirs.
    runs.sort(key=lambda r: (r["started"], r["model"]))
    return runs


def host_memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return None


def build_payload(results_dir: Path) -> dict[str, Any]:
    runs = collect_runs(results_dir)
    gpu_names = [r["env"]["gpu"] for r in runs if r["env"].get("gpu")]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scoring_fix_date": SCORING_FIX_DATE.date().isoformat(),
        "host": {
            "gpu": gpu_names[-1] if gpu_names else None,
            "memory_gib": host_memory_gib(),
        },
        "runs": runs,
    }


def read_modules(names: list[str]) -> str:
    return "\n".join((PAGE_DIR / name).read_text(encoding="utf-8") for name in names)


def render(payload: dict[str, Any], fragment: bool) -> str:
    page = (PAGE_DIR / "page.html").read_text(encoding="utf-8")
    for marker in (CSS_MARKER, JS_MARKER, DATA_MARKER):
        if marker not in page:
            raise SystemExit(f"page.html has no {marker} marker")
    script = '(function () {\n  "use strict";\n' + read_modules(JS_MODULES) + "\n  start();\n})();"
    # "</" inside a <script> block would end it early.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    body = page.replace(CSS_MARKER, read_modules(CSS_MODULES)).replace(JS_MARKER, script)
    body = body.replace(DATA_MARKER, data)
    if fragment:
        return body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, "
        "viewport-fit=cover\">\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "dashboard.html"))
    parser.add_argument("--fragment", action="store_true",
                        help="Write the page body without the html/head/body skeleton")
    parser.add_argument("--json", help="Also write the collected data to this path")
    args = parser.parse_args()

    payload = build_payload(Path(args.results))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload, args.fragment), encoding="utf-8")
    print(f"dashboard: {len(payload['runs'])} runs -> {out}")


if __name__ == "__main__":
    main()
