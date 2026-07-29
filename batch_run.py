#!/usr/bin/env python3
"""
batch_run.py — Run core_runner.py on every YAML in models/ and report
which ones pass, fail, or need attention.

Usage:
    python batch_run.py                  # run all models, stream output to console
    python batch_run.py --model name.yml # run only one model
    python batch_run.py --jobs 2         # run up to 2 models in parallel
    python batch_run.py --dry-run        # show what would run without executing

Output:
    - Live progress lines to the terminal (coloured when supported)
    - A machine-parseable summary.json written alongside this script
    - A human-readable SUMMARY.md with per-model pass/fail, errors, and stats
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Colours (auto-disable when stdout is not a TTY)
# --------------------------------------------------------------------------- #

USE_COLOR = sys.stdout.isatty()

C = {
    "reset": "\033[0m",
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}


def c(color: str, text: str) -> str:
    return f"{C[color]}{text}{C['reset']}" if USE_COLOR else text


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    """Print a timestamped, pipe-safe line to stdout."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def model_name_from_yaml(path: Path) -> str:
    """Read the `name` field from a model YAML. Fall back to filename stem."""
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data.get("name", path.stem)
    except Exception:
        pass
    return path.stem


def find_model_port(cfg: dict) -> int | None:
    """Find the port the model's server would bind to (managed mode)."""
    server = cfg.get("server", {})
    if isinstance(server, dict):
        for i, arg in enumerate(server.get("command", [])):
            if arg == "--port" and i + 1 < len(server["command"]):
                try:
                    return int(server["command"][i + 1])
                except (ValueError, IndexError):
                    pass
    return None


def find_model_mode(cfg: dict) -> str:
    server = cfg.get("server")
    if isinstance(server, dict) and server:
        return server.get("mode", "managed")
    return "external"


# --------------------------------------------------------------------------- #
# Single model run
# --------------------------------------------------------------------------- #


def run_one_model(
    model_yaml: Path,
    core_runner: Path,
    *,
    retry: bool = True,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Execute core_runner.py for one model YAML and return a result dict."""
    name = model_name_from_yaml(model_yaml)
    result = {
        "name": name,
        "yaml": str(model_yaml),
        "status": "unknown",
        "duration_s": None,
        "error": None,
        "run_dir": None,
        "summary": None,
    }

    deadline = time.time() + 3600 * 24  # generous per-model cap

    for attempt in range(1 + (max_retries if retry else 0)):
        if time.time() > deadline:
            result["status"] = "timeout"
            result["error"] = "exceeded per-model time budget (24 h)"
            return result

        if attempt > 0:
            log(f"  {c('yellow', f'Retry {attempt} for {name}...')}")
            time.sleep(2)

        attempt_start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(core_runner), "--model", str(model_yaml)],
                capture_output=True,
                text=True,
                timeout=3600 * 24,
            )
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = "process exceeded 24 h timeout"
            continue
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"subprocess exception: {exc}"
            continue

        result["duration_s"] = round(time.time() - attempt_start, 1)

        # core_runner writes summary.json to its run directory.
        # Try to find it by scanning the results tree.
        results_dir = Path("results")
        if results_dir.exists():
            for sd in sorted(results_dir.glob(f"{name}/*/summary.json"), reverse=True):
                try:
                    with open(sd) as f:
                        summary = json.load(f)
                    result["run_dir"] = str(sd.parent)
                    result["summary"] = summary
                    result["status"] = "completed" if summary.get("status") == "completed" else "failed"
                    if result["status"] == "failed" and attempt < max_retries:
                        result["error"] = summary.get("error", "status=failed")
                        continue  # retry
                    break
                except Exception:
                    pass

        if result["status"] in ("completed", "failed") or not retry:
            if not result["error"] and result["status"] == "completed":
                result["status"] = "pass"
            elif not result["error"]:
                result["status"] = "fail"
            break

        # If we get here: status was unknown → retry
        result["error"] = result.get("error") or "no summary.json found"

    if result.get("error") and result["status"] == "unknown":
        result["status"] = "error"

    return result


# --------------------------------------------------------------------------- #
# Summary generation
# --------------------------------------------------------------------------- #


def generate_summary_md(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write a human-readable SUMMARY.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total = len(results)
    pass_count = sum(1 for r in results if r["status"] == "pass")
    fail_count = sum(1 for r in results if r["status"] in ("fail", "failed"))
    error_count = sum(1 for r in results if r["status"] not in ("pass", "fail", "failed", "unknown"))

    lines = [
        "# Batch Benchmark Run Summary",
        "",
        f"**Generated:** {now}",
        f"**Models:** {total}  |  **Pass:** {pass_count}  |  **Fail:** {fail_count}  |  **Error:** {error_count}",
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
    ]

    for r in sorted(results, key=lambda x: (0 if x["status"] == "pass" else 1, x["name"])):
        status_icon = {"pass": "✅", "fail": "❌", "failed": "❌", "error": "⏰", "timeout": "⏰"}.get(
            r["status"], "?"
        )
        lines.append(f"### {status_icon} {r['name']} (`{r['yaml']}`) — {c_short(r['status'])}")
        lines.append("")
        if r.get("error"):
            lines.append(f"- **Error:** {r['error']}")
        if r.get("duration_s"):
            lines.append(f"- **Duration:** {r['duration_s']:.1f}s")
        if r.get("run_dir"):
            lines.append(f"- **Run dir:** `{r['run_dir']}`")
        if r.get("summary"):
            s = r["summary"]
            lines.append(f"- **Server mode:** {s.get('server_mode', 'N/A')}")
            if s.get("latency"):
                lat = s["latency"]
                if isinstance(lat, dict):
                    lines.append(f"  - Latency: {len(lat)} lengths tested")
            if s.get("tool_calling"):
                tc = s["tool_calling"]
                if isinstance(tc, dict):
                    score = tc.get("composite_score")
                    total_t = tc.get("total_tasks")
                    correct = tc.get("total_correct")
                    if score is not None:
                        lines.append(
                            f"  - Tool-calling: score={score}  {correct}/{total_t} correct"
                        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Quick Stats")
    lines.append("")
    for r in results:
        if r.get("summary") and r["status"] == "pass":
            name = r["name"]
            s = r["summary"]
            parts = []
            if s.get("tool_calling"):
                tc = s["tool_calling"]
                if isinstance(tc, dict) and tc.get("composite_score") is not None:
                    parts.append(f"tool_calling={tc['composite_score']}")
            if parts:
                lines.append(f"- **{name}:** {', '.join(parts)}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def c_short(status: str) -> str:
    return {"pass": "PASS", "fail": "FAIL", "failed": "FAIL", "error": "ERROR", "timeout": "TIMEOUT", "unknown": "UNKNOWN"}.get(
        status, status.upper()
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Run core_runner.py on all model YAMLs")
    parser.add_argument("--model", help="Run only this specific model YAML (relative to models/)")
    parser.add_argument("--jobs", type=int, default=1, help="Max parallel runs (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--no-retry", action="store_true", help="Do not retry failed runs")
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retry attempts on failure")
    parser.add_argument("--output-dir", default=".", help="Directory for summary output (default: .)")
    args = parser.parse_args()

    models_dir = Path("models")
    if not models_dir.is_dir():
        log(f"{c('red', 'ERROR')}: models/ directory not found")
        sys.exit(1)

    # Discover model YAMLs
    if args.model:
        yaml_path = models_dir / args.model
        if not yaml_path.exists():
            log(f"{c('red', 'ERROR')}: {yaml_path} not found")
            sys.exit(1)
        model_files = [yaml_path]
    else:
        model_files = sorted(models_dir.glob("*.yml")) + sorted(models_dir.glob("*.yaml"))
        if not model_files:
            log(f"{c('yellow', 'WARNING')}: no .yml/.yaml files found in models/")
            return

    # Dry run: show what would be executed
    if args.dry_run:
        log(f"{c('bold', 'DRY RUN')} — {len(model_files)} model(s) to benchmark")
        log("")
        for mf in model_files:
            name = model_name_from_yaml(mf)
            port = None
            mode = "external"
            try:
                import yaml

                with open(mf) as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    port = find_model_port(cfg)
                    mode = find_model_mode(cfg)
            except Exception:
                pass
            mode_str = "managed" if mode == "managed" else "external"
            port_str = f" :{port}" if port else ""
            log(f"  {name}{port_str} [{mode_str}]")
        return

    # Port conflict warning
    try:
        import yaml

        ports: dict[int, str] = {}
        for mf in model_files:
            with open(mf) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                p = find_model_port(cfg)
                if p is not None:
                    if p in ports:
                        log(
                            f"{c('yellow', 'WARNING')}: port {p} is used by "
                            f"{ports[p]} and {model_name_from_yaml(mf)} — "
                            f"managed servers will conflict!"
                        )
                    ports[p] = model_name_from_yaml(mf)
    except Exception:
        pass

    # Run models sequentially (parallel is tricky because they all share
    # the same GPU — running them one at a time is the safe default).
    log(f"{c('bold', 'BATCH RUN')} — starting {len(model_files)} model(s) ...")
    log(f"{'─' * 60}")

    start_wall = time.time()
    results: list[dict[str, Any]] = []
    core_runner = Path("core_runner.py")

    for i, mf in enumerate(model_files, 1):
        name = model_name_from_yaml(mf)
        log("")
        log(f"{c('cyan', f'[{i}/{len(model_files)}])')} {c('bold', name)} — starting ...")

        r = run_one_model(
            mf,
            core_runner,
            retry=not args.no_retry,
            max_retries=args.max_retries,
        )
        results.append(r)

        status_icon = {"pass": "PASS", "fail": "FAIL", "failed": "FAIL", "error": "ERROR"}.get(
            r["status"], r["status"].upper()
        )
        color = {"pass": "green", "fail": "red", "failed": "red", "error": "red"}.get(r["status"], "yellow")
        log(
            f"  {c(color, status_icon)} {r['status']} — "
            f"{'{:.1f}s'.format(r['duration_s']) if r['duration_s'] else 'N/A'}"
        )
        if r.get("error"):
            err_short = r["error"][:120]
            log(f"    {c('dim', err_short)}")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] in ("fail", "failed", "error", "timeout"))
    wall = time.time() - start_wall

    # Write summary artifacts
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # summary.json — machine-parseable
    summary_json = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(wall, 1),
        "total": total,
        "passed": passed,
        "failed": failed,
        "models": results,
    }
    (out / "summary.json").write_text(json.dumps(summary_json, indent=2, default=str) + "\n", encoding="utf-8")

    # SUMMARY.md — human-readable
    generate_summary_md(results, out / "SUMMARY.md")

    # Final banner
    log("")
    log(f"{'=' * 60}")
    log(f"{c('bold', 'BATCH COMPLETE')} — {passed}/{total} passed ({failed} failed) in {wall:.1f}s")
    log(f"{'=' * 60}")
    log("")
    log(f"  summary.json -> {out / 'summary.json'}")
    log(f"  SUMMARY.md   -> {out / 'SUMMARY.md'}")
    log("")

    # Per-model detail links (in SUMMARY.md)
    for r in sorted(results, key=lambda x: x["name"]):
        icon = {"pass": "PASS", "fail": "FAIL", "failed": "FAIL", "error": "ERROR"}.get(
            r["status"], r["status"].upper()
        )
        color = {"pass": "green", "fail": "red", "failed": "red", "error": "red"}.get(r["status"], "yellow")
        log(f"  {c(color, icon)} {r['name']} — {r['status']}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
