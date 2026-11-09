#!/usr/bin/env python3
"""
batch_run.py — Run core_runner.py on every YAML in models/ and report
which ones pass, fail, or need attention.

Usage:
    python batch_run.py                  # run all models, stream output to console
    python batch_run.py --model name.yml # run only one model
    python batch_run.py --dry-run        # show what would run without executing

The script runs the models one at a time. It reads only models/*.yml and
models/*.yaml, not the subfolders of models/.

Output:
    - Live progress lines and the core_runner output to the terminal
    - The core_runner output of each model in <output-dir>/logs/<name>.log
    - A machine-parseable summary.json in --output-dir (default: current directory)
    - A human-readable SUMMARY.md in --output-dir with per-model pass/fail and errors
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# core_runner.py writes its run directories here by default.
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CORE_RUNNER = ROOT / "core_runner.py"

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

# core_runner.py prints this line right after it makes the run directory.
RUN_DIR_PREFIX = "[core_runner] writing results to "
MODEL_TIMEOUT_S = 3600 * 24  # generous per-model cap


def list_run_dirs(name: str) -> set[Path]:
    """Return the run directories that exist now for one model name."""
    model_dir = RESULTS_DIR / name
    if not model_dir.is_dir():
        return set()
    return {p.resolve() for p in model_dir.iterdir() if p.is_dir()}


def stream_process(cmd: list[str], log_file: Any, timeout_s: float) -> tuple[int | None, list[str]]:
    """Run cmd and copy its output to the console and to log_file.

    Return (returncode, output lines). The return code is None when the
    process exceeds timeout_s and batch_run kills it.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, kill)
    timer.start()
    lines: list[str] = []
    try:
        for line in proc.stdout:
            lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        returncode = proc.wait()
    finally:
        timer.cancel()
    return (None if timed_out.is_set() else returncode), lines


def find_new_run_dir(name: str, before: set[Path], lines: list[str]) -> Path | None:
    """Find the run directory that this attempt made.

    Use the run-dir line that core_runner prints. If that line is missing,
    use the one directory that did not exist before the attempt.
    """
    for line in lines:
        if line.startswith(RUN_DIR_PREFIX):
            path = Path(line[len(RUN_DIR_PREFIX):].strip())
            path = (path if path.is_absolute() else ROOT / path).resolve()
            if path not in before:
                return path
    new_dirs = list_run_dirs(name) - before
    return new_dirs.pop() if len(new_dirs) == 1 else None


def run_attempt(model_yaml: Path, core_runner: Path, name: str, log_file: Any, timeout_s: float) -> dict[str, Any]:
    """Run core_runner.py once. Return the status, error, run_dir and summary."""
    before = list_run_dirs(name)
    cmd = [sys.executable, str(core_runner), "--model", str(model_yaml), "--results-dir", str(RESULTS_DIR)]
    try:
        returncode, lines = stream_process(cmd, log_file, timeout_s)
    except Exception as exc:
        return {"status": "error", "error": f"subprocess exception: {exc}"}
    if returncode is None:
        return {"status": "timeout", "error": f"process exceeded the {timeout_s:.0f} s timeout"}

    out: dict[str, Any] = {"status": "error"}
    summary = None
    run_dir = find_new_run_dir(name, before, lines)
    if run_dir is not None:
        out["run_dir"] = str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir)
        try:
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = None

    if isinstance(summary, dict):
        out["summary"] = summary
        if summary.get("status") == "completed" and returncode == 0:
            out.update(status="pass", error=None)
        else:
            out["status"] = "fail"
            out["error"] = summary.get("error") or f"status={summary.get('status')}, exit code {returncode}"
    elif returncode != 0:
        out["error"] = f"core_runner exit code {returncode}, no summary.json for this run"
    else:
        out["error"] = "no summary.json for this run"
    return out


def run_one_model(
    model_yaml: Path,
    core_runner: Path,
    *,
    retry: bool = True,
    max_retries: int = 1,
    log_dir: Path = Path("logs"),
) -> dict[str, Any]:
    """Execute core_runner.py for one model YAML and return a result dict.

    Retry on a failed status, a nonzero exit code, a timeout or a missing
    summary.json, at most max_retries times. Do not retry when retry is False.
    """
    name = model_name_from_yaml(model_yaml)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    result: dict[str, Any] = {
        "name": name,
        "yaml": str(model_yaml),
        "status": "unknown",
        "duration_s": None,
        "error": None,
        "run_dir": None,
        "summary": None,
        "log": str(log_path),
        "attempts": 0,
    }
    deadline = time.time() + MODEL_TIMEOUT_S

    with open(log_path, "w", encoding="utf-8") as log_file:
        for attempt in range(1 + (max(max_retries, 0) if retry else 0)):
            remaining = deadline - time.time()
            if remaining <= 0:
                result.update(status="timeout", error="exceeded per-model time budget (24 h)")
                break
            if attempt > 0:
                log(f"  {c('yellow', f'Retry {attempt} for {name}...')}")
                time.sleep(2)
            log_file.write(f"=== batch_run: attempt {attempt + 1} for {name} ===\n")
            start = time.time()
            # Clear the fields so that a result never shows an earlier attempt's run dir.
            result.update(run_dir=None, summary=None, error=None)
            result.update(run_attempt(model_yaml, core_runner, name, log_file, remaining))
            result["duration_s"] = round(time.time() - start, 1)
            result["attempts"] = attempt + 1
            if result["status"] == "pass":
                break

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
        if r.get("log"):
            lines.append(f"- **Log:** `{r['log']}`")
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run core_runner.py on all model YAMLs")
    parser.add_argument("--model", help="Run only this specific model YAML (relative to models/)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--no-retry", action="store_true", help="Do not retry failed runs")
    parser.add_argument("--max-retries", type=int, default=1, help="Number of retry attempts on failure")
    parser.add_argument("--output-dir", default=".", help="Directory for summary and log output (default: .)")
    args = parser.parse_args(argv)

    # Only the top level of models/: batch_run skips subfolders such as models/legacy/.
    models_dir = MODELS_DIR
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

    # Run the models one at a time. They share the GPU. core_runner stops each
    # managed server before the next model starts, so models can share a port.
    log(f"{c('bold', 'BATCH RUN')} — starting {len(model_files)} model(s) ...")
    log(f"{'─' * 60}")

    start_wall = time.time()
    results: list[dict[str, Any]] = []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, mf in enumerate(model_files, 1):
        name = model_name_from_yaml(mf)
        log("")
        log(f"{c('cyan', f'[{i}/{len(model_files)}])')} {c('bold', name)} — starting ...")

        r = run_one_model(
            mf,
            CORE_RUNNER,
            retry=not args.no_retry,
            max_retries=args.max_retries,
            log_dir=out / "logs",
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
            log(f"    {c('dim', 'log: ' + r['log'])}")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] in ("fail", "failed", "error", "timeout"))
    wall = time.time() - start_wall

    # Write summary artifacts

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
