"""The sandbox step: the model works as an agent on coding tasks, and each command runs in a locked container.

agent-eval (``dsv4-specialize/agent-eval``) owns the agent loop, the tools, the containers and the checks. This
module runs agent-eval at a pinned commit against the model under test, with the dev set and the optional held-out
set, and reads its JSON report. It also reads the prefix-cache counters of the server before and after the run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "1.0"
DEFAULTS: dict[str, Any] = {
    "agent_eval_dir": "~/Projects/dsv4-specialize/agent-eval",
    "agent_eval_commit": None,
    "heldout_dir": None,
    "heldout_commit": None,
    "families": ["C", "D"],
    "tasks": None,   # a list of task ids instead of the families, for a short check
    "seeds": 3,
    "agents": 1,
    "slots": 2,
    "container_memory": "1g",
    "max_tokens": 8192,
    "time_limit_s": 1200,
    "stop_new_after": None,
}
RUN_LINE = re.compile(r": (pass|wrong|hallucination|stopped|server_error) \(")


class SandboxError(RuntimeError):
    """The step refuses to run: a checkout, a commit or the prefix cache is not as the config needs."""


def checkout(path: Path, pinned: str | None, what: str) -> str:
    """The commit of a clean git checkout at the pinned commit. Raises SandboxError otherwise."""
    if not pinned:
        raise SandboxError(f"sandbox.{what}_commit is not set")

    def git(*args: str) -> str:
        p = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        if p.returncode:
            raise SandboxError(f"{what} at {path}: git {' '.join(args)}: {p.stderr.strip()}")
        return p.stdout.strip()

    head, want = git("rev-parse", "HEAD"), git("rev-parse", "--verify", f"{pinned}^{{commit}}")
    if head != want:
        raise SandboxError(f"{what} at {path} is at {head[:12]}, not at the pinned commit {want[:12]}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SandboxError(f"{what} at {path} has local changes")
    return head


def server_metrics(base_url: str) -> dict[str, Any] | None:
    """The cache config and the prefix-cache counters from /metrics of a vLLM server, or None."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return parse_metrics(r.read().decode())
    except (OSError, ValueError):
        return None


def parse_metrics(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"enable_prefix_caching": None, "block_size": None, "hits": 0.0, "queries": 0.0}
    for line in text.splitlines():
        m = re.match(r"([\w:]+)(?:\{(.*)\})?\s+(\S+)$", line)
        if not m:
            continue
        name, labels, value = m[1], dict(re.findall(r'(\w+)="([^"]*)"', m[2] or "")), m[3]
        if name == "vllm:cache_config_info":
            out["enable_prefix_caching"] = {"True": True, "False": False}.get(labels.get("enable_prefix_caching", ""))
            out["block_size"] = int(labels["block_size"]) if labels.get("block_size", "").isdigit() else None
        elif name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
            key = "hits" if "hits" in name else "queries"
            out[key] += float(value)
    return out


def until_time(hhmm: str, now: datetime) -> str:
    """The next local time HH:MM after now, in the format of agent_eval.py --until."""
    h, m = map(int, hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (t if t > now else t + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")


def task_count(sets: dict[str, Path], families: list[str]) -> int:
    return sum(json.loads(f.read_text())["family"] in families for d in sets.values() for f in d.glob("*/task.json"))


def job(*args: str) -> None:
    """Report to the machine-dash jobs page; nothing happens outside a `job run`."""
    if os.environ.get("MACHINE_DASH_JOB") and shutil.which("job"):
        subprocess.run(["job", *args], capture_output=True)


def run_sandbox(sandbox_cfg: dict[str, Any], *, base_url: str, served_name: str, label: str, run_dir: Path,
                command_flag: bool | None = None, allow_no_prefix_cache: bool = False) -> dict[str, Any]:
    sc = DEFAULTS | sandbox_cfg
    ae_dir = Path(sc["agent_eval_dir"]).expanduser()
    sets = {"dev": ae_dir / "tasks"}
    commits = {"agent_eval": checkout(ae_dir, sc["agent_eval_commit"], "agent_eval")}
    if sc["heldout_dir"]:
        held = Path(sc["heldout_dir"]).expanduser()
        commits["heldout"] = checkout(held, sc["heldout_commit"], "heldout")
        sets["heldout"] = held / "tasks"

    before = server_metrics(base_url)
    cache_on = before["enable_prefix_caching"] if before and before["enable_prefix_caching"] is not None \
        else command_flag
    if cache_on is False and not allow_no_prefix_cache:
        raise SandboxError("the server has prefix caching off; an agent loop sends its full history on each turn "
                           "(--sandbox-allow-no-prefix-cache runs anyway)")

    out = run_dir / "sandbox"
    env = os.environ | {"AGENT_EVAL_OUT": str(out), "AGENT_EVAL_SLOTS": str(sc["slots"]),
                        "AGENT_EVAL_MEMORY": str(sc["container_memory"]),
                        "AGENT_EVAL_TASKS": ":".join(f"{k}={v}" for k, v in sets.items())}
    argv = [sys.executable, "-u", "agent_eval.py", "point", "--name", "point",
            "--endpoint", label, base_url, served_name, "--max-tokens", str(sc["max_tokens"]),
            "--time-limit", str(sc["time_limit_s"]), "--mix", f"{label}={sc['agents']}",
            "--seeds", str(sc["seeds"]),
            *(["--tasks", ",".join(sc["tasks"])] if sc["tasks"] else ["--families", ",".join(sc["families"])])]
    if sc["stop_new_after"]:
        argv += ["--until", until_time(str(sc["stop_new_after"]), datetime.now())]
    tasks = len(sc["tasks"]) if sc["tasks"] else task_count(sets, sc["families"])
    job("set", "--steps", str(tasks * int(sc["seeds"])))

    start, done = time.time(), 0
    print(f"[sandbox] {' '.join(argv[2:])}", flush=True)
    with subprocess.Popen(argv, cwd=ae_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(f"[agent-eval] {line}", end="", flush=True)
            if RUN_LINE.search(line):
                done += 1
                job("step", str(done))
    end, after = time.time(), server_metrics(base_url)

    block = (after or before or {}).get("block_size")
    rep = subprocess.run([sys.executable, "agent_report.py", "point", "--json",
                          *(["--block-size", str(block)] if block else [])],
                         cwd=ae_dir, env=env, capture_output=True, text=True)
    if rep.returncode:
        raise RuntimeError(f"agent_report.py failed: {rep.stderr.strip()[-500:]}")
    report = json.loads(rep.stdout)

    cache: dict[str, Any] = {"enabled": cache_on, "block_size": block}
    if before and after:
        hits, queries = after["hits"] - before["hits"], after["queries"] - before["queries"]
        cache |= {"hits": hits, "queries": queries, "hit_rate": round(100 * hits / queries, 1) if queries else None}
    cache |= {k: v for k, v in report.get("cache", {}).get(label, {}).items() if k.startswith("best")}
    return {"benchmark_version": BENCHMARK_VERSION, "start_time": start, "end_time": end, "exit_code": p.returncode,
            "config": {**sc, "commits": commits, "sets": {k: str(v) for k, v in sets.items()}, "label": label,
                       "served_name": served_name},
            "prefix_cache": cache, "point_dir": str(out / "point"), "report": report}


def summary_entry(result: dict[str, Any]) -> dict[str, Any]:
    """The rates of each task set (all families) and the cache hit rate, for summary.json."""
    sets = {g["set"]: {"runs": g["runs"], "tasks": g["tasks"], "pass": g["pass"], "hallucination": g["hallucination"]}
            for g in result["report"]["groups"] if g["family"] == "all"}
    status = "completed" if result["exit_code"] == 0 else "failed"
    return {"status": status, "exit_code": result["exit_code"], "sets": sets,
            "prefix_cache_hit_rate": result["prefix_cache"].get("hit_rate"),
            "prefix_cache_best_rate": result["prefix_cache"].get("best_rate")}
