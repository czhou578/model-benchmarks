"""Tests of benchmarks/sandbox.py with a fake agent-eval checkout. No Docker, no GPU and no server."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime

import pytest

from benchmarks import sandbox

FAKE_EVAL = """import json, os, sys
os.makedirs(os.environ["AGENT_EVAL_OUT"], exist_ok=True)
with open(os.path.join(os.environ["AGENT_EVAL_OUT"], "call.json"), "w") as f:
    json.dump({"argv": sys.argv[1:], "tasks": os.environ["AGENT_EVAL_TASKS"]}, f)
print("21:00:00 point m C1 s0: pass (report, 3 turns, 1.0 s)")
print("a line that is not a run")
print("21:00:01 point m D1 s0: wrong (report, 5 turns, 2.0 s)")
"""
FAKE_REPORT = """import json, sys
group = {"set": "dev", "family": "all", "runs": 2, "tasks": 2, "pass": {"rate": 50.0, "lo": 0.0, "hi": 100.0},
         "hallucination": {"rate": 0.0, "lo": 0.0, "hi": 0.0}}
cache = {"m": {"block_size": 16, "prompt_tokens": 100, "best_hits": 64, "best_rate": 64.0}}
print(json.dumps({"argv": sys.argv[1:], "groups": [group, group | {"family": "C"}], "cache": cache}))
"""


def repo(path, files):
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def task(tid, family):
    return {f"tasks/{tid}/task.json": json.dumps({"id": tid, "family": family})}


@pytest.fixture
def env(tmp_path, monkeypatch):
    ae, held = tmp_path / "ae", tmp_path / "held"
    commit = repo(ae, {"agent_eval.py": FAKE_EVAL, "agent_report.py": FAKE_REPORT,
                       **task("C1", "C"), **task("D1", "D"), **task("L1", "L")})
    held_commit = repo(held, task("HC01", "C"))
    metrics = [{"enable_prefix_caching": True, "block_size": 16, "hits": 100.0, "queries": 200.0},
               {"enable_prefix_caching": True, "block_size": 16, "hits": 160.0, "queries": 300.0}]
    monkeypatch.setattr(sandbox, "server_metrics", lambda url: metrics.pop(0) if metrics else None)
    jobs: list[tuple[str, ...]] = []
    monkeypatch.setattr(sandbox, "job", lambda *a: jobs.append(a))
    cfg = {"agent_eval_dir": str(ae), "agent_eval_commit": commit, "heldout_dir": str(held),
           "heldout_commit": held_commit}
    return cfg, tmp_path, metrics, jobs


def run(cfg, tmp_path, **kw):
    return sandbox.run_sandbox(cfg, base_url="http://127.0.0.1:1", served_name="served", label="m",
                               run_dir=tmp_path, **kw)


def test_parse_metrics():
    text = ('# HELP vllm:cache_config_info x\nvllm:cache_config_info{block_size="2176",enable_prefix_caching="True"} 1.0\n'
            'vllm:prefix_cache_hits_total{engine="0"} 10.0\nvllm:prefix_cache_hits_total{engine="1"} 5.0\n'
            'vllm:prefix_cache_queries_total{engine="0"} 30.0\n')
    assert sandbox.parse_metrics(text) == {"enable_prefix_caching": True, "block_size": 2176, "hits": 15.0,
                                           "queries": 30.0}


def test_until_time():
    assert sandbox.until_time("04:30", datetime(2026, 11, 23, 21, 0)) == "2026-11-24T04:30"
    assert sandbox.until_time("04:30", datetime(2026, 11, 24, 3, 0)) == "2026-11-24T04:30"


def test_run_sandbox(env):
    cfg, tmp_path, _, jobs = env
    result = run(cfg | {"stop_new_after": "04:30"}, tmp_path)
    call = json.loads((tmp_path / "sandbox" / "call.json").read_text())
    argv = call["argv"]
    assert argv[argv.index("--endpoint") + 1:][:3] == ["m", "http://127.0.0.1:1", "served"]
    assert argv[argv.index("--families") + 1] == "C,D" and argv[argv.index("--mix") + 1] == "m=1"
    assert "--until" in argv and call["tasks"] == f"dev={tmp_path}/ae/tasks:heldout={tmp_path}/held/tasks"
    assert jobs == [("set", "--steps", "9"), ("step", "1"), ("step", "2")]   # (C1, D1, HC01) x 3 seeds
    assert result["report"]["argv"] == ["point", "--json", "--block-size", "16"]
    assert result["prefix_cache"] == {"enabled": True, "block_size": 16, "hits": 60.0, "queries": 100.0,
                                      "hit_rate": 60.0, "best_hits": 64, "best_rate": 64.0}
    assert result["exit_code"] == 0 and set(result["config"]["commits"]) == {"agent_eval", "heldout"}
    run(cfg | {"tasks": ["D1", "HC01"], "seeds": 1}, tmp_path)   # /metrics gives nothing now: the run goes on
    argv = json.loads((tmp_path / "sandbox" / "call.json").read_text())["argv"]
    assert argv[argv.index("--tasks") + 1] == "D1,HC01" and "--families" not in argv and jobs[3] == ("set", "--steps", "2")
    entry = sandbox.summary_entry(result)
    assert entry["status"] == "completed" and entry["sets"]["dev"]["pass"]["rate"] == 50.0


def test_refusals(env):
    cfg, tmp_path, metrics, _ = env
    with pytest.raises(sandbox.SandboxError, match="agent_eval_commit is not set"):
        run(cfg | {"agent_eval_commit": None}, tmp_path)
    with pytest.raises(sandbox.SandboxError, match="deadbeef"):
        run(cfg | {"heldout_commit": "deadbeef"}, tmp_path)
    (tmp_path / "ae" / "agent_report.py").write_text("changed")
    with pytest.raises(sandbox.SandboxError, match="local changes"):
        run(cfg, tmp_path)
    subprocess.run(["git", "-C", str(tmp_path / "ae"), "checkout", "-q", "agent_report.py"], check=True)
    metrics[0]["enable_prefix_caching"] = False
    with pytest.raises(sandbox.SandboxError, match="prefix caching off"):
        run(cfg, tmp_path)
    metrics[0] = None   # no /metrics: the flag of the server command decides
    with pytest.raises(sandbox.SandboxError, match="prefix caching off"):
        run(cfg, tmp_path, command_flag=False)
    assert run(cfg, tmp_path, command_flag=False, allow_no_prefix_cache=True)["prefix_cache"]["enabled"] is False
