"""Tests for the dashboard data collector and the page build."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import dashboard


def _write_run(root: Path, model: str, stamp: str, *, spec: bool, memory: bool) -> Path:
    run_dir = root / model / stamp
    run_dir.mkdir(parents=True)
    summary = {
        "model": model,
        "status": "completed",
        "latency": {"2048": {"ttft_median_s": 0.25, "ttft_p95_s": 0.4, "n": 10},
                    "512": {"ttft_median_s": 0.1, "n": 10}},
        "decode": {"512": {"tok_per_sec_avg": 20.0, "tok_per_sec_peak": 25.0},
                   "1024": {"tok_per_sec_avg": 22.0, "tok_per_sec_peak": 26.0}},
        "concurrency": {"per_concurrency_level": {
            "1": {"aggregate_throughput_tok_s": 20.0, "ttft": {"median_s": 0.1}},
            "16": {"aggregate_throughput_tok_s": 200.0, "ttft": {"median_s": 0.5}},
        }},
        "ttft_breakdown": {"per_length": {"2048": {
            "aggregated": {"ttft": {"median_ms": 250.0}, "scheduler_delay": {"median_ms": 0.02}},
            "per_request": [{"success": True, "post_queue_ms": v} for v in (240.0, 250.0, 260.0)],
        }}},
        "tool_calling": {"composite_score": 3.5, "total_tasks": 85, "total_correct": 50,
                         "composite_components": {"tool_accuracy": 0.9}},
        "spec_comparison": {
            "spec_enabled": {"512": {"tok_per_sec_avg": 40.0}},
            "spec_disabled": {"512": {"tok_per_sec_avg": 20.0}},
        },
        "gpu": {"gpu_power_avg_w": 30.0, "energy_wh": 5.0},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    command = ["vllm", "serve", "org/Model", "--port", "8000"]
    if spec:
        command += ["--speculative-config", '{"method":"mtp"}']
    (run_dir / "resolved_server.json").write_text(json.dumps({"command": command}))
    (run_dir / "tool_calling.json").write_text(json.dumps(
        {"category_scores": {"weather_basic": {"pass_rate": 0.5, "total": 4}}}
    ))
    if memory:
        rows = ["elapsed_s,timestamp,mem_total_gib,mem_available_gib,mem_used_gib,swap_used_gib"]
        rows += [f"{i * 5}.0,t,121.7,{100 - i},{21.7 + i},1.0" for i in range(4)]
        (run_dir / "memory_samples.csv").write_text("\n".join(rows) + "\n")
        (run_dir / "memory_guard.json").write_text(json.dumps({"floor_gib": 8.0, "warn_gib": 16.0}))
    return run_dir


def _two_runs(root: Path) -> list[dict]:
    _write_run(root, "new-model", "20261107_100000", spec=True, memory=True)
    _write_run(root, "old-model", "20260729_150000", spec=False, memory=False)
    return dashboard.collect_runs(root)


def test_collect_runs_normalizes_and_orders(tmp_path: Path) -> None:
    old, new = _two_runs(tmp_path)

    assert (old["model"], new["model"]) == ("old-model", "new-model")
    assert old["pre_fix"] is True and new["pre_fix"] is False
    assert old["server"]["speculative"] is False and new["server"]["speculative"] is True
    assert [p["x"] for p in new["latency"]] == [512, 2048]
    assert new["latency"][1]["ttft_median_ms"] == 250.0
    assert [d["avg"] for d in new["decode"]] == [20.0, 22.0]
    assert new["concurrency"][-1] == {
        "x": 16, "throughput": 200.0, "ttft_median_ms": 500.0,
        "latency_median_s": None, "success_rate": None,
    }
    assert new["gpu"]["energy_wh"] == 5.0


def test_cli_writes_the_collected_data(tmp_path: Path) -> None:
    _two_runs(tmp_path / "results")
    out = tmp_path / "data.json"
    proc = subprocess.run(
        [sys.executable, str(Path(dashboard.__file__)), "--results", str(tmp_path / "results"),
         "--out", str(tmp_path / "page.html"), "--json", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert [r["model"] for r in json.loads(out.read_text())["runs"]] == ["old-model", "new-model"]


def test_render_escapes_script_end_tags() -> None:
    payload = {"runs": [{"error": "bad </script><script>alert(1)</script>"}]}
    html = dashboard.render(payload, fragment=True)
    data_block = html.split("window.BENCH = ", 1)[1].split(";\n", 1)[0]
    assert "</script>" not in data_block
    assert json.loads(data_block.replace("<\\/", "</")) == payload


def test_render_full_page_joins_the_modules() -> None:
    html = dashboard.render({"runs": []}, fragment=False)
    assert html.startswith("<!doctype html>")
    assert "<title>Spark Inference Benchmarks</title>" in html
    for marker in (dashboard.CSS_MARKER, dashboard.JS_MARKER, dashboard.DATA_MARKER):
        assert marker not in html
    assert html.rstrip().endswith("start();\n})();\n</script>\n\n</body>\n</html>".rstrip())
