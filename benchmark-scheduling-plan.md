# Scheduling Benchmark Plan (Item #10)

> Phase 4 — Compare: Synchronous / Async / Chunked Prefill / No Chunked Prefill
> Measure: Queue delay / Fairness / Throughput / Latency

---

## 1. What the benchmark does

The existing concurrency benchmark ([`benchmarks/concurrency.py`](benchmarks/concurrency.py)) fires concurrent requests at different levels and measures aggregate throughput and per-request TTFT/latency — but it does **not** vary the scheduler configuration.

This benchmark compares **how vLLM's scheduling modes affect performance** by restarting the vLLM server with different scheduling flags and running the same concurrency sweep under each configuration.

| Metric | What it tells you |
|--------|-------------------|
| **Queue delay** | How long requests wait before the scheduler picks them up (server-side `queue_time_s`) |
| **Fairness** | How evenly requests in the same batch are served (P95/P50, max/min ratio) |
| **Throughput** | Aggregate output tok/s across all concurrent requests |
| **Latency** | Per-request TTFT and total generation time distributions |

---

## 2. Where it lives

```
benchmarks/scheduling.py   # new — standalone benchmark module
core_runner.py             # updated — --skip-scheduling flag, integration in main()
```

Follows the existing module pattern: standalone module in `benchmarks/`, imports `ModelClient` from `core_runner`.

---

## 3. Scheduling configurations to compare

vLLM exposes these scheduling-relevant flags:

| Config | Flags | Description |
|--------|-------|-------------|
| **Sync scheduling** | (default — no `--async-scheduling`) | Batch scheduling: requests are scheduled together at synchronization barriers. Simple, predictable, but can cause queue-up under burst load. |
| **Async scheduling** | `--async-scheduling` | Scheduler runs asynchronously: each request is scheduled independently without barriers. Better for bursty workloads but can introduce scheduling overhead. |
| **Chunked prefill (on)** | `--enable-chunked-prefill --max-num-batched-tokens N` | Prefill work is split into chunks of `max-num-batched-tokens` tokens, allowing interleaving with decode. Reduces TTFT for late arrivals in a batch. |
| **Chunked prefill (off)** | *(default — no flag)* | Prefill runs as a single contiguous block. Blocks the GPU until the full prefill is done. |

These are toggled via flags on the vLLM command line. The benchmark restarts the server for each configuration using `VllmServer`.

### Recommended test matrix

| Config | async-scheduling | chunked-prefill | max-num-batched-tokens |
|--------|-----------------|-----------------|------------------------|
| sync_baseline | off (default) | off (default) | — |
| async | on | off (default) | — |
| chunked_on | off (default) | on | 2048 (configurable) |
| chunked_off | off (default) | off (default) | — |

Start with these four. A full 2×2 cross-combination (async + chunked together) is a follow-up.

---

## 4. Test design

### 4.1 Request pattern

Each scheduling config gets a **full concurrency sweep**, identical to the existing concurrency benchmark:

```
Concurrency levels: [1, 2, 4, 8, 16]
Requests per level: 16
Same content-deterministic prompt (salted), max_tokens=256, temperature=0.0
```

This fires requests simultaneously and measures how the scheduler handles contention.

### 4.2 Per-config output

One JSON file per config: `scheduling_sync_baseline.json`, `scheduling_async.json`, `scheduling_chunked_on.json`, `scheduling_chunked_off.json`.

Structure mirrors `concurrency.json` with added fairness and queue-delay breakdown:

```json
{
  "config": {
    "name": "async",
    "async_scheduling": true,
    "chunked_prefill": false,
    "max_num_batched_tokens": null,
    "concurrency_levels": [1, 2, 4, 8, 16],
    "requests_per_level": 16,
    "max_tokens": 256,
    "temperature": 0.0,
    "prompt_tokens": 256
  },
  "per_concurrency_level": {
    "16": {
      "wall_time_s": 4.8,
      "total_output_tokens": 4096,
      "aggregate_throughput_tok_s": 853.3,
      "n_requests": 16,
      "n_success": 16,
      "n_failed": 0,
      "ttft": { "avg_s": 0.12, "p50_s": 0.10, "p95_s": 0.25, "min_s": 0.08, "max_s": 0.45 },
      "total_time_s": { "avg_s": 4.8, "p50_s": 4.7, "p95_s": 5.2, "min_s": 4.3, "max_s": 5.8 },
      "queue_delay_s": { "avg_s": 0.02, "p50_s": 0.01, "p95_s": 0.08, "min_s": 0.0, "max_s": 0.15 },
      "fairness": {
        "ttft_p95_p50_ratio": 2.5,
        "ttft_max_min_ratio": 5.6,
        "total_time_p95_p50_ratio": 1.24,
        "total_time_max_min_ratio": 1.35
      },
      "per_request": [
        {
          "index": 0,
          "success": true,
          "prompt_tokens": 256,
          "output_tokens": 256,
          "ttft_s": 0.10,
          "total_time_s": 4.7,
          "queue_time_s": 0.01,
          "server_ttft_s": 0.09
        },
        ...
      ]
    }
  }
}
```

### 4.3 Combined summary

`scheduling.json` — aggregated view across all configs for easy comparison:

```json
{
  "configs": ["sync_baseline", "async", "chunked_on", "chunked_off"],
  "summary": {
    "16_concurrency": {
      "aggregate_throughput_tok_s": {
        "sync_baseline": 780,
        "async": 853,
        "chunked_on": 820,
        "chunked_off": 765
      },
      "ttft_p50_s": {
        "sync_baseline": 0.11,
        "async": 0.09,
        "chunked_on": 0.08,
        "chunked_off": 0.12
      },
      "ttft_p95_s": {
        "sync_baseline": 0.22,
        "async": 0.18,
        "chunked_on": 0.14,
        "chunked_off": 0.24
      },
      "fairness_score": {
        "sync_baseline": 1.0,
        "async": 1.15,
        "chunked_on": 1.4,
        "chunked_off": 1.0
      }
    }
  }
}
```

---

## 5. Fairness metrics

The key differentiator from the concurrency benchmark. For each concurrency level within a config:

```python
def compute_fairness(ttfts: list[float], total_times: list[float]) -> dict[str, float]:
    p50_ttft = median(ttfts)
    p95_ttft = percentile(ttfts, 95)
    max_ttft = max(ttfts)
    min_ttft = min(ttfts)

    p50_total = percentile(total_times, 50)
    p95_total = percentile(total_times, 95)
    max_total = max(total_times)
    min_total = min(total_times)

    return {
        "ttft_p95_p50_ratio": round(p95_ttft / p50_ttft, 2) if p50_ttft > 0 else None,
        "ttft_max_min_ratio": round(max_ttft / min_ttft, 2) if min_ttft > 0 else None,
        "total_time_p95_p50_ratio": round(p95_total / p50_total, 2) if p50_total > 0 else None,
        "total_time_max_min_ratio": round(max_total / min_total, 2) if min_total > 0 else None,
    }
```

**Interpretation**: A ratio near 1.0 means all requests in a batch are served similarly (fair). A high ratio (3x+) means some requests get fast answers while others wait much longer (unfair).

---

## 6. Queue delay analysis

For each config, aggregate queue delay from server-side `queue_time_s` across all requests at each concurrency level:

```json
{
    "queue_delay": {
        "1": { "avg_s": 0.001, "p95_s": 0.002, "max_s": 0.005 },
        "2": { "avg_s": 0.012, "p95_s": 0.035, "max_s": 0.089 },
        "4": { "avg_s": 0.045, "p95_s": 0.120, "max_s": 0.290 },
        "8": { "avg_s": 0.089, "p95_s": 0.250, "max_s": 0.580 },
        "16": { "avg_s": 0.160, "p95_s": 0.450, "max_s": 1.100 }
    }
}
```

This shows how each scheduler handles queuing under load. Async schedulers should show lower queue delay at high concurrency.

---

## 7. Server restart strategy

The benchmark requires restarting vLLM with different scheduling flags. This follows the `--compare-spec` pattern (core_runner.py:1237-1278): stop server → modify command → restart → run benchmark → stop server.

```python
def run_scheduling_test(
    model_config: dict,
    run_dir: Path,
    configs: list[dict],     # scheduling config specs (name, flags)
    concurrency_levels: list[int],
    requests_per_level: int,
    max_tokens: int,
    temperature: float,
) -> dict:
    base_command = extract_base_command(model_config)
    results = {"configs": {}, "summary": {}}

    for cfg_spec in configs:
        config_name = cfg_spec["name"]
        print(f"[scheduling] testing: {config_name}")

        # Build vLLM command with scheduling flags
        new_command = apply_scheduling_flags(base_command, cfg_spec)

        # Stop current server, start new one with these flags
        server = make_managed_server(model_config, run_dir,
                                     log_name=f"scheduling_{config_name}")
        server.command = new_command
        server.start()
        client = make_client(model_config)
        wait_for_endpoint(client, model_config, server)

        # Run concurrency sweep
        sweep = run_concurrency_sweep(client, ...)

        # Stop server
        server.stop()

        results["configs"][config_name] = sweep

    results["summary"] = build_summary(results["configs"])
    return results
```

---

## 8. Integration into core_runner.py

### New CLI flag

```python
parser.add_argument("--skip-scheduling", action="store_true",
                    help="Skip scheduling benchmark")
```

### New YAML config keys (all optional)

```yaml
# In the model YAML
scheduling_configs:
  - name: sync_baseline
    async_scheduling: false
    chunked_prefill: false
  - name: async
    async_scheduling: true
    chunked_prefill: false
  - name: chunked_on
    async_scheduling: false
    chunked_prefill: true
    max_num_batched_tokens: 2048
  - name: chunked_off
    async_scheduling: false
    chunked_prefill: false

scheduling_concurrency_levels: [1, 2, 4, 8, 16]
scheduling_requests_per_level: 16
scheduling_max_tokens: 256
scheduling_temperature: 0.0
```

### Insertion point in main()

Place **after the concurrency benchmark** (line ~1202 in core_runner.py). The concurrency benchmark measures "what happens"; the scheduling benchmark explains "why" by comparing scheduler implementations.

```python
if not args.skip_scheduling:
    if server_mode != "managed" or server is None:
        print("[core_runner] scheduling benchmark requires managed server mode")
    else:
        from benchmarks.scheduling import run_scheduling_test

        sched_cfgs = cfg.get("scheduling_configs", _default_scheduling_configs())
        max_tokens = cfg.get("scheduling_max_tokens", 256)
        temperature = cfg.get("scheduling_temperature", 0.0)
        levels = cfg.get("scheduling_concurrency_levels",
                         cfg.get("concurrency_levels", [1,2,4,8,16]))
        reqs = cfg.get("scheduling_requests_per_level",
                       cfg.get("concurrency_requests_per_level", 16))

        print(f"[core_runner] scheduling benchmark: {len(sched_cfgs)} configs")
        sched_results = run_scheduling_test(
            cfg, run_dir, sched_cfgs, levels, reqs,
            max_tokens=max_tokens, temperature=temperature,
        )
        save_json(run_dir / "scheduling.json", sched_results["summary"])
        summary["scheduling"] = sched_results["summary"]
```

---

## 9. Implementation steps

| Step | Task | Files |
|------|------|-------|
| 1 | Create `benchmarks/scheduling.py` with `apply_scheduling_flags()` — takes base vLLM command and returns new command with scheduling flags added/removed | `benchmarks/scheduling.py` (new) |
| 2 | Implement `run_concurrency_sweep()` — reuses concurrency logic (could be shared from `benchmarks/concurrency.py` via a refactored common helper) | `benchmarks/scheduling.py` |
| 3 | Implement `compute_fairness()` — P95/P50 and max/min ratios for TTFT and total time | `benchmarks/scheduling.py` |
| 4 | Implement `build_summary()` — cross-config aggregated comparison | `benchmarks/scheduling.py` |
| 5 | Implement `run_scheduling_test()` — orchestrates server restarts, runs sweeps, aggregates results | `benchmarks/scheduling.py` |
| 6 | Add `--skip-scheduling` CLI flag + YAML config loading in `core_runner.py` | `core_runner.py` |
| 7 | Wire up in `main()` after concurrency benchmark | `core_runner.py` |
| 8 | Test with 2 configs (sync baseline + async) first | Run against test model |
| 9 | Add chunked_on and chunked_off configs | Run against test model |

---

## 10. Edge cases & failure modes

| Scenario | Handling |
|----------|----------|
| vLLM flag not recognized (e.g., `--async-scheduling` on old version) | Log warning, skip that config, continue with others |
| Server fails to start with new flags | Log error with vLLM log tail, skip config, continue |
| Port still in use after restart | Retry loop: stop → sleep(1) → check port → retry up to 3 times |
| Concurrent requests fail under high load | Record failures, include in fairness metric |
| OOM at high concurrency | Existing OOM detection in concurrency benchmark; record and continue |

---

## 11. GPU telemetry

Not required for this benchmark. The primary question is scheduler behavior, not GPU utilization. The existing concurrency benchmark doesn't include GPU telemetry either, so consistency suggests keeping it out.

---

## 12. Future extensions

- **Full 2×2 cross-combination**: async + chunked prefill together (4 configs → 16 configs).
- **Queue depth analysis**: Vary the number of queued requests vs GPU capacity.
- **Heterogeneous requests**: Mix short and long prompts in the same batch — how does the scheduler prioritize?
- **Priority scheduling**: If vLLM supports priority queues, benchmark priority enforcement.