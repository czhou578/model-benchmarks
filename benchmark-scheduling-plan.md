# Scheduling Benchmark Plan (Item #10)

> Phase 4 — Compare: Synchronous / Async / Chunked Prefill / No Chunked Prefill
> Measure: Queue delay / Fairness / Throughput / Latency

---

## 1. What the benchmark does

The existing concurrency benchmark ([`benchmarks/concurrency.py`](benchmarks/concurrency.py)) fires concurrent requests at different levels and measures aggregate throughput and per-request TTFT/latency — but it does **not** vary the scheduler configuration.

This benchmark compares **how vLLM's scheduling modes affect performance** by running the concurrency sweep under different configurations.

To minimize vLLM boot overhead (300–600s per restart), the benchmark uses vLLM's **runtime `/config` API** for flags that support runtime toggling, reducing the number of server boots.

| Metric | What it tells you |
|--------|-------------------|
| **Queue delay** | How long requests wait before the scheduler picks them up (server-side `queue_time_s`) |
| **Fairness** | How evenly requests in the same batch are served (P95/P50, max/min ratio) |
| **Throughput** | Aggregate output tok/s across all concurrent requests |
| **Latency** | Per-request TTFT and total generation time distributions |

---

## 2. Scheduling configurations (v1)

vLLM flags are split into two categories: those togglable at runtime via the `/config` API, and those requiring a full server restart.

| Config | vLLM flags | Toggle method |
|--------|-----------|---------------|
| **sync_baseline** | *(default)* | Initial server boot |
| **chunked_on** | `enable_chunked_prefill=true`, `max_num_batched_tokens=N` | Runtime `/config` API |
| **chunked_off** | `enable_chunked_prefill=false`, `max_num_batched_tokens=null` | Runtime `/config` API |

### Why not async?

Async scheduling (`--async-scheduling`) is a **hard-start flag** — vLLM does not expose it via `/config`. Adding it would require a full server restart and add 300–600s of boot time.

**Plan:** ship v1 with sync_baseline + chunked toggles (one boot, 3 configs). Add async as a "manual" entry in the results that requires a separate server restart. This gives meaningful scheduling insights — chunked prefill vs no-chunked is the practical decision most users care about.

---

## 3. Runtime config toggling

vLLM exposes a `/config` endpoint:

```bash
# GET current config
curl -X GET http://localhost:8000/config

# PUT runtime config changes
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "enable_chunked_prefill": true,
    "max_num_batched_tokens": 2048
  }'
```

After each `/config` change, the server needs a **brief stabilization period** (~2–3 seconds) before the next benchmark request fires. This is much faster than a full server restart.

### Toggle sequence

```
1. Boot vLLM with sync_baseline flags (default)
2. Wait for ready
3. Run concurrency sweep → sync_baseline results
4. PUT config: enable_chunked_prefill=true, max_num_batched_tokens=2048
5. Wait ~3s for stabilization
6. Run concurrency sweep → chunked_on results
7. PUT config: enable_chunked_prefill=false, max_num_batched_tokens=null
8. Wait ~3s for stabilization
9. Run concurrency sweep → chunked_off results
```

Total time: **~6 min boot + ~20 min benchmark** ≈ **26 min total** (down from 44 min with serial restarts).

---

## 4. Test design

### 4.1 Request pattern

Identical to the existing concurrency benchmark:

```
Concurrency levels: [1, 2, 4, 8, 16]
Requests per level: 16
Same content-deterministic prompt (salted), max_tokens=256, temperature=0.0
```

This fires requests simultaneously and measures how the scheduler handles contention under each scheduling config.

### 4.2 Per-config output

One JSON file per config: `scheduling_sync_baseline.json`, `scheduling_chunked_on.json`, `scheduling_chunked_off.json`.

Structure mirrors `concurrency.json` with added fairness and queue-delay breakdown:

```json
{
  "config": {
    "name": "chunked_on",
    "async_scheduling": false,
    "chunked_prefill": true,
    "max_num_batched_tokens": 2048,
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
  "configs": ["sync_baseline", "chunked_on", "chunked_off"],
  "summary": {
    "16_concurrency": {
      "aggregate_throughput_tok_s": {
        "sync_baseline": 780,
        "chunked_on": 853,
        "chunked_off": 765
      },
      "ttft_p50_s": {
        "sync_baseline": 0.11,
        "chunked_on": 0.08,
        "chunked_off": 0.12
      },
      "ttft_p95_s": {
        "sync_baseline": 0.22,
        "chunked_on": 0.14,
        "chunked_off": 0.24
      },
      "fairness_score": {
        "sync_baseline": 1.0,
        "chunked_on": 1.1,
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

This shows how each scheduler handles queuing under load.

---

## 7. Runtime config management

The benchmark uses vLLM's `/config` API to toggle settings between runs. This is a simple HTTP PUT:

```python
def toggle_scheduling_config(
    client: ModelClient,
    config_name: str,
    enable_chunked: bool | None = None,
    max_num_batched_tokens: int | None = None,
    stabilization_seconds: float = 3.0,
) -> None:
    """Toggle vLLM runtime scheduling config via /config API.

    Args:
        client: ModelClient connected to the running vLLM endpoint.
        config_name: Human-readable name of the target config (for logging).
        enable_chunked: If True, enable chunked prefill. If False, disable.
                        If None, leave current value unchanged.
        max_num_batched_tokens: New max_num_batched_tokens value. None = use default.
        stabilization_seconds: Seconds to wait after toggle before returning.
    """
    payload = {}
    if enable_chunked is not None:
        payload["enable_chunked_prefill"] = enable_chunked
    if max_num_batched_tokens is not None:
        payload["max_num_batched_tokens"] = max_num_batched_tokens

    if payload:
        config_url = f"{client.base_url.rstrip('/')}/config"
        resp = requests.put(config_url, json=payload, timeout=30)
        resp.raise_for_status()
        print(f"[scheduling] toggled config ({config_name}): {payload}")
        time.sleep(stabilization_seconds)
```

After each toggle, the benchmark sleeps for `stabilization_seconds` (default 3.0s) to ensure vLLM's scheduler state has settled before firing the next batch of requests.

### Handling unsupported runtime toggles

Not all scheduling flags can be toggled at runtime. If the user specifies a config that requires restart, fall back to the restart approach:

```python
def apply_scheduling_config(
    server: VllmServer | None,
    model_config: dict,
    run_dir: Path,
    spec: dict,
) -> tuple[ModelClient, VllmServer | None]:
    """Apply a scheduling config. May reuse existing server (runtime toggle)
    or restart it (hard flags). Returns (client, server)."""

    requires_restart = spec.get("async_scheduling")  # --async-scheduling needs restart

    if requires_restart:
        # Fall back to server restart (slow path)
        ...
    else:
        # Runtime toggle (fast path)
        ...
```

For v1, all configs in the default matrix use runtime toggle, so this is dead code that future-proofs the design.

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
            cfg, run_dir, server, sched_cfgs, levels, reqs,
            max_tokens=max_tokens, temperature=temperature,
        )
        save_json(run_dir / "scheduling.json", sched_results["summary"])
        summary["scheduling"] = sched_results["summary"]
```

---

## 9. Implementation steps

| Step | Task | Files |
|------|------|-------|
| 1 | Create `benchmarks/scheduling.py` with `toggle_scheduling_config()` — runtime `/config` API toggle | `benchmarks/scheduling.py` (new) |
| 2 | Implement `run_concurrency_sweep()` — reuses concurrency logic (shared with `benchmarks/concurrency.py`) | `benchmarks/scheduling.py` |
| 3 | Implement `compute_fairness()` — P95/P50 and max/min ratios for TTFT and total time | `benchmarks/scheduling.py` |
| 4 | Implement `run_scheduling_test()` — toggles runtime config between runs, aggregates results | `benchmarks/scheduling.py` |
| 5 | Add `--skip-scheduling` CLI flag + YAML config loading in `core_runner.py` | `core_runner.py` |
| 6 | Wire up in `main()` after concurrency benchmark | `core_runner.py` |
| 7 | Test with 2 configs (sync baseline + chunked_on toggle) first | Run against test model |
| 8 | Add chunked_off toggle after sync+chunked works | Run against test model |

### Dependencies

- No new Python dependencies. Uses existing `requests` import (already used in `core_runner.py`), `VllmServer` for managed lifecycle.
- Requires managed server mode (`server.mode: managed` in YAML).

---

## 10. Edge cases & failure modes

| Scenario | Handling |
|----------|----------|
| vLLM rejects runtime `/config` toggle | Log warning, skip that config, continue (fall back to restart path if async is needed) |
| Runtime toggle doesn't take effect after stabilization | Log warning, add longer stabilization (5s), continue |
| Concurrent requests fail under high load | Record failures, include in fairness metric |
| OOM at high concurrency | Existing OOM detection in concurrency benchmark; record and continue |

---

## 11. GPU telemetry

Not required for this benchmark. The primary question is scheduler behavior, not GPU utilization. The existing concurrency benchmark doesn't include GPU telemetry either, so consistency suggests keeping it out.

---

## 12. Future extensions

- **Async scheduling**: Add as a "manual" entry that requires full server restart. Document in results that it was not tested.
- **Full 2×2 cross-combination**: async + chunked prefill together. Requires restart for async.
- **Queue depth analysis**: Vary the number of queued requests vs GPU capacity.
- **Heterogeneous requests**: Mix short and long prompts in the same batch — how does the scheduler prioritize?
- **Priority scheduling**: If vLLM supports priority queues, benchmark priority enforcement.