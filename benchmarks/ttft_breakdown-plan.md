# TTFT Breakdown Benchmark — Implementation Plan

## Goal

Split client-side TTFT (wall-clock time to first token) into four
contributing factors so every millisecond of latency can be explained:

1. **Scheduler / queue delay** — time from request arrival to prefill start.
2. **Prefill time** — GPU/CPU time to process prompt tokens.
3. **First decode + scheduling** — time from end of prefill to first output token.
4. **GPU idle / power overhead** — opportunity cost of the GPU during TTFT.

## Where the data already is

`ModelClient._execute_request()` (core_runner.py:555) already extracts
`request_metrics` from vLLM's stream:

| Server field                  | GenerationResult field    | Meaning                    |
|-------------------------------|---------------------------|----------------------------|
| `queue_time_s`               | `queue_time_s`            | Scheduler delay            |
| `prompt_time_s`              | `prefill_time_s`          | Prefill (prompt) time      |
| `time_to_first_token_s`      | `time_to_first_token_s`   | Server-side TTFT           |

But `prompt_time_s` is the server-side prefill duration while the client-side
TTFT is `time_to_first_token_s`.  For a single-request benchmark (no concurrency)
they should sum to ~TTFT; with concurrency the gap is the first decode +
scheduler overhead.

**Key assumption:** prompt_time_s ≈ time the GPU spends on prefill work.  The
difference `TTFT − queue_time_s − prompt_time_s` = first decode time.  We
validate by confirming that the three non-negative components sum to within ~5 %
of the client-side TTFT (the remaining gap is network round-trip).

## Files to change

### 1. `benchmarks/ttft_breakdown.py` (new)

Same structure as `benchmarks/prefill.py` but focused on latency, not throughput:

```python
# Pseudocode — not implementation

@dataclass
class TtftRequestResult:
    index: int
    success: bool
    prompt_tokens: int
    prompt_tokens_exact: bool
    output_tokens: int

    # Client wall-clock (ms)
    ttft_ms: float
    total_time_ms: float

    # Server-side (s) — None if unsupported
    queue_time_s: float | None
    prefill_time_s: float | None
    server_ttft_s: float | None

    # Derived breakdown (ms) — computed client-side
    scheduler_delay_ms: float | None      # = queue_time_s → ms
    prefill_ms: float | None               # = prefill_time_s → ms
    first_decode_ms: float | None          # = server_ttft − queue − prefill

    cached_tokens: int
    error: str
    ...

def run_ttft_breakdown(client, prompt_lengths=None, repetitions=10) -> dict:
    """Measure TTFT breakdown across prompt lengths."""
    # 1. Probe: does the server emit request_metrics?
    # 2. For each length: build prompt → send N cold requests → collect results
    # 3. Aggregate per-length: avg/median/p95/min/max for each breakdown component
    # 4. Attach GPU telemetry if monitor is provided
    # 5. Return {config, per_length: {…}}
```

### 2. `core_runner.py` — orchestration

Add one new benchmark invocation alongside the existing ones:

```python
# Inside main(), after skip-prefill block or wherever makes sense
# (recommended: after skip-latency, before skip-decode)

if not args.skip_ttft:
    from benchmarks.ttft_breakdown import run_ttft_breakdown

    prompt_lengths = cfg.get(
        "ttft_prompt_lengths", [128, 512, 2048, 8192, 32768]
    )
    repetitions = cfg.get("ttft_repetitions", 10)

    print(f"[core_runner] TTFT breakdown over {prompt_lengths} ({repetitions} reps)")
    ttft_results = run_ttft_breakdown(client, prompt_lengths, repetitions)
    save_json(run_dir / "ttft_breakdown.json", ttft_results)
    summary["ttft_breakdown"] = ttft_results
```

Add CLI flag:

```python
parser.add_argument("--skip-ttft", action="store_true")
```

Add model config defaults (YAML):

```yaml
ttft_prompt_lengths: [128, 512, 2048, 8192, 32768]
ttft_repetitions: 10
```

### 3. `core_runner.py` — minor fix

`datetime.utcnow()` is deprecated; replace calls with:

```python
datetime.now(datetime.timezone.utc).isoformat()
```

(Found at lines 136, 1008.)

## Output schema

```json
{
  "config": {
    "benchmark_version": "1.0",
    "definition": "TTFT decomposition into scheduler delay, prefill time, and first decode overhead",
    "prompt_lengths": [128, 512, 2048, 8192, 32768],
    "repetitions": 10,
    "cache_isolation_method": "cache_salt",
    "server_metrics_supported": true,
    "start_time": "2026-07-19T...",
    "end_time": "2026-07-19T..."
  },
  "per_length": {
    "128": {
      "status": "success",
      "requested_tokens": 128,
      "actual_tokens": 128,
      "n_requests": 10,
      "n_success": 10,
      "n_failed": 0,
      "cache_hit_rate": 0.0,
      "per_request": [
        {
          "index": 0,
          "success": true,
          "prompt_tokens": 128,
          "ttft_ms": 45.2,
          "queue_time_s": 0.001,
          "prefill_time_s": 0.032,
          "server_ttft_s": 0.042,
          "scheduler_delay_ms": 1.0,
          "prefill_ms": 32.0,
          "first_decode_ms": 9.0
        },
        ...
      ],
      "aggregated": {
        "ttft":            { "avg_ms": 47.1, "median_ms": 46.5, "p95_ms": 62.0, "min_ms": 38.2, "max_ms": 89.1 },
        "scheduler_delay": { "avg_ms": 1.2,  "median_ms": 1.0,   "p95_ms": 3.5,   "min_ms": 0.3,   "max_ms": 8.2  },
        "prefill_time":    { "avg_ms": 34.5, "median_ms": 34.0,  "p95_ms": 45.0,  "min_ms": 28.0,  "max_ms": 52.0 },
        "first_decode":    { "avg_ms": 11.4, "median_ms": 11.0,  "p95_ms": 18.0,  "min_ms": 8.5,   "max_ms": 25.0 },
        "gpu": { "gpu_util_avg_pct": 12.5, "gpu_power_avg_w": 180, "energy_wh": 0.005 }
      }
    }
  }
}
```

## Validation / sanity checks

1. **Sum check**: `scheduler_delay + prefill + first_decode ≈ server_ttft`
   (within 5 %).  If not, log a warning — indicates a vLLM version mismatch.

2. **Client vs server TTFT**: `client_ttft_s` should be ≥ `server_ttft_s`
   (network adds latency).  If client < server, log a warning.

3. **Non-negative**: All breakdown components must be ≥ 0.  If negative,
   clamp to 0 and log — happens when scheduler overestimates queue time.

4. **Server metrics probe**: If the server doesn't emit `request_metrics`
   (vLLM < 0.6.0, or `--disable-log-requests` enabled), the benchmark
   should still run but note `server_metrics_supported: false` and leave the
   three sub-components as `null`.

5. **No-cache guarantee**: Each request uses a unique `cache_salt` so
   prefix cache cannot skew results.  This is critical — a cache hit
   would make `scheduler_delay` and `prefill_ms` essentially zero.

## Integration with existing benchmarks

This benchmark is complementary to `prefill_scaling`:

| Benchmark        | What it measures                        | Why it differs            |
|------------------|-----------------------------------------|---------------------------|
| `prefill_scaling`| Prefill throughput (tok/s) across lengths | Maximizes GPU compute     |
| `ttft_breakdown` | TTFT component timing breakdown         | Maximizes timing granularity |

Both use `max_tokens=1` and cache isolation.  They can run back-to-back
in the same invocation.

## Implementation order (within core_runner)

The recommended insertion point is **after the `--skip-latency` block**
(because latency.json reports client-side TTFT only) and **before the
`--skip-decode` block** (because decode.json measures throughput, not latency).

```
if not args.skip_latency          → latency.json (client TTFT, prefill TPS)
if context_lengths is present     → deep_context.json (TTFT at large contexts)
if not args.skip_ttft             → ttft_breakdown.json (TIMING BREAKDOWN) ← NEW
if not args.skip_decode           → decode.json (throughput)
if not args.skip_reasoning        → reasoning.json
if not args.skip_concurrency      → concurrency.json
if not args.skip_prefill          → prefill_scaling.json (prefill TPS)
```

## Future extensions (after v1)

- **Concurrency-aware breakdown**: Run with concurrent requests to measure
  how scheduler delay grows under load.  This overlaps with the concurrency
  benchmark (Phase 4, #9 Saturation Curve).
- **Cached-request breakdown**: Send two identical requests back-to-back to
  measure how prefix cache eliminates prefill time and reduces scheduler
  overhead (overlaps with Phase 3, #6 Prefix Cache).
- **Per-request GPU telemetry**: Embed GPU power draw per-request (via
  GpuMonitor windows) to correlate energy with each breakdown component.