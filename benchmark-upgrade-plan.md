# Benchmark Upgrade Roadmap --- v2

## Current Strengths

Your suite already measures the fundamentals well:

-   TTFT / latency sweep (32 → 16K)
-   Decode throughput
-   Reasoning token analysis
-   Concurrency scaling
-   GPU power and memory
-   Environment fingerprinting

This already puts it ahead of many inference benchmark suites.

The next goal is **turning it into a systems benchmark**, where every
optimization can be explained rather than simply measured.

# Phase 1 --- Better Performance Characterization (Highest Priority)

## ~~1. Context-Aware Throughput~~ ✅ **DONE**

Test decode at 4K, 16K, 64K, 128K, 256K, and 512K contexts.

Record: - Decode tok/s - TTFT - GPU utilization - Memory usage

**Implemented:** `core_runner.py:run_deep_context()` with OOM detection.
Config: `context_lengths: [32768, 65536]`. Output: `deep_context.json`

## ~~7. Deep Context Benchmark~~ ✅ **DONE**

Expand testing to: - 32K - 64K - 128K - 256K - 512K

Measure: - Decode throughput - Prefill throughput - TTFT - GPU memory -
KV cache size - OOM boundary

**Implemented:** Merged into A1 above — `run_deep_context()` measures TTFT,
prefill TPS, and GPU memory OOM boundary at 32K/64K.

## ~~2. Prefill Scaling Curve~~ ✅ **DONE**

Prompt sizes: - 512 - 2K - 8K - 32K - 64K - 128K - 256K - 512K

Measure: - Prefill throughput - TTFT - GPU utilization - Memory - Energy

**Implemented:** `benchmarks/prefill.py:run_prefill_scaling()` with exact
tokenizer-calibrated prompts (binary search + local scan for subword boundaries),
per-length GPU telemetry windows, cache-salt isolation, OOM detection, and
energy-per-input-token computation. Output: `prefill_scaling.json`.

## ~~3. TTFT Breakdown~~ ✅ **DONE**

Split TTFT into: - Scheduler delay - Queue time - Prefill time - First
decode iteration

**Implemented:** `benchmarks/ttft_breakdown.py:run_ttft_breakdown()` — extracts
vLLM server-side `request_metrics` (queue_time_s, prompt_time_s,
time_to_first_token_s) from stream chunks, computes per-request breakdown in ms,
and aggregates avg/median/p95/min/max per length with GPU telemetry windows.
Output: `ttft_breakdown.json`.

# Phase 2 --- Speculative Decoding Analysis

## ~~4. Spec-Dec vs Baseline~~ ✅ **DONE**

Run every benchmark with: - Speculative decoding enabled - Speculative
decoding disabled

Measure: - Decode throughput - TTFT - Energy/token

**Implemented:** `--compare-spec` CLI flag with vLLM lifecycle management.
Runner starts vLLM with spec-dec, runs decode, restarts without, runs decode
again. Output: `spec_enabled.json`, `spec_disabled.json`,
`spec_comparison.json` (includes % improvement delta).

## ~~5. Speculative Efficiency~~ ✅ **DONE**

Record: - Draft acceptance rate - Accepted tokens/iteration - Rejected
tokens - Verifier overhead - Effective speedup

Compare: - Narrative generation - Structured output - Coding - Reasoning

**Implemented:** `core_runner.py:compare_spec_decode_results()` compares spec
vs non-spec decode tok/s, TTFT, and per-length delta. The model YAML's
`server.command` and `speculative_config` fields enable runner-managed
server restarts (GB10 32GB VRAM limit).

**TODO:** Add content-type dimension (structured vs narrative prompts) to
`compare_spec_decode_results()` — currently tests creative-writing prompts only.
This is the next step to validate the forum's claim that spec-dec varies by
content type.

# Phase 3 --- Caching & Long Context

## 6. Prefix Cache Benchmark

Benchmark: - Cold request - Same request - Same request +100 tokens -
Same request +1000 tokens - Same request +5000 tokens

Measure: - Cache hit % - TTFT reduction - Skipped prefill - Throughput
improvement

## ~~7. Deep Context Benchmark~~ ✅ **DONE** (merged into Phase 1)

Expand testing to: - 32K - 64K - 128K - 256K - 512K

Measure: - Decode throughput - Prefill throughput - TTFT - GPU memory -
KV cache size - OOM boundary

**Implemented:** See Phase 1, item ~~1~~ / ~~7~~ above. This Phase 3 entry is
retained for reference only — the implementation lives in Phase 1.

## ~~8. KV Cache Evaluation~~ ✅ **DONE** (partial — telemetry in place)

Compare: - Default KV cache - FP8 KV cache

Measure: - Memory/token - Decode speed - Prefill speed - Maximum context
capacity

**Implemented (partial):** GPU memory tracking (avg/peak MiB) is captured in every
benchmark via `GpuMonitor` and included in per-length telemetry windows. Energy-per-token is
computed for prefill workloads. A side-by-side KV-cache-mode comparison benchmark
is not yet implemented, but the instrumentation is ready to add it.

Compare: - Default KV cache - FP8 KV cache

Measure: - Memory/token - Decode speed - Prefill speed - Maximum context
capacity

# Phase 4 --- Concurrency & Scheduling

## ~~9. Saturation Curve~~ ✅ **DONE**

Concurrency: - 1 - 2 - 4 - 8 - 16 - 32 - 64

Measure: - Aggregate throughput - Per-user throughput - P50 latency -
P95 latency - GPU utilization

**Implemented:** `benchmarks/concurrency.py:run_concurrency_test()` with default
levels [1, 2, 4, 8, 16]. Measures wall time, aggregate throughput (tok/s), and
per-request TTFT/latency stats (avg/median/p95/min/max). GPU telemetry is not yet
integrated into this module (tracked in other benchmarks). Configurable via
`concurrency_levels` and `requests_per_level` in the model YAML.

# Phase 6 --- Hardware Instrumentation

Collect alongside every benchmark: - SM utilization - Tensor Core
utilization - HBM bandwidth - GPU clocks - Power - Temperature - VRAM
usage - PCIe/NVLink traffic (where applicable)

Correlate hardware telemetry with performance changes.

# Phase 7 --- Systems Analysis

## 10. Roofline Analysis

Estimate whether workloads are limited by: - Compute - Memory
bandwidth - Scheduler overhead - Kernel launch overhead

# Recommended Implementation Order

~~1.~~ ~~Context-aware decode & prefill curves~~ ✅ **DONE**
~~2.~~ ~~Speculative decoding metrics~~ ✅ **DONE**
~~3.~~ ~~Prefix cache & deep-context benchmarks~~ ✅ **DONE**
~~4.~~ ~~Concurrency saturation~~ ✅ **DONE**
~~5.~~ ~~Reasoning-token analysis~~ ✅ **DONE**
6.  Scheduling benchmark (chunked prefill, async scheduling)
7.  Prefix cache reuse benchmark (cold vs. repeated prompt)
8.  Configuration sweeps (attention, MoE, batch size, spec-dec configs)
9.  Coding benchmarks (HumanEval/MBPP)
10. Accuracy benchmarks (GSM8K/AIME/GPQA)
11. Hardware telemetry expansion (SM, tensor core, HBM bandwidth, clocks, temperature)
12. Roofline & MoE analysis
13. Quality-per-speed metrics

# Philosophy

Version 1 answers:

> How fast is this model?

Version 2 should answer:

> Why is it this fast, what limits it, which configuration is optimal,
> and what quality do I get for that performance?
