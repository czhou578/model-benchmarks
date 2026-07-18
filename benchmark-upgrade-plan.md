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

## 2. Prefill Scaling Curve

Prompt sizes: - 512 - 2K - 8K - 32K - 64K - 128K - 256K - 512K

Measure: - Prefill throughput - TTFT - GPU utilization - Memory - Energy

## 3. TTFT Breakdown

Split TTFT into: - Scheduler delay - Queue time - Prefill time - First
decode iteration

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

## 7. Deep Context Benchmark

Expand testing to: - 32K - 64K - 128K - 256K - 512K

Measure: - Decode throughput - Prefill throughput - TTFT - GPU memory -
KV cache size - OOM boundary

## 8. KV Cache Evaluation

Compare: - Default KV cache - FP8 KV cache

Measure: - Memory/token - Decode speed - Prefill speed - Maximum context
capacity

# Phase 4 --- Concurrency & Scheduling

## 9. Saturation Curve

Concurrency: - 1 - 2 - 4 - 8 - 16 - 32 - 64

Measure: - Aggregate throughput - Per-user throughput - P50 latency -
P95 latency - GPU utilization

## 10. Scheduling Benchmark

Compare: - Synchronous scheduling - Async scheduling - Chunked prefill -
No chunked prefill

Measure: - Queue delay - Fairness - Throughput - Latency

# Phase 5 --- Configuration Optimization

## 11. Attention Backend Sweep

Compare supported attention backends.

Measure: - TTFT - Decode speed - Memory - GPU utilization

## 12. MoE Backend Sweep

Compare supported MoE implementations.

Measure: - Routing overhead - Throughput - Memory - Expert dispatch cost

## 13. Batch Size Sweep

Sweep: - 1024 - 4096 - 8192 - 16384 - 32768

Measure: - Throughput - Latency - P95 latency - GPU utilization

## 14. Speculative Configuration Sweep

Compare: - MTP-1 - MTP-2 - MTP-3 - MTP-4 (if supported)

# Phase 6 --- Capability Benchmarks

## 15. Coding

Add: - HumanEval - MBPP - LiveCodeBench (optional)

Measure: - Pass@1 - Generation latency - Compile failures

## 16. Reasoning

Add: - GSM8K - AIME-lite - GPQA-lite

Measure: - Accuracy - Reasoning tokens - Latency

# Phase 7 --- Hardware Instrumentation

Collect alongside every benchmark: - SM utilization - Tensor Core
utilization - HBM bandwidth - GPU clocks - Power - Temperature - VRAM
usage - PCIe/NVLink traffic (where applicable)

Correlate hardware telemetry with performance changes.

# Phase 8 --- Systems Analysis

## 18. Roofline Analysis

Estimate whether workloads are limited by: - Compute - Memory
bandwidth - Scheduler overhead - Kernel launch overhead

## 19. MoE-Specific Metrics

Record: - Active experts/token - Expert imbalance - Routing overhead -
Expert occupancy

# Recommended Implementation Order

~~1.~~ ~~Context-aware decode & prefill curves~~ ✅ **DONE**
~~2.~~ ~~Speculative decoding metrics~~ ✅ **DONE**
~~3.~~ ~~Prefix cache & deep-context benchmarks~~ ✅ **DONE**
4.  Concurrency saturation & scheduler analysis
5.  Configuration sweeps
6.  Coding & reasoning benchmarks
7.  Hardware telemetry
8.  Roofline & MoE analysis
9.  Quality-per-speed metrics

# Philosophy

Version 1 answers:

> How fast is this model?

Version 2 should answer:

> Why is it this fast, what limits it, which configuration is optimal,
> and what quality do I get for that performance?
