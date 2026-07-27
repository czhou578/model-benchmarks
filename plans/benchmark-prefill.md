You already have pieces of this benchmark, but not a trustworthy dedicated prefill curve. The current latency and deep-context routines reuse identical, prefix-sharing prompts while prefix caching is enabled, so later measurements can be cache hits rather than full prefills.

No code or files were changed.

## Recommended design

Create a standalone `prefill_scaling` benchmark covering:

- 512
- 2K
- 8K
- 32K
- 64K
- 128K
- 256K
- 512K

Keep it separate from the latency sweep because it needs cold-cache isolation, per-length GPU telemetry, configuration-limit handling, and more detailed failure reporting.

### Metric definitions

Report two kinds of throughput clearly:

- `effective_prefill_tps = actual_prompt_tokens / client_ttft`
  - End-to-end and comparable across serving stacks.
  - Includes HTTP, tokenization, queueing, scheduling, prefill, and first-token delivery.
- `engine_prefill_tps = newly_computed_prompt_tokens / server_prefill_time`
  - The preferred measure of actual model prefill speed.
  - Only available when the running vLLM version exposes the required request metrics.

Current vLLM metrics distinguish queue time, prefill time, TTFT, and newly computed KV tokens. TTFT alone is not pure prefill time. [vLLM metrics documentation](https://docs.vllm.ai/en/v0.14.0/design/metrics/)

## Implementation plan

### 1. Add a dedicated benchmark module [✓]

Create `benchmarks/prefill.py` with responsibility for:

- Prompt preparation and exact-length calibration.
- Cold-cache request execution.
- Warmups and measured repetitions.
- Per-request result capture.
- Per-length aggregation.
- OOM and unsupported-length classification.

Keep general HTTP functionality and GPU monitoring in [core_runner.py](/home/colin-spark/Projects/model-benchmarks/core_runner.py:240).

### 2. Generate exact model-token lengths [✓]

The existing `build_prompt_of_length()` uses `cl100k_base` or a word-count estimate. That is useful for rough sizing but cannot guarantee Qwen token counts.

For each target:

1. Generate a candidate document.
2. Ask the server’s tokenizer endpoint for the model-specific count.
3. Adjust or truncate until the rendered request is at or just below the target.
4. Save both `requested_prompt_tokens` and `actual_prompt_tokens`.

vLLM provides `/tokenize`; account for the chat template when calibrating chat requests. [vLLM tokenizer API](https://docs.vllm.ai/en/v0.21.0/serving/openai_compatible_server/)

Prefer a document-like workload—several differently worded passages—over repeated copies of the current creative-writing instructions. Repetition can create unrealistic attention and compression behavior.

### 3. Guarantee a cold prefill [✓]

Give every measured request a unique `cache_salt`. This preserves the prompt itself while preventing prefix-cache reuse between repetitions and lengths. vLLM documents `cache_salt` specifically as a prefix-cache isolation mechanism. [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.15.0/design/prefix_caching/)

Preflight the feature because external or older servers may reject it. Fallback order:

1. Unique `cache_salt`.
2. Managed server variant with prefix caching disabled.
3. Unique text at the beginning of every prompt, followed by length recalibration.

Do not merely append a nonce: most of the earlier prefix could still be reusable.

### 4. Minimize decode contamination [✓]

Request one output token per trial. That is enough to observe TTFT while keeping the benchmark focused on prefill.

Before measurement:

- Wait for the endpoint.
- Run one short, excluded warmup.
- Confirm server metrics and token usage are present.
- Record idle GPU power and memory for several seconds.

For each length:

- Run one excluded length-specific warmup.
- Run 5 measured cold requests initially.
- Leave a short stabilization gap between requests.
- Stop progressing to larger sizes if the server dies or a genuine memory boundary is reached.

### 5. Capture raw request data [✓]

Store every repetition, not only averages:

- Requested and actual prompt tokens.
- Newly computed versus cached tokens, when available.
- Client TTFT.
- Server TTFT.
- Queue time.
- Server prefill time.
- Effective and engine prefill TPS.
- Start/end monotonic timestamps.
- HTTP status and structured error.
- Whether token counts and server timings were exact.
- Cache-isolation method used.

This lets you recalculate summaries later and identify outliers such as the existing 32-token latency result.

### 6. Add per-length GPU telemetry [✓]

The current `GpuMonitor` produces one summary for the entire suite, so it cannot attribute power, memory, or energy to a prompt length.

Extend it to support timestamped benchmark windows and add:

- GPU utilization average and peak.
- GPU memory average and peak.
- Memory delta from the pre-benchmark idle baseline.
- Power average and peak.
- Gross energy per request.
- Energy per input token.
- Incremental energy above idle, if the idle baseline is stable.

For short prompts, a 1 Hz sampler is too coarse. Either sample closer to 100–200 ms or measure all repetitions for a length as one telemetry window and divide aggregate energy by the number of requests.

Also record vLLM KV-cache utilization when available. `nvidia-smi memory.used` alone may remain nearly constant because vLLM commonly reserves its KV-cache allocation at startup.

### 7. Define the output schema [✓]

Output written to `prefill_scaling.json` with the following structure:

- `config`: Benchmark version, definition, server/cache settings, sampling config, prompt workload identifier, cache isolation method + preflight result, start/end timestamps.
- `per_length.<length>`: Per-length record containing:
  - `status`: "success", "oom", "server_unavailable", "skipped_after_oom", "request_error"
  - `requested_tokens`, `actual_tokens`, `prompt_length_tolerance`
  - `n_requests`, `n_success`
  - `per_request[]`: Each repetition with all raw fields (tokens, TTFT, cached tokens, server timing, queue time, prefill time, cache isolation, timestamps, errors)
  - `aggregated`: TTFT (avg, median, p95, min, max), effective prefill TPS, engine prefill TPS, GPU telemetry (util, memory, power, energy, energy per token)

Add a compact reference to this object in `summary.json`; avoid duplicating all raw samples there.

### 8. Integrate configuration and orchestration [✓]

Implemented in `core_runner.py`:

- YAML config keys: `prefill_target_lengths` (default `[512, 2048, 8192, 32768, 65536]`), `prefill_repetitions` (default 5).
- GPU idle baseline captured before benchmark (5 seconds at the configured `monitor_interval_s`).
- `--skip-prefill` CLI flag present (consistent with existing `--skip-*` flags).
- Compact summary written to `summary.json`: per-length status and `n_success` only.
- The benchmark runs independently of `prompt_lengths`, `context_lengths`, and `concurrency_levels`.

Not yet done: treating `prefill_tps_avg` in `latency.json` / `deep_context.json` as legacy — three competing prefill implementations remain.

### 10. Validate in stages [ ]

1. Run a smoke test with 512 and 2K, 2 repetitions each:
   `prefill_target_lengths: [512, 2048]` and `prefill_repetitions: 2` in the YAML, then run the benchmark.
2. Verify the output: `prefill_scaling.json` exists, has a `config` block and a `per_length` block with entries for "512" and "2048", each with `"n_success": 2`.

## Acceptance criteria

Consider the benchmark complete when:

- The benchmark runs without crashing and produces `prefill_scaling.json`.
- Every successful request uses the intended model-token length within a documented tolerance.
- Prefix-cache reuse is demonstrably zero (all `cached_tokens` are 0).
- Raw per-request data and aggregate statistics are retained.

## Why do we calibrate_prompt function?

It takes a long text document and a target token count, and finds the exact character index where the document, when tokenized, contains precisely target_tokens tokens.

The algorithm: binary search + local scan

1. Binary search over the document's character positions. For each midpoint, it calls client.tokenize_prompt(source[:midpoint]).count to learn the token count. If it's exact, great — return immediately. If it's too low, the answer is somewhere to the right; if too high, to the left. This narrows down to a small window near the boundary.
2. Local scan around that window (±boundary_scan_chars, default 64). Why? Because tokenizers are discrete — the boundary between "N tokens" and "N+1 tokens" might not fall exactly at the binary-search midpoint. A small linear scan catches the exact character offset.

Why we need it (and what breaks if you remove it)

Without this function, you'd be stuck with approximate lengths. Consider:

- You generate 256KB of text, expecting ~32K tokens. The model tokenizer might render it as 34K.
- Without calibration, you can't know whether your "32K benchmark point" was actually 32K, 34K, or 29K.
- That noise defeats the purpose: the whole point of the benchmark is to cleanly attmpt length, not to a mixture of lengths with unknown error bars.

In short: if you removed calibrate_prompt, you'd need an equally precise way to find the character boundary that yields exactly target_tokens. There isn't one. The tokenizer is a black box — you can't inverse it. You have to search.

The one constraint to be aware of

The function only trims — it finds a prefix of the document that is at or below targ prepare_exact_prompt (line 144) generates progressively larger documents until one is big enough to trim down to target. This avoids synthetic padding that could look unrealistic.