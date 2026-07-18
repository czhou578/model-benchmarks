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

### 1. Add a dedicated benchmark module

Create `benchmarks/prefill.py` with responsibility for:

- Prompt preparation and exact-length calibration.
- Cold-cache request execution.
- Warmups and measured repetitions.
- Per-request result capture.
- Per-length aggregation.
- OOM and unsupported-length classification.

Keep general HTTP functionality and GPU monitoring in [core_runner.py](/home/colin-spark/Projects/model-benchmarks/core_runner.py:240).

### 2. Generate exact model-token lengths

The existing `build_prompt_of_length()` uses `cl100k_base` or a word-count estimate. That is useful for rough sizing but cannot guarantee Qwen token counts.

For each target:

1. Generate a candidate document.
2. Ask the server’s tokenizer endpoint for the model-specific count.
3. Adjust or truncate until the rendered request is at or just below the target.
4. Save both `requested_prompt_tokens` and `actual_prompt_tokens`.

vLLM provides `/tokenize`; account for the chat template when calibrating chat requests. [vLLM tokenizer API](https://docs.vllm.ai/en/v0.21.0/serving/openai_compatible_server/)

Prefer a document-like workload—several differently worded passages—over repeated copies of the current creative-writing instructions. Repetition can create unrealistic attention and compression behavior.

### 3. Guarantee a cold prefill

Give every measured request a unique `cache_salt`. This preserves the prompt itself while preventing prefix-cache reuse between repetitions and lengths. vLLM documents `cache_salt` specifically as a prefix-cache isolation mechanism. [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.15.0/design/prefix_caching/)

Preflight the feature because external or older servers may reject it. Fallback order:

1. Unique `cache_salt`.
2. Managed server variant with prefix caching disabled.
3. Unique text at the beginning of every prompt, followed by length recalibration.

Do not merely append a nonce: most of the earlier prefix could still be reusable.

### 4. Minimize decode contamination

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

### 5. Capture raw request data

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

### 6. Add per-length GPU telemetry

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

### 7. Handle configured limits before OOM

The managed NVIDIA configurations currently specify `--max-model-len 262144`, so:

- 512K is unsupported without restarting the server with a larger limit.
- A full 256K request may also exceed the limit after chat-template and output-token overhead.

Parse the resolved server configuration and classify lengths as:

- `success`
- `unsupported_model_limit`
- `oom`
- `server_unavailable`
- `request_error`
- `skipped_after_oom`

Do not classify every connection error as OOM, as `run_deep_context()` currently does.

### 8. Define the output schema

Write `prefill_scaling.json` with metadata plus per-length records:

- Benchmark version and definition.
- Server/cache settings.
- Sampling configuration.
- Prompt workload identifier.
- Raw repetitions.
- Aggregated TTFT: mean, median, P95, standard deviation.
- Effective prefill TPS: mean, median, P95.
- Engine prefill TPS when supported.
- GPU utilization, memory, power, and energy.
- Status and failure details.

Add a compact reference to this object in `summary.json`; avoid duplicating all raw samples there.

### 9. Integrate configuration and orchestration

Add a dedicated configuration section to each model YAML, conceptually containing:

- Enabled flag.
- Target lengths.
- Repetitions and warmups.
- Output token count.
- Cache isolation policy.
- Telemetry sampling interval.
- Stop-after-OOM behavior.

Add a `--skip-prefill` runner option consistent with the existing CLI. The benchmark should execute independently of `prompt_lengths` and `context_lengths`.

Once validated, treat `prefill_tps_avg` in `latency.json` and `deep_context.json` as legacy estimates. Keeping three competing prefill implementations will make reports ambiguous.

### 10. Validate in stages

1. Unit-test exact-length calibration, unique cache salts, aggregation, telemetry-window selection, and status classification.
2. Run a smoke test at 512 and 2K with two repetitions.
3. Confirm server-reported cached tokens are zero.
4. Confirm effective TPS and engine TPS are reasonably close at concurrency one.
5. Run through 64K.
6. Attempt 128K and the safe maximum below the configured model limit.
7. Only attempt 256K/512K after adjusting `--max-model-len` and checking KV-cache capacity.
8. Repeat one curve to measure run-to-run variance.

## Acceptance criteria

Consider the benchmark complete when:

- Every successful request uses the intended model-token length within a documented tolerance.
- Prefix-cache reuse is demonstrably zero.
- Raw repetitions and aggregate statistics are retained.
- GPU and energy measurements are attributable to each length.
- Unsupported lengths differ from OOM and server crashes.
- A repeated curve produces similar median throughput.
- The roadmap is updated to clarify that the old latency/deep-context numbers were preliminary effective-prefill estimates, while `prefill_scaling.json` is the canonical curve.