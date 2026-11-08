# Methodology

This page tells how `core_runner.py` takes each measurement. [results-schema.md](results-schema.md) lists the output fields. [model-config.md](model-config.md) lists the keys.

## Run sequence

1. The runner copies the model config to the run dir and writes `environment.json`.
2. In managed mode, it starts the server and writes `resolved_server.json`.
3. It waits for readiness (`ModelClient.wait_until_ready`):
   - It polls `GET /v1/models` every `ready_poll_s` seconds until the server returns HTTP 200.
   - It then sends a chat completion `"hi"` with `max_tokens` 1 until the server returns HTTP 200.
4. It starts the GPU monitor.
5. It runs the benchmarks in this order: latency, deep context, TTFT breakdown, decode, reasoning, concurrency, prefill scaling, FLOP analysis, tool calling, speculative decoding comparison.
6. It stops the GPU monitor and the managed server, and writes `summary.json`.

The readiness request in step 3 is the only warm-up for the whole run. All benchmarks send one request at a time, except the concurrency benchmark.

## Client-side TTFT

- `ModelClient._execute_request` records `start = time.time()` before it sends the HTTP request.
- The first-token time is the arrival time of the first stream event with text. In chat mode, the text is in `delta.reasoning` or `delta.content`. In completions mode, the client reads `choices[0].text`.
- TTFT is the first-token time minus `start`. It includes the HTTP connection, the server queue, and the prefill. It also includes the first decode step and the return trip of the first event.
- A reasoning model streams its thinking text in `delta.reasoning`. The first thinking token stops the TTFT clock. TTFT does not wait for the answer text.
- When no event with text arrives, TTFT is NaN. The latency and deep context benchmarks leave out that request. The decode benchmark writes `null`. The other benchmarks keep the NaN value in their statistics.

## Fixed values of each benchmark

These values have no model config key.

| Benchmark | Prompt | `max_tokens` | Requests per setting | Cache salt | Warm-up and pauses |
|---|---|---|---|---|---|
| Latency | `build_prompt_of_length(n)` | 8 | `latency_repeats` | new UUID per request | none |
| Deep context | `build_prompt_of_length(n)` | 64 | `context_repeats` | new UUID per request | none |
| TTFT breakdown | `"hello world "` × `n // 2` | 1 | `ttft_repetitions` | new UUID per request | 0.5 s pause between requests |
| Decode | `build_prompt_of_length(128)` | each `decode_lengths` value | 1 | none | none |
| Reasoning | `reasoning_prompts` | 1024 | 1 | none | none |
| Concurrency | `build_distinct_prompts`: a distinct prompt per request, calibrated to 256 tokens | `concurrency_max_tokens` | `concurrency_requests_per_level` | new UUID per request, when the server accepts `cache_salt` | none |
| Prefill scaling | document text, calibrated to the exact length | 1 | `prefill_repetitions` | new UUID per request | one warm-up request per length, then 0.5 s pauses |
| Tool calling | the task prompts | `tool_calling_max_tokens` | 1 | new UUID per request | 0.5 s pause between tasks |

All requests use temperature 0, except the concurrency requests, which use `concurrency_temperature`.

`build_prompt_of_length` repeats creative-writing text and cuts it to `n` tokens of tiktoken `cl100k_base`. The server counts tokens with the model tokenizer and adds the chat template, so its count differs from `n`.

## Repeats and percentiles

- `_percentile(data, pct)` sorts the values. It interpolates in a straight line between the two nearest ranks, at rank `(n − 1) × pct / 100`.
- `_stat_summary` uses the same rule for p95. It also gives the mean, the median, the minimum, and the maximum.
- With 10 requests, p95 lies between the 9th and the 10th sorted value, and p99 is near the maximum. With 5 requests, p95 lies between the 4th and the 5th value.
- The runner takes one sample set per setting in each run. It does not repeat a whole run, and it reports no confidence intervals.

## Cache isolation

- `ModelClient.generate(..., cache_salt=...)` puts `cache_salt` at the top level of the request body. vLLM then does not share prefix-cache blocks between requests with different salts.
- When the server returns HTTP 400, the client sends the request again with ` salt=<value>` added to the prompt text. This fallback changes the prompt and its token count a little.
- The TTFT breakdown and prefill benchmarks call `preflight_cache_salt()` once. It sends one short request with a `cache_salt`. The result goes to `cache_isolation_method`: `cache_salt` when the server accepts the field, else `text_salt`.
- The prefill benchmark records `cached_tokens` for each request. A value of 0 shows that the request missed the prefix cache.
- The decode and reasoning benchmarks send no salt. With prefix caching on, a repeated prompt can hit the cache.
- The concurrency benchmark sends a salt only when `preflight_cache_salt()` returns `True`. Then `cache_isolation_method` is `cache_salt`. Else the benchmark sends no salt, and the distinct prompt text isolates the requests. `cache_isolation_method` then is `distinct_text`.

## TTFT breakdown

- The server must run with `--enable-per-request-metrics`. vLLM then adds a `metrics` object to the last stream chunk.
- Scheduler delay is the server queue time (`queue_time_ms`).
- Post-queue time is the server TTFT minus the queue time. It covers the prefill and the first decode step.
- vLLM does not report a prefill time, so `prefill_ms` and `first_decode_ms` stay `null`.
- The prompt is `"hello world "` repeated, so the token count is approximate. `actual_tokens` holds the server count.
- After an error whose text contains `memory` or `oom`, the benchmark skips all later lengths in the list.

## Prefill scaling

- `prepare_exact_prompt` builds fixed document text. It counts tokens with the server `/tokenize` endpoint and the chat template. A binary search finds a text prefix with exactly the target token count.
- When no exact prefix exists, it uses the nearest prefix below the target. `prompt_length_tolerance` then is above 0.
- Effective prefill rate is `prompt_tokens / client TTFT`.
- Engine prefill rate uses the server prefill time when the server reports one. vLLM does not, so the rate is `prompt_tokens / (server TTFT − queue time)`. It includes the first decode step.
- After an out-of-memory error or a lost connection, the benchmark skips all later lengths in the list.
- Before this benchmark, the runner records an idle GPU baseline of `int(5 / monitor_interval_s)` samples.

## Concurrency

The benchmark holds these values constant at each level: the prompt length, `max_tokens`, the temperature, and the total number of requests (`concurrency_requests_per_level`). Only the number of worker threads changes. It is equal to the level.

Each request sends its own prompt (`prompt_mode: "distinct"`):

- `build_source_text(k)` starts prompt `k` with a subject, a setting, and a style. The subject changes with each `k`. So the text differs from the first words, and the prefix cache cannot share blocks of the prompt text. Only the chat template prefix stays the same.
- A shuffle with the fixed seed `concurrency-prompt-<k>` orders the body sentences. So the prompts are the same in each run.
- Prompt `k` goes to request `k % concurrency_requests_per_level` of the level at position `k // concurrency_requests_per_level`. No level repeats a prompt of an earlier level. So speculative methods such as suffix decoding cannot learn from earlier outputs of the same prompt.
- The benchmark builds all prompts before the first level. The build time is not part of the wall time.
- When the server `/tokenize` endpoint works, `calibrate_prompt` cuts each prompt to exactly 256 server tokens, chat template included. `prompt_length_method` then is `server_tokenize`.
- When `/tokenize` fails, the client count (`count_tokens`) sets the length. `prompt_length_method` then is `client_estimate`. The server counts can then differ between requests. A check with the Qwen3.8-27B tokenizer gave server counts from 307 to 310 tokens for 80 prompts of 256 cl100k tokens. Read `individual_requests[].prompt_tokens` for the server counts.
- Runs before this change sent one prompt, `build_prompt_of_length(256)`, to every request. Their results have no `prompt_mode`. With prefix caching on, their throughput can be too high and their TTFT can be too low. Do not compare them with distinct-prompt runs.

- Wall time runs from the first submit to the last completion. Aggregate throughput is the output tokens of all successful requests divided by the wall time.
- A request can stop at the end-of-sequence token before `max_tokens`. So output lengths can differ.
- When `concurrency_requests_per_level` is below a level, fewer requests than workers run at that level. The runner prints a warning at the start.

## Speculative decoding comparison (`--compare-spec`)

The comparison needs managed mode and `speculative_config`. It runs after all other benchmarks:

1. The runner stops the main server.
2. It starts the server with `server.command` plus `--speculative-config <speculative_config>`. It waits for readiness, runs the decode benchmark over `decode_lengths`, and stops the server.
3. It starts the server with `server.command` only, and runs the decode benchmark again.
4. It compares `tok_per_sec_avg` and `ttft_s` of the two variants for each length.

- Each variant sends one request per length.
- The GPU monitor runs during both variants. Their samples are part of the run totals.
- Caution: the runner does not remove a `--speculative-config` flag that `server.command` already contains. Both variants then use speculative decoding.
- With speculative decoding, one stream event can carry several tokens. Compare `tok_per_sec_avg` only.

## GPU samples

- `GpuMonitor` starts after the endpoint is ready, and stops at the end of the run. It does not cover the model load.
- A background thread calls `nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw`. It reads only the first GPU line. Then it waits `monitor_interval_s`.
- The real sample period is longer than `monitor_interval_s`, because the `nvidia-smi` call also takes time.
- Energy is mean power × number of samples × `monitor_interval_s`. The code assumes that each sample covers one interval.
- The TTFT breakdown and prefill benchmarks open a named window for each length. A window collects the samples taken while it is open. A short window can have no sample.
- On the DGX Spark, `nvidia-smi` reports `[N/A]` for `memory.used` and `memory.total`. `memory_guard.py` records host memory from `/proc/meminfo` instead. See [dgx-spark.md](dgx-spark.md).

## FLOP analysis

The FLOP analysis is an estimate from the model architecture. It sends no timed requests and measures nothing.

- It computes the FLOPs of one decode step at each context length in `flop_analysis_lengths`.
- It computes the FLOPs of one prefill request at each of these lengths.
- It counts 2 FLOPs per multiply-accumulate. The totals leave out the operations in `omitted_non_matmul`, for example norms and RoPE.
- Linear-attention prefill uses the 64-token chunked algorithm.

`load_architecture_config` tries these config sources in order. The first complete config that passes the dimension check wins.

1. A local file: `architecture_config_path`, or a local path in the model reference.
2. Hugging Face `AutoConfig.from_pretrained`. It runs only when no complete local config exists.
3. The `config` field of the first model in the server `/v1/models` response, if the server sends one.

Caution: the Hugging Face lookup passes `trust_remote_code=True`. It can run Python code from the model repository.

The estimator supports only the Qwen3.6-35B-A3B dimensions (`_FIXED_DIMENSIONS`) with a MoE FFN, full attention, and Gated DeltaNet linear attention. Other models get `estimate_status: unsupported` and a list of `warnings`. `estimate_status` is `exact_from_config` when `model_type` starts with `qwen3_5_moe` or `qwen3_6_moe`, else `approximate`.

## Limits

- One machine: the results in this repo come from one DGX Spark. `environment.json` records no CPU, OS, or host memory data.
- Client-side timing: TTFT and the rates use the client wall clock. They include HTTP and network time.
- Small samples: the default repeats are 10 or 5. The decode and reasoning benchmarks send one request per setting. The runner has no seed option and no run-level repeats.
- No model revision: `resolved_server.json` records the server command, which names the model repository. `environment.json` records the GPU, driver, CUDA, torch, and vLLM versions. No file records a model revision or a weight checksum.
- Version source: `environment.json` reads the torch and vLLM versions from the Python that runs `core_runner.py`. A managed server from another environment can use other versions.
- External mode: the runner writes no `resolved_server.json` and no `vllm.log`. The run dir does not record the server settings.
- Token estimates: reasoning token counts use tiktoken `cl100k_base`, not the model tokenizer.
- FLOP analysis: it supports one model architecture.
