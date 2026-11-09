# How a run works

Read this page first. It tells what one run does, step by step, and which file does each step. [methodology.md](methodology.md) tells how each benchmark measures. The [code layout table](../README.md#code-layout) in the README lists the files.

## What a run is

- A run takes one model config, a YAML file in `models/`.
- A run writes one run dir: `results/<name>/<YYYYMMDD_HHMMSS>/`. `<name>` is the `name` key of the model config. `--results-dir` changes the `results/` folder.
- This command starts a run: `python3 core_runner.py --model models/<file>.yml`.
- `core_runner.py` holds the command line, the run state (`Run`), one `step_*` function for each benchmark, and `run_benchmarks()`. `run_benchmarks()` sets the step order and the skip rules.
- Each benchmark has a module in `benchmarks/`. The shared code is in `harness/`.

## Life cycle of a run

The steps below follow `main()` and `run_benchmarks()` in `core_runner.py`.

### Set-up

1. **Load the config.** `load_model_config()` reads the YAML file. `make_run_dir()` makes the run dir. The runner prints `[core_runner] writing results to <run dir>`. Output: `model_config.yml`, an exact copy of the config file.
2. **Record the environment.** `collect_environment()` in `harness/environment.py` reads the GPU name, the driver, the CUDA toolkit, and the torch and vLLM versions. Output: `environment.json`.
3. **Start vLLM (managed mode only).** `make_managed_server()` makes a `VllmServer` from `vllm_server.py`. `VllmServer.start()` runs `server.command` with `server.environment` added. It refuses to start when the port is already in use. Output: `resolved_server.json` and `vllm.log`.
   - In external mode, the runner does not start or stop a server. It writes no `resolved_server.json` and no `vllm.log`.
   - `--server-mode` overrides `server.mode`. When neither sets a mode, the mode is `external`.
4. **Wait for readiness.** `ModelClient.wait_until_ready()` in `harness/client.py` does two checks:
   1. It polls `GET /v1/models` until the server returns HTTP 200.
   2. It sends the prompt `"hi"` with `max_tokens` 1 until the server returns HTTP 200.
   - The time limit is `server.startup_timeout_s`, else `ready_timeout_s`, else 600 s. The wait between two probes is `ready_poll_s` (default 3 s).
   - In managed mode, the run stops when the vLLM process exits. The error holds the last 30 lines of `vllm.log`.
   - The `"hi"` request is the only warm-up of the run. Only prefill scaling adds its own warm-up requests.
5. **Start the GPU monitor.** `GpuMonitor` in `harness/telemetry.py` calls `nvidia-smi` in a background thread every `monitor_interval_s` (default 1 s). The samples thus do not include the first model load.

### Benchmarks

6. **Run the benchmarks.** `run_benchmarks()` calls the steps in the order of the next table. Each step writes `<key>.json` to the run dir and adds `<key>` to the summary (`Run.record()`).

| # | Benchmark | Step function | Module in `benchmarks/` | Runs when | Output file |
|---|---|---|---|---|---|
| 1 | Latency sweep | `step_latency` | `latency.py` | no `--skip-latency` | `latency.json` |
| 2 | Deep context | `step_deep_context` | `latency.py` | `context_lengths` is set and no `--skip-deep-context` | `deep_context.json` |
| 3 | TTFT breakdown | `step_ttft_breakdown` | `ttft_breakdown.py` | no `--skip-ttft` | `ttft_breakdown.json` |
| 4 | Decode | `step_decode` | `decode.py` | no `--skip-decode` | `decode.json` |
| 5 | Reasoning | `step_reasoning` | `reasoning.py` | no `--skip-reasoning` | `reasoning.json` |
| 6 | Concurrency | `step_concurrency` | `concurrency.py` | no `--skip-concurrency` | `concurrency.json` |
| 7 | Prefill scaling | `step_prefill` | `prefill.py` | no `--skip-prefill` | `prefill_scaling.json` |
| 8 | FLOP analysis | `step_flops` | `architecture_flops.py` | no `--skip-flops` | `flops_analysis.json` |
| 9 | Tool calling | `step_tool_calling` | `tool_calling.py` | no `--skip-tool-calling` | `tool_calling.json` |
| 10 | Spec comparison | `step_spec_comparison` | `decode.py` | `--compare-spec` | `spec_comparison.json` and five more files |

The next table gives the default parameters in the code. A model config can change most of them. [model-config.md](model-config.md) lists the keys. Prompt lengths are in tokens.

| # | Prompt | `max_tokens` | Requests | Cache salt | Pauses |
|---|---|---|---|---|---|
| 1 | Creative text, 32, 128, 512, 2048, 8192, 16384 tokens | 8 | 10 per length | new for each request | none |
| 2 | Creative text, the `context_lengths` values (no default) | 64 (`deep_context_output_tokens`) | 5 per length | new for each request | none |
| 3 | `"hello world "` repeated, about 128, 512, 2048, 8192, 32768 tokens | 1 | 10 per length | new for each request | 0.5 s between requests |
| 4 | Creative text, 128 tokens | 512, 1024, 2048 | 1 per `max_tokens` value | none | none |
| 5 | Two built-in prompts | 1024 | 1 per prompt | none | none |
| 6 | A distinct prompt of 256 tokens for each request | 256 | 5 per level, at 1, 2, 4, 8, and 16 workers | new for each request, when the server accepts `cache_salt` | none |
| 7 | Document text, exactly 512, 2048, 8192 tokens | 1 | 1 warm-up, then 5 per length | new for each request | 5 s idle baseline (`idle_baseline_s`) before the step, 0.5 s between requests |
| 8 | No timed requests. Estimates at 512, 2048, 8192, 32768 tokens | – | – | – | – |
| 9 | The task prompts of `datasets/tool_calling_tasks.yaml` (85 tasks) | 4096 | 1 per task | new for each request | 0.5 s between tasks |
| 10 | As decode (4) | As decode (4) | 1 per length, for each of two servers | none | none |

- The Qwen3.8-27B config sets 16 requests per concurrency level, six reasoning prompts, and `context_lengths` of 32768 and 65536.
- With `tool_calling_task_set: lite`, the tool-calling step runs at most 8 tasks. The read timeout of each tool-calling request is 600 s.
- All requests use temperature 0, except the concurrency requests, which use `concurrency_temperature` (default 0).

#### Idle baseline before prefill scaling

`step_prefill` records `int(idle_baseline_s / monitor_interval_s)` idle samples, one each `monitor_interval_s`. At the defaults (5 s and 1 s), this is 5 samples in about 5 s. When `monitor_interval_s` is 0 or less, the step stops with a `ValueError`. The prefill GPU windows use this baseline for `energy_above_idle_wh`. `--skip-prefill` also skips the baseline.

#### Errors in a step

- An unhandled error in a step stops the run. The `finally` block then writes `summary.json` with `status: failed`.
- Tool calling is the exception. `step_tool_calling` catches the error and prints `[core_runner] warning: tool-calling benchmark failed: ...`. The summary gets `tool_calling: {status: failed, error: ...}`. The run dir gets no `tool_calling.json`, and the next step runs.
- A failed tool-calling request, for example a timeout, does not raise an error. The task counts as failed, and `failure_modes` counts the cause.
- Deep context records a failed length with `status: oom` or `request_error` and goes on to the next length.
- TTFT breakdown and prefill scaling skip all later lengths after an out-of-memory error (`skipped_after_oom`). Prefill scaling also skips them after a lost connection.

#### Spec comparison (`--compare-spec`)

CAUTION: The spec comparison runs last. Without managed mode or `speculative_config`, it raises `ValueError` after all other benchmarks, and the run status becomes `failed`.

1. `step_spec_comparison` stops the main server and writes `resolved_server.json` again.
2. It starts vLLM with `server.command` plus `--speculative-config <speculative_config>`. It waits for readiness, runs `run_decode_speed()` over `decode_lengths`, and stops the server. Output: `vllm_spec_enabled.log`, `resolved_server_spec_enabled.json`, `spec_enabled.json`.
3. It does the same with `server.command` only. Output: `vllm_spec_disabled.log`, `resolved_server_spec_disabled.json`, `spec_disabled.json`.
4. `compare_spec_decode_results()` compares `tok_per_sec_avg` and `ttft_s` for each length. Output: `spec_comparison.json`.

The GPU monitor runs during both variants. Their samples include the two model loads.

### End of the run

7. **Clean up in the `finally` block.** This block runs after success, after an error, and after SIGTERM:
   1. It stops the GPU monitor. Output: `gpu_samples.csv` and the `gpu` block of the summary.
   2. When the decode step ran, `decode_energy_summary()` computes the GPU energy in the decode window. Output: `decode_energy` in the summary.
   3. In managed mode, it stops vLLM. `VllmServer.stop()` sends SIGTERM to the process group, and SIGKILL after `server.shutdown_timeout_s` (default 30 s). Output: `resolved_server.json` again, with the stop time and the exit code.
   4. It writes `summary.json`. `status` is `completed`, or `failed` with the `error` text.

The runner turns SIGTERM into a `KeyboardInterrupt`. Thus a stop from `memory_guard.py` also goes through this block. The status then is `failed`.

The published run dirs do not hold `*.log` files or `gpu_samples.csv`. `.gitignore` leaves them out.

### The memory guard

`memory_guard.py` wraps the whole `core_runner.py` command. Use it for each large run on a DGX Spark. [dgx-spark.md](dgx-spark.md#memory-guard) gives the default limits and an example command.

1. The guard reads `MemAvailable` from `/proc/meminfo`. It refuses to start below `--min-start-gib`.
2. It starts the command and prints its output. It finds the run dir in the line `writing results to <run dir>`.
3. It writes one memory sample every `--interval-s` seconds to a CSV file. Without `--csv`, the file is a temporary file.
4. When `MemAvailable` stays below `--floor-gib` for `--grace-samples` samples in a row, it sends SIGTERM to the command. It waits up to 180 s, then kills the command.
5. At the end, it moves the temporary file to `memory_samples.csv` in the run dir. With `--csv`, it copies the file and keeps the original. It also writes `memory_guard.json` there, with `stopped_for_memory` and `min_available_gib`.
6. It exits with code 3 when it refuses to start or when it stops the run.

`batch_run.py` runs `core_runner.py` without the memory guard.

### The dashboard

`dashboard.py` reads every `results/*/*/summary.json` and writes `reports/dashboard.html`. It also reads `environment.json`, `resolved_server.json`, `tool_calling.json`, `memory_samples.csv`, and `memory_guard.json`. [dashboard.md](dashboard.md) tells how to build and read the page.

## Cache isolation

vLLM can keep a prefix cache. A repeated prompt then skips most of its prefill, and TTFT shows a cache hit. To measure a full prefill, the harness sends a new random `cache_salt` with each request. vLLM does not share cached blocks between requests with different salts.

- `ModelClient.generate()` puts `cache_salt` at the top level of the request body.
- When the server rejects the field with HTTP 400, the client adds ` salt=<value>` to the prompt text instead.
- These steps salt each request: latency sweep, deep context, TTFT breakdown, prefill scaling, and tool calling.
- The concurrency step salts each request only when the server accepts `cache_salt`. Else the distinct prompts keep the requests apart.
- These requests have no salt: the readiness request, decode, reasoning, and the spec comparison. With prefix caching on, their TTFT can show a cache hit.
- The prefill warm-up request uses the fixed salt `warmup_<length>`.

The Qwen3.8-27B config also turns prefix caching off with `--no-enable-prefix-caching`. [methodology.md](methodology.md#cache-isolation) gives the details.

## Glossary

| Term | Meaning |
|---|---|
| TTFT (client) | Time to first token, from the client clock. It starts just before the HTTP request and stops at the first stream event with text. It includes the HTTP time, the queue time, the prefill, and the first decode step. The first reasoning token stops the clock. |
| TTFT (server), post-queue time | vLLM `time_to_first_token_ms`: from the first scheduling of the request to the first token. It does not include the queue time. `ttft_breakdown.json` stores it as `post_queue_ms`. vLLM sends it only with `--enable-per-request-metrics`. |
| Queue time, scheduler delay | vLLM `queue_time_ms`: from the arrival of the request in the queue to its first scheduling. |
| Prefill | The forward pass over all prompt tokens. It fills the KV cache and comes before the first output token. |
| Decode | The generation of the output tokens, one step at a time, after the prefill. The decode rate is in tokens per second. |
| Chunked prefill | `--enable-chunked-prefill`. vLLM splits a long prefill into chunks of at most `--max-num-batched-tokens` tokens. |
| KV cache | The memory that holds the attention keys and values of the tokens of each request. |
| KV pin | `--kv-cache-memory-bytes`, which sets the KV cache size directly. The Qwen3.8-27B config uses `16G`. See [dgx-spark.md](dgx-spark.md#kv-cache-size). |
| Prefix caching | vLLM reuses the KV cache blocks of an earlier request with the same prompt start. `--enable-prefix-caching` and `--no-enable-prefix-caching` set it. |
| `cache_salt` | A vLLM request field. Requests with different salts do not share prefix-cache blocks. |
| NVFP4 | The 4-bit floating-point weight format of the published checkpoints, for example `nvidia/Qwen3.8-27B-NVFP4`. |
| MTP | Multi-token prediction. The checkpoint has an extra layer that proposes the next tokens. The configs use it with `{"method":"mtp","num_speculative_tokens":3}` (MTP K=3). |
| Speculative decoding | A draft method, for example MTP, proposes several tokens. The model checks them in one forward pass and keeps the accepted tokens. One stream event can then carry several tokens. |
| Acceptance rate | The fraction of the proposed tokens that the model keeps. vLLM prints it in its log as `SpecDecoding metrics`. The harness does not record it. |
| Reasoning parser | `--reasoning-parser qwen3`. vLLM sends the thinking text in `delta.reasoning` and the answer in `delta.content`. |
| Unified memory | The CPU and the GPU of the DGX Spark share 121 GB of memory. `nvidia-smi` reports the GPU memory as `[N/A]`. |
| Managed mode, external mode | In managed mode, the runner starts and stops vLLM from `server.command`. In external mode, it uses a server that already runs. |
| Composite score | The tool-calling score from 0 to 6: the sum of six components with weight 1. See [tool-calling.md](tool-calling.md#category-scores-and-composite). |
| Run dir | The output folder of one run, `results/<name>/<YYYYMMDD_HHMMSS>/`. |

## Where to go next

- [methodology.md](methodology.md): how each benchmark measures.
- [results-schema.md](results-schema.md): each file and field in a run dir.
- [model-config.md](model-config.md): each key of a model config.
- [comparability.md](comparability.md): which published results you can compare.
- [adding-a-benchmark.md](adding-a-benchmark.md): how to add a benchmark.
- [dgx-spark.md](dgx-spark.md): DGX Spark setup and the memory guard.
