# Results schema

One run writes one run dir: `<results-dir>/<name>/<YYYYMMDD_HHMMSS>/`. [model-config.md](model-config.md) tells how the runner names it. This page lists each file in the run dir and its fields. [methodology.md](methodology.md) tells how the runner takes each measurement.

## Files

| File | Written by | When |
|---|---|---|
| `model_config.yml` | `main`, a copy of the `--model` file | always |
| `environment.json` | `collect_environment` | always |
| `resolved_server.json` | `VllmServer.save_metadata` | managed mode |
| `vllm.log` | `VllmServer.start`, stdout and stderr of vLLM | managed mode |
| `latency.json` | `run_latency_sweep` | without `--skip-latency` |
| `deep_context.json` | `run_deep_context` | when `context_lengths` is set |
| `ttft_breakdown.json` | `run_ttft_breakdown` | without `--skip-ttft` |
| `decode.json` | `run_decode_speed` | without `--skip-decode` |
| `reasoning.json` | `run_reasoning_benchmark` | without `--skip-reasoning` |
| `concurrency.json` | `run_concurrency_test` | without `--skip-concurrency` |
| `prefill_scaling.json` | `run_prefill_scaling` | without `--skip-prefill` |
| `flops_analysis.json` | `run_flops_analysis` | always |
| `tool_calling.json` | `run_tool_calling_benchmark` | when the benchmark ends without an error |
| `spec_enabled.json`, `spec_disabled.json`, `spec_comparison.json` | `main`, `compare_spec_decode_results` | with `--compare-spec` |
| `resolved_server_spec_enabled.json`, `resolved_server_spec_disabled.json` | `VllmServer.save_metadata` | with `--compare-spec` |
| `vllm_spec_enabled.log`, `vllm_spec_disabled.log` | `VllmServer.start` | with `--compare-spec` |
| `gpu_samples.csv` | `GpuMonitor.stop` | when the monitor took at least one sample |
| `summary.json` | `main`, in its `finally` block | always, also after a failure |
| `memory_samples.csv`, `memory_guard.json` | `memory_guard.py` | only when `memory_guard.py` wraps the run |

## Units and known ambiguities

Read these points before you compare numbers.

- A field that ends in `_s` holds seconds. A field that ends in `_ms` holds milliseconds. The `ttft_*_s` fields are seconds.
- `tok_per_sec_*`, `*_tps*`, and `*_tok_s` fields are tokens per second. The fields in `reasoning.json` are token counts, not rates.
- Exception: in `prefill_scaling.json`, the `effective_prefill_tps` and `engine_prefill_tps` blocks hold tokens per second, but their keys end in `_s`. `prefill.py` calls `stat_summary` with the default suffix `s`.
- Client TTFT starts before the client sends the HTTP request (`start` in `ModelClient._execute_request`). In chat mode, it stops at the first stream event with text in `delta.reasoning` or `delta.content`. The `has_text` check in the same function sets this time. So the first reasoning token counts as the first token.
- `prefill_tps_avg` in `latency.json` and `deep_context.json` is the mean of `prompt_tokens / client TTFT`. It includes network time, queue time, and the first decode step. It is not an engine prefill rate.
- `prefill_scaling.json` is a curve of prefill time and prefill rate against prompt length. It is not a GPU power curve. Its `gpu` block holds the GPU data of each length window.
- In `decode.json`, `tok_per_sec_peak`, `tok_per_sec_min`, and `tok_per_sec_median` are `1 / gap` between two stream events with text (`instant_rates` in `run_decode_speed`). One event can carry more than one token, for example with speculative decoding. These three fields then count events per second. Use `tok_per_sec_avg` for the token rate.
- `output_truncated` in `decode.json` is `true` when `actual_output_tokens` differs from `requested_output_tokens`. A `true` value means that the model stopped before `max_tokens`. It does not mean that `max_tokens` cut the output.
- Old runs have wrong energy values. An old run has no `duration_s` in its `gpu` block. Do not compare old runs with newer runs. The dashboard does not show the old per-token value.
  - `energy_per_token_wh` in `summary.json` is the GPU energy of the whole run divided by the decode output tokens only. In the 2026-11-07 Qwen3.8-27B run, it is 0.0158 Wh/token. An estimate of the decode window from the power trace gives 0.0009 Wh/token. So the old value is about 17 times too high. Newer runs write `decode_energy` instead.
  - Every `energy_wh` is mean power × number of samples × `monitor_interval_s`. The real sample period is about 1.07 s, so the value is about 6% too low.
  - In a GPU window block, `idle_base_power_w` comes from the window samples, not from the idle samples. It equals `gpu_power_avg_w`, and `energy_above_idle_wh` is near 0.
  - In a prefill window, `energy_per_input_token_wh` is the window energy divided by the tokens of one prompt. The window holds the warm-up request and all repetitions. Newer runs write `window_energy_per_input_token_wh` instead.
- Token counts come from the server `usage` block. Without it, the client estimates them with tiktoken `cl100k_base`, or with 0.75 tokens per word (`count_tokens`). An `*_exact` flag tells which source applies.

## summary.json

| Field | Unit | Meaning |
|---|---|---|
| `model` | | `name` from the model config. |
| `run_dir` | | Path of the run dir. |
| `server_mode` | | `managed` or `external`. |
| `status` | | `running`, `completed`, or `failed`. |
| `error` | | Exception type and message. Present only when `status` is `failed`. |
| `latency`, `deep_context`, `ttft_breakdown`, `decode`, `reasoning`, `concurrency`, `spec_comparison` | | Full copy of the file with the same name. |
| `prefill_scaling` | | `status` and `n_success` for each length. |
| `flops_analysis` | | `status` (from `estimate_status`) and `model` (from `architecture.model`). |
| `tool_calling` | | `status` (`completed` or `failed`). When `completed`: `composite_score`, `composite_components`, `total_tasks`, `total_correct`, `categories` (names only), `failure_modes`. When `failed`: `error` (exception type and message) only. Runs before this field have no `status`. |
| `gpu` | | GPU data of the whole run. See the next table. |
| `decode_energy` | | GPU energy of the decode benchmark. See the next table. Present when the decode benchmark and the GPU monitor ran. |
| `energy_per_token_wh` | Wh/token | Old runs only. Whole-run energy / decode output tokens. Do not use it. |

`decode_energy` (`decode_energy_summary`):

| Field | Unit | Meaning |
|---|---|---|
| `window_start_unix`, `window_end_unix` | s, Unix time | Start of the first decode request and end of the last one. |
| `duration_s` | s | Length of the decode window. |
| `energy_wh` | Wh | GPU energy in the decode window. |
| `output_tokens` | tokens | Sum of `actual_output_tokens` in `decode`. |
| `energy_per_output_token_wh` | Wh/token | `energy_wh` / `output_tokens`. It includes the short prefill of each 128-token decode prompt. |

GPU data (`GpuMonitor.stop` for the whole run, `_window_summary` for one window):

| Field | Unit | Meaning |
|---|---|---|
| `samples_csv` | | Path of `gpu_samples.csv`. Whole run only. |
| `num_samples` | count | Number of samples. |
| `duration_s` | s | Whole run: first sample to last sample. Window: window start to window stop. |
| `gpu_util_avg_pct`, `gpu_util_peak_pct` | % | nvidia-smi `utilization.gpu`. |
| `gpu_mem_util_avg_pct`, `gpu_mem_util_peak_pct` | % | nvidia-smi `utilization.memory`. |
| `gpu_mem_used_avg_mib`, `gpu_mem_used_peak_mib` | MiB | nvidia-smi `memory.used`. `null` on the GB10, which reports `[N/A]`. |
| `gpu_power_avg_w`, `gpu_power_peak_w` | W | nvidia-smi `power.draw`. |
| `energy_wh` | Wh | Power integrated over the sample timestamps (`gpu_energy_wh`). See [methodology.md](methodology.md#gpu-samples). |
| `idle_base_power_w` | W | Window only, after an idle baseline. Mean power of the idle samples. |
| `energy_above_idle_wh` | Wh | Window only. `energy_wh` − `idle_base_power_w` × `duration_s`. It can be negative. |
| `window_input_tokens` | tokens | Prefill windows only. Input tokens of the warm-up request and of all successful repetitions. |
| `window_energy_per_input_token_wh` | Wh/token | Prefill windows only. `energy_wh` / `window_input_tokens`. The energy also includes the pauses between the requests. |
| `energy_per_input_token_wh` | Wh/token | Old runs only. See the ambiguities above. |

## environment.json

| Field | Meaning |
|---|---|
| `timestamp` | Start of the run, ISO 8601 in UTC. |
| `gpu_name`, `driver_version` | nvidia-smi `name` and `driver_version` of the first GPU. |
| `gpu_memory_total` | nvidia-smi `memory.total` as raw text. `[N/A]` on the GB10. |
| `cuda_version` | Last line of `nvcc --version`, the CUDA toolkit on `PATH`. |
| `torch_version`, `vllm_version` | `__version__` in the Python that runs core_runner.py. A managed server can use another Python. |

The file records no model revision and no weight checksum.

## resolved_server.json

The runner writes it after it starts a managed server, and again after it stops the server. In a `--compare-spec` run, the last write occurs before the variants start. `resolved_server_spec_*.json` has the same fields.

| Field | Unit | Meaning |
|---|---|---|
| `command`, `command_display` | | Server argument list, and the same list as one shell string. |
| `base_url`, `host`, `port` | | From `endpoint.base_url`. |
| `environment` | | `server.environment`. Names with `TOKEN`, `KEY`, `SECRET`, or `PASSWORD` show `<redacted>`. |
| `pid`, `return_code` | | Process ID and exit code. |
| `started_at_unix`, `stopped_at_unix` | s, Unix time | Start and stop of the server process. |
| `log_path` | | Path of the vLLM log. |

## latency.json and deep_context.json

Each file maps the requested length, as a string, to one entry.

| Field | Unit | Meaning |
|---|---|---|
| `requested_prompt_tokens` (latency), `requested_context_tokens` (deep context) | tokens | Target length. The client builds the prompt with tiktoken, not with the model tokenizer. |
| `prompt_tokens_exact` | bool | The server reported `usage.prompt_tokens`. The files do not store that count. |
| `n` | count | Requests with a valid TTFT. |
| `ttft_avg_s`, `ttft_median_s`, `ttft_p95_s` | s | Client TTFT statistics. |
| `ttft_p99_s` | s | Latency only. |
| `prefill_tps_avg` | tokens/s | Mean of `prompt_tokens / client TTFT`. |
| `status` | | Deep context only. `success`, `oom`, or `request_error`. The code sets `oom` on a connection error, or on an HTTP error whose text contains `memory` or `oom`. It sets `request_error` on any other error. After an error, the benchmark stops the requests for that length and goes on to the next length. The statistics use only the requests before the error. |
| `error` | | Deep context only. Exception type and message. For an HTTP error, the message holds the status code and the server body. Present only when `status` is not `success`. |

## ttft_breakdown.json

`config`: `benchmark_version`, `definition`, `server_requirement`, `prompt_lengths`, `repetitions`, `cache_isolation_method` (`cache_salt` or `text_salt`), `max_tokens` (1), `temperature`, `start_time`, `end_time`.

`per_length.<length>`:

| Field | Unit | Meaning |
|---|---|---|
| `status` | | `success`, `oom`, or `skipped_after_oom`. |
| `requested_tokens` | tokens | Target length. The prompt is `"hello world "` repeated `length // 2` times, so the length is approximate. |
| `actual_tokens` | tokens | `prompt_tokens` of the first successful request. |
| `n_requests`, `n_success`, `n_failed` | count | Request counts. |
| `per_request[].ttft_ms`, `per_request[].total_time_ms` | ms | Client TTFT and total request time. |
| `per_request[].server_ttft_s`, `per_request[].queue_time_s` | s | vLLM `time_to_first_token_ms` and `queue_time_ms`, in seconds. The server TTFT starts when the scheduler first schedules the request. It does not include the queue time. |
| `per_request[].scheduler_delay_ms` | ms | Server queue time. |
| `per_request[].post_queue_ms` | ms | Time from scheduling to the first token. It is equal to the server TTFT. It covers prefill and the first decode step. |
| `per_request[].prefill_time_s`, `prefill_ms`, `first_decode_ms` | s, ms | `null`. vLLM does not report a prefill time. |
| `per_request[]` other fields | | `index`, `success`, `prompt_tokens`, `output_tokens`, `error`. |
| `aggregated.<part>` | ms | `avg_ms`, `median_ms`, `p95_ms`, `min_ms`, `max_ms` for `ttft`, `scheduler_delay`, `post_queue`, `prefill_time`, `first_decode`. |
| `aggregated.gpu` | | GPU window data. `{}` when the window has no sample. |

The server fields are `null` when vLLM runs without `--enable-per-request-metrics`.

## decode.json

Each entry is keyed by the requested output length. `spec_enabled.json` and `spec_disabled.json` use the same schema.

| Field | Unit | Meaning |
|---|---|---|
| `requested_output_tokens` | tokens | `max_tokens` of the request. |
| `actual_output_tokens` | tokens | `usage.completion_tokens`. It includes reasoning tokens. |
| `output_tokens_exact` | bool | The server reported `usage.completion_tokens`. |
| `output_truncated` | bool | `actual_output_tokens` differs from `requested_output_tokens`. See the ambiguities above. |
| `ttft_s` | s | Client TTFT. |
| `decode_time_s` | s | Total request time − TTFT. |
| `tok_per_sec_avg` | tokens/s | `actual_output_tokens / decode_time_s`. |
| `tok_per_sec_peak`, `tok_per_sec_min`, `tok_per_sec_median` | events/s | `1 / gap` between stream events. See the ambiguities above. |
| `output_text_preview` | | First 512 characters of the output. |
| `reasoning_tokens`, `answer_tokens` | tokens, estimated | `count_tokens` of the reasoning text and of the answer text. |
| `reasoning_ratio` | ratio | `reasoning_tokens / answer_tokens`. `null` when there is no answer text. |

## reasoning.json

Each request uses `max_tokens` 1024. All counts come from `count_tokens`, not from the model tokenizer. The text in `delta.reasoning` is the thinking text, and the text in `delta.content` is the answer text.

`decode.json` uses a fallback when the stream has no reasoning text: `analyze_reasoning_tokens` splits the output by tags or by marker phrases. `reasoning.json` uses this fallback only when the output is empty.

| Field | Unit | Meaning |
|---|---|---|
| `per_prompt[].thinking_tokens`, `per_prompt[].answer_tokens` | tokens | Counts of the reasoning text and of the answer text. |
| `per_prompt[].ratio_thinking_to_answer` | ratio | Thinking tokens / answer tokens. |
| `per_prompt[].reasoning_text_preview`, `answer_text_preview`, `prompt_preview` | | First 128, 128, and 80 characters. |
| `thinking_tokens_avg`, `thinking_tokens_median`, `thinking_tokens_max` | tokens | Statistics over all prompts. |
| `answer_tokens_avg`, `answer_tokens_median`, `answer_tokens_max` | tokens | Statistics over all prompts. |

## concurrency.json

`config`:

| Field | Unit | Meaning |
|---|---|---|
| `concurrency_levels`, `requests_per_level`, `max_tokens`, `temperature` | | The run settings. |
| `prompt_mode` | | `distinct`: each request sends its own prompt. Runs before this field sent one prompt to every request. Read a missing field as `shared`. |
| `n_distinct_prompts` | count | Number of prompts, one for each request of the run. |
| `prompt_tokens` | tokens | Target length of each prompt. In old runs, the client estimate of the one prompt. |
| `prompt_length_method` | | `server_tokenize` (server `/tokenize` count, chat template included) or `client_estimate` (`count_tokens`). |
| `prompt_tokens_exact` | | `true` when all prompts have exactly `prompt_tokens` tokens by `prompt_length_method`. |
| `prompt_tokens_min`, `prompt_tokens_max` | tokens | Range of the prompt lengths by `prompt_length_method`. |
| `cache_isolation_method` | | `cache_salt` (new salt per request) or `distinct_text` (no salt). |

`per_concurrency_level.<level>`:

| Field | Unit | Meaning |
|---|---|---|
| `wall_time_s` | s | Time from the first submit to the last completion. |
| `total_output_tokens` | tokens | Sum over successful requests. |
| `aggregate_throughput_tok_s` | tokens/s | `total_output_tokens / wall_time_s`. |
| `n_requests`, `n_success`, `n_failed` | count | Request counts. |
| `success_rate` | fraction | `n_success / n_requests`. |
| `ttft`, `total_time_s` | s | `avg_s`, `median_s`, `p95_s`, `min_s`, `max_s` of client TTFT and of request time. |
| `individual_requests[]` | | `index`, `prompt_index`, `success`, `prompt_tokens` (server count), `output_tokens`, `ttft_s`, `total_time_s`, `error`. A failed request has `null` counts and times. |

## prefill_scaling.json

`config`: `benchmark_version`, `definition`, `target_lengths`, `cache_isolation_method`, `cache_preflight_supported`, `repetitions`, `max_tokens` (1), `max_output_tokens` (1), `temperature`, `prompt_workload`, `server_cache_settings`, `start_time`, `end_time`. The code writes `server_cache_settings.prefix_caching` as the constant `true`. It does not read the server flag.

`per_length.<length>`:

| Field | Unit | Meaning |
|---|---|---|
| `status` | | `success`, `oom`, `server_unavailable`, `request_error`, or `skipped_after_oom`. A `request_error` entry holds only `status` and `error`. |
| `requested_tokens`, `actual_tokens` | tokens | Target length, and the length after calibration with the server `/tokenize` endpoint. |
| `prompt_length_tolerance` | tokens | `requested_tokens − actual_tokens`. |
| `n_requests`, `n_success` | count | Measured requests. The warm-up request is not counted. |
| `per_request[].client_ttft_s`, `total_time_s` | s | Client TTFT and total request time. |
| `per_request[].effective_prefill_tps` | tokens/s | `prompt_tokens / client_ttft_s`. |
| `per_request[].engine_prefill_tps` | tokens/s | `prompt_tokens` / server prefill time. vLLM reports none, so the code uses the server TTFT. The server TTFT does not include the queue time. |
| `per_request[].server_ttft_s`, `queue_time_s`, `prefill_time_s` | s | Server timings. `prefill_time_s` is `null` with vLLM. |
| `per_request[].cached_tokens` | tokens | `usage.prompt_tokens_details.cached_tokens`. 0 means that the request missed the prefix cache. |
| `per_request[].start_time`, `end_time` | s | Monotonic clock values, not Unix time. |
| `per_request[]` other fields | | `index`, `success`, `prompt_tokens`, `prompt_tokens_exact`, `cache_isolation_method`, `error`. |
| `aggregated.ttft` | s | `avg_s`, `median_s`, `p95_s`, `min_s`, `max_s` of client TTFT. |
| `aggregated.effective_prefill_tps`, `aggregated.engine_prefill_tps` | tokens/s | The same five statistics. The keys end in `_s`, but the values are rates. |
| `aggregated.gpu` | | GPU window data with the idle and energy fields. |

## flops_analysis.json

| Field | Unit | Meaning |
|---|---|---|
| `estimate_status` | | `exact_from_config`, `approximate`, or `unsupported`. |
| `architecture` | | `model` label, `layer_counts`, `ffn`, `token_mixers`, `is_hybrid`. |
| `flops.decode.<length>` | FLOPs | One decode step at this context length. |
| `flops.prefill.<length>` | FLOPs | One prefill request of this length. |
| `flops.*.<length>.total`, `matmul_flops`, `non_matmul_flops` | FLOPs | Totals. Other keys hold the FLOPs of each component. |
| `flops.*.<length>.state_traffic_bytes` | bytes | Linear-attention state read and written. |
| `flops.*.<length>.omitted_non_matmul` | | Operations that the totals leave out. |
| `assumptions` | | Rules of the estimate. |
| `warnings` | | Present when `estimate_status` is `unsupported`. It tells why. |

## tool_calling.json

Top-level fields: `config`, `total_tasks`, `total_correct`, `composite_score` (0 to 6), `composite_components`, `category_scores`, `per_task`, `failure_modes`. [tool-calling.md](tool-calling.md) describes the fields and the scoring.

## spec_comparison.json

| Field | Unit | Meaning |
|---|---|---|
| `config.output_lengths` | tokens | `decode_lengths` of the model config. |
| `spec_enabled`, `spec_disabled` | | Copies of `spec_enabled.json` and `spec_disabled.json`. |
| `delta.<length>.tok_s_improvement_pct` | % | (enabled − disabled) / disabled × 100 of `tok_per_sec_avg`. |
| `delta.<length>.ttft_change_pct` | % | The same formula for `ttft_s`. |

## CSV files and logs

| File | Columns or content |
|---|---|
| `gpu_samples.csv` | `gpu_mem_total_mib`, `gpu_mem_used_mib`, `gpu_mem_util_pct`, `gpu_power_w`, `gpu_util_pct`, `t` (Unix time, s). One row per sample. The memory columns are empty on the GB10. |
| `memory_samples.csv` | `elapsed_s`, `timestamp`, `mem_total_gib`, `mem_available_gib`, `mem_used_gib` (total − available), `swap_used_gib`. From `/proc/meminfo`. |
| `memory_guard.json` | `floor_gib`, `warn_gib`, `interval_s`, `start_available_gib`, `min_available_gib`, `stopped_for_memory`, `return_code`. |
| `vllm.log`, `vllm_spec_*.log` | stdout and stderr of the vLLM process. |
| `model_config.yml` | Exact copy of the model config file. |

## Runs before 2026-11-07

Fixes merged on 2026-11-07 changed several files. Run dirs from before that date differ from this page:

- `prefill_scaling.json` holds `status: request_error` for each length. The client then had no `tokenize_prompt` method.
- The server command has no `--enable-per-request-metrics`, so all server timing fields are `null`.
- `ttft_breakdown.json` has no `post_queue_ms` field.
- The run dir has no `tool_calling.json`. The `tool_calling` entry in `summary.json` has no `failure_modes`.

The dashboard marks these runs "Pre-fix scoring" (`SCORING_FIX_DATE` in dashboard.py).

## Runs with the old TTFT split

Before the TTFT split fix, the client set `post_queue_ms` to server TTFT − queue time. The vLLM server TTFT does not include the queue time, so this value subtracts the queue time two times. The error is equal to the queue time. It is small at concurrency 1 and larger with queueing.

- `ttft_breakdown.json` in these runs holds `post_queue_ms` and `aggregated.post_queue` values that are too low.
- `prefill_scaling.json` in these runs holds `engine_prefill_tps` values that are too high.
- The dashboard does not read `post_queue_ms`. It computes the post-queue median from `per_request[].server_ttft_s`, so it shows the correct value for old and new runs.
