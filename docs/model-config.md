# Model config reference

A model config is a YAML file in `models/`. It tells `core_runner.py` how to reach the model server. It also sets the parameters of each benchmark. The folder `models/legacy/` holds the configs of the July 2026 runs. [models/legacy/README.md](../models/legacy/README.md) gives their status.

Run one model config:

```bash
python core_runner.py --model models/qwen3.8_27b_nvidia_nvfp4.yml
```

## Run dir

The runner writes all output of one run to one run dir:

```
<results-dir>/<name>/<YYYYMMDD_HHMMSS>/
```

- `--results-dir` sets the root folder. The default is `results/` under the repo root.
- `<name>` is the `name` key of the model config.
- The timestamp is the local time at the start of the run (`make_run_dir`).
- The runner does not reuse a run dir. A second run of the same model config in the same second fails.

[results-schema.md](results-schema.md) describes each file in the run dir.

## The `name` scheme

Use a lowercase name in this form:

```
<model>-<size>-<provider>-<quant>[-<variant>]
```

| Part | Example | Meaning |
|---|---|---|
| `model` | `qwen3.6` | Model family and version. |
| `size` | `35b-a3b` | Total parameters. For a MoE model, add the active parameters. |
| `provider` | `nvidia` | The organization that publishes the checkpoint. |
| `quant` | `nvfp4` | Weight quantization. |
| `variant` | `nospec` | Optional. A server setting that differs from the base model config. |

Examples: `qwen3.8-27b-nvidia-nvfp4`, `qwen3.6-35b-a3b-unsloth-nvfp4-nospec`.

The code does not check the scheme. It uses `name` as a folder name, so do not put `/` in it. The runner also writes `name` as `model` in `summary.json`, and the dashboard shows it as the run label.

## Server modes

| Mode | What the runner does |
|---|---|
| `external` | It connects to a server that already runs at `endpoint.base_url`. |
| `managed` | It starts `server.command` as a child process, waits for it, and stops it at the end of the run. |

- `server.mode` sets the mode. The default is `external`.
- `--server-mode managed` or `--server-mode external` overrides `server.mode`.
- In managed mode, the runner refuses to start when the port of `endpoint.base_url` is already open.
- In managed mode, `endpoint.base_url` must include a host and a port. A `--port` value in `server.command` must match that port (`VllmServer.__init__`).
- To stop a managed server, the runner sends SIGTERM to the process group. After `server.shutdown_timeout_s`, it sends SIGKILL.

## Required keys

| Mode | Required keys |
|---|---|
| external | `name`, `endpoint.base_url` |
| managed | `name`, `endpoint.base_url` with host and port, `server.command`, and `server.mode: managed` or `--server-mode managed` |

`endpoint.model_name` falls back to `name` (`make_client`). Set it when `name` differs from the model name that the server serves.

The runner does not validate keys. It ignores an unknown key, so a misspelled key has no effect and the default applies.

## Keys

"Used by" names the benchmark or the part of the runner that reads the key.

| Key | Type | Default | Required | Meaning | Used by |
|---|---|---|---|---|---|
| `name` | string | none | yes | Run dir folder name. | all |
| `run_label` | string | none | no | Free text that tells the purpose of the run, for example `prefill rerun`. The runner writes it to `summary.json`. It does not change the run dir. The dashboard shows it on the run card. | runner, dashboard |
| `server.mode` | `managed` or `external` | `external` | no | Server mode. | runner |
| `server.command` | list of strings | none | managed mode | Command that starts the server, one argument per item. | runner, FLOP analysis |
| `server.environment` | map | `{}` | no | Extra environment variables for the server process. | runner (managed) |
| `server.startup_timeout_s` | number, s | `ready_timeout_s` | no | Time limit for readiness. It wins over `ready_timeout_s` in both modes. | runner |
| `server.shutdown_timeout_s` | number, s | 30 | no | Wait after SIGTERM before SIGKILL. | runner (managed) |
| `endpoint.base_url` | URL | none | yes | Server root, for example `http://127.0.0.1:8000`. Do not add `/v1`. The client adds it. | all |
| `endpoint.model_name` | string | `name` | no | The `model` field of each request. | all |
| `endpoint.chat` | bool | `true` | no | `true` uses `/v1/chat/completions`. `false` uses `/v1/completions`. | client |
| `endpoint.api_key` | string | none | no | Sent as `Authorization: Bearer <key>`. | client |
| `ready_timeout_s` | number, s | 600 | no | Time limit for readiness when `server.startup_timeout_s` is not set. | runner |
| `ready_poll_s` | number, s | 3.0 | no | Wait between two readiness probes. | runner |
| `monitor_interval_s` | number, s | 1.0 | no | GPU sample interval. Use a value above 0. | GPU monitor, prefill idle baseline |
| `idle_baseline_s` | number, s | 5.0 | no | Length of the GPU idle baseline before prefill scaling. | prefill idle baseline |
| `prompt_lengths` | list of int, tokens | `[32, 128, 512, 2048, 8192, 16384]` | no | Prompt lengths. | latency |
| `latency_repeats` | int | 10 | no | Requests per prompt length. | latency |
| `context_lengths` | list of int, tokens | none | no | Long prompt lengths. The benchmark runs only when this key is set. | deep context |
| `context_repeats` | int | 5 | no | Requests per context length. | deep context |
| `deep_context_output_tokens` | int | 64 | no | `max_tokens` of each deep context request. | deep context |
| `ttft_prompt_lengths` | list of int, tokens | `[128, 512, 2048, 8192, 32768]` | no | Exact prompt lengths. | TTFT breakdown |
| `ttft_repetitions` | int | 10 | no | Requests per prompt length. | TTFT breakdown |
| `decode_lengths` | list of int, tokens | `[512, 1024, 2048]` | no | `max_tokens` values. | decode, `--compare-spec` |
| `reasoning_prompts` | list of strings | two built-in prompts | no | Prompts. | reasoning |
| `reasoning_max_tokens` | int | 4096 | no | `max_tokens` of each request. 1024 cut the longest Qwen3.8 and Qwen3.6 prompts before the answer. | reasoning |
| `concurrency_levels` | list of int | `[1, 2, 4, 8, 16]` | no | Parallel workers at each level. | concurrency |
| `concurrency_requests_per_level` | int | 5 | no | Total requests at each level. The runner warns when this is below the largest level. | concurrency |
| `concurrency_max_tokens` | int | 256 | no | `max_tokens` of each request. | concurrency |
| `concurrency_temperature` | float | 0.0 | no | Sampling temperature. | concurrency |
| `prefill_target_lengths` | list of int, tokens | `[512, 2048, 8192]` | no | Exact prompt lengths. | prefill scaling |
| `prefill_repetitions` | int | 5 | no | Measured requests per length, after one warm-up request. | prefill scaling |
| `flop_analysis_lengths` | list of int, tokens | `[512, 2048, 8192, 32768]` | no | Sequence lengths for the decode and prefill estimates. | FLOP analysis |
| `architecture_config_path` | path | none | no | A Hugging Face `config.json` file or its folder. A relative path starts at the folder of the model config. | FLOP analysis |
| `local_model_path`, `model_path`, `model_id` | string | none | no | Model reference for the architecture lookup. The first key that is set wins. | FLOP analysis |
| `architecture_trust_remote_code` | bool | false | no | Lets the Hugging Face config lookup run Python code from the model repository. Set it only for a repository that you trust. | FLOP analysis |
| `tool_calling_tasks` | path | `datasets/tool_calling_tasks.yaml` | no | Task file. A relative path starts at the current directory. When the file is not there, the path starts at the repo root. | tool calling |
| `tool_calling_task_set` | `full` or `lite` | `full` | no | Task subset. | tool calling |
| `tool_calling_max_tokens` | int | 4096 | no | `max_tokens` of each request. | tool calling |
| `tool_calling_chat_template_kwargs` | map | none | no | Sent as `chat_template_kwargs` in each request. | tool calling |
| `tool_calling_request_timeout_s` | float | 600 | no | Read timeout of each request, in seconds. 4096 tokens at 12.3 tok/s take about 330 s, so 120 s was too short. | tool calling |
| `tool_calling_repeats` | int | 1 | no | Number of passes over the task set. With more than 1, `tool_calling.json` adds the mean and the spread of the composite and the pass count. | tool calling |
| `speculative_config` | JSON string or map | none | for `--compare-spec` | Value of `--speculative-config` for the spec-enabled server. | `--compare-spec` |

Notes:

- `resolved_server.json` replaces the value of each secret-like `server.environment` variable with `<redacted>` (`is_secret_name` in `vllm_server.py`). A name is secret-like when it contains `SECRET` or `PASSWORD`. It is also secret-like when it is, or ends in `_` plus, one of `TOKEN`, `KEY`, `APIKEY`, `KEY_ID`, `PASS`, `PASSWD`, `CREDENTIAL`, `CREDENTIALS`. Thus `HF_TOKEN` and `OPENAI_API_KEY` show `<redacted>`, and `VLLM_USE_FASTOKENS` shows its value.
- WARNING: The rule does not find every secret. Do not put a secret in a variable with another name, because `resolved_server.json` then shows it.
- The tool-calling benchmark always uses `/v1/chat/completions`. It ignores `endpoint.chat`.
- Write `speculative_config` as a JSON string in single quotes, or as a YAML map. The runner sends a map as JSON (`json.dumps`) and a string as it is (`speculative_config_arg` in core_runner.py).
- With `--compare-spec`, the runner removes the speculative decoding flags of `server.command` from both variants (`strip_spec_flags`). These flags are `--speculative-config`, `-sc`, `--speculative-config.<key>`, `--spec-method`, `--spec-model`, and `--spec-tokens`, each with its value. The enabled variant then adds `--speculative-config` with `speculative_config`. The runner prints a warning, and `spec_comparison.json` lists the removed arguments in `config.removed_server_flags`.
- The FLOP analysis looks for a model reference in this order: `local_model_path`, `model_path`, `model_id`, `endpoint.model_name`, then the argument after `serve` in `server.command` (`_model_reference`).
- Some benchmark values have no key, for example the `max_tokens` of the latency requests. [methodology.md](methodology.md) lists them.

## Example: external mode

Start the server first. Add `--enable-per-request-metrics` to the `vllm serve` command, so the TTFT breakdown gets server timings.

```yaml
name: qwen3.8-27b-nvidia-nvfp4
endpoint:
  base_url: "http://127.0.0.1:8000"
  model_name: "nvidia/Qwen3.8-27B-NVFP4"
```

## Example: managed mode

```yaml
name: qwen3.8-27b-nvidia-nvfp4
server:
  mode: managed
  command: [vllm, serve, nvidia/Qwen3.8-27B-NVFP4, --host, "127.0.0.1", --port, "8000",
            --enable-per-request-metrics, --reasoning-parser, qwen3,
            --enable-auto-tool-choice, --tool-call-parser, qwen3_coder]
  startup_timeout_s: 900
endpoint:
  base_url: "http://127.0.0.1:8000"
  model_name: "nvidia/Qwen3.8-27B-NVFP4"
speculative_config: '{"method":"mtp","num_speculative_tokens":3}'
```

Quote each number in `server.command`. `VllmServer` accepts only strings. The files in `models/` are complete examples.
