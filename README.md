# Model Benchmarks

This Python harness measures how a local LLM performs when vLLM serves it. It records latency, decode speed, throughput under load, reasoning tokens, and tool-calling quality. The published results come from an NVIDIA DGX Spark (GB10 GPU, 121 GB unified memory). The harness is for people who tune vLLM on a single node. Use it to compare weight sources, quantizations, and server flags with one method.

## What it measures

A run executes the benchmarks below in sequence. Each benchmark writes one file to the run dir, `results/<name>/<timestamp>/`.

| Benchmark | Output file | Main metric | Unit |
|---|---|---|---|
| Latency sweep | `latency.json` | Median and p99 TTFT, prompts of 32 to 16K tokens | s |
| Deep context | `deep_context.json` | Median TTFT at each `context_lengths` value, for example 32K and 64K tokens | s |
| TTFT breakdown | `ttft_breakdown.json` | Server queue time and post-queue time | ms |
| Decode speed | `decode.json` | Mean decode rate at 512, 1K, and 2K output tokens | tokens/s |
| Reasoning | `reasoning.json` | Thinking tokens and answer tokens per prompt | tokens |
| Concurrency | `concurrency.json` | Aggregate throughput at 1 to 16 parallel requests | tokens/s |
| Prefill scaling | `prefill_scaling.json` | Effective and engine prefill throughput per prompt length | tokens/s |
| FLOP analysis | `flops_analysis.json` | Estimated FLOPs per decode token and per prefill (Qwen3.6-35B-A3B only) | FLOPs |
| Tool calling | `tool_calling.json` | Tasks passed (of 85) and composite score (0 to 6) | count, score |
| Spec comparison | `spec_comparison.json` | Decode speed with speculative decoding on and off | tokens/s, % |

Deep context runs only when the model config sets `context_lengths`. Spec comparison runs only with `--compare-spec`.

The FLOP analysis accepts only the dimensions of Qwen3.6-35B-A3B. For other models, for example Qwen3.8-27B, `flops_analysis.json` holds `estimate_status: unsupported`, empty FLOP tables, and `warnings`.

A run also writes `model_config.yml`, `environment.json`, `gpu_samples.csv`, and `summary.json`. In managed mode, it also writes `resolved_server.json` and the vLLM log. The published run dirs in `results/` do not include `gpu_samples.csv` or the vLLM logs. See [docs/results-schema.md](docs/results-schema.md) for each file and field.

## Requirements

- An NVIDIA GPU with a driver and `nvidia-smi`.
- vLLM. The model configs target vLLM 0.33.0. The newest published run uses vLLM 0.33.0, and the older runs use 0.25.1. This harness drives vLLM; it does not install it.
- Python 3.10 or later.
- Hugging Face access to the model weights, and disk space for them.
- On a DGX Spark: the setup notes in [docs/dgx-spark.md](docs/dgx-spark.md).

## Install

```bash
git clone https://github.com/czhou578/model-benchmarks.git
cd model-benchmarks
pip install -e .            # to run benchmarks
pip install -e ".[dev]"     # to run tests and lint
```

The runtime dependencies are `requests`, `PyYAML`, and `tiktoken`.

## Quickstart

CAUTION: A large model can fill the unified memory of a DGX Spark and make the host unresponsive. Read [docs/dgx-spark.md](docs/dgx-spark.md) before the first run.

Run these commands from the repo root. In the copy, edit `name:`, the `vllm serve` command, and `endpoint.model_name`.

```bash
cp models/qwen3.8_27b_nvidia_nvfp4.yml models/my_model.yml   # copy an example model config
python3 memory_guard.py -- \
    python3 -u core_runner.py --model models/my_model.yml    # start vLLM, run, stop vLLM
python3 dashboard.py                                         # write reports/dashboard.html
```

`memory_guard.py` stops the run when the available host memory stays below 8 GiB. It refuses to start below 40 GiB. Without the guard, run `python3 core_runner.py --model models/my_model.yml`. [docs/dgx-spark.md](docs/dgx-spark.md#memory-guard) explains the guard options.

Open `reports/dashboard.html` in a browser. To see new runs on each reload, run `python3 serve_dashboard.py` and open http://127.0.0.1:8080/dashboard.html.

`python3 batch_run.py` runs every model config in `models/` in sequence. It skips the July 2026 configs in [models/legacy/](models/legacy/README.md). A full batch takes hours. See [docs/model-config.md](docs/model-config.md) for all model config keys.

## External mode

In external mode, the harness uses an OpenAI-compatible server that already runs. It does not start or stop the server.

```yaml
name: my-model-8b-myorg-fp8
server:
  mode: external
endpoint:
  base_url: "http://127.0.0.1:8000"
  model_name: "my-org/MyModel"
  # api_key: "..."   # only if the server needs a key
```

- Set `server.mode: external` in the model config, or pass `--server-mode external`.
- Start vLLM with `--enable-per-request-metrics`. Without it, the TTFT breakdown has no server data.
- Run the harness on the server host. GPU telemetry reads the local `nvidia-smi`.
- Do not use `--compare-spec`. It needs managed mode.

The harness sends vLLM request fields such as `cache_salt`. Another server can ignore them. Then its TTFT values can include prefix-cache hits.

## Common flags

`core_runner.py`:

| Flag | Effect |
|---|---|
| `--model PATH` | Run this model config. Required. |
| `--skip-latency`, `--skip-ttft`, `--skip-decode` | Skip the latency sweep, the TTFT breakdown, or decode speed. |
| `--skip-reasoning`, `--skip-concurrency`, `--skip-prefill` | Skip reasoning, concurrency, or prefill scaling. |
| `--skip-deep-context`, `--skip-flops`, `--skip-tool-calling` | Skip deep context, the FLOP analysis, or tool calling. |
| `--server-mode {managed,external}` | Override `server.mode` from the model config. |
| `--compare-spec` | Restart vLLM with and without `speculative_config`, then compare decode speed. |
| `--results-dir DIR` | Write run dirs under `DIR`. The default is `results/` in the repo root. |

`batch_run.py`:

| Flag | Effect |
|---|---|
| `--model FILE` | Run one model config. The path is relative to `models/`. |
| `--dry-run` | List the model configs and their ports, then stop. |
| `--max-retries N` | Retry a failed run `N` times. The default is 1. |
| `--no-retry` | Do not retry a failed run. |
| `--output-dir DIR` | Write `summary.json`, `SUMMARY.md` and `logs/<name>.log` to `DIR`. The default is the current directory. |

`batch_run.py` runs the model configs one at a time. It reads only the top level of `models/`, not its subfolders. It passes only `--model` and `--results-dir` to `core_runner.py`. A run fails when `core_runner.py` exits with a nonzero code, writes no `summary.json`, or writes a status other than `completed`. `batch_run.py` then retries the run. It shows the `core_runner.py` output on the console and writes it to `logs/<name>.log`.

## Published results

The write-ups in [docs/results/](docs/results/) describe past runs. Each write-up names its date and the harness state. Fixes to the harness divide the runs into eras. Before you compare two runs, read [docs/comparability.md](docs/comparability.md).

Test system of the published runs, from `environment.json` in the run dirs. All runs use an NVIDIA GB10 (DGX Spark, 121 GB unified memory) and CUDA 13.0.

| Era | Runs | Driver | PyTorch | vLLM |
|---|---|---|---|---|
| E0 | 2026-07-29, five Qwen3.6-35B-A3B runs | 580.159.03 | 2.11.0+cu130 | 0.25.1 |
| E1 | 2026-11-07, two Qwen3.8-27B runs | 580.173.02 | 2.11.0+cu130 | 0.25.1 |
| E2 | 2026-11-08, one Qwen3.8-27B run | 580.173.02 | 2.14.0+cu130 | 0.33.0 |

## Code layout

| Path | Role |
|---|---|
| `core_runner.py` | The command line, the run state, and the step order of one run. Start here. |
| `benchmarks/` | One module per benchmark: `latency.py` (latency sweep and deep context), `ttft_breakdown.py`, `decode.py` (decode and the spec comparison), `reasoning.py`, `concurrency.py`, `prefill.py`, `architecture_flops.py`, `tool_calling.py`. |
| `harness/` | Shared code: `client.py` (HTTP client and stream reader), `prompts.py`, `stats.py`, `telemetry.py` (GPU monitor and energy), `environment.py`, `provenance.py` (schema version, git commit, server vLLM version), `paths.py`. |
| `vllm_server.py` | Starts and stops a managed vLLM server. |
| `memory_guard.py` | Runs a command and stops it when the free memory falls below a floor. |
| `batch_run.py` | Runs each model config in `models/`, one at a time. |
| `dashboard.py`, `serve_dashboard.py`, `reports/dashboard/` | Build and serve the HTML dashboard. |
| `models/` | Model configs. `models/legacy/` holds the July 2026 configs. |
| `datasets/tool_calling_tasks.yaml` | The 85 tool-calling tasks. |
| `results/` | The published run dirs. |
| `tests/` | Unit tests. They need no GPU and no server. |

The modules in `benchmarks/` import from `harness/` and from other modules in `benchmarks/`. They do not import `core_runner.py` at the top level. `core_runner.py` imports from both folders. [docs/adding-a-benchmark.md](docs/adding-a-benchmark.md) tells how to add a benchmark.

## Documentation

- [docs/overview.md](docs/overview.md): start here. How a run works, step by step, and a glossary.
- [docs/model-config.md](docs/model-config.md): the keys of a model config.
- [docs/results-schema.md](docs/results-schema.md): the files and fields in a run dir.
- [docs/methodology.md](docs/methodology.md): how each benchmark measures.
- [docs/tool-calling.md](docs/tool-calling.md): the tool-calling tasks and the scorer.
- [docs/adding-a-benchmark.md](docs/adding-a-benchmark.md): how to add a benchmark.
- [docs/dashboard.md](docs/dashboard.md): how to build, serve, and read the dashboard.
- [docs/dgx-spark.md](docs/dgx-spark.md): DGX Spark setup and the memory guard.
- [docs/comparability.md](docs/comparability.md): which published results you can compare, and the noise.
- [docs/results/](docs/results/): published result write-ups.
- [docs/archive/design/](docs/archive/design/): archived design notes from the development of the harness, with a status for each note.

## Contributing

- Run `ruff check .` and `pytest -q` before you open a pull request.
- CI runs `ruff check .` and `pytest -q` on each push to `main` and on each pull request.
- Keep each pull request small, with one topic.

See [LICENSE](LICENSE).
