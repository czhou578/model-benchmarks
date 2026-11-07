# Model Benchmarks

This Python harness measures how a local LLM performs when vLLM serves it. It records latency, decode speed, throughput under load, reasoning tokens, and tool-calling quality. The published results come from an NVIDIA DGX Spark (GB10 GPU, 121 GB unified memory). The harness is for people who tune vLLM on a single node. Use it to compare weight sources, quantizations, and server flags with one method.

## What it measures

A run executes the benchmarks below in sequence. Each benchmark writes one file to the run dir, `results/<name>/<timestamp>/`.

| Benchmark | Output file | Main metric | Unit |
|---|---|---|---|
| Latency sweep | `latency.json` | Median and p99 TTFT, prompts of 32 to 16K tokens | s |
| Deep context | `deep_context.json` | Median TTFT at 32K and 64K tokens | s |
| TTFT breakdown | `ttft_breakdown.json` | Server queue time and post-queue time | ms |
| Decode speed | `decode.json` | Mean decode rate at 512, 1K, and 2K output tokens | tokens/s |
| Concurrency | `concurrency.json` | Aggregate throughput at 1 to 16 parallel requests | tokens/s |
| Reasoning | `reasoning.json` | Thinking tokens and answer tokens per prompt | tokens |
| Prefill scaling | `prefill_scaling.json` | Effective and engine prefill throughput per prompt length | tokens/s |
| FLOP analysis | `flops_analysis.json` | Estimated FLOPs per decode token and per prefill | FLOPs |
| Tool calling | `tool_calling.json` | Tasks passed (of 85) and composite score (0 to 6) | count, score |
| Spec comparison | `spec_comparison.json` | Decode speed with speculative decoding on and off | tokens/s, % |

Deep context runs only when the model config sets `context_lengths`. Spec comparison runs only with `--compare-spec`. A run also writes `environment.json`, `gpu_samples.csv`, `summary.json`, and the vLLM log. See [docs/results-schema.md](docs/results-schema.md) for each file and field.

## Requirements

- An NVIDIA GPU with a driver and `nvidia-smi`.
- vLLM. The published results use vLLM 0.25.1. This harness drives vLLM; it does not install it.
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
python3 core_runner.py --model models/my_model.yml           # start vLLM, run, stop vLLM
python3 dashboard.py                                         # write reports/dashboard.html
```

Open `reports/dashboard.html` in a browser. To see new runs on each reload, run `python3 serve_dashboard.py` and open http://127.0.0.1:8080/dashboard.html.

`python3 batch_run.py` runs every model config in `models/` in sequence. A full batch takes hours. See [docs/model-config.md](docs/model-config.md) for all model config keys.

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
| `--server-mode {managed,external}` | Override `server.mode` from the model config. |
| `--compare-spec` | Restart vLLM with and without `speculative_config`, then compare decode speed. |
| `--results-dir DIR` | Write run dirs under `DIR`. The default is `results/` in the repo root. |

FLOP analysis and tool calling have no skip flag.

`batch_run.py`:

| Flag | Effect |
|---|---|
| `--model FILE` | Run one model config. The path is relative to `models/`. |
| `--dry-run` | List the model configs and their ports, then stop. |
| `--jobs N` | Run up to `N` model configs in parallel. The default is 1. |
| `--max-retries N` | Retry a failed run `N` times. The default is 1. |
| `--no-retry` | Do not retry a failed run. |
| `--output-dir DIR` | Write `summary.json` and `SUMMARY.md` to `DIR`. The default is the current directory. |

Keep `--jobs 1` on a single GPU. Parallel runs share the GPU memory and the server ports. `batch_run.py` passes only `--model` to `core_runner.py`.

## Published results

The write-ups in [docs/results/](docs/results/) describe past runs. Each write-up names its date and the harness state. Runs before 2026-11-07 use the earlier tool-calling scorer and cache-salt placement. Do not compare their TTFT or tool-calling numbers with newer runs. [docs/dashboard.md](docs/dashboard.md) explains why.

Test system, from `environment.json` in the run dirs:

| Component | Value |
|---|---|
| GPU | NVIDIA GB10 (DGX Spark, 121 GB unified memory) |
| Driver | 580.173.02 (2026-11 runs), 580.159.03 (2026-07 runs) |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.25.1 |

## Documentation

- [docs/model-config.md](docs/model-config.md): the keys of a model config.
- [docs/results-schema.md](docs/results-schema.md): the files and fields in a run dir.
- [docs/methodology.md](docs/methodology.md): how each benchmark measures.
- [docs/tool-calling.md](docs/tool-calling.md): the tool-calling tasks and the scorer.
- [docs/adding-a-benchmark.md](docs/adding-a-benchmark.md): how to add a benchmark.
- [docs/dashboard.md](docs/dashboard.md): how to build, serve, and read the dashboard.
- [docs/dgx-spark.md](docs/dgx-spark.md): DGX Spark setup and the memory guard.
- [docs/results/](docs/results/): published result write-ups.
- [docs/design/](docs/design/): design notes from the development of the harness.

## Contributing

- Run `ruff check .` and `pytest -q` before you open a pull request.
- CI runs `ruff check .` and `pytest -q` on each push and pull request.
- Keep each pull request small, with one topic.

See [LICENSE](LICENSE).
