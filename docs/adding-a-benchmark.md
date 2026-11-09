# Add a benchmark

This page tells how `core_runner.py` runs each benchmark, and how to add a new one.

## Current benchmarks

The runner runs the benchmarks in the order of this table. The function `run_benchmarks()` in core_runner.py sets this order.

| Benchmark | Code | Output file in the run dir | `summary.json` key | Control |
|---|---|---|---|---|
| Latency | `run_latency_sweep` in `benchmarks/latency.py` | `latency.json` | `latency`, full copy | `--skip-latency` |
| Deep context | `run_deep_context` in `benchmarks/latency.py` | `deep_context.json` | `deep_context`, full copy | runs only when `context_lengths` is set |
| TTFT breakdown | `benchmarks/ttft_breakdown.py` | `ttft_breakdown.json` | `ttft_breakdown`, full copy | `--skip-ttft` |
| Decode | `run_decode_speed` in `benchmarks/decode.py` | `decode.json` | `decode`, full copy | `--skip-decode` |
| Reasoning | `run_reasoning_benchmark` in `benchmarks/reasoning.py` | `reasoning.json` | `reasoning`, full copy | `--skip-reasoning` |
| Concurrency | `benchmarks/concurrency.py` | `concurrency.json` | `concurrency`, full copy | `--skip-concurrency` |
| Prefill scaling | `benchmarks/prefill.py` | `prefill_scaling.json` | `prefill_scaling`, status and `n_success` per length | `--skip-prefill` |
| FLOP analysis | `benchmarks/architecture_flops.py` | `flops_analysis.json` | `flops_analysis`, status and model label | always runs |
| Tool calling | `benchmarks/tool_calling.py` | `tool_calling.json` | `tool_calling`, scores only | always runs |
| Speculative decoding comparison | `compare_spec_decode_results` in `benchmarks/decode.py` | `spec_enabled.json`, `spec_disabled.json`, `spec_comparison.json` | `spec_comparison`, full copy | runs only with `--compare-spec` |

Other command-line options: `--model` (required), `--server-mode`, `--results-dir`, and `--compare-spec`. [model-config.md](model-config.md) and [methodology.md](methodology.md) describe them.

An unhandled error in a benchmark stops the run. The runner then writes `summary.json` with `status: failed` and the `error` text. The tool-calling benchmark is the exception. The runner catches its error, prints a warning, and continues.

## How the runner wires a benchmark

Each benchmark has one step function in core_runner.py, for example `step_decode(run)`. `run_benchmarks(run)` checks the skip flags and calls the steps in order. The `Run` object holds the state of the run: `cfg`, `args`, `run_dir`, `summary`, `client`, `monitor`, and `server`.

A step function does these things:

1. It reads its parameters with `run.cfg.get("<key>", <default>)`.
2. It calls the run function of its module with `run.client`, the parameters, and optionally `gpu_monitor=run.monitor`.
3. It calls `run.record("<name>", result)`. This writes `<name>.json` to the run dir and adds the `<name>` entry to the summary. Use `summary_entry=` to put a short form in the summary.

The modules in `benchmarks/` import only from `harness/`, never from `core_runner`. Keep it so: an import of `core_runner` from a benchmark makes a circular import.

## Conventions

- Name the output file `<name>.json`, in lowercase with underscores. Use the same `<name>` as the `summary.json` key.
- Return a dict. Put the parameters, `benchmark_version`, `start_time`, and `end_time` in a `config` block, as `ttft_breakdown.py` and `prefill.py` do.
- Key a per-length map by the length as a string, for example `"2048"`. The dashboard sorts these keys as integers (`by_int_key`).
- Use `stat_summary(values, suffix)` from `harness/stats.py` for avg, median, p95, min, and max. Set `suffix` to the unit of the values, for example `"s"` or `"ms"`.
- Send `cache_salt=uuid.uuid4().hex` to `client.generate` when each request must miss the prefix cache.
- Put a short summary in `summary.json` when the full result is large. `prefill_scaling` and `tool_calling` do this.
- For GPU data per step, call `gpu_monitor.start_window(name)` and `gpu_monitor.stop_window(name)`. The second call returns the window summary.

## Steps

1. Create `benchmarks/<name>.py` with a function `run_<name>(client, ..., gpu_monitor=None) -> dict`.
2. Add `--skip-<name>` to the argument parser in `main()`.
3. Add a function `step_<name>(run)` to core_runner.py. Call it from `run_benchmarks()` at the point in the order where the benchmark must run.
4. Add new keys to the model configs that need values other than the defaults.
5. Add tests in `tests/test_<name>.py`. Add the step to `ALL_STEPS` in `tests/test_core_runner_main.py`.
6. Add the new keys to [model-config.md](model-config.md) and the new fields to [results-schema.md](results-schema.md).
7. Optional: show the metric in the dashboard. The next section tells how.

| File | Change |
|---|---|
| `benchmarks/<name>.py` | New module. |
| `core_runner.py` | Skip flag, `step_<name>()`, the call in `run_benchmarks()`. |
| `models/*.yml` | New keys, only where the defaults do not fit. |
| `tests/test_<name>.py` | New tests. |
| `docs/model-config.md`, `docs/results-schema.md` | New keys and fields. |
| `dashboard.py`, `reports/dashboard/*.js` | Only to show the metric. |

## Show a new metric in the dashboard

`dashboard.py` finds run dirs by itself. It reads each `*/*/summary.json` under the folder that `--results` sets. The default folder is `results/`. It skips a run dir whose name does not match `%Y%m%d_%H%M%S`.

`dashboard.py` does not find metrics by itself. `collect_run()` reads a fixed list of keys and files:

| Source | What `dashboard.py` reads |
|---|---|
| `summary.json` | `model`, `status`, `error`, `latency`, `deep_context`, `ttft_breakdown`, `decode`, `concurrency`, `reasoning`, `tool_calling`, `spec_comparison`, `gpu`, `decode_energy` |
| `environment.json` | `gpu_name`, `vllm_version`, `torch_version`, `driver_version` |
| `resolved_server.json` | `command` |
| `tool_calling.json` | `category_scores`, `failure_modes` |
| `memory_samples.csv`, `memory_guard.json` | Host memory data from `memory_guard.py` |

A new key in `summary.json` does not show in the page until you change two places:

1. In `dashboard.py`, add a function that turns the summary entry into rows, for example `{"x": length, "value": ...}`. Add its output to the dict that `collect_run()` returns.
2. In a JS module in `reports/dashboard/`, for example `bars.js` or `lines.js`, add a `CHARTS.push({order, id, title, sub, draw})` entry. The `draw` function reads the new field from each run.

Then add a test to `tests/test_dashboard.py`, and rebuild the page with `python dashboard.py`. [dashboard.md](dashboard.md) describes the page.

## Run the tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests do not need a GPU or a model server. They replace the HTTP layer. Some tests patch `requests.post` with `monkeypatch`. Other tests give the benchmark a fake client object with a `generate` method.
