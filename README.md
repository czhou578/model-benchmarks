# Model Benchmarks

An automated benchmarking suite for large language models — designed to compare different model weights, serving configurations, and inference parameters across a battery of production-relevant metrics.

Built for GPU-level inference tuning (vLLM on single-node systems like NVIDIA DGX Spark), but extensible to any OpenAI-compatible endpoint.

---

## Quick Start

```bash
# Run all models defined in models/*.yml
python batch_run.py

# Run a single model
python batch_run.py --model qwen3.6_35b_unsloth_nvfp4.yml

# Dry run (see what would execute without running)
python batch_run.py --dry-run
```

Each run produces a timestamped directory under `results/<model_name>/<timestamp>/` containing structured JSON outputs.

---

## Purpose

This suite answers a fundamental question: **how much does the serving configuration and weight source matter, beyond just the model name?**

The same base model (Qwen3.6-35B-A3B-NVFP4) has been benchmarked across five different configurations — different weight providers (NVIDIA, unsloth, RedHat), different MOE backends, with and without speculative decoding, with and without prefix caching. The data reveals that **config differences can produce 2–4× performance swings** — dwarfing the differences between model variants.

### Key Questions

- **Tool calling readiness**: How well do models handle tool use in production-like scenarios?
- **Bottleneck identification**: Is prefill or decode the true bottleneck for a given use case?
- **Speculative decoding ROI**: Does MTP spec decoding help or hurt depending on the config?
- **Long context cost**: What happens at 32K and 64K context?
- **Concurrency scaling**: How does throughput and latency scale with multiple simultaneous users?
- **Reasoning overhead**: How much chain-of-thought does the model incur per task?

---

## Benchmark Suite

Each run executes 10 benchmarks. Every benchmark writes a JSON file to the run directory.

| Benchmark | File | Measures |
|-----------|------|----------|
| **Latency Sweep** | `latency.json` | TTFT across prompt lengths 32–16K tokens (avg, median, p95, p99, prefill TPS) |
| **Deep Context** | `deep_context.json` | TTFT & prefill throughput at 32K and 64K context lengths |
| **TTFT Breakdown** | `ttft_breakdown.json` | Queue time, prefill time, and decode overlap components |
| **Decode Speed** | `decode.json` | Token generation rate at 512/1K/2K output lengths (avg, peak, min, median) |
| **Concurrency** | `concurrency.json` | Throughput & latency at concurrency levels 1, 2, 4, 8, 16 |
| **Reasoning** | `reasoning.json` | Thinking token count, answer token count, think/answer ratio |
| **Tool Calling** | `tool_calling.json` | 85 tasks across 16 categories with multi-layered scoring (see below) |
| **Prefill Scaling** | `prefill_scaling.json` | GPU power, memory, utilization during prefill |
| **FLOP Analysis** | `flops_analysis.json` | Theoretical compute requirements from model architecture |
| **Spec Comparison** | `spec_comparison.json` | Speculative vs. baseline decode (when run with `--compare-spec`) |

### Tool Calling: 3 Phases, 85 Tasks

The most comprehensive benchmark. Tests tool selection, parameter validation, multi-tool orchestration, and error recovery across three phases:

| Phase | Category | What It Tests |
|-------|----------|---------------|
| **Phase 1** | Single-tool calls | Tool selection + parameter validation |
| **Phase 2** | Multi-tool chaining | Multi-turn chains — model must orchestrate multiple tools in sequence |
| **Phase 3** | Schema compliance & Error recovery | Strict JSON Schema constraints + model self-correction after errors |

**Scoring** produces three levels of output:
- **Per-task**: `correct`, `tool_correct`, `params_complete`, `params_valid`, `details`
- **Per-category**: pass rate, tool accuracy, param completeness/validity averages
- **Aggregate**: weighted **composite score** across 6 dimensions (tool accuracy, param completeness, param correctness, multi-tool, schema compliance, refusal) + **failure mode counts**

[See the blog post](BLOG_POST.md) for a full breakdown of results.

### GPU Telemetry

The `GpuMonitor` class samples `nvidia-smi` every second during benchmark execution, capturing:
- GPU utilization (%) and memory usage (MiB) — average and peak
- Power draw (W) — average and peak
- Energy consumed (Wh) — total and incremental above idle

All raw samples are written to `gpu_samples.csv`.

---

## Architecture

```
models/*.yml          → YAML configs: vLLM server flags + benchmark sweep params
core_runner.py        → Orchestration: start/stop vLLM, run benchmarks, collect telemetry
benchmarks/*.py       → Individual benchmark implementations
vllm_server.py        → Managed vLLM lifecycle (start, stop, health-check)
batch_run.py          → Batch orchestrator: discover YAMLs, run sequentially, produce summary
```

**Workflow:** The runner starts a vLLM server (managed mode) or connects to a running server (external mode), waits for the model to load, then runs each benchmark in sequence. Results are written as structured JSON.

---

## Adding a New Model

Create a YAML file in `models/`:

```yaml
name: my-model-name
server:
  mode: managed            # "managed" = auto-start vLLM; "external" = connect to running server
  command:
    - vllm
    - serve
    - my-org/MyModel
    - --host
    - "127.0.0.1"
    - --port
    - "8000"
    # ... all vLLM flags
  startup_timeout_s: 600
  shutdown_timeout_s: 30

endpoint:
  base_url: "http://127.0.0.1:8000"
  model_name: "MyModel"

# Optional: override benchmark parameters
# prompt_lengths: [32, 128, 512, 2048, 8192, 16384]
# decode_lengths: [512, 1024, 2048]
# concurrency_levels: [1, 2, 4, 8, 16]
```

```bash
python batch_run.py --model my-model-name.yml
```

---

## Adding Tool-Calling Tasks

Tasks are defined in `datasets/tool_calling_tasks.yaml`. Each task specifies:
- `id` — unique identifier
- `category` — scoring category (`single_tool`, `multi_tool`, `error_recovery`, etc.)
- `prompt` — the user prompt
- `tools` — available tool definitions (JSON Schema)
- `expected` — expected tool name and parameters
- `scoring` — category-specific scoring rules (e.g. `strategy: "assumption"`)

---

## Output Structure

```
results/
└── <model_name>/
    └── <YYYYMMDD_HHMMSS>/
        ├── environment.json       # GPU, driver, CUDA, PyTorch, vLLM versions
        ├── model_config.yml       # Config used for this run
        ├── resolved_server.json   # Full vLLM command and metadata
        ├── latency.json           # TTFT sweep across prompt lengths
        ├── deep_context.json      # 32K/64K context performance
        ├── ttft_breakdown.json    # TTFT component analysis
        ├── decode.json            # Decode speed (avg/peak/min/median tok/s)
        ├── concurrency.json       # Multi-level concurrency throughput & latency
        ├── reasoning.json         # Chain-of-thought token analysis
        ├── tool_calling.json      # 85-task tool calling results
        ├── prefill_scaling.json   # GPU telemetry during prefill
        ├── flops_analysis.json    # Theoretical FLOP estimates
        ├── gpu_samples.csv        # Per-second GPU telemetry (raw)
        ├── vllm.log               # vLLM server stdout/stderr
        └── summary.json           # Aggregate run summary
```

---

## Results

| Metric | Best Config | Value |
|--------|-------------|-------|
| Tool accuracy | unsloth (baseline) | **41.2%** (35/85) |
| Decode speed | unsloth (spec) | **85.6 tok/s** avg, 181 peak |
| Concurrency peak | nvidia (baseline) | **552 tok/s** at l16 |
| Consistent latency | RedHat (Docker) | Most deterministic at short prompts |

See [SUMMARY.md](SUMMARY.md) for aggregated results. See [BLOG_POST.md](BLOG_POST.md) for a detailed analysis with comparison tables.

---

## Dependencies

```
pip install requests pyyaml
```

Optional (more accurate token counting):
```
pip install tiktoken
```

Requires: vLLM installed and available on the system path, `nvidia-smi` for GPU telemetry.

---

## Design Decisions

- **Sequential by default** — Models share a single GPU; parallel runs can cause port conflicts or OOM.
- **JSON-first output** — Every benchmark writes structured JSON for easy programmatic comparison.
- **Managed vLLM lifecycle** — The runner starts and stops vLLM itself in managed mode. In external mode, it connects to a pre-existing process.
- **Cache-salt support** — Random cache salts prevent prefix-cache from contaminating latency measurements.
- **Idempotent runs** — Each run writes to a new timestamped directory. Old results are never overwritten.
