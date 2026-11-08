# DGX Spark Notes

The published results come from an NVIDIA DGX Spark. This page lists the facts about that system that affect a run.

## Unified memory

- The GPU is an NVIDIA GB10.
- The CPU and the GPU share 121 GB of unified memory.
- `nvidia-smi` reports GPU memory as `[N/A]`. Thus `environment.json` records `gpu_memory_total` as `[N/A]`.

The vLLM server, the harness, and the operating system use the same memory pool. A KV cache pool that is too large can leave too little memory for the host.

## vLLM environment

Install vLLM in its own virtual environment. The model configs target vLLM 0.33.0 in a venv at `~/venvs/vllm033`. This path is only an example. Use your own path in the commands below.

1. Make the venv and install vLLM:

   ```bash
   uv venv --python 3.12 ~/venvs/vllm033
   uv pip install -p ~/venvs/vllm033/bin/python "vllm==0.33.0" --torch-backend=cu130
   ~/venvs/vllm033/bin/vllm download-kernels
   ```

2. Install the harness into the same venv: `uv pip install -p ~/venvs/vllm033/bin/python -e .`
3. Put the venv `bin` folder first on `PATH`. Then the `vllm` command in the model config starts the vLLM from that venv.
4. Start the harness with the Python of that venv.

The vLLM 0.33.0 wheel for aarch64 uses CUDA 13.0 and torch 2.14.0. Its CUDA arch list includes the `12.0f` family target, which runs on the sm_121 GPU of the GB10. The first server start compiles kernels and can take several minutes.

The published runs use vLLM 0.25.1 in `~/venvs/vllm025`. A gate run of the Qwen3.8-27B suite on 2026-11-07 compared the two versions. vLLM 0.33.0 was equal or faster in decode, MTP decode, TTFT, concurrency throughput, and prefill. Tool calling stayed within the run-to-run noise. The dashboard marks the vLLM version of each run. See [dashboard.md](dashboard.md).

## Prefix caching

vLLM 0.33.0 turns on prefix caching by default, also for hybrid models. In the vLLM 0.25.1 runs, the Qwen3.8-27B server had prefix caching off. A model config without a prefix caching flag can thus change its behaviour on 0.33.0.

Add `--no-enable-prefix-caching` to `server.command` to keep prefix caching off. The Qwen3.8-27B config and the two `no_spec` Qwen3.6 configs have this flag. The other Qwen3.6 configs set `--enable-prefix-caching`.

## Environment variables

A model config can set environment variables for the vLLM process in `server.environment`. For example, `models/qwen3.8_27b_nvidia_nvfp4.yml` sets `VLLM_USE_FASTOKENS: "0"`.

On the test system, `~/.bashrc` exports `VLLM_USE_FASTOKENS=1`. The vLLM venv has no `fastokens` package, so vLLM then fails at startup. Look for `VLLM_*` variables in your shell profile before a run.

## KV cache size

On the GB10, vLLM 0.25.1 estimates the CUDA graph memory as a negative number (-18 GiB). With `--gpu-memory-utilization` alone, the Qwen3.8-27B run got a 47.5 GiB KV pool. Only 35 GiB of memory stayed free.

Set the KV pool size with `--kv-cache-memory-bytes`. The Qwen3.8-27B model config uses `16G`. This pool holds 512,023 tokens without speculative decoding and 458,752 tokens with MTP K=3. Both are more than the 262K maximum model length.

With this flag, vLLM 0.33.0 skips its memory profile. The vLLM 0.33.0 gate run kept the 16G pin, so it does not show whether 0.33.0 still estimates the CUDA graph memory wrongly. Keep the pin. It also keeps the KV capacity equal to the vLLM 0.25.1 runs.

## Memory guard

CAUTION: A run that fills the unified memory can make the host unresponsive. Start each large run through `memory_guard.py`.

`memory_guard.py` runs a command and reads `/proc/meminfo`, because `nvidia-smi` gives no GPU memory value. The guard does these steps:

1. It refuses to start when `MemAvailable` is below `--min-start-gib` (default 40).
2. It samples the memory every `--interval-s` seconds (default 5) into a CSV file.
3. It logs a warning when `MemAvailable` is below `--warn-gib` (default 16).
4. It sends SIGTERM to the command when `MemAvailable` stays below `--floor-gib` (default 8). The limit is `--grace-samples` low samples in a row (default 3).
5. It copies `memory_samples.csv` and `memory_guard.json` into the run dir.

`core_runner.py` handles SIGTERM and stops its managed vLLM server. The guard exits with code 3 when it refuses to start or when it stops the run.

Example, with the venv path from above:

```bash
PATH=~/venvs/vllm033/bin:$PATH ~/venvs/vllm033/bin/python memory_guard.py \
    --floor-gib 8 --warn-gib 16 --min-start-gib 60 -- \
    ~/venvs/vllm033/bin/python -u core_runner.py \
    --model models/qwen3.8_27b_nvidia_nvfp4.yml --compare-spec
```

The guard finds the run dir in the line `writing results to <run dir>` that `core_runner.py` prints. The `-u` flag turns off the output buffer of Python. Then the guard prints each log line at once.

The dashboard reads `memory_samples.csv` for its memory headroom chart. See [dashboard.md](dashboard.md).
