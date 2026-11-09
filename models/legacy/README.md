# Legacy model configs

These five configs describe the Qwen3.6-35B-A3B NVFP4 runs of July 2026. The repo keeps them as a record of those runs.

`batch_run.py` does not run them. It reads only the top level of `models/`.

## Status

The configs ran on vLLM 0.25.1. Nobody has tested them on vLLM 0.33.0, the current target. They differ from the current Qwen3.8-27B config in these ways:

- They do not set `VLLM_USE_FASTOKENS: "0"`. vLLM fails at startup on a host that exports `VLLM_USE_FASTOKENS=1` without the `fastokens` package.
- They do not pin the KV cache with `--kv-cache-memory-bytes`. Four of them set `--gpu-memory-utilization 0.70`, about 85 GB of the 121 GB unified memory.
- Three of them set `--enable-prefix-caching`. The decode and reasoning benchmarks send no `cache_salt`, so their requests can hit the prefix cache.
- `qwen3.6_35b_unsloth_nvfp4.yml` puts `--speculative-config` in `server.command`. With `--compare-spec`, its "disabled" variant still uses MTP.

| Config | Weights | Speculative decoding | Prefix caching | Published run |
|---|---|---|---|---|
| `qwen3.6_35b_nvidia_nvfp4.yml` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `speculative_config` key, for `--compare-spec` only | on | `results/qwen3.6-35b-a3b-nvidia-nvfp4/20260729_151852` |
| `qwen3.6_35b_nvidia_nvfp4_no_spec.yml` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | none | off | `results/qwen3.6-35b-a3b-nvidia-nvfp4-nospec/20260729_153414` |
| `qwen3.6_35b_redhat_nvfp4.yml` | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | none | on | `results/qwen3.6-35b-a3b-redhat-nvfp4/20260729_164741` |
| `qwen3.6_35b_unsloth_nvfp4.yml` | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | MTP in `server.command` | on | `results/qwen3.6-35b-a3b-unsloth-nvfp4/20260729_161944` |
| `qwen3.6_35b_unsloth_nvfp4_no_spec.yml` | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | none | off | `results/qwen3.6-35b-a3b-unsloth-nvfp4-nospec/20260729_155440` |

## The configs and the July runs

Later PRs changed these configs. For example, they added `--enable-per-request-metrics` and renamed `name`. The files in a run dir record what that run used:

- `model_config.yml` is the exact model config of the run.
- `resolved_server.json` holds the exact server command.

## Run a legacy config

CAUTION: These configs can fill the unified memory of a DGX Spark and make the host unresponsive. Read [docs/dgx-spark.md](../../docs/dgx-spark.md) before you run one.

1. Copy the config to `models/`.
2. Add `VLLM_USE_FASTOKENS: "0"` to `server.environment`.
3. Replace `--gpu-memory-utilization` with a `--kv-cache-memory-bytes` pin. Use [qwen3.8_27b_nvidia_nvfp4.yml](../qwen3.8_27b_nvidia_nvfp4.yml) as the example.
4. Start the run through `memory_guard.py`, as [docs/dgx-spark.md](../../docs/dgx-spark.md) shows.
