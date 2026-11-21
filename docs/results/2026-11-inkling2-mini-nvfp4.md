# Inkling 2 Mini NVFP4: First Run (November 2026)

This write-up covers one full run of `thinkingmachines/Inkling-2-Mini` on the DGX Spark, on 2026-11-20. It compares the run with run D of [2026-11-qwen3.8-27b-nvfp4.md](2026-11-qwen3.8-27b-nvfp4.md), the Qwen3.8-27B baseline.

The run dir is not in the repo. Nobody can check these numbers against published data. The numbers below come from the local run dir `results/inkling2-mini-26b-a3b-thinkingmachines-nvfp4/20261120_182702`.

## The model

Inkling 2 Mini is a looped MoE model from Thinking Machines, released on 2026-11-20.

- 24 layers. Layers 6 to 17 run three times per token with shared weights, so each token passes through 48 layers.
- 26.4B parameters, 3.4B active per pass. Each MoE layer routes each token to 8 of 128 experts.
- Mixed NVFP4 and MXFP8 weights, about 20 GB. The checkpoint includes a DSpark speculator (`mtp.safetensors`).
- A hybrid of local (window 1024) and global attention. The context limit is 1M tokens. The vLLM recipe sets 262,144 tokens on the DGX Spark.

## The run

| Setting | Value |
|---|---|
| Model config | [models/inkling2_mini_26b_a3b_thinkingmachines_nvfp4.yml](../../models/inkling2_mini_26b_a3b_thinkingmachines_nvfp4.yml) |
| Server | Docker image `vllm/vllm-openai:inkling-2-mini` (arm64), pinned by digest `sha256:c2dc759f…` |
| vLLM | `0.1.dev25855+g7cb5dae22` (`vllm_server_version`). No vLLM release loads the model yet. The support is in vllm-project/vllm pull request 69028. |
| Harness | commit `d6d7ac0`, `harness.schema_version` 5 (era E5), `git_dirty` false |
| Driver / CUDA | 580.173.02 / 13.0 (the image has torch 2.14.0+cu130) |
| Scope | full suite, `--compare-spec` |
| Run time | 44.9 min |
| Memory guard | floor 8 GiB, warning 16 GiB, 5 s samples. Free at start 118.24 GiB, minimum 71.28 GiB. |

The model config in the repo starts docker through `sh -c`, so that the mounts can use `$HOME`. The run used the same docker arguments with absolute paths. The vLLM arguments are the same.

### Server flags

| Setting | Value |
|---|---|
| KV cache | `--kv-cache-memory-bytes 20G`, `--kv-cache-dtype fp8_e4m3`. The pool holds 302,275 tokens without DSpark and 285,129 tokens with DSpark. |
| Limits | `--max-model-len 262144`, `--max-num-seqs 16`, `--max-num-batched-tokens 8192` |
| MoE and graphs | `--moe-backend humming`, decode CUDA graphs for 1, 2, 4, 8, and 16 requests, no `torch.compile` |
| Attention | Selected by the model: `INKLING_SCONV_SWA` and `FLASH_ATTN` (version 2) |
| Other flags | `--no-enable-prefix-caching`, `--limit-mm-per-prompt '{"image": 0}'`, `--enable-per-request-metrics`, `--generation-config vllm` |
| Parsers | `--reasoning-parser inkling`, `--tool-call-parser inkling` |
| Spec decoding | None in the main server. The spec-on server adds `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`. |
| Reasoning effort | The chat template default, 0.9 ("high") |

The 16G KV pin of the Qwen3.8-27B config is too small. One request of 262,144 tokens with the DSpark draft needs 18.39 GiB, so vLLM refuses to start.

## Results

"Qwen3.8" is run D, `results/qwen3.8-27b-nvidia-nvfp4/20261109_143107`. Read [Comparability](#comparability) before you use the ratios.

### Latency (client TTFT, median)

| Prompt tokens | Inkling 2 Mini | Qwen3.8 |
|---|---|---|
| 32 | 0.157 s | 0.115 s |
| 128 | 0.192 s | 0.130 s |
| 512 | 0.273 s | 0.213 s |
| 2048 | 0.674 s | 0.650 s |
| 8192 | 2.317 s | 3.038 s |
| 16384 | 4.652 s | 6.403 s |
| 32768 (deep context) | 9.773 s | 14.515 s |
| 65536 (deep context) | 21.986 s | 36.369 s |

Source: `latency.json` (10 requests per length) and `deep_context.json` (5 requests per length). Inkling is slower on short prompts and faster from 8192 tokens. The p99 at 512 tokens is 2.07 s, against a median of 0.27 s. The first requests of a new shape compile Triton kernels, which is a probable cause.

### Decode and speculative decoding (tok/s, mean)

| Output tokens | Inkling decode | Inkling DSpark off | Inkling DSpark on | Qwen3.8 decode | Qwen3.8 MTP on |
|---|---|---|---|---|---|
| 512 | 37.63 | 37.72 | 39.21 (+4.0%) | 12.22 | 24.56 |
| 1024 | 37.42 | 37.41 | 36.19 (−3.3%) | 12.21 | 22.91 |
| 2048 | 36.99 | 37.15 | 44.49 (+19.8%) | 12.20 | 23.34 |

Source: `tok_per_sec_avg` in `decode.json` and `spec_comparison.json`.

- Inkling decodes about three times faster than Qwen3.8. It is a MoE with 3.4B active parameters per pass, and Qwen3.8 is a dense 27B model.
- DSpark gives little gain at temperature 0. In the spec-on server log, the mean acceptance length per 10 s interval has a median of 1.99, with a range of 1.80 to 2.97, over 8 intervals. MTP doubles the decode rate of Qwen3.8.

### Concurrency throughput (tok/s)

| Level | Inkling 2 Mini | Qwen3.8 |
|---|---|---|
| 1 | 36.8 | 12.2 |
| 2 | 56.7 | 23.3 |
| 4 | 87.7 | 44.7 |
| 8 | 136.4 | 82.9 |
| 16 | 219.7 | 145.3 |

Source: `aggregate_throughput_tok_s` in `concurrency.json`. Each level sent 16 requests with a distinct prompt of 256 tokens and 256 output tokens. All requests succeeded.

### Prefill (engine tok/s, median of 5)

| Prompt tokens | Inkling 2 Mini | Qwen3.8 |
|---|---|---|
| 512 | 2496 | 2874 |
| 2048 | 3629 | 3391 |
| 8192 | 4137 | 2738 |

Source: `engine_prefill_tps` in `prefill_scaling.json`. The effective rate is `null` for Inkling, see [Caveats](#caveats). The TTFT breakdown gives a server TTFT after the queue of 129, 206, 563, 1977, and 8751 ms at 128, 512, 2048, 8192, and 32768 tokens.

### Reasoning (6 prompts)

| Metric | Inkling 2 Mini | Qwen3.8 |
|---|---|---|
| Thinking tokens: avg / median / max | 664.2 / 198.5 / 3080 | 180.8 / 73.0 / 713 |
| Answer tokens: avg | 216.3 | 378.3 |
| Prompts cut by the 4096-token cap | 1 | 0 |

Source: `reasoning.json`. The palindrome code prompt thought for 3080 tokens and ended with `finish_reason: length`. The model card warns that the model can stay in long reasoning loops.

### Tool calling

| Metric | Inkling 2 Mini | Qwen3.8 |
|---|---|---|
| Passed | 49/85 (57.6%) | 51/85 (60.0%) |
| Composite | 4.7416 of 6 | 4.8338 of 6 |
| `single_tool` | 18/21 | 18/21 |
| `multi_tool` | 8/16 | 9/16 |
| `schema_compliance` | 14/20 | 14/20 |
| `error_recovery` | 3/17 | 3/17 |
| `ambiguous`, `refusal` | 4/9, 2/2 | 5/9, 2/2 |

Source: `tool_calling.json`. The difference of 2 tasks is inside the noise of about ±10 points for 85 tasks. The most frequent failure modes of Inkling are `wrong_turn_count` (18), `wrong_tool_sequence` (14), and `error_no_retry` (14).

### Energy

Decode energy per output token: 0.00025 Wh for Inkling (0.9033 Wh for 3584 tokens) and 0.00088 Wh for Qwen3.8. Source: `decode_energy` in `summary.json`.

## Comparability

- The two runs differ in the model, the vLLM build, the server flags, and the harness era (E5 against E4). The ratios show the model and the runtime together.
- Era E5 changes only the clarification check of the `ambiguous` tasks and the `request_failed` count. The other metrics use the same method as E4. See [../comparability.md](../comparability.md).
- The KV pins differ: 20G for Inkling and 16G for Qwen3.8.
- The Docker build is not a vLLM release. A later vLLM release can give other numbers.

## Caveats

- The client TTFT of the TTFT breakdown is `NaN`, and the effective prefill rate is `null`. Both benchmarks send `max_tokens=1`. The one output token of Inkling opens the thinking block and has no visible text, so the client sees no first token. The server timings and the engine prefill rate are valid.
- `environment.json` records `torch_version: null`, because PyTorch is in the container, not on the host.
- The FLOP analysis gives `status: unsupported`. The estimator does not know the looped `inkling2_loop` architecture.
- All benchmarks use temperature 0. The default of the model is 1.0, and the DSpark acceptance depends on the temperature.
- The harness sends text only. The run does not measure image input.
