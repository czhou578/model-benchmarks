# Qwen3.6-35B-A3B-NVFP4 Benchmark Report — Trial Runs Comparison (Part 1)

> **Date:** 2026-07-15
> **GPU:** NVIDIA GB10 (DGX Spark) — Driver 580.159.03, CUDA 13.0
> **Framework:** vLLM (serving all three models via OpenAI-compatible endpoint)

---

## 1. What Was Compared

Three independently-built variants of the **Qwen3.6-35B-A3B-NVFP4** model were benchmarked on the same hardware (GB10 GPU):

| Run # | Source / Variant | Serving Configuration | Run Timestamp (UTC) |
|-------|-----------------|----------------------|---------------------|
| A | **NVIDIA official** — `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM with MTP speculative decoding, flashinfer, fastsafetensors, FP8 KV cache | 2026-07-14 19:22:56 |
| B | **RedHat converted** — `qwen3.6-35b-a3b-nvfp4` | Standard vLLM serve (no speculative decoding visible in config) | 2026-07-14 16:20:03 |
| C | **Unsloth converted** — `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | vLLM with flashinfer_b12x MOE backend, FP8 KV cache | 2026-07-14 19:08:44 |

All three runs were performed at **temperature = 0** with identical benchmark prompts.

---

## 2. Environment Consistency

All three runs were executed on the **identical GPU hardware** (NVIDIA GB10, driver 580.159.03, CUDA 13.0), ensuring a fair comparison. The only meaningful difference is the model source / quantization path and the vLLM serving flags.

---

## 3. Decode-Speed Results (tokens/sec)

The core throughput metric — average tokens generated per second — shows the widest spread across all metrics:

| Decode Length | A: NVIDIA (tok/s) | B: RedHat (tok/s) | C: Unsloth (tok/s) | Winner |
|---------------|-------------------|-------------------|-------------------|--------|
| **512 tokens** | 169.96 | **1,353.45** | 74.50 | **RedHat** (~8× NVIDIA) |
| **1,024 tokens** | 174.33 | 339.47 | 405.46 | **Unsloth** |
| **2,048 tokens** | 176.73 | 474.42 | 435.74 | **RedHat** |

### Key observations:

- **RedHat (Run B) dominates at short outputs** (512 tokens): 1,353 tok/s vs NVIDIA's 170 tok/s — an **~8× speedup**. This is the most striking result.
- **Unsloth (Run C) leads at longer outputs** (1,024 / 2,048 tokens): 405–436 tok/s vs NVIDIA's 174–177 tok/s — a **~2.3–2.5× improvement**.
- **NVIDIA's throughput is flat and consistent** across all output lengths (170 → 174 → 177 tok/s), which is expected from a well-optimized reference implementation.
- **RedHat's performance is variable**: 1,353 tok/s at 512 tokens but drops to 339 tok/s at 1,024 tokens. This non-monotonic behavior suggests the model may hit a throughput wall at longer sequences, or there may be an issue with the serving configuration (e.g., early token-limit cutoff).

### Output completeness (actual vs requested tokens):

| Run | 512 requested / actual | 1,024 requested / actual | 2,048 requested / actual |
|-----|----------------------|-------------------------|-------------------------|
| A: NVIDIA | 512 / 512 ✅ | 1,024 / 1,024 ✅ | 2,048 / 2,048 ✅ |
| B: RedHat | 512 / 512 ✅ | 1,024 / **696** ❌ | 2,048 / **554** ❌ |
| C: Unsloth | 512 / 512 ✅ | 1,024 / **596** ❌ | 2,048 / **693** ❌ |

**NVIDIA is the only model that reliably generates all requested tokens.** Both RedHat and Unsloth terminate early at longer output lengths, producing only ~30–60% of the requested tokens. This is a significant reliability concern for production use.

---

## 4. Power Consumption & Energy Efficiency

| Metric | A: NVIDIA | B: RedHat | C: Unsloth |
|--------|-----------|-----------|------------|
| Avg GPU Power | 41.76 W | 39.41 W | 46.03 W |
| Peak GPU Power | 67.34 W | 68.33 W | 68.72 W |
| Total Energy (Wh) | 0.696 | 0.887 | 1.227 |
| Energy / Token | 0.000194 Wh | 0.000503 Wh | 0.000681 Wh |

### Key observations:

- **NVIDIA is by far the most energy-efficient model** — ~2.6× better energy/token than RedHat and ~3.5× better than Unsloth.
- The NVIDIA run consumed only **0.70 Wh** total for the entire benchmark, while Unsloth consumed **1.23 Wh** (77% more) and RedHat consumed **0.89 Wh** (28% more).
- Peak power draw is nearly identical across all three (~67–69 W), meaning the difference is in sustained average draw — NVIDIA runs cooler and more efficiently over time.
- Despite RedHat's higher throughput, its energy efficiency is worse because it terminates early (fewer useful tokens produced per joule).

---

## 5. Reasoning Capability

All three models were tested with two prompts:
1. *"Solve: if a train travels 60 miles in 45 minutes, what is its speed in mph? Show your reasoning."*
2. *"A farmer has 17 sheep, all but 9 die. How many are left? Explain your reasoning step by step."*

### Token counts:

| Run | Thinking Tokens | Answer Tokens (Prompt 1) | Answer Tokens (Prompt 2) |
|-----|----------------|------------------------|------------------------|
| A: NVIDIA | 0 | 0 | 0 |
| B: RedHat | 0 | 170 | 117 |
| C: Unsloth | 0 | 144 | 0 |

### Key observations:

- **No model generated any thinking tokens** (`<anthropic...>` tags) — all produced direct answers, suggesting either the reasoning parser didn't detect thinking tags or the models weren't prompted/finetuned for chain-of-thought.
- **NVIDIA produced zero answer tokens on both prompts** — the model returned empty responses. This is a significant quality concern.
- **RedHat produced non-trivial answers** on both prompts (170 and 117 tokens), indicating some reasoning capability.
- **Unsloth produced one answer** (144 tokens on the math prompt) but returned empty on the sheep puzzle.
- All models show a **thinking-to-answer ratio of 0.0**, meaning none leveraged the Qwen3-style reasoning token mechanism in these trials.

---

## 6. Prefill / First-Token Latency

All three runs show **null values** for every first-token latency and prefill throughput metric (TTFT, prefill tokens/sec at all prompt lengths from 32 to 16,384). This indicates:

- The latency sweep either failed to capture data, the server did not return token-level timing, or the measurement infrastructure did not record it during these trial runs.
- This is a **data gap** that should be addressed in the next round of benchmarking — prefill latency at varying context lengths is a critical metric for real-world serving.

---

## 7. Summary Comparison

| Dimension | A: NVIDIA | B: RedHat | C: Unsloth | Verdict |
|-----------|-----------|-----------|------------|---------|
| **Decode Throughput (512 tok)** | 170 | **1,353** | 75 | RedHat wins on speed |
| **Decode Throughput (2,048 tok)** | 177 | 474 | **436** | RedHat wins; Unsloth close |
| **Output Completeness** | 100% ✅ | 30–60% ❌ | 34–37% ❌ | **NVIDIA only** |
| **Energy Efficiency** | **Best** (0.000194 Wh/tok) | Worse (0.000503) | Worst (0.000681) | **NVIDIA** |
| **Reasoning Quality** | Empty responses | Some answers (170+117 tok) | Partial (144 tok) | RedHat wins |
| **Latency Data** | Missing | Missing | Missing | All need fix |
| **Power Stability** | Stable, low | Variable | Variable | NVIDIA |

---

## 8. Conclusions & Recommendations

### What the data tells us:

1. **NVIDIA's official quantization is the most reliable and energy-efficient** but is **~2–8× slower** than the community-converted variants. The trade-off is clear: NVIDIA prioritizes correctness and consistency over raw throughput.

2. **RedHat's conversion delivers the highest raw throughput** (especially at short outputs) but suffers from early termination — the model cuts off at ~60% of requested length. This is a critical issue that needs investigation (likely a generation stopping criterion, max-length config, or EOS token behavior).

3. **Unsloth's conversion is competitive at longer outputs** (~435 tok/s at 2,048 tokens) but has the same early-termination problem. Unsloth's "Fast" model appears to use optimized attention/backends (flashinfer_b12x) that help at scale.

4. **None of the three models demonstrated reasoning capability** in these trial prompts. This could be due to the reasoning parser not detecting the model's output format, or the models not being in reasoning mode. Further investigation is needed.

5. **Prefill latency was not measured** for any run — this is a critical gap for a complete benchmark picture.

### Recommended next steps:

1. **Fix the latency sweep** — ensure TTFT and prefill throughput are being captured for all prompt lengths.
2. **Investigate early termination** in RedHat and Unsloth — check EOS token probabilities, `max_tokens` handling, and generation stopping criteria.
3. **Test reasoning mode** — explicitly invoke chain-of-thought prompts and verify the reasoning parser correctly identifies thinking tokens.
4. **Increase latency run repeats** from 0 recorded samples toward the planned 10+ for statistical significance.
5. **Add concurrency testing** — measure throughput with multiple simultaneous requests.
6. **Extend to accuracy benchmarks** — HumanEval, MMLU, or similar to compare capability, not just speed.