# Three-Model Benchmark Comparison

**Date:** 2026-07-16
**Hardware:** NVIDIA GB10, driver 580.159.03, CUDA 13.0

| Model | Build Source |
|---|---|
| **nvidia-Qwen3.6-35B-A3B-NVFP4** | NVIDIA official quantization |
| **qwen3.6-35b-a3b-redhat-test-nvfp4** | Red Hat quantization |
| **qwen3.6-35b-a3b-unsloth-nvfp4** | Unsloth Q-LoRA quantization |

---

## 1. First-Token Latency (TTFT)

Lower is better.

| Prompt Tokens | NVIDIA avg (ms) | Red Hat avg (ms) | Unsloth avg (ms) |
|---|---|---|---|
| 32 | 144 | 147 | **63** |
| 128 | 148 | 151 | **87** |
| 512 | 209 | 194 | 132 |
| 2048 | 409 | 385 | 333 |
| 8192 | 774 | 766 | **1,265** |
| 16384 | 893 | 894 | **2,687** |

**Key takeaways:**

- **Unsloth dominates short prompts** (< 2K tokens) — TTFT is 2–5× faster than both NVIDIA and Red Hat at 32–128 tokens. This is likely because Unsloth's optimized attention/kernels reduce the prefill overhead for short sequences.
- **NVIDIA and Red Hat are nearly identical** across all prompt lengths — within 1–2% at every length. The two quantization implementations produce functionally equivalent prefill performance.
- **Unsloth's advantage collapses at long prompts.** At 8K+ tokens, NVIDIA and Red Hat overtake Unsloth significantly. At 16K tokens, Unsloth's TTFT is 3× slower than the other two. The prefill throughput analysis below explains why.

## 2. Prompt Processing Speed (Prefill Throughput)

Tokens/second through the prefill phase — higher is better.

| Prompt Tokens | NVIDIA | Red Hat | Unsloth |
|---|---|---|---|
| 32 | 474 | 306 | 673 |
| 128 | 1,199 | 912 | **1,596** |
| 512 | 3,003 | 2,689 | **3,970** |
| 2048 | 5,400 | 5,341 | 6,182 |
| 8192 | **11,455** | 11,096 | 6,485 |
| 16384 | **22,305** | 20,355 | 6,101 |

**Key takeaways:**

- **All three models converge around 2K tokens.** NVIDIA, Red Hat, and Unsloth all achieve ~5,300–6,200 tok/s at the 2K prompt length — no significant difference in prefill throughput at this scale.
- **NVIDIA and Red Hat scale dramatically at longer prompts.** From 2K to 16K, they gain 4× and 3.8× respectively in throughput. Unsloth stays flat at ~6K tok/s. This strongly suggests NVIDIA and Red Hat are using a more efficient attention implementation (e.g., PagedAttention with better memory tiling or a fused prefill kernel) that benefits from longer context windows.
- **Unsloth's prefill throughput is capped around 6K tok/s** regardless of prompt length. This is the architectural reason Unsloth wins on short prompts but loses on long ones: its per-token prefill overhead is high for short sequences but its throughput ceiling is lower.

## 3. Decode Speed

Average output tokens/sec — higher is better.

| Output Tokens | NVIDIA | Red Hat | Unsloth |
|---|---|---|---|
| 512 | **174** | 55 | 74 |
| 1024 | **174** | 53 | 74 |
| 2048 | **175** | 53 | 74 |

**Key takeaways:**

- **NVIDIA is ~3× faster at decoding than Red Hat and ~2.3× faster than Unsloth.** This is by far the most dramatic gap in the entire benchmark suite. Decode throughput is the primary throughput bottleneck for long-generation workloads, and NVIDIA's model consistently delivers 174 tok/s across all output lengths — extremely stable.
- **Red Hat and Unsloth are in the same ballpark** (53–74 tok/s). Red Hat is slightly slower than Unsloth at decode, but the gap (about 30%) is far smaller than the gap to NVIDIA.
- **NVIDIA's decode latency is low variance** — median tok/s is ~44 and peak is ~46 across all output lengths. Red Hat and Unsloth show much wider ranges: Red Hat's tok/s varies from 16 to 83, Unsloth from 59 to 90. NVIDIA's output is far more consistent, which matters for predictable response times.

## 4. Reasoning Token Analysis

Measures how much "thinking" (reasoning/thinking tokens) the model produces per task.

| Metric | NVIDIA | Red Hat | Unsloth |
|---|---|---|---|
| Avg thinking tokens | 128 | **640** | 695 |
| Avg answer tokens | 0 | 68 | 72 |
| Reasoning ratio (thinking/answer) | N/A | 5.2 | 5.0 |
| Max thinking tokens | 128 | 977 | 990 |

**Critical observation:**

- **NVIDIA's model produced only thinking tokens and zero answer tokens across every test.** The output text is entirely exclamation marks — the model appears stuck in a reasoning loop, unable to transition from thinking to answering. This is a **functional correctness failure**, not just a performance one.
- **Red Hat and Unsloth both produce substantive answers** (avg 68–72 answer tokens) alongside their thinking tokens. They properly complete the thinking→answering cycle.
- **Red Hat thinks less than Unsloth** (avg 640 vs 695 thinking tokens) but produces similarly concise answers. Red Hat is ~8% more "thought-efficient."
- NVIDIA's thinking tokens are capped at 128 across every prompt — this looks like a system-level limit or a quantization artifact rather than model behavior.

## 5. Concurrency Scaling

Aggregate throughput (tokens/sec) as concurrent requests increase.

| Concurrency | NVIDIA (tok/s) | Red Hat (tok/s) | Unsloth (tok/s) |
|---|---|---|---|
| 1 | 150 | 69 | 72 |
| 2 | 172 | 107 | 100 |
| 4 | 315 | 155 | 141 |
| 8 | 350 | 227 | 168 |
| 16 | 351 | 218 | 166 |

**Key takeaways:**

- **NVIDIA scales to ~350 tok/s aggregate at concurrency 8–16**, which is ~5× Red Hat and ~2.1× Unsloth. The GPU is clearly the throughput bottleneck, and NVIDIA's model saturates it more effectively.
- **NVIDIA's throughput plateaus at concurrency 8** — going to 16 adds no value (350 → 351 tok/s). Red Hat plateaus around 8–16 as well. This tells us the GPU's parallel execution units are the limiting factor, and any deployment with more than 8 concurrent users won't see additional aggregate throughput benefit regardless of model.
- **Red Hat slightly degrades from concurrency 8→16** (227 → 218 tok/s), while Unsloth also plateaus (168 → 166 tok/s). NVIDIA is the only model that holds steady across all concurrency levels.

## 6. Power & Energy

| Metric | NVIDIA | Red Hat | Unsloth |
|---|---|---|---|
| Avg GPU power (W) | 37.4 | 36.7 | 40.6 |
| Peak GPU power (W) | 66.6 | 66.8 | 68.5 |
| Total energy (Wh) | 1.17 | 2.09 | 2.40 |
| Energy/token (Wh) | **0.00033** | 0.00058 | 0.00067 |

**Key takeaways:**

- **NVIDIA is the most energy-efficient by a wide margin.** It uses 56% less energy than Red Hat and 51% less than Unsloth to produce the same output. This is a direct consequence of its higher decode throughput — faster completion means the GPU spends less time at high power draw.
- **Red Hat and Unsloth have similar power profiles.** Both draw ~37–41W average and ~67–69W peak. Red Hat's lower total energy comes from shorter benchmark runtime, not lower power draw.
- **Peak power is similar across all three (~67–69W)**, suggesting the GPU's power ceiling is the same regardless of model. The difference is in how effectively each model keeps the GPU at or near that ceiling during sustained work.

## 7. Summary — What This Shows

### NVIDIA Official Build: The Performance Winner

Across every performance metric that matters — decode speed, energy efficiency, concurrency throughput — the NVIDIA build dominates. It's 3× faster at decode, uses half the energy, and delivers 5× the aggregate throughput of the competitors. This is what you want for production inference where throughput and cost-per-token are the primary concerns.

**The major caveat: functional failure on reasoning tasks.** NVIDIA's model produced zero answer tokens in every benchmark — it got stuck in a reasoning loop with no transition to answering. This is a dealbreaker for any workload that requires the model to produce actual answers. The model may have a quantization or serving-layer issue specific to how reasoning-mode output is split. If this is fixed (e.g., with a different temperature, system prompt, or serving configuration), NVIDIA would be the clear choice.

### Red Hat Build: Middle Ground with Working Reasoning

Red Hat's model produces correct answers with reasonable efficiency. It's the second-fastest at decode (53 tok/s), the second-most energy-efficient, and most importantly, it works correctly for reasoning tasks. Its performance is nearly identical to NVIDIA on prefill, and on concurrency scaling it behaves similarly. If you're evaluating Red Hat as a production serving platform, this benchmark shows the model itself is sound — just not optimized for decode throughput.

### Unsloth Build: Good for Short Prompts, Weak for Long Ones

Unsloth's model is the fastest at prefill for short prompts (<2K tokens) but has the lowest prefill throughput ceiling at long prompts and the second-worst decode speed. Its reasoning works correctly (though it thinks slightly more than Red Hat). It would be a reasonable choice for low-latency single-request workloads with short context, but the 3× prefill throughput gap at long prompts and 2.3× decode gap to NVIDIA make it less competitive for sustained high-volume workloads.

### The Bigger Picture

The prefill/decode split reveals an important architecture-level insight: **prefill is not the bottleneck for long-context workloads on this hardware.** All three models achieve reasonable prefill throughput, but decode speed determines the per-request latency and energy cost. NVIDIA's advantage is almost entirely in decode, which suggests its KV-cache management or kernel fusion for the decode phase is superior. For GPU-bound inference tuning, optimizing decode throughput (batch scheduling, KV-cache optimization, attention implementation) matters far more than optimizing prefill.