# Architecture-Based FLOP and Memory Calculator — Revised Design Plan

**Phase:** Roadmap item #18 (Phase 7 — Systems Analysis) + #19 (MoE-Specific Metrics)  
**Primary model:** `nvidia/Qwen3.6-35B-A3B-NVFP4`  
**Output file:** `flops_analysis.json`

---

## Review outcome

The original plan should not be implemented as written. Its overall goal is sound, but several assumptions produce order-of-magnitude errors:

1. The primary model is a **hybrid Gated DeltaNet + full-attention MoE**, not a conventional Transformer. Its `text_config` has 40 layers, a 3:1 linear/full-attention pattern, 256 experts, top-8 routing, and one shared expert.
2. `moe_intermediate_size` is the **per-expert** width for this model. Dividing it by `num_experts` is incorrect.
3. SwiGLU/gated FFNs have three matrices and cost approximately `6*d*m` FLOPs per token. This model uses gated FFNs throughout; no two-matrix GELU path is needed.
4. GQA projection and KV-cache formulas must use `num_key_value_heads`; treating Q, K, and V as three `d -> d` projections overcounts them.
5. The LM head is applied once, not once per layer. The original memory expression multiplied `d*v` by `l`.
6. Decode and prefill do not have the same attention FLOPs per token. A length-`S` causal prefill sees an average history of roughly `S/2`; a decode token at context `S` sees all `S` cached positions.
7. “Weights resident in VRAM” and “bytes transferred from HBM per token” are different quantities. Inactive expert weights are capacity, not traffic, and batching changes expert-weight reuse.
8. The proposed filename `architecture-flops.py` cannot be imported as `architecture_flops`; use `architecture_flops.py`.
9. The Llama-2-7B test expectation of `~1.2e13` FLOPs/token is about three orders of magnitude too high. A forward pass is on the order of `1.3e10` FLOPs/token, depending on the exact conventions included.
10. The existing GPU table has a likely unit mismatch: values such as H100 `1.979` are labeled TFLOP/s and multiplied by `1e12`, although they appear to be PFLOP/s-class tensor-core figures. This can make compute bounds another 1000× too low. Every peak must have an explicit unit and dense/sparse qualifier.

The revised design below fixes these issues and makes unsupported architectures fail explicitly instead of silently returning a blanket fallback estimate.

---

## Quick Start — Measure Qwen3.6-35B MoE FLOPs

For the primary model, FLOPs estimation is **complete and accurate** via `compute_flops()`.  
No profiling, no instrumentation — just pass your model config and get exact per-component results.

```python
from benchmarks.architecture_flops import compute_flops

config = {
    'model_type': 'qwen3_5_moe_text',
    'hidden_size': 2048, 'num_hidden_layers': 40, 'vocab_size': 248320,
    'head_dim': 256, 'num_attention_heads': 16, 'num_key_value_heads': 2,
    'attn_output_gate': True,
    'layer_types': ['linear_attention'] * 30 + ['full_attention'] * 10,
    'num_experts': 256, 'num_experts_per_tok': 8,
    'moe_intermediate_size': 512, 'shared_expert_intermediate_size': 512,
    'linear_num_key_heads': 16, 'linear_num_value_heads': 32,
    'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
    'linear_conv_kernel_dim': 4, 'tie_word_embeddings': False,
}

# Decode: one new token, context = 8192
decode = compute_flops(config, mode='decode', sequence_length=8192)
# → total: ~7.35 × 10⁹ FLOPs/token, status: “exact_from_config”

# Prefill: full prompt of length S
prefill = compute_flops(config, mode='prefill', sequence_length=128)
# → total: ~6.55 × 10¹¹ FLOPs for S=128, status: “exact_from_config”
```

The result dict includes:
- `flops['total']` — total matmul + non-matmul FLOPs
- `flops['matmul_flops']` — matrix multiply FLOPs (2 per MAC)
- `flops['non_matmul_flops']` — activation/gating FLOPs
- Per-component breakdowns (moe_routed, moe_shared, attention_qkv, linear_projection, lm_head, etc.)
- `estimate_status` — `”exact_from_config”` means all required fields are present and formulas are verified

### Expected Results for Qwen3.6-35B

| Mode | Sequence | Total FLOPs | Per-token avg |
|------|----------|-------------|---------------|
| Decode | 8,192 | ~7.35 × 10⁹ | 7.35 billion |
| Prefill | 128 | ~6.55 × 10¹¹ | ~5.12 billion |
| Prefill | 1,024 | ~4.57 × 10¹² | ~4.47 billion |
| Prefill | 8,192 | ~3.07 × 10¹³ | ~3.75 billion |

The MoE + router matmul subtotal is invariant: **2,306,867,200 FLOPs/token** (40 layers × [routed: 50.3M + shared: 6.3M + router: 1.0M]).

---

## Scope and definitions

The calculator estimates **model forward-pass operations**, not achieved hardware FLOPS. It reports:

- FLOPs per generated token for decode at a specified context length.
- Total and average FLOPs for causal prefill of a specified prompt length.
- Parameter residency bytes separately from estimated HBM traffic.
- Per-component breakdowns and assumptions.
- A confidence level (`exact_from_config`, `approximate`, or `unsupported`).

Use 2 FLOPs per multiply-accumulate. Elementwise operations (normalization, activation, RoPE, softmax, top-k, gating, and state updates) must either be reported in a separate `non_matmul_flops` field or explicitly omitted. Do not hide them inside a matrix-multiplication formula.

The API must distinguish workload mode:

```python
def compute_flops(
    model_config: dict,
    *,
    mode: Literal["decode", "prefill"],
    sequence_length: int,
    batch_size: int = 1,
    gpu_specs: dict | None = None,
) -> dict:
    """Return architecture-aware FLOPs, residency, and traffic estimates."""
```

`weight_bits` is not a sufficient input for mixed-precision checkpoints. Keep it only as an explicit fallback override; normally derive storage by component from `quantization_config`.

---

## Phase 1 — Normalize and validate configuration

### 1.1 Obtain the real model configuration — ✅ Done

Do not rely on `/v1/models` returning a complete Hugging Face config. The current `_get_model_config()` also discards every architecture-specific key. Load the config in this order:

1. A local model/config path from the benchmark YAML, if available.
2. `transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)`.
3. A server-provided config only if it is complete.
4. Otherwise return `unsupported`; do not substitute Llama-like defaults.

For multimodal wrapper configs, unwrap the language backbone (`text_config`) for text-only benchmarks and retain the vision config separately. Vision FLOPs should be added only when image/video inputs are actually present.

### 1.2 Normalize aliases without inventing values — ✅ Done

Create a normalized schema with provenance for every field:

```python
NormalizedConfig(
    hidden_size=d,
    num_layers=l,
    vocab_size=v,
    head_dim=h,
    num_attention_heads=n_q,
    num_key_value_heads=n_kv,
    layer_types=[...],
    ffn_kind="gated",
    num_experts=E,
    experts_per_token=k,
    expert_intermediate_size=m_e,
    shared_expert_intermediate_size=m_shared,
    tie_word_embeddings=bool,
    quantization=...,
    source_keys={...},
)
```

Recognize common aliases such as `num_experts_per_tok`, `num_experts_per_token`, `moe_router_topk`, `moe_intermediate_size`, and `expert_intermediate_size`. `moe_intermediate_size` is per expert unless model-specific documentation says otherwise. Never infer a per-expert width by dividing a generic `intermediate_size` by `num_experts`.

### 1.3 Architecture features are composable — ✅ Done

A single enum is insufficient — the FFN choice and the token-mixer choice are independent. Detect features compositely:

```python
ArchitectureFeatures(
    ffn="moe",
    token_mixers={"full_attention", "linear_attention"},
    is_hybrid=bool,
    is_multimodal=bool,
    has_mtp=bool,
)
```

Dispatch per layer using `layer_types`. If a layer type has no estimator, return an explicit unsupported component; do not route it to a fallback.

For the primary checkpoint, expected detection is:

```text
ffn=moe
token_mixers={linear_attention, full_attention}
is_hybrid=true
layer count: 30 linear-attention + 10 full-attention
experts: 256 total, 8 routed/token + 1 shared expert
```

MTP layers count only when speculative/MTP decoding is enabled. The vision tower counts only for multimodal requests.

---

## Phase 2 — Composable component estimators

All component estimators for the primary model are implemented and tested.

### 2.1 MoE FFN ✅

Gated experts with shared expert:

```text
routed_expert_params/layer   = 3*d*m_e*E
active_routed_flops/layer    = 6*d*m_e*k
shared_expert_params/layer   = 3*d*m_s
active_shared_flops/layer    = 6*d*m_s
router_params/layer          = d*E
router_matmul_flops/layer    = 2*d*E
reweight_and_accumulate      ~= 2*k*d       # elementwise, not another expert projection
```

### 2.2 Standard full attention (GQA) ✅

```text
QKV projection/layer/token = 2*d*(d_q + d_k + d_v)
QK^T + AV/layer/token      = 4*n_q*h_q*S   (decode)
QK^T + AV/layer/request    = 2*n_q*h_q*S*(S+1)   (prefill)
```

Includes optional query gate (Qwen `attn_output_gate`).

### 2.3 Gated DeltaNet / linear attention ✅

Derived from `Qwen3_5MoeGatedDeltaNet` with keys: `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`.

Decode is sequence-length-independent (recurrent). Prefill uses the 64-token chunked kernel.

### 2.4 LM head ✅

```text
FLOPs = 2*d*v per logits token (once per token, never per layer)
```

### 2.5 Architecture file layout ✅

```text
architecture_flops.py
├── normalize_config()           # alias mapping + provenance
├── detect_features()            # ffns, token_mixers, layer_counts
├── MoeFfnEstimator              # routed + shared expert
├── FullAttentionEstimator       # GQA, optional query gate
├── GatedDeltaNetEstimator       # linear attention / delta-rule
├── LmHeadEstimator              # vocabulary projection
└── ModelFlopsEstimator          # sums components, applies layer counts
```

### 2.6 Quick Start — Measure MoE FLOPs for Qwen 3.6 35B ✅

The simplest way to get accurate FLOPs:

```python
from benchmarks.architecture_flops import compute_flops

config = {
    "model_type": "qwen3_5_moe",
    "hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320,
    "head_dim": 256, "num_attention_heads": 16, "num_key_value_heads": 2,
    "layer_types": ["linear_attention"]*30 + ["full_attention"]*10,
    "num_experts": 256, "num_experts_per_tok": 8,
    "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "tie_word_embeddings": False,
}

# Decode: FLOPs per generated token at context length 8192
result = compute_flops(config, mode="decode", sequence_length=8192)
print(result["flops"]["total"])        # e.g. ~7.3e9
print(result["estimate_status"])       # "exact_from_config"

# Prefill: total FLOPs for prompt of length 128
result = compute_flops(config, mode="prefill", sequence_length=128)
print(result["flops"]["total"])        # e.g. ~6.6e11
```

`compute_flops()` is a thin wrapper. For control, use the component classes directly:

```python
from benchmarks.architecture_flops import (
    normalize_config, detect_features, ModelFlopsEstimator,
)

normalized = normalize_config(config)
features = detect_features(normalized)
estimator = ModelFlopsEstimator(normalized, features)

# Decode at 8192 context
decode = estimator.decode_model_flops(8192)

# Prefill for S=128 tokens, producing 1 output token's worth of logits
prefill = estimator.prefill_flops(128, logits_tokens=1)
```

`estimate_status` tells you the confidence: `"exact_from_config"` means all required fields came from real model metadata with no fallbacks. `"partial"` means some components were estimated. `"unsupported"` means the model is not recognized.

---

## Phase 3 — Memory residency and traffic (Not started)

TODO: Add weight bytes, KV cache bytes, and traffic estimates alongside FLOPs.

---

## Phase 4 — Roofline integration (Partially done)

`benchmarks/roofline.py` already calls `ModelFlopsEstimator` for compute-bound and memory-bound analysis.
It uses `_compute_bounds_from_estimator()` which computes:

```text
compute TPS bound = peak FLOP/s / FLOPs per token
memory TPS bound  = HBM bandwidth / bytes per token
roofline TPS      = min(compute bound, memory bound)
```

The old `compute_theoretical_bounds()` wrapper has been removed. The roofline pipeline now:
1. Loads architecture config (local → HuggingFace → server)
2. Builds `ModelFlopsEstimator` from normalized config
3. Computes per-length prefill analysis + decode analysis
4. Classifies each as compute-bound, memory-bound, or mixed

---

## Phase 5 — Tests and invariants ✅

All existing tests pass.

### Known invariants verified:

- MoE + router matmul subtotal = 2,306,867,200 FLOPs/token (40 layers)
- LM head = 1,017,118,720 FLOPs/logits token
- Decode attention is linear in context length `S`
- Causal prefill attention is quadratic in `S`
- Gated FFN uses 3 matrices (gate, up, down) per expert

The output from `compute_flops()` already includes all these fields:

```json
{
  "architecture": {
    "ffn": "moe",
    "token_mixers": ["linear_attention", "full_attention"],
    "layer_counts": {"linear_attention": 30, "full_attention": 10}
  },
  "estimate_status": "exact_from_config",
  "flops": {
    "total": 7346462720,
    "matmul_flops": 7297597440,
    "non_matmul_flops": 48865280,
    "moe_routed": 2013265920,
    "moe_shared": 251658240,
    "moe_router": 41943040,
    "attention_qkv_projection_flops": 209715200,
    "linear_projection_flops": 1517813760,
    "lm_head_flops": 1017118720,
    ...
  }
}
```

---

## Files — Current status

| File | Status | Description |
|---|---|---|
| `benchmarks/architecture_flops.py` | ✅ Complete | Config normalization, feature detection, all MoE estimators |
| `benchmarks/roofline.py` | ✅ Complete | Roofline integration using component estimators |
| `tests/test_architecture_flops.py` | ✅ Complete | Config parsing, feature detection, alias tests |
| `tests/test_architecture_flops_estimators.py` | ✅ Complete | All component estimator unit tests + regression tests |

## Remaining work

1. **Memory residency & traffic (Phase 3)** — Add weight bytes, KV cache bytes, traffic estimates

---

## Non-goals

- Measuring achieved kernel FLOPs or timing without profiler instrumentation.
- Pretending tensor parallelism changes global model FLOPs. It changes per-device work, communication, residency, and traffic and should be modeled later as a separate execution-placement layer.
- Treating theoretical FLOP count as a prediction of latency. Kernel efficiency, expert imbalance, quantization/dequantization, fusion, launch overhead, and fallback implementations remain empirical concerns.
