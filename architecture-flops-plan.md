# Architecture-Based FLOP and Memory Calculator — Revised Design Plan

**Phase:** Roadmap item #18 (Phase 7 — Systems Analysis) + #19 (MoE-Specific Metrics)  
**Primary model:** `nvidia/Qwen3.6-35B-A3B-NVFP4`  
**Output file:** `flops_analysis.json`

---

## Review outcome

The original plan should not be implemented as written. Its overall goal is sound, but several assumptions produce order-of-magnitude errors:

1. The primary model is a **hybrid Gated DeltaNet + full-attention MoE**, not a conventional Transformer MoE. Its `text_config` has 40 layers, a 3:1 linear/full-attention pattern, 256 experts, top-8 routing, and one shared expert.
2. `moe_intermediate_size` is the **per-expert** width for this model. Dividing it by `num_experts` is incorrect.
3. SwiGLU/gated FFNs have three matrices and cost approximately `6*d*m` FLOPs per token. `4*d*m` applies to a two-matrix FFN such as GELU, not to Qwen/Llama-style gated FFNs.
4. GQA projection and KV-cache formulas must use `num_key_value_heads`; treating Q, K, and V as three `d -> d` projections overcounts them.
5. The LM head is applied once, not once per layer. The original memory expression multiplied `d*v` by `l`.
6. Decode and prefill do not have the same attention FLOPs per token. A length-`S` causal prefill sees an average history of roughly `S/2`; a decode token at context `S` sees all `S` cached positions.
7. “Weights resident in VRAM” and “bytes transferred from HBM per token” are different quantities. Inactive expert weights are capacity, not traffic, and batching changes expert-weight reuse.
8. MLA means **Multi-head Latent Attention** and is associated with architectures such as DeepSeek. It is not “Multi-Level Attention,” and Llama 4 should not be used as the default MLA example.
9. The proposed filename `architecture-flops.py` cannot be imported as `architecture_flops`; use `architecture_flops.py`.
10. The Llama-2-7B test expectation of `~1.2e13` FLOPs/token is about three orders of magnitude too high. A forward pass is on the order of `1.3e10` FLOPs/token, depending on the exact conventions included.
11. The existing GPU table has a likely unit mismatch: values such as H100 `1.979` are labeled TFLOP/s and multiplied by `1e12`, although they appear to be PFLOP/s-class tensor-core figures. This can make compute bounds another 1000× too low. Every peak must have an explicit unit and dense/sparse qualifier.

The revised design below fixes these issues and makes unsupported architectures fail explicitly instead of silently returning a dense estimate.

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

### 1.1 Obtain the real model configuration

Do not rely on `/v1/models` returning a complete Hugging Face config. The current `_get_model_config()` also discards every architecture-specific key. Load the config in this order:

1. A local model/config path from the benchmark YAML, if available.
2. `transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)`.
3. A server-provided config only if it is complete.
4. Otherwise return `unsupported`; do not substitute Llama-like defaults.

For multimodal wrapper configs, unwrap the language backbone (`text_config`) for text-only benchmarks and retain the vision config separately. Vision FLOPs should be added only when image/video inputs are actually present.

### 1.2 Normalize aliases without inventing values

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
    ffn_kind="gated" | "two_matrix",
    intermediate_size=m_dense,
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

### 1.3 Architecture features are composable

A single enum such as `dense | moe | mla` is insufficient because MoE is an FFN choice while MLA, full attention, and linear attention are token-mixer choices. Detect independent features:

```python
ArchitectureFeatures(
    ffn="dense" | "moe",
    token_mixers={"full_attention", "mla", "linear_attention"},
    is_hybrid=bool,
    is_multimodal=bool,
    has_mtp=bool,
)
```

Dispatch per layer using `layer_types`. If a layer type has no estimator, return an explicit unsupported component; do not route it to dense attention.

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

Implement components rather than mutually exclusive whole-model subclasses:

```text
architecture_flops.py
├── normalize_config()
├── detect_features()
├── DenseFfnEstimator
├── MoeFfnEstimator
├── FullAttentionEstimator
├── MlaEstimator
├── GatedDeltaNetEstimator
├── LmHeadEstimator
└── ModelFlopsEstimator       # sums the actual per-layer components
```

### 2.1 Dense FFN

Let `d` be hidden size and `m` intermediate size.

| FFN type | Weight parameters | Matmul FLOPs/token |
|---|---:|---:|
| Two-matrix GELU/ReLU | `2*d*m` | `4*d*m` |
| Gated/SwiGLU (`gate`, `up`, `down`) | `3*d*m` | `6*d*m` |

Activation and elementwise multiply costs are smaller but should be listed under `non_matmul_flops` when modeled.

### 2.2 MoE FFN

Let `E` be total routed experts, `k` routed experts per token, `m_e` the per-expert width, and `m_s` the optional shared-expert width. For gated experts:

```text
routed_expert_params/layer   = 3*d*m_e*E
active_routed_flops/layer    = 6*d*m_e*k
shared_expert_params/layer   = 3*d*m_s
active_shared_flops/layer    = 6*d*m_s
router_params/layer          = d*E
router_matmul_flops/layer    = 2*d*E
reweight_and_accumulate      ~= 2*k*d       # elementwise, not another expert projection
```

Use the corresponding two-matrix coefficient for non-gated experts. Softmax and top-k are not multiplied by `k`; report them as an implementation-dependent `O(E)` selection cost.

For the primary checkpoint (`d=2048`, `l=40`, `E=256`, `k=8`, `m_e=512`, `m_s=512`), the gated FFN matmul sanity check is:

```text
routed experts/layer = 6*2048*512*8   = 50,331,648 FLOPs
shared expert/layer  = 6*2048*512     =  6,291,456 FLOPs
router/layer         = 2*2048*256     =  1,048,576 FLOPs
40-layer subtotal                         2,306,867,200 FLOPs/token
```

This subtotal excludes token mixers, LM head, norms, activations, top-k, and optional MTP/vision work. It is a useful invariant, not the complete model answer.

### 2.3 Standard full attention, including GQA

Let:

```text
d_q  = n_q  * h_q
d_k  = n_kv * h_k
d_v  = n_kv * h_v
```

For ordinary ungated Q projection:

```text
QKV projection/layer/token = 2*d*(d_q + d_k + d_v)
output projection          = 2*d_q*d
```

Some models project an additional query/output gate. Represent that extra projection explicitly from the implementation/config rather than assuming three `d -> d` projections.

For decode with one new token and cached context length `S`:

```text
QK^T + AV/layer/token = 4*n_q*h_q*S
```

For a causal prefill of `S` tokens, counting only the triangular causal region:

```text
QK^T + AV/layer/request = 2*n_q*h_q*S*(S + 1)
average per prompt token = 2*n_q*h_q*(S + 1)
```

These formulas logically repeat GQA K/V heads across query groups for attention math, while projection and KV storage use only `n_kv` heads.

### 2.4 Gated DeltaNet / linear attention

This estimator is mandatory for the primary model. It must be derived from the actual `Qwen3_5MoeGatedDeltaNet` projections, convolution, recurrent-state update, and output projection using:

- `linear_num_key_heads`
- `linear_num_value_heads`
- `linear_key_head_dim`
- `linear_value_head_dim`
- `linear_conv_kernel_dim`
- the implementation's state shape and gate projections

Do not approximate these 30 layers as quadratic full attention. Report separately:

```text
linear_projection_flops
short_conv_flops
recurrent_state_flops
linear_output_projection_flops
linear_state_bytes_read
linear_state_bytes_written
```

Decode should have sequence-length-independent recurrent-state work (apart from implementation overhead). Prefill should use the chunked/parallel kernel's operation count. Until these formulas and tests are implemented, the primary model's total result must be marked `partial`, with the known MoE, full-attention, and LM-head terms shown rather than a fabricated total.

### 2.5 Multi-head Latent Attention (MLA)

Treat MLA as another token-mixer component, detected from model-specific keys such as `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, and `qk_rope_head_dim`.

Do not use the old formula. MLA's cached state is commonly a compressed KV latent plus the decoupled RoPE key component, so a representative per-token cache size is based on:

```text
kv_lora_rank + qk_rope_head_dim
```

not `2*kv_lora_dim`. Projection and attention FLOPs depend on whether weight matrices are absorbed into the attention computation. Implement MLA from a specific supported architecture (initially DeepSeek-style), document the execution form, and test against that implementation.

### 2.6 LM head

For an untied dense vocabulary projection:

```text
params = d*v
FLOPs/generated token = 2*d*v
```

Apply it once per evaluated token, never once per transformer layer. It is not necessarily negligible: for the primary model, `2*2048*248320 = 1,017,118,720` FLOPs per token.

For prefill, most serving paths do not materialize logits for every prompt position. The API therefore needs `logits_tokens` (default `1` for serving-style prefill), rather than blindly charging `S` LM heads.

---

## Phase 3 — Memory residency and traffic

Do not expose a single ambiguous `bytes_per_token` calculation. Return two sections:

```json
{
  "residency_bytes": {"weights": 0, "kv_or_state": 0, "other": 0},
  "traffic_bytes": {"weights": 0, "kv_or_state_read": 0, "kv_or_state_write": 0, "activations": 0},
  "traffic_assumptions": {"mode": "decode", "batch_size": 1, "cache_dtype": "fp8"}
}
```

### 3.1 Standard-attention KV cache

With `b_cache` bytes per stored element:

```text
KV residency/layer/sequence = S*(d_k + d_v)*b_cache
decode KV read/layer/step    ~= S*(d_k + d_v)*b_cache
decode KV write/layer/step   = (d_k + d_v)*b_cache
```

Do not hardcode FP16: the current benchmark explicitly requests FP8 KV cache. Include allocator/block overhead separately if known.

### 3.2 Weight traffic and MoE batching

Parameter bytes are not automatically HBM bytes per token:

- Single-token, batch-1 decode usually streams the dense/shared weights and the selected experts.
- For batch size `B`, dense weights may be reused across the batch, so ideal per-token traffic is roughly dense weight bytes divided by `B`.
- MoE traffic depends on the **union of experts selected in each layer**, not simply `k*B` or `k` per token. Report a lower bound (perfect reuse), an expected estimate if a routing model is supplied, and an upper bound capped at all `E` experts.
- Inactive expert weights belong in residency/capacity, not traffic.

### 3.3 Quantization

NVFP4 checkpoints can have FP4 expert/dense weights, FP8 linear-attention weights or activations, higher-precision norms/router/head tensors, and scale metadata. A model-name heuristic of `weight_bits=4` is not accurate enough for roofline traffic.

Read component-level quantization groups when available. Include scales/metadata, or fall back to actual tensor/checkpoint byte counts and mark the traffic estimate approximate. Select the GPU compute ceiling matching the operation's actual precision; do not use TF32 peak for NVFP4/FP8 kernels by default.

### 3.4 Activations

Activation traffic depends on mode, batch size, fusion, and kernel implementation. Do not use `3*l*d*s*4` for decode or assume FP32 Q/K/V buffers. Provide either:

- a documented logical-tensor lower bound, or
- a kernel-aware estimate with explicit dtype and fusion assumptions.

---

## Phase 4 — Roofline integration

Keep compatibility wrappers only if they require the caller to select `mode`; otherwise they perpetuate the current decode/prefill ambiguity.

```python
def estimate_flops_per_token(model_config, sequence_length, *, mode):
    return compute_flops(
        model_config,
        mode=mode,
        sequence_length=sequence_length,
    )["flops"]
```

The roofline calculation should expose bounds rather than false precision:

```text
compute TPS bound = applicable peak FLOP/s / FLOPs per token
memory TPS bound  = HBM bandwidth / estimated traffic bytes per token
roofline TPS      = min(compute bound, memory bound)
```

The ridge point is `peak FLOP/s / bandwidth` in FLOP/byte. Ensure `classify_workload()` compares arithmetic intensity to that quantity. If an existing helper returns the reciprocal (bytes/FLOP), fix the helper or invert it before comparison.

Normalize GPU specifications before using them. Prefer raw `peak_flops_per_second` values, or consistently store true TFLOP/s values and multiply by `1e12`. Add validation tests for known orders of magnitude and record whether each peak assumes structured sparsity. A dense-model roofline must not use a sparse peak unless the executed kernels actually exploit that sparsity.

Add these output fields:

```json
{
  "architecture": {
    "ffn": "moe",
    "token_mixers": ["linear_attention", "full_attention"],
    "layer_counts": {"linear_attention": 30, "full_attention": 10},
    "config_source": "huggingface_text_config"
  },
  "estimate_status": "exact_from_config | approximate | partial | unsupported",
  "assumptions": [],
  "warnings": []
}
```

Do not call the integration a drop-in replacement until result keys and semantics have compatibility tests. Also decide on one output name (`roofline.json` or `flops_analysis.json`) and avoid documenting both as the same artifact.

---

## Phase 5 — Tests and invariants

### 5.1 Formula unit tests

Use small synthetic configs whose results can be calculated by hand:

1. Two-matrix dense FFN: `4*d*m` per layer/token.
2. Gated dense FFN: `6*d*m` per layer/token.
3. GQA projections: `2*d*(d_q+d_k+d_v)`, proving that `n_kv` changes projection FLOPs.
4. Decode attention: linear in context length `S`.
5. Causal prefill attention: quadratic total and approximately half the per-token attention work of decode at the same final `S`.
6. LM head: exactly once per logits token.
7. KV bytes: proportional to `n_kv`, cache dtype bytes, layers, and sequence length.
8. MoE: per-expert width is never divided by `E`; shared expert always executes.

### 5.2 Known-config regression tests

**Llama-2-7B-like gated MHA, `S=2048`:** expect a result on the order of `1.3e10` FLOPs per decode token, not `1.2e13`. Pin an exact value only after deciding whether embeddings are tied and which elementwise operations are counted.

**Primary Qwen checkpoint:** assert configuration parsing and the MoE subtotal:

```text
d=2048, l=40, E=256, k=8, m_e=512, m_s=512, v=248320
30 linear-attention layers, 10 full-attention layers
MoE + router matmul subtotal = 2,306,867,200 FLOPs/token
LM head = 1,017,118,720 FLOPs/logits token
```

Add a regression test for the Gated DeltaNet estimator based on the supported Transformers/vLLM implementation before declaring the primary model estimate complete.

### 5.3 Sanity checks

- Compare sequence-independent matmul FLOPs with `~2 * active matrix parameters`, adjusting for tied embeddings, router/shared experts, and whether the LM head runs.
- Total parameters derived from config should be within a documented tolerance of checkpoint tensor counts.
- Component totals must sum exactly to the reported total.
- No estimator may use a default architecture value without emitting a warning and reducing confidence.
- Decode and prefill results must carry different mode labels and cannot be compared as if their traffic assumptions were identical.

---

## Files to create or modify

| File | Action | Description |
|---|---|---|
| `benchmarks/architecture_flops.py` | **Create** | Config normalization and composable component estimators |
| `benchmarks/roofline.py` | **Modify** | Mode-aware integration; separate residency from traffic |
| `core_runner.py` | **Modify** | Load complete model config and add analysis entry point |
| `benchmarks/__init__.py` | **Modify** | Register the new analysis |
| `tests/test_architecture_flops.py` | **Create** | Hand-calculated formula and known-config regression tests |

---

## Implementation order

1. Normalize/unwrap full configs and preserve source-key provenance.
2. Implement dense/gated FFN, MoE + shared expert, standard MHA/GQA, and LM head.
3. Implement the Qwen Gated DeltaNet estimator; do not claim primary-model completeness before this step.
4. Implement decode/prefill-specific KV/state residency and traffic.
5. Add mixed-quantization storage and precision-aware GPU ceilings.
6. Integrate with `roofline.py` behind compatibility tests.
7. Add DeepSeek-style MLA as a separate, tested token-mixer implementation.
8. Audit and normalize GPU peak units and dense-versus-sparse throughput labels before publishing roofline TPS.

---

## Non-goals

- Measuring achieved kernel FLOPs or timing without profiler instrumentation.
- Pretending tensor parallelism changes global model FLOPs. It changes per-device work, communication, residency, and traffic and should be modeled later as a separate execution-placement layer.
- Treating theoretical FLOP count as a prediction of latency. Kernel efficiency, expert imbalance, quantization/dequantization, fusion, launch overhead, and fallback implementations remain empirical concerns.
