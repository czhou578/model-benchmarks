# Qwen3.6-35B FLOP Counter — Focused Plan

**Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4`
**Workload:** text-only inference
**Output:** `flops_analysis.json`

## Goal

Calculate forward-pass FLOPs for this one model:

- Decode: FLOPs for one generated token at context length `S`.
- Prefill: total and average FLOPs for a prompt of length `S`.
- Show a per-component breakdown.
- Use the model's real hybrid layer layout.

This is an analytical operation count, not measured hardware utilization or a latency prediction.

## Fixed model contract

The calculator supports only the Qwen text backbone identified by
`model_type=qwen3_5_moe_text`. Qwen3.6 uses this Transformers architecture
name.

Required configuration:

| Field | Expected checkpoint value |
|---|---:|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 40 |
| `vocab_size` | 248320 |
| `head_dim` | 256 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 |
| `attn_output_gate` | true |
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 32 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| `tie_word_embeddings` | false |

The 40 `layer_types` entries must resolve to:

- 30 `linear_attention` layers
- 10 `full_attention` layers

Load the configuration from a local `config.json`, Hugging Face
`AutoConfig`, or a complete server response. Unwrap `text_config` when the
checkpoint has a multimodal wrapper. Missing or mismatched required fields
must return `unsupported`; do not use architecture defaults.

## Counting conventions

- One multiply-accumulate is 2 FLOPs.
- Matrix/convolution FLOPs are reported under `matmul_flops`.
- Counted elementwise work is reported under `non_matmul_flops`.
- Unmodeled operations such as softmax, top-k implementation details, RoPE,
  normalization, and transcendental gate functions are listed explicitly in
  `omitted_non_matmul`.
- Decode `sequence_length` is the cached context seen by one new token.
- Prefill `sequence_length` is the complete causal prompt length.
- Serving-style prefill defaults to `logits_tokens=1`.

## Required formulas

### 1. MoE FFN on all 40 layers

For `d=2048`, `E=256`, `k=8`, `m_e=512`, and `m_s=512`:

```text
routed experts/layer = 6*d*m_e*k = 50,331,648
shared expert/layer  = 6*d*m_s   =  6,291,456
router/layer         = 2*d*E     =  1,048,576
shared gate/layer    = 2*d       =      4,096
```

The first three terms must preserve this regression invariant:

```text
40 * (routed + shared + router) = 2,306,867,200 FLOPs/token
```

Report routed reweight/accumulation separately as approximately `2*k*d`
elementwise FLOPs per layer. The shared-expert gate also performs a
`d -> 1` projection and an elementwise multiply.

### 2. Full GQA on 10 layers

```text
d_q = num_attention_heads * head_dim
d_k = d_v = num_key_value_heads * head_dim

QKV projection/token = 2*d*(d_q + d_k + d_v)
Qwen query gate       = 2*d*d_q
output projection     = 2*d_q*d
decode QK^T + AV      = 4*num_attention_heads*head_dim*S
prefill QK^T + AV     = 2*num_attention_heads*head_dim*S*(S+1)
```

The query-gate term is mandatory because this checkpoint has
`attn_output_gate=true`.

### 3. Gated DeltaNet on 30 layers

Use the operation shapes from `Qwen3_5MoeGatedDeltaNet`:

```text
key_dim       = linear_num_key_heads * linear_key_head_dim       = 2048
value_dim     = linear_num_value_heads * linear_value_head_dim   = 4096
conv_dim      = 2*key_dim + value_dim                            = 8192
state_elements= linear_num_value_heads
                * linear_key_head_dim
                * linear_value_head_dim                          = 524,288
```

Count these components separately:

- `linear_projection_flops`: combined QKV, z, a, and b projections
- `short_conv_flops`: depthwise causal convolution over `conv_dim`
- `recurrent_state_flops`: delta-rule state reductions/updates
- `linear_output_projection_flops`: `value_dim -> hidden_size`

Decode uses the recurrent single-token rule and is independent of context
length. Prefill uses the Transformers reference 64-token chunk algorithm;
keep its detailed tiled-operation formula in code rather than duplicating it
in this plan.

### 4. LM head once

```text
LM-head FLOPs = logits_tokens * 2*d*vocab_size
```

For one logits token:

```text
2 * 2048 * 248320 = 1,017,118,720 FLOPs
```

Weight tying changes parameter residency, not the projection computation.

## Model composition

Never build a synthetic layer containing both attention types and multiply it
by 40. Apply these exact multipliers:

| Component | Multiplier |
|---|---:|
| MoE FFN | 40 layers |
| Gated DeltaNet | 30 layers |
| Full GQA | 10 layers |
| LM head | once |

Every reported total must satisfy:

```text
total = matmul_flops + non_matmul_flops
```

and the component breakdown must sum exactly to those totals.

## Minimal API

```python
def compute_flops(
    model_config: dict,
    *,
    mode: Literal["decode", "prefill"],
    sequence_length: int,
    logits_tokens: int = 1,
) -> dict:
    """Return Qwen3.6-35B text-inference FLOPs."""
```

Required result shape:

```json
{
  "estimate_status": "exact_from_config",
  "architecture": {
    "model": "qwen3.6-35b-a3b",
    "layer_counts": {
      "linear_attention": 30,
      "full_attention": 10
    }
  },
  "flops": {
    "mode": "decode",
    "sequence_length": 8192,
    "matmul_flops": 0,
    "non_matmul_flops": 0,
    "total": 0,
    "omitted_non_matmul": [],
    "components": {}
  },
  "assumptions": [],
  "warnings": []
}
```

No GPU specifications, weight precision, bandwidth, or batch-size arguments
are needed to count global model FLOPs.

## Implementation status

- [x] Load and unwrap the real text configuration.
- [x] Validate required Qwen fields without defaults.
- [x] Reject configs whose fixed dimensions do not match this checkpoint.
- [x] Implement MoE, gated GQA, Gated DeltaNet, and LM-head estimators.
- [x] Apply the 40/30/10/1 component multipliers.
- [x] Support decode and 64-token-chunk prefill.
- [x] Expose configurable `logits_tokens`.
- [x] Add exact component and known-config regression tests.
- [x] Simplify the public API to the signature above by removing unused
      `batch_size` and `gpu_specs` arguments.
- [x] Add a direct benchmark entry point that writes `flops_analysis.json`.
- [x] Invoke FLOP calculation directly instead of only through roofline reporting.
- [x] Add a regression test for the final JSON schema and output file.

## Files

| File | Purpose |
|---|---|
| `benchmarks/architecture_flops.py` | The Qwen-only calculator and config validation |
| `core_runner.py` | Invoke the calculator and write `flops_analysis.json` |
| `tests/test_architecture_flops_estimators.py` | Formula, composition, and output regressions |

`benchmarks/roofline.py` is not part of this deliverable. It may consume
`flops_analysis.json` later, but FLOP calculation must work without it.

## Acceptance criteria

1. The real checkpoint config returns `exact_from_config`.
2. Detection returns 30 linear-attention and 10 full-attention layers.
3. The MoE invariant is exactly `2,306,867,200` FLOPs/token.
4. The LM head is exactly `1,017,118,720` FLOPs per logits token.
5. Decode attention grows linearly with context length.
6. Causal full-attention prefill grows quadratically with prompt length.
7. DeltaNet decode is context-length-independent.
8. DeltaNet prefill follows the 64-token chunk count.
9. The 40/30/10/1 composition is covered by regression tests.
10. `flops_analysis.json` matches the documented schema.

## Explicit non-goals

- Dense Transformer or non-MoE models
- MLA or other attention architectures
- Vision/video tower FLOPs
- MTP/speculative decoding FLOPs
- Training/backward-pass FLOPs
- Weight residency, KV-cache capacity, or HBM traffic
- Quantization/dequantization cost
- GPU peak FLOP/s, roofline bounds, throughput, or latency prediction
- Tensor-parallel communication or per-device work
- Runtime profiling or achieved hardware FLOPS
