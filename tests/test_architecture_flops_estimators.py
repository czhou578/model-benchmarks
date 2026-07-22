"""Regression tests for the Qwen3.6-35B applicable Phase 2 estimators."""

from __future__ import annotations

import unittest

from benchmarks.architecture_flops import (
    FullAttentionEstimator,
    GatedDeltaNetEstimator,
    LmHeadEstimator,
    ModelFlopsEstimator,
    MoeFfnEstimator,
    compute_flops,
    detect_features,
    normalize_config,
)


def primary_text_config():
    """Exact text_config dimensions from nvidia/Qwen3.6-35B-A3B-NVFP4."""
    return {
        "model_type": "qwen3_6_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 248320,
        "head_dim": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "attn_output_gate": True,
        "layer_types": (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            * 10
        ),
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": False,
    }


class MoeFfnEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.moe = MoeFfnEstimator(
            d=2048,
            num_experts=256,
            experts_per_token=8,
            expert_intermediate_size=512,
            shared_expert_intermediate_size=512,
            ffn_kind="gated",
        )

    def test_plan_matmul_invariant(self):
        breakdown = self.moe.decode_flops_breakdown()
        self.assertEqual(breakdown["routed_experts"], 50_331_648)
        self.assertEqual(breakdown["shared_expert"], 6_291_456)
        self.assertEqual(breakdown["router"], 1_048_576)
        subtotal = sum(
            breakdown[name]
            for name in ("routed_experts", "shared_expert", "router")
        )
        self.assertEqual(subtotal * 40, 2_306_867_200)

    def test_qwen_shared_gate_and_elementwise_terms(self):
        result = self.moe.decode_flops()
        self.assertEqual(result.matmul_breakdown["shared_expert_gate"], 4_096)
        self.assertEqual(
            result.non_matmul_breakdown["routed_reweight_and_accumulate"],
            2 * 8 * 2048,
        )
        self.assertEqual(
            result.non_matmul_breakdown["shared_expert_gate_multiply"], 2048
        )
        self.assertTrue(any("top-k" in item for item in result.omitted_non_matmul))

    def test_parameter_counts_use_per_expert_width(self):
        params = self.moe.parameter_breakdown()
        self.assertEqual(
            params["routed_experts"], 256 * 3 * 2048 * 512
        )
        self.assertEqual(params["shared_expert"], 3 * 2048 * 512)
        self.assertEqual(params["router"], 2048 * 256)
        self.assertEqual(params["shared_expert_gate"], 2048)


class FullAttentionEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.attention = FullAttentionEstimator(
            d=2048,
            num_heads=16,
            num_kv_heads=2,
            head_dim=256,
            query_gate=True,
        )

    def test_gqa_projection_and_qwen_gate(self):
        breakdown = self.attention.projection_flops()
        d_q, d_k, d_v = 4096, 512, 512
        self.assertEqual(
            breakdown["attention_qkv_projection_flops"],
            2 * 2048 * (d_q + d_k + d_v),
        )
        self.assertEqual(
            breakdown["attention_query_gate_projection_flops"],
            2 * 2048 * d_q,
        )
        self.assertEqual(
            breakdown["attention_output_projection_flops"],
            2 * d_q * 2048,
        )

    def test_decode_attention_is_linear_in_context(self):
        self.assertEqual(
            self.attention.decode_attn_flops(4096),
            4 * self.attention.decode_attn_flops(1024),
        )

    def test_causal_prefill_attention_formula(self):
        S = 1024
        result = self.attention.prefill_flops(S)
        self.assertEqual(
            result.matmul_breakdown["attention_score_flops"],
            2 * 16 * 256 * S * (S + 1),
        )


class GatedDeltaNetEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.delta = GatedDeltaNetEstimator(
            d=2048,
            num_key_heads=16,
            num_value_heads=32,
            key_head_dim=128,
            value_head_dim=128,
            kernel_size=4,
        )

    def test_qwen_dimensions_and_decode_breakdown(self):
        self.assertEqual(self.delta.key_dim, 2048)
        self.assertEqual(self.delta.value_dim, 4096)
        self.assertEqual(self.delta.conv_dim, 8192)
        self.assertEqual(self.delta.state_elements, 32 * 128 * 128)

        result = self.delta.decode_flops()
        self.assertEqual(
            result.matmul_breakdown["linear_projection_flops"],
            2 * 2048 * (8192 + 4096 + 64),
        )
        self.assertEqual(
            result.matmul_breakdown["short_conv_flops"],
            2 * 8192 * 4,
        )
        self.assertEqual(
            result.matmul_breakdown["recurrent_state_flops"],
            4 * 32 * 128 * 128,
        )
        self.assertEqual(
            result.matmul_breakdown["linear_output_projection_flops"],
            2 * 4096 * 2048,
        )

    def test_decode_is_sequence_length_independent(self):
        self.assertEqual(
            self.delta.decode_flops().matmul_flops,
            self.delta.decode_flops().matmul_flops,
        )

    def test_chunked_prefill_scales_by_chunk_count(self):
        one_chunk = self.delta.prefill_flops(64)
        two_chunks = self.delta.prefill_flops(128)
        self.assertEqual(two_chunks.matmul_flops, 2 * one_chunk.matmul_flops)
        self.assertEqual(
            one_chunk.matmul_breakdown["linear_projection_flops"],
            self.delta.projection_flops(tokens=64)["linear_projection_flops"],
        )

    def test_state_shape_drives_cache_bytes(self):
        expected = 4 * (
            32 * 128 * 128 + (2 * 2048 + 4096) * 4
        )
        self.assertEqual(self.delta.state_bytes(), expected)


class LmHeadEstimatorTests(unittest.TestCase):
    def test_untied_projection(self):
        lm = LmHeadEstimator(d=2048, vocab_size=248320, tied=False)
        result = lm.decode_flops()
        self.assertEqual(result.matmul_flops, 1_017_118_720)
        self.assertGreater(result.weight_bytes, 0)

    def test_tying_changes_residency_not_compute(self):
        lm = LmHeadEstimator(d=2048, vocab_size=248320, tied=True)
        result = lm.decode_flops()
        self.assertEqual(result.matmul_flops, 1_017_118_720)
        self.assertEqual(result.weight_bytes, 0)

    def test_prefill_logits_tokens(self):
        lm = LmHeadEstimator(d=2048, vocab_size=248320)
        self.assertEqual(
            lm.prefill_flops(logits_tokens=3).matmul_flops,
            3 * 1_017_118_720,
        )


class ModelFlopsEstimatorTests(unittest.TestCase):
    def setUp(self):
        normalized = normalize_config(primary_text_config(), config_source="test")
        self.estimator = ModelFlopsEstimator(
            normalized, detect_features(normalized)
        )

    def test_decode_uses_40_30_10_1_multipliers(self):
        result = self.estimator.decode_model_flops(8192)
        moe = self.estimator._moe.decode_flops()
        attention = self.estimator._full_attn.decode_flops(8192)
        delta = self.estimator._delta.decode_flops()

        self.assertEqual(
            result["moe_routed"],
            moe.matmul_breakdown["routed_experts"] * 40,
        )
        self.assertEqual(
            result["attention_qkv_projection_flops"],
            attention.matmul_breakdown["attention_qkv_projection_flops"] * 10,
        )
        self.assertEqual(
            result["linear_projection_flops"],
            delta.matmul_breakdown["linear_projection_flops"] * 30,
        )
        self.assertEqual(result["lm_head_flops"], 1_017_118_720)

    def test_prefill_uses_40_30_10_and_configurable_logits(self):
        S = 128
        result = self.estimator.prefill_flops(S, logits_tokens=3)
        attention = self.estimator._full_attn.prefill_flops(S)
        delta = self.estimator._delta.prefill_flops(S)
        self.assertEqual(
            result["attention_score_flops"],
            attention.matmul_breakdown["attention_score_flops"] * 10,
        )
        self.assertEqual(
            result["recurrent_state_flops"],
            delta.matmul_breakdown["recurrent_state_flops"] * 30,
        )
        self.assertEqual(result["lm_head_flops"], 3 * 1_017_118_720)
        self.assertEqual(result["logits_tokens"], 3)

    def test_totals_and_status(self):
        result = self.estimator.decode_model_flops(8192)
        self.assertEqual(
            result["total"],
            result["matmul_flops"] + result["non_matmul_flops"],
        )
        self.assertEqual(result["estimate_status"], "exact_from_config")
        self.assertEqual(
            result["layer_counts"],
            {"linear_attention": 30, "full_attention": 10},
        )

    def test_public_compute_api(self):
        result = compute_flops(
            primary_text_config(),
            mode="prefill",
            sequence_length=128,
            logits_tokens=2,
        )
        self.assertEqual(result["estimate_status"], "exact_from_config")
        self.assertEqual(result["flops"]["logits_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
