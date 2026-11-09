"""Tests for the FLOP analysis: the July 2026 Qwen3.6-35B-A3B golden values and Qwen3.8-27B.

The configs come from tests/fixtures. No Hugging Face download, no server.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.architecture_flops import run_flops_analysis

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CONFIG_35B = FIXTURES / "qwen3.6_35b_a3b_config.json"
CONFIG_27B = FIXTURES / "qwen3.8_27b_config.json"
GOLDEN_35B = ROOT / "results/qwen3.6-35b-a3b-nvidia-nvfp4/20260729_151852/flops_analysis.json"
LENGTHS = [512, 2048, 8192, 32768]


def _analyze(config_path: Path, lengths: list[int] = LENGTHS) -> dict:
    return run_flops_analysis(
        {"architecture_config_path": str(config_path)},
        context_lengths=lengths, prefill_lengths=lengths,
    )


def test_qwen36_35b_matches_the_july_run():
    golden = json.loads(GOLDEN_35B.read_text())
    result = _analyze(CONFIG_35B)

    assert result["estimate_status"] == golden["estimate_status"] == "exact_from_config"
    assert result["flops"]["decode"] == golden["flops"]["decode"]
    assert result["flops"]["prefill"] == golden["flops"]["prefill"]
    for key in ("layer_counts", "ffn", "token_mixers", "is_hybrid"):
        assert result["architecture"][key] == golden["architecture"][key]
    assert result["architecture"]["model"] == "qwen3_5_moe"


def test_qwen38_27b_decode_by_hand():
    result = _analyze(CONFIG_27B, [512])
    assert result["estimate_status"] == "exact_from_config"
    assert result["architecture"]["model"] == "qwen3_5"
    assert result["architecture"]["ffn"] == "dense"
    assert result["architecture"]["layer_counts"] == {"linear_attention": 48, "full_attention": 16}

    d, ffn, vocab = 5120, 17408, 248320
    # Full attention: 24 query heads and 4 KV heads of dim 256, 16 layers.
    d_q, d_kv = 24 * 256, 4 * 256
    # Gated DeltaNet: 16 key heads and 48 value heads of dim 128, 48 layers.
    key_dim, value_dim, value_heads = 16 * 128, 48 * 128, 48
    conv_dim = 2 * key_dim + value_dim
    expected = {
        "dense_ffn": 64 * 3 * 2 * d * ffn,
        "attention_qkv_projection_flops": 16 * 2 * d * (d_q + 2 * d_kv),
        "attention_output_projection_flops": 16 * 2 * d_q * d,
        "attention_query_gate_projection_flops": 16 * 2 * d * d_q,
        "attention_score_flops": 16 * 4 * 24 * 256 * 512,
        "linear_projection_flops": 48 * 2 * d * (conv_dim + value_dim + 2 * value_heads),
        "short_conv_flops": 48 * 2 * conv_dim * 4,
        "linear_output_projection_flops": 48 * 2 * value_dim * d,
        "recurrent_state_flops": 48 * 4 * value_heads * 128 * 128,
        "lm_head_flops": 2 * d * vocab,
    }
    decode = result["flops"]["decode"]["512"]
    for name, value in expected.items():
        assert decode[name] == value, name
    assert decode["dense_ffn"] == 34_225_520_640
    assert decode["matmul_flops"] == sum(expected.values()) == 51_600_162_816


def test_qwen38_27b_prefill_dense_ffn_scales_with_length():
    prefill = _analyze(CONFIG_27B, [2048])["flops"]["prefill"]["2048"]
    assert prefill["dense_ffn"] == 2048 * 64 * 3 * 2 * 5120 * 17408
    assert prefill["lm_head_flops"] == 2 * 5120 * 248320
