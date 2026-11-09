"""Tests for the FLOP analysis: the July 2026 Qwen3.6-35B-A3B golden values and Qwen3.8-27B.

The configs come from tests/fixtures. No Hugging Face download, no server.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

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
    for key in ("layer_counts", "ffn", "token_mixers", "is_hybrid"):
        assert result["architecture"][key] == golden["architecture"][key]
    assert result["architecture"]["model"] == "qwen3_5_moe"


# The July runs multiplied these prefill terms by the prompt length two times.
_MIXER_TERMS = (
    "attention_qkv_projection_flops", "attention_output_projection_flops",
    "attention_query_gate_projection_flops", "attention_score_flops",
    "linear_projection_flops", "short_conv_flops", "linear_output_projection_flops",
    "recurrent_state_flops",
)


def test_qwen36_35b_prefill_counts_the_length_once():
    golden = json.loads(GOLDEN_35B.read_text())["flops"]["prefill"]
    result = _analyze(CONFIG_35B)["flops"]["prefill"]

    for length in LENGTHS:
        old, new = golden[str(length)], result[str(length)]
        for name in ("moe_routed", "moe_shared", "moe_router", "moe_shared_gate", "lm_head_flops"):
            assert new[name] == old[name], name
        for name in _MIXER_TERMS:
            assert new[name] == pytest.approx(old[name] / length, rel=1e-12), name
        for name, value in old["non_matmul_breakdown"].items():
            scale = 1 if name.startswith("moe_") else length
            assert new["non_matmul_breakdown"][name] == pytest.approx(value / scale, rel=1e-12)
    # A prefill costs about S decode steps without their lm_head (one logits token).
    decode = json.loads(GOLDEN_35B.read_text())["flops"]["decode"]["512"]
    per_token = decode["total"] - decode["lm_head_flops"]
    assert result["512"]["total"] == pytest.approx(512 * per_token, rel=0.05)


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


def test_qwen38_27b_prefill_by_hand():
    S, d = 2048, 5120
    prefill = _analyze(CONFIG_27B, [S])["flops"]["prefill"][str(S)]
    assert prefill["dense_ffn"] == S * 64 * 3 * 2 * d * 17408
    assert prefill["lm_head_flops"] == 2 * d * 248320
    # 16 full-attention layers, each over all S tokens.
    assert prefill["attention_qkv_projection_flops"] == 16 * S * 2 * d * (6144 + 2 * 1024)
    assert prefill["attention_score_flops"] == 16 * 2 * 24 * 256 * S * (S + 1)
    # 48 linear-attention layers, each over all S tokens.
    assert prefill["linear_projection_flops"] == 48 * S * 2 * d * (10240 + 6144 + 96)
    assert prefill["linear_output_projection_flops"] == 48 * S * 2 * 6144 * d


def _fake_transformers(monkeypatch, calls: list[dict]) -> None:
    config = json.loads(CONFIG_27B.read_text())

    class AutoConfig:
        @staticmethod
        def from_pretrained(reference, **kwargs):
            calls.append({"reference": reference, **kwargs})
            return config

    module = types.ModuleType("transformers")
    module.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.mark.parametrize(("opt_in", "expected"), [(None, False), (True, True)])
def test_hugging_face_lookup_needs_an_opt_in_for_remote_code(monkeypatch, opt_in, expected):
    calls: list[dict] = []
    _fake_transformers(monkeypatch, calls)
    cfg = {"model_id": "Qwen/Qwen3.8-27B"}
    if opt_in is not None:
        cfg["architecture_trust_remote_code"] = opt_in

    result = run_flops_analysis(cfg, context_lengths=[512], prefill_lengths=[512])

    assert calls == [{"reference": "Qwen/Qwen3.8-27B", "trust_remote_code": expected}]
    assert result["estimate_status"] == "exact_from_config"


def test_missing_transformers_gives_a_clear_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = run_flops_analysis({"model_id": "Qwen/Qwen3.8-27B"})

    assert result["estimate_status"] == "unsupported"
    assert any("transformers is not installed" in w for w in result["warnings"])
