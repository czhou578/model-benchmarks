import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.architecture_flops import (
    detect_features,
    load_architecture_config,
    missing_required_fields,
    normalize_config,
    run_flops_analysis,
    validate_fixed_dimensions,
)


def dense_config(**overrides):
    config = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "tie_word_embeddings": True,
    }
    config.update(overrides)
    return config


def primary_text_config():
    return {
        "model_type": "qwen3_6_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 248320,
        "head_dim": 256,
        "attn_output_gate": True,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "layer_types": ["linear_attention"] * 30 + ["full_attention"] * 10,
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


class NormalizeConfigTests(unittest.TestCase):
    def test_aliases_preserve_provenance_and_expert_width(self):
        config = dense_config(
            num_experts=256,
            num_experts_per_token=8,
            moe_intermediate_size=512,
            intermediate_size=None,
        )
        normalized = normalize_config(config)

        self.assertEqual(normalized.experts_per_token, 8)
        self.assertEqual(normalized.expert_intermediate_size, 512)
        self.assertEqual(
            normalized.source_keys["expert_intermediate_size"],
            "moe_intermediate_size",
        )
        self.assertNotEqual(normalized.expert_intermediate_size, 512 // 256)
        self.assertEqual(normalized.head_dim, 8)
        self.assertEqual(
            normalized.source_keys["head_dim"],
            "derived:hidden_size/num_attention_heads",
        )

    def test_multimodal_wrapper_unwraps_text_and_retains_vision(self):
        wrapper = {
            "model_type": "vision_text_wrapper",
            "text_config": dense_config(),
            "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24},
            "quantization_config": {"quant_method": "fp4"},
        }
        normalized = normalize_config(wrapper, config_source="huggingface")
        features = detect_features(normalized)

        self.assertEqual(normalized.hidden_size, 64)
        self.assertEqual(normalized.source_keys["hidden_size"], "text_config.hidden_size")
        self.assertEqual(normalized.quantization, {"quant_method": "fp4"})
        self.assertEqual(
            normalized.source_keys["quantization"],
            "wrapper.quantization_config",
        )
        self.assertEqual(normalized.vision_config["hidden_size"], 1024)
        self.assertTrue(features.is_multimodal)

    def test_values_are_not_filled_with_architecture_defaults(self):
        normalized = normalize_config({"hidden_size": 64})

        self.assertIsNone(normalized.num_layers)
        self.assertIsNone(normalized.num_attention_heads)
        self.assertIsNone(normalized.num_key_value_heads)
        self.assertIn("num_layers", missing_required_fields(normalized))


class FeatureDetectionTests(unittest.TestCase):
    def test_primary_checkpoint_features_are_composable(self):
        normalized = normalize_config(primary_text_config())
        features = detect_features(normalized)

        self.assertEqual(features.ffn, "moe")
        self.assertEqual(
            features.token_mixers,
            frozenset({"linear_attention", "full_attention"}),
        )
        self.assertTrue(features.is_hybrid)
        self.assertEqual(
            features.layer_counts,
            {"linear_attention": 30, "full_attention": 10},
        )
        self.assertEqual(normalized.num_experts, 256)
        self.assertEqual(normalized.experts_per_token, 8)
        self.assertEqual(normalized.expert_intermediate_size, 512)
        self.assertEqual(normalized.shared_expert_intermediate_size, 512)
        self.assertEqual(missing_required_fields(normalized), [])

    def test_unknown_layer_type_is_explicitly_unsupported(self):
        config = primary_text_config()
        config["layer_types"] = ["linear_attention"] * 39 + ["mystery_mixer"]
        normalized = normalize_config(config)
        features = detect_features(normalized)

        self.assertEqual(features.unsupported_layer_types, ("mystery_mixer",))
        self.assertIn("supported_layer_types", missing_required_fields(normalized))

    def test_mtp_detected_without_mla(self):
        config = dense_config(
            num_nextn_predict_layers=1,
        )
        features = detect_features(normalize_config(config))

        self.assertTrue(features.has_mtp)


class DimensionValidationTests(unittest.TestCase):
    def test_primary_checkpoint_passes(self):
        normalized = normalize_config(primary_text_config())
        self.assertEqual(validate_fixed_dimensions(normalized), [])

    def test_wrong_hidden_size_is_rejected(self):
        config = dense_config(hidden_size=4096, num_experts=256, num_experts_per_token=8, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("hidden_size" in m for m in mismatches))

    def test_wrong_num_layers_is_rejected(self):
        config = dense_config(num_hidden_layers=32, num_experts=256, num_experts_per_token=8, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("num_layers" in m for m in mismatches))

    def test_wrong_vocab_size_is_rejected(self):
        config = dense_config(vocab_size=32000, num_experts=256, num_experts_per_token=8, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("vocab_size" in m for m in mismatches))

    def test_wrong_head_dim_is_rejected(self):
        config = dense_config(head_dim=128, num_experts=256, num_experts_per_token=8, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("head_dim" in m for m in mismatches))

    def test_wrong_num_attention_heads_is_rejected(self):
        config = dense_config(num_attention_heads=32, num_experts=256, num_experts_per_token=8, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("num_attention_heads" in m for m in mismatches))

    def test_wrong_num_experts_is_rejected(self):
        config = dense_config(num_experts=128, num_experts_per_token=8, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("num_experts" in m for m in mismatches))

    def test_wrong_experts_per_token_is_rejected(self):
        config = dense_config(num_experts=256, num_experts_per_token=4, moe_intermediate_size=512)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("experts_per_token" in m for m in mismatches))

    def test_wrong_expert_intermediate_size_is_rejected(self):
        config = dense_config(num_experts=256, num_experts_per_token=8, moe_intermediate_size=1024)
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertTrue(any("expert_intermediate_size" in m for m in mismatches))

    def test_multiple_mismatches_reported(self):
        config = dense_config(
            hidden_size=4096, num_hidden_layers=32, vocab_size=32000,
            num_experts=128, num_experts_per_token=4, moe_intermediate_size=1024,
        )
        normalized = normalize_config(config)
        mismatches = validate_fixed_dimensions(normalized)
        self.assertGreater(len(mismatches), 1)
        field_names = [m.split(":")[0] for m in mismatches]
        self.assertIn("hidden_size", field_names)
        self.assertIn("num_experts", field_names)

    def test_missing_fields_skip_not_fail(self):
        # When a dimension field is None, it should be skipped (missing_required_fields handles it)
        normalized = normalize_config({"hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320, "head_dim": 256, "num_attention_heads": 16, "num_key_value_heads": 2})
        mismatches = validate_fixed_dimensions(normalized)
        self.assertEqual(mismatches, [])


class ConfigLoadingTests(unittest.TestCase):
    def test_local_config_has_precedence_and_is_relative_to_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(primary_text_config()))
            auto_calls = []

            result = load_architecture_config(
                {
                    "architecture_config_path": "config.json",
                    "_config_dir": str(root),
                    "model_id": "remote/model",
                },
                server_config=dense_config(hidden_size=32),
                auto_config_loader=lambda model_id: auto_calls.append(model_id),
            )

        self.assertEqual(result.status, "exact_from_config")
        self.assertTrue(result.config_source.startswith("local:"))
        self.assertEqual(result.normalized_config.hidden_size, 2048)
        self.assertEqual(auto_calls, [])

    def test_huggingface_precedes_complete_server_config(self):
        result = load_architecture_config(
            {"model_id": "remote/model"},
            server_config=dense_config(hidden_size=32),
            auto_config_loader=lambda model_id: primary_text_config(),
        )

        self.assertEqual(result.config_source, "huggingface")
        self.assertEqual(result.normalized_config.hidden_size, 2048)

    def test_wrong_dimension_falls_through_to_next_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # local config: all required fields present but wrong dimensions
            local = primary_text_config()
            local["hidden_size"] = 512
            (root / "config.json").write_text(json.dumps(local))
            result = load_architecture_config(
                {
                    "architecture_config_path": "config.json",
                    "_config_dir": str(root),
                    "model_id": "remote/model",
                },
                auto_config_loader=lambda model_id: primary_text_config(),
            )

        self.assertEqual(result.config_source, "huggingface")
        self.assertEqual(result.normalized_config.hidden_size, 2048)

    def test_complete_server_config_is_last_supported_source(self):
        def unavailable(_model_id):
            raise OSError("offline")

    def test_complete_server_config_is_last_supported_source(self):
        def unavailable(_model_id):
            raise OSError("offline")

        result = load_architecture_config(
            {"model_id": "remote/model"},
            server_config=primary_text_config(),
            auto_config_loader=unavailable,
        )

        self.assertEqual(result.config_source, "server")
        self.assertEqual(result.normalized_config.hidden_size, 2048)
        self.assertTrue(any("offline" in warning for warning in result.warnings))

    def test_incomplete_sources_return_unsupported_without_defaults(self):
        result = load_architecture_config(
            {}, server_config={"hidden_size": 4096}
        )

        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.normalized_config)
        self.assertTrue(any("incomplete" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()


class RunFlopsAnalysisTests(unittest.TestCase):
    def test_primary_checkpoint_returns_exact_from_config(self):
        result = run_flops_analysis(primary_text_config())
        self.assertEqual(result["estimate_status"], "exact_from_config")
        self.assertIn("decode", result["flops"])
        self.assertIn("prefill", result["flops"])
        self.assertIn("assumptions", result["flops"])

    def test_output_schema_has_required_top_level_keys(self):
        result = run_flops_analysis(primary_text_config())
        for key in ("estimate_status", "architecture", "flops", "assumptions", "warnings"):
            self.assertIn(key, result)

    def test_output_schema_has_model_and_layer_counts(self):
        result = run_flops_analysis(primary_text_config())
        self.assertEqual(result["architecture"]["model"], "qwen3.6-35b-a3b")
        self.assertEqual(
            result["architecture"]["layer_counts"],
            {"linear_attention": 30, "full_attention": 10},
        )
        self.assertEqual(result["architecture"]["ffn"], "moe")

    def test_decode_and_prefill_keys_are_sequence_lengths(self):
        result = run_flops_analysis(primary_text_config())
        decode_keys = sorted(result["flops"]["decode"].keys(), key=int)
        prefill_keys = sorted(result["flops"]["prefill"].keys(), key=int)
        self.assertEqual(decode_keys, prefill_keys)
        for key in ("512", "2048", "8192", "32768"):
            self.assertIn(key, result["flops"]["decode"])
            self.assertIn(key, result["flops"]["prefill"])

    def test_incomplete_config_returns_unsupported(self):
        result = run_flops_analysis({"hidden_size": 64})
        self.assertEqual(result["estimate_status"], "unsupported")
        self.assertEqual(result["flops"]["decode"], {})
        self.assertEqual(result["flops"]["prefill"], {})

    def test_wrong_dimensions_returns_unsupported(self):
        config = dense_config(hidden_size=512, num_experts=256, num_experts_per_token=8, moe_intermediate_size=512)
        result = run_flops_analysis(config)
        self.assertEqual(result["estimate_status"], "unsupported")
        self.assertTrue(any("dimensions" in w for w in result["warnings"]))

    def test_custom_lengths(self):
        result = run_flops_analysis(
            primary_text_config(),
            context_lengths=[1024, 4096],
            prefill_lengths=[128, 1024],
        )
        self.assertIn("1024", result["flops"]["decode"])
        self.assertIn("4096", result["flops"]["decode"])
        self.assertIn("128", result["flops"]["prefill"])
        self.assertNotIn("512", result["flops"]["decode"])
        self.assertNotIn("512", result["flops"]["prefill"])
