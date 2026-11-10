"""The FLOP analysis entry point: run_flops_analysis, its status and warnings."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from benchmarks.architecture_flops.config import (
    ConfigLoadResult,
    detect_features,
    missing_required_fields,
    normalize_config,
)
from benchmarks.architecture_flops.loader import load_architecture_config
from benchmarks.architecture_flops.model import ModelFlopsEstimator


def _model_label(raw: Mapping[str, Any]) -> str:
    """Return the text model_type of the config, for example ``qwen3_5_moe``.

    The ``_text`` suffix of a multimodal text config is removed. The label
    names the architecture family. It does not name the parameter count.
    """
    model_type = str(raw.get("model_type", "unknown")).lower()
    return model_type.removesuffix("_text")


def run_flops_analysis(
    model_config: dict[str, Any],
    *,
    auto_config_loader: Callable[[str], Any] | None = None,
    context_lengths: list[int] | None = None,
    prefill_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Return a complete FLOPs analysis document for the model config.

    If the config is already complete (has all required fields) it is used
    directly.  Otherwise the function resolves the config from a local file
    or from Hugging Face via ``load_architecture_config``. All dimensions
    come from the config.

    Args:
        model_config: the benchmark YAML config dict or a complete model config.
        auto_config_loader: optional callable(str) -> config for HF resolution.
        context_lengths: decode context lengths to analyze.
        prefill_lengths: prefill prompt lengths to analyze.
    """
    if context_lengths is None:
        context_lengths = [512, 2048, 8192, 32768]
    if prefill_lengths is None:
        prefill_lengths = [512, 2048, 8192, 32768]

    # Fast path: if the config is already complete (has all required fields),
    # use it directly instead of trying file/HF lookups.
    required = (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
    )
    if all(k in model_config and model_config[k] is not None for k in required):
        normalized = normalize_config(model_config)
        missing = missing_required_fields(normalized)
        if missing:
            return {
                "estimate_status": "unsupported",
                "architecture": {"model": _model_label(model_config)},
                "flops": {"decode": {}, "prefill": {}},
                "assumptions": [],
                "warnings": ["missing configuration fields: " + ", ".join(missing)],
            }
        features = detect_features(normalized)
        config_result = ConfigLoadResult(
            status="exact_from_config",
            config_source="provided_config",
            raw_config=dict(model_config),
            normalized_config=normalized,
            features=features,
        )
    else:
        config_result = load_architecture_config(
            model_config,
            auto_config_loader=auto_config_loader,
        )

    if config_result.status == "unsupported":
        return {
            "estimate_status": "unsupported",
            "architecture": {"model": _model_label(model_config) if isinstance(model_config, dict) else "unknown"},
            "flops": {"decode": {}, "prefill": {}},
            "assumptions": [],
            "warnings": list(config_result.warnings),
        }

    assert config_result.normalized_config is not None
    assert config_result.features is not None
    normalized = config_result.normalized_config
    features = config_result.features

    try:
        estimator = ModelFlopsEstimator(normalized, features)
    except ValueError as exc:
        return {
            "estimate_status": "unsupported",
            "architecture": {"model": _model_label(normalized.raw_text_config)},
            "flops": {"decode": {}, "prefill": {}},
            "assumptions": [],
            "warnings": [str(exc)],
        }

    decode_results: dict[str, Any] = {}
    prefill_results: dict[str, Any] = {}
    for length in sorted(set(context_lengths)):
        key = str(length)
        decode_results[key] = estimator.decode_model_flops(length)

    for length in sorted(set(prefill_lengths)):
        key = str(length)
        prefill_results[key] = estimator.prefill_flops(length)

    return {
        "estimate_status": estimator.estimate_status,
        "architecture": {
            "model": _model_label(normalized.raw_text_config),
            "layer_counts": dict(features.layer_counts),
            "ffn": features.ffn,
            "token_mixers": sorted(features.token_mixers),
            "is_hybrid": features.is_hybrid,
        },
        "flops": {
            "decode": decode_results,
            "prefill": prefill_results,
        },
        "assumptions": estimator.assumptions,
    }
