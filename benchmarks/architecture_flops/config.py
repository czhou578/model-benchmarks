"""Architecture configuration loading, normalization, and feature detection.

This module deliberately contains no fallback architecture.  Callers either get
fields supported by their source configuration or an explicit ``unsupported``
result describing what is missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedConfig:
    hidden_size: int | None = None
    num_layers: int | None = None
    vocab_size: int | None = None
    head_dim: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    layer_types: tuple[str, ...] = ()
    ffn_kind: str | None = None
    intermediate_size: int | None = None
    num_experts: int | None = None
    experts_per_token: int | None = None
    expert_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    tie_word_embeddings: bool | None = None
    quantization: dict[str, Any] | None = None
    source_keys: dict[str, str] = field(default_factory=dict)
    config_source: str = "unknown"
    vision_config: dict[str, Any] | None = None
    raw_text_config: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_types"] = list(self.layer_types)
        return result


@dataclass(frozen=True)
class ArchitectureFeatures:
    ffn: str | None
    token_mixers: frozenset[str]
    is_hybrid: bool
    is_multimodal: bool
    has_mtp: bool
    layer_counts: dict[str, int]
    unsupported_layer_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ffn": self.ffn,
            "token_mixers": sorted(self.token_mixers),
            "is_hybrid": self.is_hybrid,
            "is_multimodal": self.is_multimodal,
            "has_mtp": self.has_mtp,
            "layer_counts": dict(self.layer_counts),
            "unsupported_layer_types": list(self.unsupported_layer_types),
        }


@dataclass(frozen=True)
class ConfigLoadResult:
    status: str
    config_source: str | None
    raw_config: dict[str, Any] | None
    normalized_config: NormalizedConfig | None
    features: ArchitectureFeatures | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_source": self.config_source,
            "raw_config": self.raw_config,
            "normalized_config": (
                self.normalized_config.to_dict() if self.normalized_config else None
            ),
            "features": self.features.to_dict() if self.features else None,
            "warnings": list(self.warnings),
        }


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_layer_type(layer_type: str) -> str | None:
    value = layer_type.lower().replace("-", "_")
    if value in {"full_attention", "attention", "self_attention"}:
        return "full_attention"
    if value in {
        "linear_attention",
        "gated_deltanet",
        "gated_delta_net",
        "linear",
    }:
        return "linear_attention"
    return None


def detect_features(config: NormalizedConfig) -> ArchitectureFeatures:
    """Detect independent FFN and token-mixer features from normalized config."""
    raw = config.raw_text_config
    ffn = "moe" if config.num_experts is not None else (
        "dense" if config.intermediate_size is not None else None
    )

    layer_counts: dict[str, int] = {}
    unsupported: list[str] = []
    for layer_type in config.layer_types:
        canonical = _canonical_layer_type(layer_type)
        if canonical is None:
            unsupported.append(layer_type)
        else:
            layer_counts[canonical] = layer_counts.get(canonical, 0) + 1

    token_mixers = set(layer_counts)
    linear_keys = {
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    }
    if linear_keys.intersection(raw):
        token_mixers.add("linear_attention")
    if not config.layer_types and config.num_attention_heads is not None:
        token_mixers.add("full_attention")
        if config.num_layers is not None:
            layer_counts["full_attention"] = config.num_layers

    mtp_keys = ("num_nextn_predict_layers", "num_mtp_modules", "mtp_depth")
    has_mtp = any((_integer(raw.get(key)) or 0) > 0 for key in mtp_keys)

    return ArchitectureFeatures(
        ffn=ffn,
        token_mixers=frozenset(token_mixers),
        is_hybrid=len(token_mixers) > 1,
        is_multimodal=config.vision_config is not None,
        has_mtp=has_mtp,
        layer_counts=layer_counts,
        unsupported_layer_types=tuple(unsupported),
    )


def missing_required_fields(config: NormalizedConfig) -> list[str]:
    """Return fields required to safely dispatch architecture estimators."""
    required = (
        "hidden_size",
        "num_layers",
        "vocab_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "ffn_kind",
    )
    missing = [name for name in required if getattr(config, name) is None]
    features = detect_features(config)
    if features.ffn == "moe":
        for name in ("num_experts", "experts_per_token", "expert_intermediate_size"):
            if getattr(config, name) is None:
                missing.append(name)
    elif features.ffn == "dense" and config.intermediate_size is None:
        missing.append("intermediate_size")
    elif features.ffn is None:
        missing.append("ffn_dimensions")
    if not features.token_mixers:
        missing.append("token_mixer")
    if config.layer_types and config.num_layers is not None:
        if len(config.layer_types) != config.num_layers:
            missing.append("layer_types_length")
    if features.unsupported_layer_types:
        missing.append("supported_layer_types")
    return missing
