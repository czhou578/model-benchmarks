"""Architecture configuration loading, normalization, and feature detection.

This module deliberately contains no fallback architecture.  Callers either get
fields supported by their source configuration or an explicit ``unsupported``
result describing what is missing.
"""

from __future__ import annotations

from collections.abc import Mapping
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


_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("hidden_size", "d_model", "n_embd"),
    "num_layers": ("num_hidden_layers", "num_layers", "n_layer"),
    "vocab_size": ("vocab_size", "n_vocab"),
    "head_dim": ("head_dim", "attention_head_dim"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads", "n_head_kv"),
    "layer_types": ("layer_types", "layers_block_type", "layer_type"),
    "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
    "num_experts": ("num_experts", "n_routed_experts", "num_local_experts"),
    "experts_per_token": (
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_router_topk",
        "top_k",
    ),
    # Both names denote a per-expert width.  It must never be divided by E.
    "expert_intermediate_size": (
        "expert_intermediate_size",
        "moe_intermediate_size",
        "moe_ffn_hidden_size",
    ),
    "shared_expert_intermediate_size": (
        "shared_expert_intermediate_size",
        "shared_expert_ffn_hidden_size",
        "moe_shared_expert_intermediate_size",
    ),
    "tie_word_embeddings": ("tie_word_embeddings",),
    "quantization": ("quantization_config",),
}

_GATED_MODEL_PREFIXES = (
    "deepseek",
    "gemma",
    "llama",
    "mistral",
    "mixtral",
    "qwen",
)


def _as_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("model configuration must be a mapping or expose to_dict()")


def _pick(
    config: Mapping[str, Any], field_name: str, source_prefix: str
) -> tuple[Any, str | None]:
    for key in _ALIASES[field_name]:
        if key in config and config[key] is not None:
            return config[key], f"{source_prefix}{key}"
    return None, None


def normalize_config(
    model_config: Mapping[str, Any] | Any,
    *,
    config_source: str = "provided_config",
) -> NormalizedConfig:
    """Normalize aliases while retaining exact source-key provenance.

    Multimodal wrappers are unwrapped through ``text_config``.  The vision
    configuration is retained, but does not affect text-only feature detection.
    """
    outer = _as_dict(model_config)
    text_value = outer.get("text_config")
    if text_value is not None:
        text = _as_dict(text_value)
        prefix = "text_config."
    else:
        text = outer
        prefix = ""

    vision_value = outer.get("vision_config")
    vision = _as_dict(vision_value) if vision_value is not None else None
    source_keys: dict[str, str] = {}
    values: dict[str, Any] = {}
    for field_name in _ALIASES:
        value, source_key = _pick(text, field_name, prefix)
        values[field_name] = value
        if source_key:
            source_keys[field_name] = source_key
    if values["quantization"] is None and text is not outer:
        value, source_key = _pick(outer, "quantization", "wrapper.")
        values["quantization"] = value
        if source_key:
            source_keys["quantization"] = source_key

    layer_types_value = values["layer_types"]
    if isinstance(layer_types_value, str):
        layer_types = (layer_types_value,)
    elif isinstance(layer_types_value, (list, tuple)):
        layer_types = tuple(str(item) for item in layer_types_value)
    else:
        layer_types = ()

    head_dim = _integer(values["head_dim"])
    n_heads = _integer(values["num_attention_heads"])
    hidden_size = _integer(values["hidden_size"])
    if (
        head_dim is None
        and hidden_size is not None
        and n_heads
        and hidden_size % n_heads == 0
    ):
        head_dim = hidden_size // n_heads
        source_keys["head_dim"] = (
            "derived:hidden_size/num_attention_heads"
        )

    ffn_kind: str | None = None
    explicit_ffn_key = (
        "ffn_kind" if text.get("ffn_kind") is not None else "mlp_type"
    )
    explicit_ffn = text.get(explicit_ffn_key)
    if explicit_ffn in {"gated", "swiglu", "glu"}:
        ffn_kind = "gated"
        source_keys["ffn_kind"] = f"{prefix}{explicit_ffn_key}"
    elif explicit_ffn in {"two_matrix", "gelu", "relu"}:
        ffn_kind = "two_matrix"
        source_keys["ffn_kind"] = f"{prefix}{explicit_ffn_key}"
    else:
        model_type = str(text.get("model_type", outer.get("model_type", ""))).lower()
        if model_type.startswith(_GATED_MODEL_PREFIXES):
            ffn_kind = "gated"
            model_type_key = "model_type" if "model_type" in text else "wrapper.model_type"
            source_keys["ffn_kind"] = f"derived:{model_type_key}={model_type}"

    quantization = values["quantization"]
    if quantization is not None and not isinstance(quantization, Mapping):
        quantization = {"value": quantization}
    elif isinstance(quantization, Mapping):
        quantization = dict(quantization)

    return NormalizedConfig(
        hidden_size=hidden_size,
        num_layers=_integer(values["num_layers"]),
        vocab_size=_integer(values["vocab_size"]),
        head_dim=head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=_integer(values["num_key_value_heads"]),
        layer_types=layer_types,
        ffn_kind=ffn_kind,
        intermediate_size=_integer(values["intermediate_size"]),
        num_experts=_integer(values["num_experts"]),
        experts_per_token=_integer(values["experts_per_token"]),
        expert_intermediate_size=_integer(values["expert_intermediate_size"]),
        shared_expert_intermediate_size=_integer(
            values["shared_expert_intermediate_size"]
        ),
        tie_word_embeddings=(
            bool(values["tie_word_embeddings"])
            if values["tie_word_embeddings"] is not None
            else None
        ),
        quantization=quantization,
        source_keys=source_keys,
        config_source=config_source,
        vision_config=vision,
        raw_text_config=text,
    )


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
    if (
        config.layer_types
        and config.num_layers is not None
        and len(config.layer_types) != config.num_layers
    ):
        missing.append("layer_types_length")
    if features.unsupported_layer_types:
        missing.append("supported_layer_types")
    return missing
