"""Architecture configuration loading, normalization, and feature detection.

This module deliberately contains no fallback architecture.  Callers either get
fields supported by their source configuration or an explicit ``unsupported``
result describing what is missing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


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


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    if head_dim is None and hidden_size is not None and n_heads:
        if hidden_size % n_heads == 0:
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
    if value in {"mla", "multi_head_latent_attention"}:
        return "mla"
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
    mla_keys = {"kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"}
    if mla_keys.issubset(raw):
        token_mixers.add("mla")
    if not config.layer_types and config.num_attention_heads is not None:
        token_mixers.add("mla" if "mla" in token_mixers else "full_attention")
        if config.num_layers is not None:
            mixer = "mla" if "mla" in token_mixers else "full_attention"
            layer_counts[mixer] = config.num_layers

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


def _model_reference(benchmark_config: Mapping[str, Any]) -> str | None:
    for key in ("local_model_path", "model_path", "model_id"):
        value = benchmark_config.get(key)
        if isinstance(value, str) and value:
            return value
    endpoint = benchmark_config.get("endpoint")
    if isinstance(endpoint, Mapping):
        value = endpoint.get("model_name")
        if isinstance(value, str) and value:
            return value
    server = benchmark_config.get("server")
    if isinstance(server, Mapping):
        command = server.get("command")
        if isinstance(command, list):
            try:
                serve_index = command.index("serve")
                value = command[serve_index + 1]
                if isinstance(value, str) and value:
                    return value
            except (ValueError, IndexError):
                pass
    return None


def _local_config_path(benchmark_config: Mapping[str, Any]) -> Path | None:
    explicit = benchmark_config.get("architecture_config_path")
    if isinstance(explicit, str) and explicit:
        path = Path(explicit).expanduser()
        config_dir = benchmark_config.get("_config_dir")
        if not path.is_absolute() and isinstance(config_dir, str):
            path = Path(config_dir) / path
        return path / "config.json" if path.is_dir() else path
    reference = _model_reference(benchmark_config)
    if reference:
        path = Path(reference).expanduser()
        if path.exists():
            return path / "config.json" if path.is_dir() else path
    return None


def _load_local(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"model config at {path} is not a JSON object")
    return value


def load_architecture_config(
    benchmark_config: Mapping[str, Any],
    *,
    server_config: Mapping[str, Any] | None = None,
    auto_config_loader: Callable[[str], Any] | None = None,
) -> ConfigLoadResult:
    """Load full architecture config in the Phase 1 precedence order."""
    warnings: list[str] = []
    candidates: list[tuple[str, dict[str, Any]]] = []

    local_path = _local_config_path(benchmark_config)
    if local_path is not None:
        try:
            candidates.append((f"local:{local_path}", _load_local(local_path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"could not load local model config {local_path}: {exc}")

    model_reference = _model_reference(benchmark_config)
    local_complete = bool(
        candidates
        and not missing_required_fields(
            normalize_config(candidates[0][1], config_source=candidates[0][0])
        )
    )
    if model_reference and not local_complete:
        try:
            if auto_config_loader is None:
                from transformers import AutoConfig

                loaded = AutoConfig.from_pretrained(
                    model_reference, trust_remote_code=True
                )
            else:
                loaded = auto_config_loader(model_reference)
            candidates.append(("huggingface", _as_dict(loaded)))
        except Exception as exc:  # dependency, connectivity, or remote config error
            warnings.append(
                f"could not load Hugging Face config for {model_reference}: {exc}"
            )

    if server_config is not None:
        try:
            candidates.append(("server", _as_dict(server_config)))
        except TypeError as exc:
            warnings.append(f"server config was invalid: {exc}")

    for source, raw in candidates:
        normalized = normalize_config(raw, config_source=source)
        missing = missing_required_fields(normalized)
        if not missing:
            features = detect_features(normalized)
            return ConfigLoadResult(
                status="exact_from_config",
                config_source=source,
                raw_config=raw,
                normalized_config=normalized,
                features=features,
                warnings=tuple(warnings),
            )
        warnings.append(f"{source} config is incomplete: missing {', '.join(missing)}")

    return ConfigLoadResult(
        status="unsupported",
        config_source=None,
        raw_config=None,
        normalized_config=None,
        features=None,
        warnings=tuple(warnings or ("no model configuration source was available",)),
    )

