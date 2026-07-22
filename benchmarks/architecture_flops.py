"""Architecture configuration loading, normalization, and feature detection.

This module deliberately contains no fallback architecture.  Callers either get
fields supported by their source configuration or an explicit ``unsupported``
result describing what is missing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast


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


# Fixed dimensions that every accepted Qwen3.6-35B MoE config must match.
_FIXED_DIMENSIONS: dict[str, int] = {
    "hidden_size": 2048,
    "num_layers": 40,
    "vocab_size": 248320,
    "head_dim": 256,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "num_experts": 256,
    "experts_per_token": 8,
    "expert_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
}


def validate_fixed_dimensions(config: NormalizedConfig) -> list[str]:
    """Return a list of dimension mismatches for the Qwen3.6-35B checkpoint.

    Returns an empty list when all fixed dimensions match.  Mismatches
    indicate that the config describes a different model variant and must
    be rejected rather than silently accepted.
    """
    mismatches: list[str] = []
    for field_name, expected in _FIXED_DIMENSIONS.items():
        actual = getattr(config, field_name, None)
        if actual is None:
            continue  # already reported by missing_required_fields
        if actual != expected:
            mismatches.append(
                f"{field_name}: expected {expected}, got {actual}"
            )
    return mismatches


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
    local_complete = False
    if candidates:
        first_norm = normalize_config(candidates[0][1], config_source=candidates[0][0])
        local_complete = not missing_required_fields(
            first_norm
        ) and not validate_fixed_dimensions(first_norm)
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
            dimension_errors = validate_fixed_dimensions(normalized)
            if dimension_errors:
                warnings.append(
                    f"{source} config dimensions do not match "
                    f"Qwen3.6-35B: {'; '.join(dimension_errors)}"
                )
                continue
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


# --------------------------------------------------------------------------- #
# Phase 2 — Composable component estimators
# --------------------------------------------------------------------------- #

# Results from a single component's layer-level computation.
# All quantities are per-layer, per-token (decode) or per-request (prefill).


@dataclass(frozen=True)
class ComponentResult:
    """FLOP and memory-traffic breakdown for a single layer component.

    Component estimators return per-layer work. Decode values cover one new
    token; prefill values cover the complete prompt request. Model-level
    composition applies the architecture's actual layer counts.
    """

    matmul_flops: float = 0.0              # matrix-multiply FLOPs (2 per MAC)
    non_matmul_flops: float = 0.0          # activations, gating, elementwise
    matmul_breakdown: dict[str, float] = field(default_factory=dict)
    non_matmul_breakdown: dict[str, float] = field(default_factory=dict)
    omitted_non_matmul: tuple[str, ...] = ()
    weight_bytes: float = 0.0              # weight bytes streamed from HBM
    kv_read_bytes: float = 0.0             # KV / recurrent state read
    kv_write_bytes: float = 0.0            # KV / recurrent state written
    activation_bytes: float = 0.0          # activation footprint (peak)

    @property
    def total_flops(self) -> float:
        return self.matmul_flops + self.non_matmul_flops

    @property
    def total_bytes(self) -> float:
        return (
            self.weight_bytes
            + self.kv_read_bytes
            + self.kv_write_bytes
            + self.activation_bytes
        )


# ---- GatedDeltaNet (linear attention) estimator ----


class GatedDeltaNetEstimator:
    """Qwen3.5/3.6 MoE Gated DeltaNet operation-count estimator.

    Prefill models the Transformers reference chunked delta-rule with a
    64-token chunk. Decode models the recurrent delta-rule. The recurrent
    state shape is [num_value_heads, key_head_dim, value_head_dim].
    """

    def __init__(
        self,
        d: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        kernel_size: int = 4,
        chunk_size: int = 64,
    ):
    
        self.d = d
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.state_elements = num_value_heads * key_head_dim * value_head_dim
        self.kernel_size = kernel_size
        self.chunk_size = chunk_size

        # Compatibility aliases for aggregate reporting.
        self.dk = self.key_dim
        self.dv = self.value_dim

    def projection_flops(self, tokens: int = 1) -> dict[str, float]:
        """QKV, z/a/b, depthwise-convolution, and output projection FLOPs."""
        return {
            "linear_projection_flops": (
                tokens
                * 2
                * self.d
                * (self.conv_dim + self.value_dim + 2 * self.num_value_heads)
            ),
            "short_conv_flops": tokens * 2 * self.conv_dim * self.kernel_size,
            "linear_output_projection_flops": (
                tokens * 2 * self.value_dim * self.d
            ),
        }

    def decode_flops(self) -> ComponentResult:
        """Per-token FLOPs for the recurrent single-token decode kernel."""
        breakdown = self.projection_flops()
        # kv_mem and output each reduce a [K, V] state.
        breakdown["recurrent_state_flops"] = 4 * self.state_elements
        non_matmul_breakdown: dict[str, float] = {
            # Decay state, form/add the outer-product update, then gate delta.
            "recurrent_state_update": float(3 * self.state_elements),
            "delta_gate_and_residual": float(2 * self.value_dim),
        }
        return ComponentResult(
            matmul_flops=sum(breakdown.values()),
            non_matmul_flops=sum(non_matmul_breakdown.values()),
            kv_read_bytes=self.state_bytes(),
            kv_write_bytes=self.state_bytes(),
            activation_bytes=4 * self.conv_dim * self.kernel_size,
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "q/k L2 normalization",
                "SiLU/softplus/sigmoid/exp gates",
                "RMSNorm and output gate",
            ),
        )

    def prefill_flops(self, S: int) -> ComponentResult:
        """Total reference chunk-kernel FLOPs for a prompt of length S."""
        if S <= 0:
            raise ValueError("sequence length must be positive")
        C = self.chunk_size
        chunks = (S + C - 1) // C
        H = self.num_value_heads
        K = self.key_head_dim
        V = self.value_head_dim

        breakdown = self.projection_flops(tokens=S)
        # Matrix operations in torch_chunk_gated_delta_rule. The final tile is
        # padded to C tokens, matching the reference implementation.
        within_chunk = (
            2 * H * chunks * C * C * K
            + 2 * H * chunks * C * C * V
            + 2 * H * chunks * C * C * K
        )
        cross_chunk = chunks * (
            2 * H * C * C * K
            + 2 * H * C * K * V
            + 2 * H * C * K * V
            + 2 * H * C * C * V
            + 2 * H * K * C * V
        )
        triangular = 2 * H * chunks * sum(i * i for i in range(1, C))
        breakdown["recurrent_state_flops"] = (
            within_chunk + cross_chunk + triangular
        )

        padded_tokens = chunks * C
        non_matmul_breakdown: dict[str, float] = {
            "chunk_decay_and_updates": float(
                padded_tokens
                * (3 * self.state_elements + 2 * self.value_dim)
            ),
        }
        return ComponentResult(
            matmul_flops=sum(breakdown.values()),
            non_matmul_flops=sum(non_matmul_breakdown.values()),
            kv_read_bytes=0,
            kv_write_bytes=self.state_bytes(),
            activation_bytes=4 * padded_tokens * self.conv_dim,
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "q/k L2 normalization",
                "chunk decay exponentials and masking",
                "SiLU/softplus/sigmoid gates and RMSNorm",
            ),
        )

    def state_bytes(self) -> float:
        """FP32 recurrent and convolution cache bytes per sequence."""
        return 4 * (self.state_elements + self.conv_dim * self.kernel_size)


# ---- FullAttentionEstimator (standard self-attention with GQA) ----


class FullAttentionEstimator:
    """Standard GQA with optional Qwen query/output gate projection."""

    def __init__(
        self,
        d: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        query_gate: bool = False,
    ):
        self.d = d
        self.n_q = num_heads
        self.n_kv = num_kv_heads
        self.h = head_dim
        self.d_q = num_heads * head_dim
        self.d_k = num_kv_heads * head_dim
        self.d_v = num_kv_heads * head_dim
        self.query_gate = query_gate

    def projection_flops(self, tokens: int = 1) -> dict[str, float]:
        breakdown: dict[str, float] = {
            "attention_qkv_projection_flops": float(
                tokens * 2 * self.d * (self.d_q + self.d_k + self.d_v)
            ),
            "attention_output_projection_flops": float(
                tokens * 2 * self.d_q * self.d
            ),
        }
        if self.query_gate:
            breakdown["attention_query_gate_projection_flops"] = float(
                tokens * 2 * self.d * self.d_q
            )
        return breakdown

    def parameter_count(self) -> int:
        return (
            self.d * (self.d_q + self.d_k + self.d_v)
            + self.d_q * self.d
            + (self.d * self.d_q if self.query_gate else 0)
        )

    def decode_flops(self, S: int | None = None) -> ComponentResult:
        breakdown = self.projection_flops()
        if S is not None:
            breakdown["attention_score_flops"] = float(4 * self.n_q * self.h * S)
        non_matmul_breakdown: dict[str, float] = {
            "attention_gate_multiply": float(self.d_q if self.query_gate else 0),
        }
        return ComponentResult(
            matmul_flops=sum(breakdown.values()),
            non_matmul_flops=sum(non_matmul_breakdown.values()),
            weight_bytes=float(2 * self.parameter_count()),
            kv_read_bytes=float(
                (self.d_k + self.d_v) * 2 * S if S is not None else 0
            ),
            kv_write_bytes=float((self.d_k + self.d_v) * 2),
            activation_bytes=float(4 * (self.d_q + self.d_k + self.d_v)),
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "attention softmax",
                "Q/K normalization and RoPE",
                "query-gate sigmoid",
            ),
        )

    def decode_attn_flops(self, S: int) -> float:
        return 4 * self.n_q * self.h * S

    def prefill_flops(self, S: int) -> ComponentResult:
        if S <= 0:
            raise ValueError("sequence length must be positive")
        breakdown = self.projection_flops(tokens=S)
        breakdown["attention_score_flops"] = float(
            2 * self.n_q * self.h * S * (S + 1)
        )
        non_matmul_breakdown: dict[str, float] = {
            "attention_gate_multiply": float(S * self.d_q if self.query_gate else 0),
        }
        return ComponentResult(
            matmul_flops=sum(breakdown.values()),
            non_matmul_flops=sum(non_matmul_breakdown.values()),
            weight_bytes=float(2 * self.parameter_count()),
            kv_read_bytes=0.0,
            kv_write_bytes=float(S * (self.d_k + self.d_v) * 2),
            activation_bytes=float(4 * S * (self.d_q + self.d_k + self.d_v)),
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "attention softmax",
                "Q/K normalization and RoPE",
                "query-gate sigmoid",
            ),
        )

    def kv_cache_bytes(self, S: int, dtype_bytes: int = 2) -> float:
        return S * (self.d_k + self.d_v) * dtype_bytes


# ---- MoeFfnEstimator ----


class MoeFfnEstimator:
    """Qwen MoE FFN with routed experts, shared expert, and shared gate."""

    def __init__(
        self,
        d: int,
        num_experts: int,
        experts_per_token: int,
        expert_intermediate_size: int,
        shared_expert_intermediate_size: int | None = None,
        ffn_kind: str = "gated",
    ) -> None:
        self.d = d
        self.E = num_experts
        self.k = experts_per_token
        self.m_e = expert_intermediate_size
        self.m_s = shared_expert_intermediate_size
        self.gated = ffn_kind == "gated"
        self.n_matrices = 3 if self.gated else 2

    def parameter_breakdown(self) -> dict[str, int]:
        result = {
            "routed_experts": self.E * self.n_matrices * self.d * self.m_e,
            "router": self.d * self.E,
        }
        if self.m_s is not None:
            result["shared_expert"] = self.n_matrices * self.d * self.m_s
            result["shared_expert_gate"] = self.d
        return result

    def decode_flops_breakdown(self) -> dict[str, float]:
        result: dict[str, float] = {
            "routed_experts": (
                self.k * self.n_matrices * 2 * self.d * self.m_e
            ),
            "shared_expert": (
                self.n_matrices * 2 * self.d * self.m_s
                if self.m_s is not None
                else 0.0
            ),
            "router": 2 * self.d * self.E,
            "shared_expert_gate": 2 * self.d if self.m_s is not None else 0.0,
        }
        return result

    def decode_flops(self) -> ComponentResult:
        breakdown = self.decode_flops_breakdown()
        non_matmul_breakdown: dict[str, float] = {
            "routed_reweight_and_accumulate": float(2 * self.k * self.d),
            "shared_expert_gate_multiply": float(self.d if self.m_s is not None else 0),
        }
        return ComponentResult(
            matmul_flops=sum(breakdown.values()),
            non_matmul_flops=sum(non_matmul_breakdown.values()),
            weight_bytes=float(2 * sum(self.parameter_breakdown().values())),
            activation_bytes=float(
                4 * self.k * self.m_e
                + (4 * self.m_s if self.m_s is not None else 0)
            ),
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "router softmax and top-k selection (implementation-dependent O(E))",
                "expert activation functions",
                "shared-expert gate sigmoid",
            ),
        )


# ---- LmHeadEstimator ----


class LmHeadEstimator:
    """Vocabulary projection; tying changes residency, not projection FLOPs."""

    def __init__(self, d: int, vocab_size: int, tied: bool = False):
        self.d = d
        self.v = vocab_size
        self.tied = tied

    def decode_flops(self) -> ComponentResult:
        matmul = 2 * self.d * self.v
        return ComponentResult(
            matmul_flops=matmul,
            weight_bytes=0 if self.tied else 2 * self.d * self.v,
            matmul_breakdown={"lm_head_flops": matmul},
            omitted_non_matmul=("logits post-processing / sampling",),
        )

    def prefill_flops(self, logits_tokens: int = 1) -> ComponentResult:
        if logits_tokens < 0:
            raise ValueError("logits_tokens must be non-negative")
        matmul = logits_tokens * 2 * self.d * self.v
        return ComponentResult(
            matmul_flops=matmul,
            weight_bytes=0 if self.tied else 2 * self.d * self.v,
            matmul_breakdown={"lm_head_flops": matmul},
            omitted_non_matmul=("logits post-processing",),
        )


# ---- ModelFlopsEstimator ----


class ModelFlopsEstimator:
    """Compose Qwen MoE components using their actual per-layer mixer counts."""

    def __init__(
        self,
        normalized: "NormalizedConfig",
        features: "ArchitectureFeatures",
    ):
        self.normalized = normalized
        self.features = features
        required = {
            "hidden_size": normalized.hidden_size,
            "num_layers": normalized.num_layers,
            "vocab_size": normalized.vocab_size,
            "head_dim": normalized.head_dim,
            "num_attention_heads": normalized.num_attention_heads,
            "num_key_value_heads": normalized.num_key_value_heads,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Cannot estimate FLOPs; missing " + ", ".join(missing)
            )

        d = normalized.hidden_size
        l = normalized.num_layers
        v = normalized.vocab_size
        h = normalized.head_dim
        n_q = normalized.num_attention_heads
        n_kv = normalized.num_key_value_heads
        assert d is not None
        assert l is not None
        assert v is not None
        assert h is not None
        assert n_q is not None
        assert n_kv is not None
        self._d = int(d)
        self._l = int(l)
        self._v = int(v)
        self._h = int(h)
        self._n_q = int(n_q)
        self._n_kv = int(n_kv)
        self._gated = normalized.ffn_kind == "gated"
        self._layer_counts = dict(features.layer_counts)
        self.full_attention_layers = self._layer_counts.get("full_attention", 0)
        self.linear_attention_layers = self._layer_counts.get("linear_attention", 0)
        if sum(self._layer_counts.values()) != self._l:
            raise ValueError("layer-type counts do not sum to num_layers")

        raw = normalized.raw_text_config
        self._full_attn: FullAttentionEstimator | None = None
        if self.full_attention_layers:
            self._full_attn = FullAttentionEstimator(
                self._d,
                self._n_q,
                self._n_kv,
                self._h,
                query_gate=bool(raw.get("attn_output_gate", False)),
            )

        self._delta: GatedDeltaNetEstimator | None = None
        linear_keys = {
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        }
        if self.linear_attention_layers:
            if not linear_keys.issubset(raw):
                raise ValueError("linear-attention dimensions are incomplete")
            self._delta = GatedDeltaNetEstimator(
                self._d,
                int(raw["linear_num_key_heads"]),
                int(raw["linear_num_value_heads"]),
                int(raw["linear_key_head_dim"]),
                int(raw["linear_value_head_dim"]),
                int(raw["linear_conv_kernel_dim"]),
            )

        self._moe: MoeFfnEstimator | None = None
        if features.ffn == "moe":
            if (
                normalized.num_experts is None
                or normalized.experts_per_token is None
                or normalized.expert_intermediate_size is None
            ):
                raise ValueError("MoE dimensions are incomplete")
            self._moe = MoeFfnEstimator(
                self._d,
                int(normalized.num_experts),
                int(normalized.experts_per_token),
                int(normalized.expert_intermediate_size),
                (
                    int(normalized.shared_expert_intermediate_size)
                    if normalized.shared_expert_intermediate_size is not None
                    else None
                ),
                ffn_kind="gated" if self._gated else "two_matrix",
            )
        else:
            raise ValueError("this Phase 2 path supports MoE FFNs only")

        self._lm = LmHeadEstimator(
            self._d,
            self._v,
            tied=bool(normalized.tie_word_embeddings),
        )
        model_type = str(raw.get("model_type", "")).lower()
        self.estimate_status = (
            "exact_from_config"
            if model_type.startswith("qwen3_5_moe")
            or model_type.startswith("qwen3_6_moe")
            else "approximate"
        )
        self.assumptions = [
            "2 FLOPs per multiply-accumulate",
            "Linear-attention prefill uses the 64-token chunked reference kernel algorithm",
            "listed omitted_non_matmul operations are excluded from totals",
        ]

    @staticmethod
    def _add_scaled(
        target: dict[str, float],
        values: Mapping[str, float],
        scale: int,
        prefix: str = "",
    ) -> None:
        for name, value in values.items():
            target[prefix + name] = target.get(prefix + name, 0.0) + value * scale

    def _finalize(
        self,
        *,
        mode: str,
        sequence_length: int,
        matmul: dict[str, float],
        non_matmul: dict[str, float],
        omitted: set[str],
    ) -> dict[str, Any]:
        matmul_total = sum(matmul.values())
        non_matmul_total = sum(non_matmul.values())
        result: dict[str, Any] = {
            **{name: round(value, 1) for name, value in matmul.items()},
            "mode": mode,
            "sequence_length": sequence_length,
            "num_layers": self._l,
            "layer_counts": dict(self._layer_counts),
            "matmul_flops": round(matmul_total, 1),
            "non_matmul_flops": round(non_matmul_total, 1),
            "non_matmul_breakdown": {
                name: round(value, 1) for name, value in non_matmul.items()
            },
            "omitted_non_matmul": sorted(omitted),
            "total": round(matmul_total + non_matmul_total, 1),
            "estimate_status": self.estimate_status,
            "assumptions": list(self.assumptions),
        }
        return result

    def decode_model_flops(self, context_length: int) -> dict[str, Any]:
        if context_length < 0:
            raise ValueError("context_length must be non-negative")
        matmul: dict[str, float] = {}
        non_matmul: dict[str, float] = {}
        omitted: set[str] = set()

        assert self._moe is not None
        moe = self._moe.decode_flops()
        moe_names = {
            "routed_experts": "moe_routed",
            "shared_expert": "moe_shared",
            "router": "moe_router",
            "shared_expert_gate": "moe_shared_gate",
        }
        for name, value in moe.matmul_breakdown.items():
            matmul[moe_names[name]] = value * self._l
        self._add_scaled(non_matmul, moe.non_matmul_breakdown, self._l, "moe_")
        omitted.update(moe.omitted_non_matmul)

        if self._full_attn:
            attention = self._full_attn.decode_flops(context_length)
            self._add_scaled(
                matmul,
                attention.matmul_breakdown,
                self.full_attention_layers,
            )
            self._add_scaled(
                non_matmul,
                attention.non_matmul_breakdown,
                self.full_attention_layers,
            )
            omitted.update(attention.omitted_non_matmul)

        delta: ComponentResult | None = None
        if self._delta:
            delta = self._delta.decode_flops()
            self._add_scaled(
                matmul,
                delta.matmul_breakdown,
                self.linear_attention_layers,
            )
            self._add_scaled(
                non_matmul,
                delta.non_matmul_breakdown,
                self.linear_attention_layers,
            )
            omitted.update(delta.omitted_non_matmul)

        lm = self._lm.decode_flops()
        self._add_scaled(matmul, lm.matmul_breakdown, 1)
        omitted.update(lm.omitted_non_matmul)
        result = self._finalize(
            mode="decode",
            sequence_length=context_length,
            matmul=matmul,
            non_matmul=non_matmul,
            omitted=omitted,
        )
        if delta is not None:
            result["state_traffic_bytes"] = {
                "linear_state_bytes_read": delta.kv_read_bytes * self.linear_attention_layers,
                "linear_state_bytes_written": delta.kv_write_bytes * self.linear_attention_layers,
            }
        return result

    def decode_flops(self, context_length: int) -> dict[str, Any]:
        """Compatibility alias returning the correctly composed model total."""
        return self.decode_model_flops(context_length)

    def prefill_flops(self, S: int, logits_tokens: int = 1) -> dict[str, Any]:
        if S <= 0:
            raise ValueError("sequence length must be positive")
        matmul: dict[str, float] = {}
        non_matmul: dict[str, float] = {}
        omitted: set[str] = set()

        assert self._moe is not None
        moe = self._moe.decode_flops()
        moe_names = {
            "routed_experts": "moe_routed",
            "shared_expert": "moe_shared",
            "router": "moe_router",
            "shared_expert_gate": "moe_shared_gate",
        }
        for name, value in moe.matmul_breakdown.items():
            matmul[moe_names[name]] = value * S * self._l
        self._add_scaled(
            non_matmul, moe.non_matmul_breakdown, S * self._l, "moe_"
        )
        omitted.update(moe.omitted_non_matmul)

        if self._full_attn:
            attention = self._full_attn.prefill_flops(S)
            self._add_scaled(
                matmul,
                attention.matmul_breakdown,
                self.full_attention_layers,
            )
            self._add_scaled(
                non_matmul,
                attention.non_matmul_breakdown,
                self.full_attention_layers,
            )
            omitted.update(attention.omitted_non_matmul)

        delta: ComponentResult | None = None
        if self._delta:
            delta = self._delta.prefill_flops(S)
            self._add_scaled(
                matmul,
                delta.matmul_breakdown,
                self.linear_attention_layers,
            )
            self._add_scaled(
                non_matmul,
                delta.non_matmul_breakdown,
                self.linear_attention_layers,
            )
            omitted.update(delta.omitted_non_matmul)

        lm = self._lm.prefill_flops(logits_tokens=logits_tokens)
        self._add_scaled(matmul, lm.matmul_breakdown, 1)
        omitted.update(lm.omitted_non_matmul)
        result = self._finalize(
            mode="prefill",
            sequence_length=S,
            matmul=matmul,
            non_matmul=non_matmul,
            omitted=omitted,
        )
        result["logits_tokens"] = logits_tokens
        if delta is not None:
            _delta = delta
            result["state_traffic_bytes"] = {
                "linear_state_bytes_read": float(_delta.kv_read_bytes * self.linear_attention_layers),
                "linear_state_bytes_written": float(_delta.kv_write_bytes * self.linear_attention_layers),
            }
        return result

    def weight_bytes(self, dtype_bytes: int = 2) -> dict[str, float]:
        out: dict[str, float] = {}
        assert self._moe is not None
        _moe = self._moe
        moe_params = _moe.parameter_breakdown()
        for name, params in moe_params.items():
            out["moe_" + name] = params * self._l * dtype_bytes

        if self._full_attn:
            out["full_attention"] = (
                self._full_attn.parameter_count()
                * self.full_attention_layers
                * dtype_bytes
            )

        if self._delta:
            delta_params = (
                self._d
                * (
                    self._delta.conv_dim
                    + self._delta.value_dim
                    + 2 * self._delta.num_value_heads
                )
                + self._delta.conv_dim * self._delta.kernel_size
                + self._delta.value_dim * self._d
                + 2 * self._delta.num_value_heads
                + self._delta.value_head_dim
            )
            out["linear_attention"] = (
                delta_params * self.linear_attention_layers * dtype_bytes
            )

        if not self._lm.tied:
            out["lm_head"] = self._d * self._v * dtype_bytes
        return out

    def kv_state_bytes(self, S: int, dtype_bytes: int = 2) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._full_attn:
            out["full_attention_kv"] = (
                self._full_attn.kv_cache_bytes(S, dtype_bytes)
                * self.full_attention_layers
            )
        if self._delta:
            out["linear_attention_state"] = (
                self._delta.state_bytes() * self.linear_attention_layers
            )
        return out


def compute_flops(
    model_config: dict[str, Any],
    *,
    mode: Literal["decode", "prefill"],
    sequence_length: int,
    logits_tokens: int = 1,
) -> dict[str, Any]:
    """Return Qwen3.6-35B text-inference FLOPs.

    Args:
        model_config: model configuration dict.
        mode: ``"decode"`` or ``"prefill"``.
        sequence_length: context length for decode, or prompt length for prefill.
        logits_tokens: number of logits positions (prefill default: 1).

    Returns:
        Dict with ``estimate_status``, ``architecture``, ``flops``,
        ``assumptions``, and ``warnings``.
    """
    normalized = normalize_config(model_config)
    missing = missing_required_fields(normalized)
    if missing:
        return {
            "estimate_status": "unsupported",
            "warnings": ["missing configuration fields: " + ", ".join(missing)],
        }
    dimension_errors = validate_fixed_dimensions(normalized)
    if dimension_errors:
        return {
            "estimate_status": "unsupported",
            "warnings": [
                "config dimensions do not match Qwen3.6-35B: "
                + "; ".join(dimension_errors)
            ],
        }
    features = detect_features(normalized)
    try:
        estimator = ModelFlopsEstimator(normalized, features)
    except ValueError as exc:
        return {"estimate_status": "unsupported", "warnings": [str(exc)]}

    flops = (
        estimator.decode_model_flops(sequence_length)
        if mode == "decode"
        else estimator.prefill_flops(sequence_length, logits_tokens=logits_tokens)
    )
    return {
        "architecture": features.to_dict(),
        "estimate_status": estimator.estimate_status,
        "flops": flops,
        "assumptions": estimator.assumptions,
        "warnings": [],
    }


def _model_label(raw: dict[str, Any]) -> str:
    """Derive a short model label from the config."""
    model_type = str(raw.get("model_type", "unknown")).lower()
    # Strip "qwen3_5_moe" / "qwen3_6_moe" prefix → "qwen3.5-35b-a3b"
    if model_type.startswith("qwen3_6"):
        return "qwen3.6-35b-a3b"
    if model_type.startswith("qwen3_5"):
        return "qwen3.5-35b-a3b"
    return model_type.replace("_", "-")


def run_flops_analysis(
    model_config: dict[str, Any],
    *,
    context_lengths: list[int] | None = None,
    prefill_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Return a complete FLOPs analysis document for the Qwen3.6-35B config.

    Runs decode and prefill estimates across a range of sequence lengths and
    returns a JSON-serialisable dict with the documented output schema.
    """
    if context_lengths is None:
        context_lengths = [512, 2048, 8192, 32768]
    if prefill_lengths is None:
        prefill_lengths = [512, 2048, 8192, 32768]

    normalized = normalize_config(model_config)
    missing = missing_required_fields(normalized)
    if missing:
        return {
            "estimate_status": "unsupported",
            "architecture": {"model": _model_label(normalized.raw_text_config)},
            "flops": {"decode": {}, "prefill": {}},
            "assumptions": [],
            "warnings": ["missing configuration fields: " + ", ".join(missing)],
        }

    dimension_errors = validate_fixed_dimensions(normalized)
    if dimension_errors:
        return {
            "estimate_status": "unsupported",
            "architecture": {"model": _model_label(normalized.raw_text_config)},
            "flops": {"decode": {}, "prefill": {}},
            "assumptions": [],
            "warnings": [
                "config dimensions do not match Qwen3.6-35B: "
                + "; ".join(dimension_errors)
            ],
        }

    features = detect_features(normalized)
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
            "assumptions": estimator.assumptions,
        },
        "assumptions": estimator.assumptions,
        "warnings": list(estimator.assumptions),
    }


# --------------------------------------------------------------------------- #
# CLI entry point — standalone analysis
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json
    import sys

    from core_runner import load_model_config as _load_model_config

    cfg_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/qwen3.6_35b_redhat_nvfp4.yml")
    )
    cfg = _load_model_config(cfg_path)
    result = run_flops_analysis(cfg)
    print(json.dumps(result, indent=2, default=str))
