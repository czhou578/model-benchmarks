"""Model-level FLOP composition: ModelFlopsEstimator."""

from __future__ import annotations

from typing import Any, Mapping

from benchmarks.architecture_flops.attention import FullAttentionEstimator
from benchmarks.architecture_flops.component import ComponentResult
from benchmarks.architecture_flops.config import ArchitectureFeatures, NormalizedConfig
from benchmarks.architecture_flops.linear_attention import GatedDeltaNetEstimator


# --------------------------------------------------------------------------- #
# Composable component estimators
# --------------------------------------------------------------------------- #


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


# ---- DenseFfnEstimator ----


class DenseFfnEstimator:
    """Dense FFN: gate, up, and down projections (gated), or up and down."""

    def __init__(self, d: int, intermediate_size: int, ffn_kind: str = "gated") -> None:
        self.d = d
        self.m = intermediate_size
        self.n_matrices = 3 if ffn_kind == "gated" else 2

    def parameter_count(self) -> int:
        return self.n_matrices * self.d * self.m

    def decode_flops(self) -> ComponentResult:
        matmul = float(2 * self.parameter_count())
        return ComponentResult(
            matmul_flops=matmul,
            weight_bytes=float(2 * self.parameter_count()),
            activation_bytes=float(4 * self.m),
            matmul_breakdown={"dense_ffn": matmul},
            omitted_non_matmul=("FFN activation function and gate multiply",),
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
    """Compose Qwen components (MoE or dense FFN) with the per-layer mixer counts."""

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
        n_layers = normalized.num_layers
        v = normalized.vocab_size
        h = normalized.head_dim
        n_q = normalized.num_attention_heads
        n_kv = normalized.num_key_value_heads
        assert d is not None
        assert n_layers is not None
        assert v is not None
        assert h is not None
        assert n_q is not None
        assert n_kv is not None
        self._d = int(d)
        self._l = int(n_layers)
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
        self._dense: DenseFfnEstimator | None = None
        if features.ffn == "dense" and normalized.intermediate_size is not None:
            self._dense = DenseFfnEstimator(
                self._d,
                int(normalized.intermediate_size),
                ffn_kind="gated" if self._gated else "two_matrix",
            )
        elif features.ffn == "moe":
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
            raise ValueError("FFN dimensions are incomplete")

        self._lm = LmHeadEstimator(
            self._d,
            self._v,
            tied=bool(normalized.tie_word_embeddings),
        )
        model_type = str(raw.get("model_type", "")).lower()
        self.estimate_status = (
            "exact_from_config"
            if model_type.startswith(("qwen3_5", "qwen3_6"))
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

    def _compose_components(
        self,
        components: list[dict[str, Any]],
        *,
        mode: str,
        sequence_length: int,
    ) -> dict[str, Any]:
        matmul: dict[str, float] = {}
        non_matmul: dict[str, float] = {}
        omitted: set[str] = set()

        for comp in components:
            result = comp["result"]
            name_map = comp.get("name_map")
            if name_map:
                for name, value in result.matmul_breakdown.items():
                    matmul[name_map.get(name, name)] = value * comp["matmul_m"]
            else:
                self._add_scaled(matmul, result.matmul_breakdown, comp["matmul_m"])
            self._add_scaled(
                non_matmul, result.non_matmul_breakdown,
                comp["non_matmul_m"], comp.get("nmm_prefix", ""),
            )
            omitted.update(result.omitted_non_matmul)

        result = self._finalize(
            mode=mode,
            sequence_length=sequence_length,
            matmul=matmul,
            non_matmul=non_matmul,
            omitted=omitted,
        )
        return result

    def _ffn_component(self, tokens: int) -> dict[str, Any]:
        """The FFN of all layers for *tokens* tokens, MoE or dense."""
        scale = tokens * self._l
        if self._dense is not None:
            return {
                "result": self._dense.decode_flops(),
                "matmul_m": scale, "non_matmul_m": scale,
            }
        assert self._moe is not None
        moe_names = {
            "routed_experts": "moe_routed",
            "shared_expert": "moe_shared",
            "router": "moe_router",
            "shared_expert_gate": "moe_shared_gate",
        }
        return {
            "result": self._moe.decode_flops(),
            "matmul_m": scale, "non_matmul_m": scale,
            "name_map": moe_names, "nmm_prefix": "moe_",
        }

    def decode_model_flops(self, context_length: int) -> dict[str, Any]:
        if context_length < 0:
            raise ValueError("context_length must be non-negative")

        components: list[dict[str, Any]] = [self._ffn_component(1)]
        if self._full_attn:
            components.append({
                "result": self._full_attn.decode_flops(context_length),
                "matmul_m": self.full_attention_layers,
                "non_matmul_m": self.full_attention_layers,
            })
        if self._delta:
            delta_result = self._delta.decode_flops()
            components.append({
                "result": delta_result,
                "matmul_m": self.linear_attention_layers,
                "non_matmul_m": self.linear_attention_layers,
            })
        components.append({
            "result": self._lm.decode_flops(),
            "matmul_m": 1, "non_matmul_m": 1,
        })

        result = self._compose_components(
            components,
            mode="decode",
            sequence_length=context_length,
        )
        if self._delta:
            result["state_traffic_bytes"] = {
                "linear_state_bytes_read": delta_result.kv_read_bytes * self.linear_attention_layers,
                "linear_state_bytes_written": delta_result.kv_write_bytes * self.linear_attention_layers,
            }
        return result

    def decode_flops(self, context_length: int) -> dict[str, Any]:
        """Compatibility alias returning the correctly composed model total."""
        return self.decode_model_flops(context_length)

    def prefill_flops(self, S: int, logits_tokens: int = 1) -> dict[str, Any]:
        if S <= 0:
            raise ValueError("sequence length must be positive")

        # The FFN result covers one token. The attention and linear-attention
        # prefill results already cover all S tokens of one layer, so they
        # scale by the layer count only.
        components: list[dict[str, Any]] = [self._ffn_component(S)]
        if self._full_attn:
            components.append({
                "result": self._full_attn.prefill_flops(S),
                "matmul_m": self.full_attention_layers,
                "non_matmul_m": self.full_attention_layers,
            })
        if self._delta:
            delta_result = self._delta.prefill_flops(S)
            components.append({
                "result": delta_result,
                "matmul_m": self.linear_attention_layers,
                "non_matmul_m": self.linear_attention_layers,
            })
        components.append({
            "result": self._lm.prefill_flops(logits_tokens=logits_tokens),
            "matmul_m": 1, "non_matmul_m": 1,
        })

        result = self._compose_components(
            components,
            mode="prefill",
            sequence_length=S,
        )
        result["logits_tokens"] = logits_tokens
        if self._delta:
            result["state_traffic_bytes"] = {
                "linear_state_bytes_read": float(delta_result.kv_read_bytes * self.linear_attention_layers),
                "linear_state_bytes_written": float(delta_result.kv_write_bytes * self.linear_attention_layers),
            }
        return result

    def weight_bytes(self, dtype_bytes: int = 2) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._dense is not None:
            out["dense_ffn"] = self._dense.parameter_count() * self._l * dtype_bytes
        else:
            assert self._moe is not None
            for name, params in self._moe.parameter_breakdown().items():
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
