"""FFN FLOP estimators: MoE (routed and shared experts) and dense."""

from __future__ import annotations

from benchmarks.architecture_flops.component import ComponentResult


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
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "router softmax and top-k selection (implementation-dependent O(E))",
                "expert activation functions",
                "shared-expert gate sigmoid",
            ),
        )


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
            matmul_breakdown={"dense_ffn": matmul},
            omitted_non_matmul=("FFN activation function and gate multiply",),
        )
