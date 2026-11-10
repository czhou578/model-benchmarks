"""Full attention (GQA) FLOP estimator."""

from __future__ import annotations

from benchmarks.architecture_flops.component import ComponentResult


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
            matmul_breakdown=breakdown,
            non_matmul_breakdown=non_matmul_breakdown,
            omitted_non_matmul=(
                "attention softmax",
                "Q/K normalization and RoPE",
                "query-gate sigmoid",
            ),
        )
