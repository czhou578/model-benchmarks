"""Gated DeltaNet (linear attention) FLOP estimator."""

from __future__ import annotations

from benchmarks.architecture_flops.component import ComponentResult


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
