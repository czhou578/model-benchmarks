"""ComponentResult: the FLOP and byte record of one layer component."""

from __future__ import annotations

from dataclasses import dataclass, field

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
