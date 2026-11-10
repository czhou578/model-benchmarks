"""ComponentResult: the FLOP record of one layer component."""

from __future__ import annotations

from dataclasses import dataclass, field

# Results from a single component's layer-level computation.
# All quantities are per-layer, per-token (decode) or per-request (prefill).


@dataclass(frozen=True)
class ComponentResult:
    """FLOP breakdown for a single layer component.

    Only the linear-attention estimator sets ``kv_read_bytes`` and
    ``kv_write_bytes``. They give the FP32 recurrent state traffic, which the
    model reports as ``state_traffic_bytes``.

    Component estimators return per-layer work. Decode values cover one new
    token; prefill values cover the complete prompt request. Model-level
    composition applies the architecture's actual layer counts.
    """

    matmul_flops: float = 0.0              # matrix-multiply FLOPs (2 per MAC)
    non_matmul_flops: float = 0.0          # activations, gating, elementwise
    matmul_breakdown: dict[str, float] = field(default_factory=dict)
    non_matmul_breakdown: dict[str, float] = field(default_factory=dict)
    omitted_non_matmul: tuple[str, ...] = ()
    kv_read_bytes: float = 0.0             # recurrent state read (linear attention)
    kv_write_bytes: float = 0.0            # recurrent state written (linear attention)

    @property
    def total_flops(self) -> float:
        return self.matmul_flops + self.non_matmul_flops
