"""lm_head (vocabulary projection) FLOP estimator."""

from __future__ import annotations

from benchmarks.architecture_flops.component import ComponentResult


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
            matmul_breakdown={"lm_head_flops": matmul},
            omitted_non_matmul=("logits post-processing / sampling",),
        )

    def prefill_flops(self, logits_tokens: int = 1) -> ComponentResult:
        if logits_tokens < 0:
            raise ValueError("logits_tokens must be non-negative")
        matmul = logits_tokens * 2 * self.d * self.v
        return ComponentResult(
            matmul_flops=matmul,
            matmul_breakdown={"lm_head_flops": matmul},
            omitted_non_matmul=("logits post-processing",),
        )
