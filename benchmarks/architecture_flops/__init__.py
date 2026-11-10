"""FLOP estimates from a model's config.json.

The package re-exports its public names, so callers import them from
``benchmarks.architecture_flops``.
"""

from benchmarks.architecture_flops.analysis import run_flops_analysis
from benchmarks.architecture_flops.attention import FullAttentionEstimator
from benchmarks.architecture_flops.component import ComponentResult
from benchmarks.architecture_flops.config import (
    ArchitectureFeatures,
    ConfigLoadResult,
    NormalizedConfig,
    detect_features,
    missing_required_fields,
    normalize_config,
)
from benchmarks.architecture_flops.ffn import DenseFfnEstimator, MoeFfnEstimator
from benchmarks.architecture_flops.linear_attention import GatedDeltaNetEstimator
from benchmarks.architecture_flops.lm_head import LmHeadEstimator
from benchmarks.architecture_flops.loader import load_architecture_config
from benchmarks.architecture_flops.model import ModelFlopsEstimator

__all__ = [
    "ArchitectureFeatures",
    "ComponentResult",
    "ConfigLoadResult",
    "DenseFfnEstimator",
    "FullAttentionEstimator",
    "GatedDeltaNetEstimator",
    "LmHeadEstimator",
    "ModelFlopsEstimator",
    "MoeFfnEstimator",
    "NormalizedConfig",
    "detect_features",
    "load_architecture_config",
    "missing_required_fields",
    "normalize_config",
    "run_flops_analysis",
]
