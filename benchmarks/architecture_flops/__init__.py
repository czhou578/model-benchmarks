"""FLOP estimates from a model's config.json.

The package re-exports its public names, so callers import them from
``benchmarks.architecture_flops``.
"""

from benchmarks.architecture_flops.analysis import run_flops_analysis
from benchmarks.architecture_flops.config import (
    ArchitectureFeatures,
    ConfigLoadResult,
    NormalizedConfig,
    detect_features,
    missing_required_fields,
)
from benchmarks.architecture_flops.loader import load_architecture_config
from benchmarks.architecture_flops.model import (
    ComponentResult,
    DenseFfnEstimator,
    FullAttentionEstimator,
    GatedDeltaNetEstimator,
    LmHeadEstimator,
    ModelFlopsEstimator,
    MoeFfnEstimator,
    normalize_config,
)

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
