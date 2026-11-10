"""CLI entry point: ``python -m benchmarks.architecture_flops [config.yml]``."""

import argparse
import json
from pathlib import Path

from benchmarks.architecture_flops import run_flops_analysis
from core_runner import load_model_config as _load_model_config

_DEFAULT_CONFIG = Path("models/legacy/qwen3.6_35b_redhat_nvfp4.yml")

parser = argparse.ArgumentParser(
    prog="python -m benchmarks.architecture_flops",
    description="Print the FLOP analysis of one model config as JSON.",
)
parser.add_argument(
    "config",
    nargs="?",
    type=Path,
    default=_DEFAULT_CONFIG,
    help=f"the model config YAML file (default: {_DEFAULT_CONFIG})",
)
args = parser.parse_args()
cfg = _load_model_config(args.config)
result = run_flops_analysis(cfg)
print(json.dumps(result, indent=2, default=str))
