"""CLI entry point: ``python -m benchmarks.architecture_flops <config.yml>``."""

import json
import sys
from pathlib import Path

from benchmarks.architecture_flops import run_flops_analysis
from core_runner import load_model_config as _load_model_config

cfg_path = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/legacy/qwen3.6_35b_redhat_nvfp4.yml")
)
cfg = _load_model_config(cfg_path)
result = run_flops_analysis(cfg)
print(json.dumps(result, indent=2, default=str))
