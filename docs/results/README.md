# Published Results

Each file in this folder is a write-up of past runs. The file name starts with the year and month of the runs.

| Write-up | Runs |
|---|---|
| [2026-07-qwen3.6-35b-a3b-nvfp4.md](2026-07-qwen3.6-35b-a3b-nvfp4.md) | Five configs of Qwen3.6-35B-A3B NVFP4 |
| [2026-07-three-model-comparison.md](2026-07-three-model-comparison.md) | NVIDIA, Red Hat, and Unsloth quantizations of Qwen3.6-35B-A3B |

The raw data of a run is in its run dir, `results/<name>/<timestamp>/`. The `name:` in the model config sets the folder name. Use the lowercase scheme `<model>-<size>-<provider>-<quant>[-<variant>]`, for example `qwen3.8-27b-nvidia-nvfp4`.

Runs before 2026-11-07 use pre-fix scoring. See [../dashboard.md](../dashboard.md) for the details.
