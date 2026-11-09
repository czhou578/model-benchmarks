# Published Results

Each file in this folder is a write-up of past runs. The file name starts with the year and month of the runs.

| Write-up | Runs |
|---|---|
| [2026-11-qwen3.8-27b-nvfp4.md](2026-11-qwen3.8-27b-nvfp4.md) | Four runs of Qwen3.8-27B NVFP4 on vLLM 0.25.1 and 0.33.0, with the current E4 baseline (run D) |
| [2026-07-qwen3.6-35b-a3b-nvfp4.md](2026-07-qwen3.6-35b-a3b-nvfp4.md) | Five configs of Qwen3.6-35B-A3B NVFP4, one with MTP spec decoding (corrected 2026-11-09) |

[../archive/results/](../archive/results/) holds write-ups without published run data. Nobody can check their numbers.

The raw data of a run is in its run dir, `results/<name>/<timestamp>/`. The `name:` in the model config sets the folder name. Use the lowercase scheme `<model>-<size>-<provider>-<quant>[-<variant>]`, for example `qwen3.8-27b-nvidia-nvfp4`.

Fixes to the harness divide the runs into eras. See [../comparability.md](../comparability.md) before you compare two runs.
