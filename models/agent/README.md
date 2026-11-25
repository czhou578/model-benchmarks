# Agent model configs

These configs serve a model for the sandbox step only ([docs/sandbox.md](../../docs/sandbox.md)). Run them with
`--only-sandbox`. `batch_run.py` skips this folder unless `--model` names a config in it.

They differ from the speed configs in `models/`: prefix caching is on, the server uses the sampling values of the
model (`--generation-config auto`), and the server takes 8 sequences for 8 agents.

| Config | Weights | Status |
|---|---|---|
| `gpt_oss_20b_openai_mxfp4.yml` | `openai/gpt-oss-20b` | Not run yet |
| `qwen3.6_35b_a3b_nvidia_nvfp4.yml` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Not run yet |

Before a run, set `sandbox.agent_eval_commit` to the commit of the agent-eval checkout. `heldout_commit: v1` needs
the tag `v1` in `czhou578/agent-eval-heldout`.
