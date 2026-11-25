# The sandbox step

## In simple words

The other steps measure how fast a model is and how well it calls mocked tools. The sandbox step lets the model work
as an agent on real coding tasks. Each command that the model writes runs in a locked Docker container with no
network. At the end, hidden tests decide if the task passes. The step gives the pass rate, the hallucination rate,
and the minutes and tokens for each task.

The step does not hold its own agent code. It runs agent-eval (`dsv4-specialize/agent-eval`) at a pinned commit and
stores its results in the run dir.

## What the step does

1. It checks that the agent-eval checkout, and the held-out checkout when the config names one, are at their pinned
   commits with no local changes. A pin is a commit or a tag. The step records the full commit.
2. It reads `/metrics` of the server. It refuses to run when the server has prefix caching off, because an agent
   loop sends its full history again on each turn. Without `/metrics`, it reads the flag of the managed server
   command. `--sandbox-allow-no-prefix-cache` runs anyway.
3. It runs `agent_eval.py point` with the model as `--endpoint`. Each task runs with each seed.
4. It calls `job step` for each finished task run, when it runs inside `job run`.
5. It sends the token counts of each finished task run to Token Tracker, when the push is on. See
   [Token usage on Token Tracker](#token-usage-on-token-tracker).
6. It reads `/metrics` again and runs `agent_report.py --json` with the block size of the server.
7. It writes `sandbox.json` and adds the rates of each task set to `summary.json`.

A failure of the step goes to `summary.json` as `"status": "failed"` with the error, and the run continues.

## The task sets

| Set | Tasks | Where |
|---|---|---|
| `dev` | C1 to C8 and D1 to D8 (code and debug) | `agent-eval/tasks` in `czhou578/spark-moe-specialize`, public |
| `heldout` | 20 in version `v1`, 40 in `v2` | `czhou578/agent-eval-heldout`, private |

The held-out score is the main score. The dev tasks are public, so a future model can train on them. Scores
compare only within one version of each set.

## Requirements

- A checkout of agent-eval with task sets (`--endpoint` and `AGENT_EVAL_TASKS`, spark-moe-specialize#109).
- The Docker image `agent-eval-sandbox:1`, built from `agent-eval/sandbox`.
- For the held-out set: a clone of `czhou578/agent-eval-heldout`.
- A model config with a `sandbox` section and prefix caching on. The configs in `models/agent/` are examples.

## Run modes

| Mode | Server | Agents | Slots × memory | Peak memory |
|---|---|---|---|---|
| Worker mode | External: a running worker | 1 | 1 × 1 GiB | 1 to 3 GiB more |
| Window mode | Managed small MoE, DeepSeek stopped | 8 | 8 × 2 GiB | about 58 GiB |
| Large window mode | Managed large MoE, both workers stopped | 4 | 4 × 2 GiB | about 84 to 104 GiB |

CAUTION: In worker mode, the `pair-watch` session stops Qwen below 2 GiB of free memory. Start the run through
`memory_guard.py --floor-gib 3 --min-start-gib 3.5`, with one slot. Then the guard stops the run first.

Keep the agents and the limits the same for all models of one size class. More agents slow each agent, so a time
limit stops more tasks.

Examples:

```bash
# Worker mode: the Qwen worker, through an external config with a sandbox section
job run --desc "sandbox, Qwen worker, model-benchmarks" "sandbox-qwen" -- \
  python memory_guard.py --floor-gib 3 --min-start-gib 3.5 -- \
  python -u core_runner.py --model my_qwen_worker.yml --only-sandbox

# Window mode: two agent configs, one after the other
job run --desc "sandbox, two small MoE models, model-benchmarks" "sandbox-window" -- \
  python memory_guard.py -- python batch_run.py --no-retry --only-sandbox \
  --model agent/gpt_oss_20b_openai_mxfp4.yml --model agent/qwen3.6_35b_a3b_nvidia_nvfp4.yml
```

Use `--no-retry` with `batch_run.py`: a retry makes a new run dir and runs all tasks again.

## Read the results

- Each rate has a 95% interval from a bootstrap over the tasks, with the seeds of a task averaged first. With 16
  tasks, the interval of one rate is about ±24 points at a 50% rate. A difference inside the interval is noise.
- The difference between two models is more precise, because both run the same tasks:
  `python3 agent_report.py <point> --ref <point>` in agent-eval.
- `prefix_cache.hit_rate` is the measured hit rate. `prefix_cache.best_rate` is the best rate for the block size of
  the server. A hybrid model with linear-attention layers gets large blocks in vLLM 0.33 (2,176 tokens for
  Qwen3.6-35B-A3B), so its hit rate is low on short prompts. Only a measured rate below the best rate shows a
  problem, for example a chat template that changes earlier turns.
- A hybrid model computes up to one block again on each turn. Compare its minutes per task with a non-hybrid model
  with care.

## Token usage on Token Tracker

agent-eval calls the model server directly, so [Token Tracker](https://github.com/czhou578/token-tracker) does not
see the tokens of the agents. The step sends one usage event for each model turn to Token Tracker, with the provider
`agent-eval`. It sends counts, model names, task ids and seeds only.

The push is on when `sandbox.token_tracker` is `true` (the default) and the machine has an ingest token of Token
Tracker. Set `sandbox.token_tracker: false` in the model config to turn the push off. A failed push gives one warning
and does not fail the step. [token-usage.md](token-usage.md) has the fields, the switches and the command for a push
by hand.

## Isolation

CAUTION: The model writes the commands, and nobody reads them before they run. Do not mount the Docker socket, the
home folder or a GPU device into a sandbox container.

Each command runs in a new container with no network, a read-only root, a tmpfs of 1 GiB on `/tmp`, the memory
limit with no swap, 256 processes, a non-root user, no capabilities and no new privileges. The agent loop runs on
the host, so the containers need no route to the model server.
