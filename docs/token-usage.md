# Token usage of the sandbox step on Token Tracker

## In simple words

In the sandbox step, agent-eval calls the model server directly. Thus [Token Tracker](https://github.com/czhou578/token-tracker)
does not see the tokens that the agents use. `benchmarks/token_usage.py` reads the transcripts of agent-eval and
sends the token counts to Token Tracker. Then the tokens show on the Token Tracker dashboard.

## What the module sends

The module sends one usage event for each model turn of each finished run.

| Field | Value |
|---|---|
| `provider` | `agent-eval` |
| `model` | The served model name of the endpoint. |
| `project` | `model-benchmarks/sandbox` from the sandbox step. `agent-eval` from the command line, or the value of `--project`. |
| `thread_id` | `<point>/<run folder>`, for example `<model>/<run time>/<model>-<task>-s<seed>` from the sandbox step. |
| `source_id` | `agent-eval:<thread_id>:<turn number>` |
| `prompt_tokens` | `prompt_tokens` of the server minus the cached tokens. |
| `cache_read_tokens` | `prompt_tokens_details.cached_tokens` of the server, or 0. |
| `completion_tokens` | `completion_tokens` of the server. The reasoning tokens are in this count. |
| `total_tokens` | `prompt_tokens` plus `completion_tokens` of the server. |
| `latency_ms` | `seconds` of the turn. |

The module sends counts, model names, task ids and seeds only. It sends no prompt and no transcript text.

Token Tracker keeps one row for each `source_id`. Thus a second push of the same point adds nothing.

A transcript has no time for each turn, so the module calculates the time:

- The end of the run is the change time of `transcript.json`.
- The start of the run is the end minus `seconds` of the run.
- A turn ends at the start plus the `seconds` of the turns up to it.

The time between the turns (the tools) is not in this result, so a turn can show some seconds too early. The times
are wrong if a later command (for example `agent_eval.py recheck`) writes the transcript again before the first push.

## Turn the push on and off

The sandbox step pushes after each finished run and one more time at the end, if these two conditions are true:

- `sandbox.token_tracker` in the model config is `true`. This is the default.
- The machine has an ingest token of Token Tracker: the file `~/.tokentracker/ingest_token`, or the environment
  variable `TOKEN_TRACKER_INGEST_TOKEN`. The command `tracker ingest-token` of Token Tracker makes the file.

To turn the push off for a model, set the key in its config:

```yaml
sandbox:
  token_tracker: false
```

A machine without an ingest token sends nothing, and the step prints no warning.

The step sends to `http://127.0.0.1:8765`. Set `TOKEN_TRACKER_URL` for another address. The events get the name of
this machine as the host. Set `TOKEN_TRACKER_HOSTNAME` for another name.

A failed push does not fail the sandbox step: the step prints one warning and continues. The last push sends each
run that an earlier push did not send.

## Push a point by hand

Use the command line for a point that agent-eval ran outside model-benchmarks, or after a failed push:

```bash
python -m benchmarks.token_usage RESULTS/G1 --model qwen=second-model-fit --dry-run
```

- `--dry-run` prints the totals and sends nothing. Remove it to send the events.
- `--model NAME=SERVED_NAME` shows the agent-eval model `NAME` under the served model name. Give it for each model.
- `--point NAME` sets the point name in the thread id. The default is the folder name.

Give each point a different name. A second point with the same name replaces the rows of the first point.

To see the totals of the provider, ask the API of Token Tracker:

```bash
curl -s 'http://127.0.0.1:8765/api/stats?provider=agent-eval'
```
