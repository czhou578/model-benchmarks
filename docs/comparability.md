# Which results you can compare

The harness changed between the published runs. Some fixes changed what a metric measures. This page puts the published runs into eras. It tells which metrics you can compare across an era boundary, and how large the noise is.

Each run dir records its versions in `environment.json`. [results-schema.md](results-schema.md) lists the fields that old runs lack.

## Eras

The PR merge times are in PST.

| Era | Dates | Starts with | vLLM, torch | Published run dirs | What changed at the start | Do not compare with later eras |
|---|---|---|---|---|---|---|
| E0 | 2026-07-29 | The first published runs | 0.25.1, 2.11.0+cu130 | The five dirs `results/qwen3.6-35b-a3b-*/20260729_*` | – | Client TTFT, TTFT breakdown, prefill scaling, concurrency, tool calling, energy |
| E1 | 2026-11-07 | #42 and #44, merged 2026-11-06 21:42 | 0.25.1, 2.11.0+cu130 | `results/qwen3.8-27b-nvidia-nvfp4/20261107_094900` (full run), `results/qwen3.8-27b-nvidia-nvfp4-prefill/20261107_125906` (prefill rerun) | `cache_salt` reaches vLLM. Server timings. New tool-calling scorer. Prefill scaling works. | Concurrency, energy |
| E2 | 2026-11-08 | #67 to #69, merged 2026-11-07 17:35. #70, merged 2026-11-08 13:26 | 0.33.0, 2.14.0+cu130 | `results/qwen3.8-27b-nvidia-nvfp4/20261108_132808` | Fixes to the TTFT split and the energy integral. A distinct concurrency prompt for each request. Configs on vLLM 0.33.0 with `--no-enable-prefix-caching`. | Tool calling, with care |
| E3 | from 2026-11-09 | #73 to #75, merged 2026-11-09 12:29 to 12:32 | 0.33.0 in the configs | None yet | 600 s tool-calling timeout. `tool_calling.status` in `summary.json`. | – |

### E0: July 2026, Qwen3.6-35B-A3B

- The client put `cache_salt` inside an `extra_body` object, and vLLM ignored it. The latency sweep and deep context sent no salt. Thus their TTFT can include prefix-cache hits. Three of the five servers had `--enable-prefix-caching`.
- `cache_isolation_method` in these files says `cache_salt`, but no request had a working salt.
- The TTFT breakdown has no server timings. All server fields are `null`.
- Prefill scaling holds `status: request_error` for each length. #45 fixed this.
- Tool calling used the old scorer. The run dirs have no `tool_calling.json`. The old scorer had three defects:
  - In the 33 multi-tool and error-recovery tasks, the chat history put the tool result before the tool call.
  - Each task allowed only 256 output tokens. A thinking model often used all of them before its first tool call.
  - Three of the six composite components were always 0. The composite read a `single_tool` category that no task had.
- Every concurrency request sent one shared prompt.
- The energy values use the old method. `energy_per_token_wh` is far too high.

### E1: 2026-11-07, Qwen3.8-27B on vLLM 0.25.1

- The full run started before #45 (merged 2026-11-07 12:57). Its `prefill_scaling.json` holds only `request_error`. The prefill rerun started after #45 and gives the E1 prefill data.
- The prefill rerun used the `lite` tool-calling set. Do not compare its tool-calling score with a full run.
- The stored `post_queue_ms` is the server TTFT minus the queue time. It is low by the queue time, about 0.015 ms in this run. `server_ttft_s` is correct.
- The energy values use the old method. `energy_wh` is about 6% low, and `energy_per_token_wh` is about 17 times too high.
- Every concurrency request sent one shared prompt of 308 server tokens. E2 sends a distinct prompt of exactly 256 server tokens, chat template included.

### E2: 2026-11-08, Qwen3.8-27B on vLLM 0.33.0

- vLLM, torch, and the harness all changed at this boundary. Use E1 and E2 together only to measure the vLLM upgrade.
- The tool-calling request timeout was 120 s. The task `schema_title_max_length_fail` failed with `Read timed out. (read timeout=120)`. A 4096-token answer at about 12.3 tokens/s takes about 330 s.

### E3: from 2026-11-09

- #73 raised the tool-calling timeout to 600 s (`tool_calling_request_timeout_s`). A task that timed out in E2 can pass in E3.
- #73 also changed the `convert_currency` mock to compute `amount` × rate, and a failed multi-turn request now fails its task. Thus tool-calling results can change a little.
- `summary.json` records `tool_calling.status`, and `failure_modes` has `request_timeout`.
- #74 changed the HTTP error handling and deep context errors. #75 fixed `batch_run.py`. Neither changes the numbers of a run without errors.
- #76 to #88 move code into `harness/` and `benchmarks/`. Their descriptions state no behavior change.

## Metrics across era boundaries

"Yes" means that the harness measures the metric with the same method on both sides. The vLLM version can still change the number. E1 and E2 also differ in the vLLM and torch versions. Read a difference between them as an effect of the vLLM upgrade.

| Metric | E0 and later eras | E1 and E2 | E2 and E3 |
|---|---|---|---|
| Client TTFT: latency, deep context | No: cache hits | Yes | Yes |
| TTFT breakdown | No: no server data | Yes, from `server_ttft_s` | Yes |
| Decode, with and without MTP | Yes | Yes | Yes |
| Reasoning tokens | Yes | Yes | Yes |
| Concurrency | No: shared prompt | No: shared prompt, other length | Yes |
| Prefill scaling | No: no data | Yes, from the prefill rerun | Yes |
| Tool calling | No: old scorer | Yes, inside the noise | With care: timeout and mock |
| Energy | No: old method | No: old method | Yes |

## How the dashboard marks the eras

`dashboard.py` and `reports/dashboard/runs.js` give a run card these badges:

| Badge | Rule in the code | Boundary |
|---|---|---|
| Pre-fix scoring | The run dir name is before 2026-11-07 (`SCORING_FIX_DATE`). | E0 and E1 |
| Shared concurrency prompt | `concurrency.json` has no `prompt_mode`. | E1 and E2 |
| Older vLLM | `vllm_version` is older than the newest vLLM of the shown runs. | E1 and E2 |

The dashboard also corrects two old values:

- The "Server-side TTFT split" table computes the post-queue median from `server_ttft_s`. It shows the correct value for E1.
- The page does not show `energy_per_token_wh`. A run card shows "Decode energy" only from `decode_energy`, which E2 and later runs have.

Known limits:

- "Pre-fix scoring" uses a date, not a schema version. A run from an old checkout after that date gets no badge.
- "Older vLLM" compares only the shown runs. When you hide the newer run, the badge goes away.
- No badge marks the E2 and E3 boundary or the old whole-run energy. "Run energy" in the "All runs" table is about 6% low for E0 and E1.
- The note above the charts says that the pre-fix concurrency numbers are not affected. This is not correct. Every E0 run sent one shared prompt. Use the "Shared concurrency prompt" badge.
- The "Prefix cache" column reads only the `--enable-prefix-caching` flag. vLLM 0.33.0 turns prefix caching on by default. A vLLM 0.33.0 run with no flag thus shows "off", which is not correct.

## Noise

The harness does not repeat a run. Each published number comes from one run. [methodology.md](methodology.md#repeats-and-percentiles) gives the repeats inside a run.

### Tool calling

- The full set has 85 tasks. The Qwen3.8-27B runs pass about 60% of them.
- At 85 tasks and a 60% pass rate, the 95% binomial confidence interval of the pass rate is about ±10 points, or about ±9 tasks: 1.96 × √(0.6 × 0.4 / 85) ≈ 0.10.
- This interval tells how well 85 tasks estimate the model ability. It does not come from repeated runs.

### Reference runs

Three runs of Qwen3.8-27B NVFP4 give the only same-model reference. The gate run is the vLLM 0.33.0 migration gate of 2026-11-07. It used the E1 harness, and it is not published.

| Run | vLLM | Tool tasks passed | Composite | Decode, no spec | TTFT median at 2K |
|---|---|---|---|---|---|
| `20261107_094900` (E1) | 0.25.1 | 52 of 85 | 5.03 | 12.17 tok/s | 793 ms |
| Gate run | 0.33.0 | 50 of 85 | 4.78 | 12.18 tok/s | 645 ms |
| `20261108_132808` (E2) | 0.33.0 | 50 of 85 | 4.78 | 12.32 tok/s | 650 ms |

- The gate run and the E2 run use the same vLLM and the same tool-calling code. Each task got the same result in both runs. The same task timed out at 120 s in both runs.
- Between these two runs, decode differs by 1.1% and TTFT at 2K by 0.8%.
- From vLLM 0.25.1 to 0.33.0, six tasks changed result: four to fail and two to pass. TTFT at 2K fell by 19%. This is an effect of vLLM, not noise.

### Rules of thumb

1. Compare a metric across eras only where the table above says "Yes".
2. Before you compare two runs, read the vLLM and torch versions in `environment.json`.
3. Treat a tool-calling difference below about 9 tasks of 85 as noise. Do not rank two models on a smaller gap.
4. Expect a vLLM upgrade alone to change a few tool-calling tasks.
5. On one vLLM version, treat a decode or TTFT difference below about 2% as noise.
6. Do not compare a `lite` tool-calling run with a full run.
7. Compare the whole-run `energy_wh` only between runs with the same benchmarks.
