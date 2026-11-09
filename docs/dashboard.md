# Dashboard

`dashboard.py` reads every run dir under `results/` and writes one self-contained HTML page. The page needs no server and no network. The script needs only the Python standard library.

## Build the page

Run these commands from the repo root:

```bash
python3 dashboard.py                      # write reports/dashboard.html
python3 dashboard.py --out /tmp/d.html    # write the page to another path
python3 dashboard.py --results my_runs/   # read run dirs from another folder
python3 dashboard.py --fragment           # write the page body only, to embed it
python3 dashboard.py --json data.json     # also write the collected data as JSON
```

Open `reports/dashboard.html` in a browser.

## Serve the page

`serve_dashboard.py` builds the page and serves it over HTTP. It rebuilds the page on each request for the page. Thus a browser reload after a run shows the new run.

```bash
python3 serve_dashboard.py                  # http://127.0.0.1:8080/dashboard.html
python3 serve_dashboard.py --port 8090      # use another port
```

The script also takes `--results` and `--out`, with the same meaning as in `dashboard.py`. Stop the server with Ctrl+C. If the port is in use, the script prints the page URL and stops. It assumes that a dashboard server already runs on that port.

WARNING: The server has no access control. With `--host 0.0.0.0`, every host on the network can read your results. Use that option only on a trusted network.

```bash
python3 serve_dashboard.py --host 0.0.0.0   # reachable from other machines
```

### Serve the page from VS Code

The "Dashboard" entry in `.vscode/launch.json` runs `serve_dashboard.py`.

1. Open the Run and Debug view.
2. Select **Dashboard**.
3. Press the green arrow, or F5.

VS Code forwards the port and opens the page in your browser. Reload the page after a run. To stop the server, click the red square in the debug toolbar.

## What the page shows

- One card per run. Each card shows the start time, the vLLM version, the `run_label` from `summary.json` when the run has one, and the badges below.
- The chart legends add the vLLM version to each run label when the shown runs use more than one version. The page reads the version from `environment.json`: `vllm_server_version`, else `vllm_version`.
- The card numbers: decode speed, TTFT at 2K tokens, peak throughput, tool tasks passed, lowest free memory, and mean GPU power.
- Charts for decode speed, TTFT, throughput under load, long-context TTFT, thinking tokens, and the MTP speed-up.
- The decode chart shows the peak rate only when `rate_method` is `token_delta`. An older run counts stream events, not tokens, so its peak shows as "–".
- Charts for tool-calling scores, the tool-calling pass rate by category, and memory headroom.
- Two tables: "All runs" with the era, the key numbers, and the server settings, and "Server-side TTFT split". The "Prefix cache" column shows the flag in the server command, or "default" when the command has no flag.
- A Table toggle on each chart. The toggle shows the chart values as a table.
- Run filter chips above the charts. Each chip shows or hides one run.

The memory headroom chart needs `memory_samples.csv` from `memory_guard.py`. See [dgx-spark.md](dgx-spark.md).

The two tool-calling charts leave out runs with the `lite` task set. A lite composite is at most 3 of 6, so it is not comparable with a full run. See [tool-calling.md](tool-calling.md). The card and the "All runs" table mark a lite run. The page reads the task set from `config.task_set` in `tool_calling.json`. Without that file, the page treats the run as full.

A card can show these badges:

| Badge | Meaning |
|---|---|
| Completed, Failed | The `status` in `summary.json`. |
| MTP spec on, No spec decode | The vLLM command in `resolved_server.json` has a `--speculative-config` flag, or not. |
| Era E0 to Era EN | The harness era of the run. See the next section. |
| Older vLLM | The run used an older vLLM version than the newest shown run. Compare runs on different versions with care. |
| Pre-fix scoring | The run is from era E0, before the fixes of 2026-11-07. See the next section. |
| Shared concurrency prompt | The run sent one prompt to every concurrency request. Its concurrency numbers are not comparable with newer runs. See `prompt_mode` in [results-schema.md](results-schema.md). |
| Stopped for memory | `memory_guard.py` stopped the run because free memory was too low. |

## Pre-fix scoring and other eras

The badges and the note above the charts mark old runs. The page reads the era from `harness.schema_version` in `summary.json`, else from `provenance.json` in the run dir, else from the run dir date. [comparability.md](comparability.md#how-the-dashboard-marks-the-eras) gives the rule, the metrics of each era that you can compare, and the known limits of the badges.

## Page source

The page source is in `reports/dashboard/`: `page.html`, the CSS modules, and the JS modules. `dashboard.py` joins them in the order of `CSS_MODULES` and `JS_MODULES`, and it embeds the run data. The JS modules share one function scope. `core.js` defines `start()`, which runs after the last module.
