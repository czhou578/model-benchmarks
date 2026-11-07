# Managed vLLM Server Lifecycle Plan

## Goal

Make every benchmark configuration self-contained and reproducible by storing
the exact vLLM command in the model YAML. `core_runner.py` will start the
configured server, wait until the model can answer a request, run the
benchmarks, and always stop the server it owns.

The runner must not kill an arbitrary process. Managed mode requires the
configured port to be free and only terminates the process group that it
started. External mode remains available for benchmarking a server started by
the user.

## Configuration

Add a `server` section to each managed model configuration:

```yaml
server:
  mode: managed
  command:
    - vllm
    - serve
    - organization/model-name
    - --host
    - "127.0.0.1"
    - --port
    - "8000"
  environment: {}
  startup_timeout_s: 600
  shutdown_timeout_s: 30
```

The command is an argument list rather than a shell string. This preserves
spaces and JSON arguments without invoking a shell or relying on `.split()`.

## Implementation

1. Add a standalone `VllmServer` class in `vllm_server.py`.
   - Validate the command and endpoint port.
   - Refuse to start if the endpoint port is already occupied.
   - Launch vLLM in its own process group.
   - Send stdout and stderr directly to a run-specific log file.
   - Detect early server exits and include the log tail in errors.
   - Stop the owned process group with `SIGTERM`, escalating to `SIGKILL` only
     after the configured timeout.

2. Integrate the class into `core_runner.py`.
   - Select managed or external mode from YAML, with a CLI override.
   - Start managed vLLM before constructing and probing the model client.
   - Keep the existing HTTP and one-token completion readiness checks.
   - Check that the managed process remains alive while readiness is polled.
   - Start GPU measurement only after readiness and warm-up complete.
   - Stop GPU measurement and vLLM in `finally` blocks.

3. Record reproducibility artifacts in every result directory.
   - Copy the input model YAML to `model_config.yml`.
   - Save the resolved command, environment-variable names, PID, port, and
     lifecycle timing to `resolved_server.json`.
   - Save all server output to `vllm.log`.
   - Redact environment values whose names look like credentials.

4. Reuse the lifecycle for speculative decoding.
   - Stop the ordinary managed server before restarting on the same port.
   - Run speculative and non-speculative variants as separate owned servers.
   - Compare already-collected result dictionaries rather than requesting
     results from a server that has been stopped.

5. Add preflight validation.
   - Require the command's `--port` to match `endpoint.base_url`.
   - Warn when concurrency requests are lower than the requested concurrency.
   - Refuse invalid server modes, empty commands, and non-string arguments.

6. Add focused tests.
   - Successful process start and cleanup.
   - Refusal when a port is occupied.
   - Early child-process failure with useful log output.
   - Arguments are preserved without shell parsing.

## CLI

Managed mode is selected by the model YAML:

```bash
python core_runner.py --model models/model.yml
```

An already-running endpoint can be used without changing the YAML:

```bash
python core_runner.py --model models/model.yml --server-mode external
```

## Acceptance Criteria

- A normal run starts exactly the command represented in its YAML.
- Benchmarks do not begin until the configured model answers a completion.
- Ctrl-C, benchmark failures, and readiness timeouts clean up runner-owned
  vLLM processes.
- An unrelated process on the configured port is never terminated.
- The result directory contains the model config, resolved server metadata,
  and vLLM log needed to reproduce or diagnose the run.
