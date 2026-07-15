# Concurrency Benchmark Module

`benchmarks/concurrency.py`

## Implementation Plan

### What this adds
Measures throughput and latency degradation as request concurrency increases — i.e., "if I fire N requests simultaneously, how does aggregate throughput and per-request latency change?" This is the final item from the report's "Recommended next steps."

### Why it matters
- vLLM processes requests in batches; at low concurrency the GPU is underutilized, at high concurrency it hits the `--max-num-seqs` limit and TTFT/latency degrade
- Shows where the throughput ceiling is for your specific GPU config

### Files to create/modify

**New files:**
1. `benchmarks/__init__.py` — empty package marker
2. `benchmarks/concurrency.py` — the concurrency test module (~150 lines)

**Modified files:**
3. `core_runner.py` — add import, CLI flag, and orchestration block
4. `models/*.yml` — add `concurrency_levels` / `concurrency_requests_per_level` config fields

### Key design decisions
- **Separate module**: `benchmarks/concurrency.py` imports from `core_runner` (direction: concurrency → core_runner, so no circular import at runtime)
- **Reusable prompts**: uses `core_runner.build_prompt_of_length(256)` for deterministic, tokenizer-aware prompts
- **Lightweight dataclass**: `ConcurrentRequestResult` captures only throughput/latency fields (no `output_text`, no `reasoning_text`)
- **Error resilient**: each request caught individually, failures tracked but don't abort the batch
- **ThreadPoolExecutor**: one fresh pool per concurrency level, `max_workers=level` (the concurrency level)
- **5 requests per level** (user-selected)
- **Default sweep**: `[1, 2, 4, 8, 16]` concurrency levels

### Implementation steps

1. **Create `benchmarks/__init__.py`** — empty file (package marker)

2. **Create `benchmarks/concurrency.py`** — the core module:
   - `ConcurrentRequestResult` dataclass (success, index, prompt_tokens, output_tokens, ttft_s, total_time_s, error)
   - `CONCURRENCY_PROMPT = build_prompt_of_length(256)` (imported)
   - `_stat_summary()` helper — wraps `core_runner._percentile` for avg/median/p95/min/max
   - `run_concurrency_test()` main function
   - `if __name__ == "__main__"` — standalone smoke test

3. **Update `core_runner.py`**:
   - Add `from benchmarks.concurrency import run_concurrency_test` at top
   - Add `--skip-concurrency` CLI flag to argument parser
   - Add orchestration block in `main()` after reasoning benchmark:
     ```python
     if not args.skip_concurrency:
         levels = cfg.get("concurrency_levels", [1, 2, 4, 8, 16])
         reqs = cfg.get("concurrency_requests_per_level", 5)
         max_tok = cfg.get("concurrency_max_tokens", 256)
         temp = cfg.get("concurrency_temperature", 0.0)
         print(f"[core_runner] concurrency test: levels={levels} ({reqs} reqs/level)")
         concurrency_results = run_concurrency_test(client, levels, reqs, max_tok, temp)
         save_json(run_dir / "concurrency.json", concurrency_results)
         summary["concurrency"] = concurrency_results
     ```

4. **Update model YAMLs** — add these fields to all three model configs:
   ```yaml
   concurrency_levels: [1, 2, 4, 8, 16]
   concurrency_requests_per_level: 5
   ```

### Expected output structure (`concurrency.json`)

```json
{
  "config": {
    "concurrency_levels": [1, 2, 4, 8, 16],
    "requests_per_level": 5,
    "max_tokens": 256,
    "temperature": 0.0,
    "prompt_tokens": 256
  },
  "per_concurrency_level": {
    "4": {
      "wall_time_s": 5.23,
      "total_output_tokens": 1280,
      "aggregate_throughput_tok_s": 244.7,
      "n_requests": 5,
      "n_success": 5,
      "n_failed": 0,
      "success_rate": 1.0,
      "ttft": {"avg_s": 0.045, "median_s": 0.042, "p95_s": 0.067, "min_s": 0.031, "max_s": 0.067},
      "total_time_s": {"avg_s": 0.512, "median_s": 0.508, "p95_s": 0.531, "min_s": 0.495, "max_s": 0.531},
      "individual_requests": [{"index": 0, "success": true, "prompt_tokens": 256, "output_tokens": 256, "ttft_s": 0.031, "total_time_s": 0.495, "error": ""}, ...]
    }
  }
}
```

### Verification
- `python -m py_compile benchmarks/concurrency.py` — syntax check
- Run the full benchmark: `python core_runner.py --model models/qwen3.6_35b_redhat_nvfp4.yml`
- Check `concurrency.json` in the latest run output directory
- Confirm the `--skip-concurrency` flag works