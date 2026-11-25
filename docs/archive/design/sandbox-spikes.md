# Sandbox spikes

Status: Phase 0 notes of the sandbox step ([sandbox.md](../../sandbox.md)). S1 is done. S2 (gVisor) is optional and
not done.

## S1: prefix cache of the Qwen worker in an agent loop

Date: 2026-11-23, 21:14 to 21:17 local time. Job: `s1-prefix`.

### Question

Does the prefix cache of the server keep the earlier turns of an agent loop? An agent loop sends the full history
again on each turn. When the cache misses, the server computes the full prompt again on each turn.

### Setup

| Item | Value |
|---|---|
| Server | Qwen worker, `second-model-fit` on `127.0.0.1:8889`, vLLM 0.33.0, prefix caching on, MTP with 3 tokens |
| Model | Pruned Qwen3.6-35B-A3B (`qwen-k208`): 40 layers, 30 linear-attention and 10 full-attention |
| Tasks | C1, C2, C3, D1, D2: the five Python tasks of the dev set that are not traps. One seed (0). One agent. |
| Sandbox | One slot (`AGENT_EVAL_SLOTS=1`), 1 GiB for each container |
| Memory guard | `memory_guard.py --floor-gib 3 --min-start-gib 3.5` |
| Command | `agent_eval.py point --name mb-s1-prefix --mix qwen=1 --seeds 1 --tasks C1,C2,C3,D1,D2` |
| Output | `dsv4-specialize/results/agent-eval/mb-s1-prefix/`: rows, transcripts, `/metrics` before and after, memory samples |

The guard needed `--min-start-gib 3.5`, not 5 as planned. With both workers up, only 3.9 GiB was available.

### Results

| Measure | Value |
|---|---|
| Requests | 47 (the sum of the turns of the five runs, so no other client used the server) |
| Prompt tokens (`prefix_cache_queries_total`) | 162,510 |
| Cache hits (`prefix_cache_hits_total`) | 84,864 |
| Hit rate | 52.2% |
| Best possible hits for the block size | 84,864 (52.2%) |
| Wall time | 153 s |
| Lowest available memory | 3.82 GiB |
| Task results | 4 pass (C2, C3, D1, D2), 1 hallucination (C1) |

For each task, the hits and the prompt tokens were:

| Task | Turns | Prompt tokens | Hits | Hit rate |
|---|---|---|---|---|
| C1 | 8 | 20,446 | 8,704 | 43% |
| C2 | 7 | 18,665 | 6,528 | 35% |
| C3 | 18 | 93,547 | 63,104 | 67% |
| D1 | 6 | 12,362 | 2,176 | 18% |
| D2 | 8 | 17,490 | 4,352 | 25% |

### Cause of the low hit rate

The hit rate is below the first target of 80%. The cause is the block size, not the chat template.

- The model is a hybrid of linear attention and full attention. With prefix caching on, vLLM 0.33 sets the mamba
  cache mode to `align` and the attention block size to 2,176 tokens. The server log and `vllm:cache_config_info`
  in `/metrics` show both values.
- A cache hit needs a full block. A prompt of fewer than 2,176 tokens gets no hit. The first prompt of each task has
  only 1,215 to 1,368 tokens, so the system prompt and the tool schemas are never a hit.
- The best possible hits for each turn are the earlier prompt of the same task, rounded down to a full block. The
  sum of these values over all turns is 84,864 tokens. The server reported exactly 84,864 hits.

Thus the cache keeps every full block of the earlier turns. The template keeps the prefix stable from turn to turn.

### Consequences

- Change the pass criterion of S1: the measured hits must equal the best possible hits for the block size of the
  server. A fixed target of 80% fails each hybrid model with short prompts, also when its cache works correctly.
- The hit rate grows with the prompt length. C3 has 18 turns and reaches 67%.
- A hybrid model computes up to one block (2,176 tokens) again on each turn. The server log shows a prompt
  throughput of about 1,700 tok/s, so this adds up to about 1.3 s per turn or less. The pass rate does not change,
  but the minutes per task are higher than for a model with small blocks, for example gpt-oss-20b. Compare the
  minutes per task of a hybrid model and of a non-hybrid model with care.
- `vllm:cache_config_info` gives the block size, so the sandbox step can compute the best possible hits from the
  transcripts and record both values.
- `enable_mamba_shared_prefix_checkpoint` is `False` on this server. A later check can test whether it gives more
  hits for hybrid models.
