This is a fantastic long-term project, and it's actually something that's surprisingly missing from the open-source community. Most benchmark suites focus only on accuracy (MMLU, GPQA, HLE, etc.) or only on serving performance (vLLM benchmarks). Very few combine **real-world usability**, **system performance**, and **reasoning quality** into one reproducible benchmark.

Given that you already have a **DGX Spark**, local **vLLM**, and experience with NanoGPT and orchestration, I'd build this as a modular framework rather than a collection of scripts.

---

# High-level Architecture

```
benchmark-suite/
│
├── models/
│   ├── qwen3-35b.yaml
│   ├── llama4.yaml
│   └── ...
│
├── benchmarks/
│   ├── latency/
│   ├── coding/
│   ├── reasoning/
│   ├── swe/
│   ├── hle/
│   └── system/
│
├── datasets/
│   ├── humaneval/
│   ├── mbpp/
│   ├── swebench/
│   ├── hle/
│   └── prompts/
│
├── runners/
│   ├── vllm_runner.py
│   ├── llama_runner.py
│   ├── sglang_runner.py
│   └── api_runner.py
│
├── metrics/
│   ├── latency.py
│   ├── memory.py
│   ├── power.py
│   ├── code.py
│   └── reasoning.py
│
├── reports/
│
└── benchmark.py
```

Every benchmark becomes a plugin.

Instead of modifying the benchmark every time a new model comes out, you only add

```
models/qwen3_40b.yaml
```

and run

```
python benchmark.py --model qwen3_40b
```

---

# Phase 1 — Build the Core Runner

This is the foundation.

The benchmark runner should:

1. Start a model
2. Wait until ready
3. Run every benchmark
4. Save raw outputs
5. Compute metrics
6. Generate report

Every run should produce

```
results/

qwen3-35b/

    latency.json

    code.json

    swe.json

    hle.json

    memory.json

    power.json

    summary.json
```

Never overwrite previous runs.

---

# Phase 2 — Standardize the Environment

This is arguably the most important step.

Otherwise benchmarks are meaningless.

Always record

GPU

Driver version

CUDA version

PyTorch version

vLLM version

CPU

RAM

Kernel

Ambient temperature (optional)

Clock speeds

Power mode

Example:

```
GPU:
DGX Spark

CUDA:
13.0

Driver:
585.xx

vLLM:
0.23

torch:
2.9

Kernel:
Ubuntu 26.04
```

Store this automatically.

---

# Phase 3 — First Token Latency

Definition:

Time between

```
POST /generate
```

and

```
first generated token
```

Measure

```
100 runs

Average

Median

95th percentile

99th percentile
```

Prompt lengths

```
32 tokens

128

512

2048

8192

16384
```

Graph

```
Prompt length

↓

Latency
```

This reveals KV-cache behavior.

---

# Phase 4 — Prompt Processing Speed

Measure

```
Input tokens

/

prefill time
```

Output

```
tokens/sec
```

Use increasingly long prompts.

```
128

512

2048

8192

16384

32768

65536
```

Graph

```
Prompt size

↓

Prefill throughput
```

---

# Phase 5 — Decode Speed

Probably the easiest benchmark.

Generate

```
512 tokens

1024 tokens

2048 tokens
```

Record

Average tok/sec

Peak tok/sec

Minimum tok/sec

Median tok/sec

---

# Phase 6 — Memory Usage

Record continuously.

Every second collect

```
nvidia-smi

GPU memory

GPU utilization

Temperature

Clocks
```

Also

CPU RAM

Swap

Store as

```
CSV
```

Then plot

```
Memory vs Time
```

---

# Phase 7 — Power Consumption

Use

```
nvidia-smi --query-gpu=power.draw
```

every second.

Collect

Average watts

Peak watts

Energy (Wh)

Energy/token

This is an underrated benchmark.

A model producing

```
60 tok/sec

at

120W
```

is much more efficient than

```
75 tok/sec

at

300W
```

---

# Phase 8 — Reasoning Token Count

This is unique.

Many reasoning models generate

```
<think>

...

</think>
```

Measure

Thinking tokens

Answer tokens

Ratio

```
thinking

/

final answer
```

Also measure

```
Average thinking length

Maximum

Median
```

This tells you whether newer models reason more efficiently.

---

# Phase 9 — Code Correctness

Start with

## HumanEval

Then

MBPP

Then

LiveCodeBench

Eventually

SWE-Lancer

Run

```
temperature=0

temperature=0.2

temperature=0.8
```

Measure

Pass@1

Pass@5

Compilation failures

Runtime failures

Average generation time

---

# Phase 10 — Deep-SWE

This is much harder.

Need

Docker

Git

Sandbox

For each issue

Clone repository

Give model issue

Allow edits

Run tests

Compute

Resolved

Not resolved

Runtime

Tokens

Cost

Deep-SWE is one of the closest benchmarks to real software engineering.

---

# Phase 11 — Humanity's Last Exam

Probably the hardest reasoning benchmark.

Need

Official questions

Evaluation harness

Run

```
temperature=0
```

Collect

Accuracy

Reasoning length

Latency

Token count

---

# Phase 12 — Build Beautiful Reports

Automatically generate

```
HTML

Markdown

JSON
```

Include plots.

Example

```
Decode Speed

██████████

Latency

██████

Power

███████████

Memory

██████

Reasoning

███████

Code

█████████
```

---

# Phase 13 — Leaderboard

Store

```
SQLite
```

Every run becomes

```
Model

Date

Version

Scores
```

Now you can compare

| Model | First Token | Decode | HumanEval | SWE | HLE |
| ----- | ----------- | ------ | --------- | --- | --- |
| Qwen  |             |        |           |     |     |
| Llama |             |        |           |     |     |
| Gemma |             |        |           |     |     |

---

# Phase 14 — Visualization

Use Plotly.

Generate

Latency curves

Memory graphs

Power graphs

Radar charts

Scatter plots

Scaling curves

Trend lines

---

# Phase 15 — Continuous Benchmarking

This is where it becomes really useful.

Whenever a model releases:

```
Download

Serve

Run suite

Generate report

Update leaderboard
```

One command

```
python benchmark.py \
    --model qwen3-40b
```

Everything happens automatically.

---

# Suggested Development Roadmap

I would build the suite in this order:

| Phase | Goal                                           | Difficulty  | Estimated Time |
| ----- | ---------------------------------------------- | ----------- | -------------- |
| 1     | Core benchmark runner and configuration system | Medium      | 2–3 days       |
| 2     | Latency (first token, prefill, decode)         | Easy        | 1–2 days       |
| 3     | Memory and power monitoring                    | Easy        | 1 day          |
| 4     | Reporting (JSON, HTML, charts)                 | Medium      | 2–3 days       |
| 5     | HumanEval and MBPP integration                 | Medium      | 2–4 days       |
| 6     | LiveCodeBench integration                      | Medium–Hard | 2–3 days       |
| 7     | Deep-SWE / SWE-bench harness                   | Hard        | 1–2 weeks      |
| 8     | Humanity's Last Exam integration               | Medium      | 2–3 days       |
| 9     | SQLite database and leaderboard                | Easy        | 1–2 days       |
| 10    | CI automation for running new models           | Medium      | 2–3 days       |

## Future enhancements

Once the core suite is stable, there are several metrics that would make it stand out from existing benchmarks:

* **Time-to-first-correct-answer:** Combine latency and accuracy by measuring how long it takes a model to produce its first correct solution.
* **Energy efficiency:** Report joules per generated token and joules per solved benchmark task, not just average watts.
* **Long-context performance:** Measure throughput, latency, and accuracy as prompt lengths grow from a few thousand tokens to the model's maximum context window.
* **Concurrency and serving scalability:** Benchmark throughput and latency with multiple simultaneous requests to evaluate serving efficiency.
* **Determinism and stability:** Run the same prompt repeatedly (especially at temperature 0) and quantify output consistency.
* **Cost modeling:** Even for local inference, estimate GPU-hours, energy cost, and tokens per dollar-equivalent to compare models fairly.
* **Structured output reliability:** Test JSON/function-calling success rates and schema adherence for agentic workflows.
* **Vision and multimodal benchmarks:** If you evaluate multimodal models later, add OCR, chart understanding, image reasoning, and visual question answering.
* **Regression tracking:** Automatically compare each run against previous versions and highlight statistically significant improvements or regressions.

This roadmap results in a benchmark suite that measures three complementary dimensions:

* **Performance:** latency, throughput, memory usage, power, scalability.
* **Capability:** coding, software engineering, reasoning, knowledge, long-context tasks.
* **Efficiency:** reasoning token count, energy per token, and energy per solved task.

Together, those metrics provide a much more complete picture of how practical an open-source model is than any single existing benchmark.

The model is repeating itself — every response starts with nearly the same sentence ("It looks like your message got cut off..."). This is a classic repetition loop that happens when the model has exhausted its meaningful vocabulary and starts cycling through common phrases.

The cutoff is consistent (~500–565 tokens) — the model reliably hits this wall regardless of whether you ask for 1024 or 2048 tokens. That's the EOS probability becoming overwhelming after the repetition starts.

At 512 tokens it completes successfully — which is why the throughput at 512 is so high (1,353 tok/s). The model completes all 512 clean tokens before the repetition loop kicks in. The speculative decoding (MTP, 3 tokens) is working perfectly for those first 512 tokens.

The real problem isn't EOS — it's vocabulary exhaustion. The model runs out of meaningful content at ~500 tokens, enters a repetition loop, and then EOS probability skyrockets and cuts it off. This is a model quality issue, not a benchmarking bug.

The benchmark prompt needs to be a meaningful, non-repetitive prompt that doesn't invite the model to comment on the structure:

The output_text_preview includes the reasoning preamble. For a cleaner comparison, you'd want to separate the reasoning tokens from the answer tokens in the report. The current runner captures them together. The analyze_reasoning_tokens() function at line 442 looks for <think>... tags, but this model's output is plain text (Here's a thinking thinking sequence), so it won't be detected.

Want me to update the runner to handle this Qwen3-style reasoning output (plain text before the answer) so the report can separately show reasoning vs. answer quality?