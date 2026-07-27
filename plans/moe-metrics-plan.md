# MoE-Specific Metrics — Measurement Plan

**Phase:** Roadmap item #19 (Phase 7 — Systems Analysis)
**Model:** `Qwen3.6-35B-A3B-NVFP4` (35B total / 3.5B active per token, top-2 expert routing)

---

## What We Want to Measure

| Metric | Question it answers |
|---|---|
| **Active experts per token** | How many experts fire per input token? |
| **Expert imbalance** | Are experts evenly loaded, or is work concentrated? |
| **Routing overhead** | How much time is spent in the gating network vs computation? |
| **Expert occupancy** | How much GPU capacity is wasted by expert parallelism/competition? |

---

## 1. Active Experts per Token

### What it is

For a top-k gating with `k=2`, each input token routes to exactly 2 experts. This is a **static property** of the model architecture.

### How to measure

| Approach | Feasibility | What you get |
|---|---|---|
| **Static config** | Easy | `num_experts_per_token` from `config.json` |
| **vLLM `/metrics` endpoint** | Medium | `vllm:num_tokens_to_schedule` + `vllm:num_tokens_total` gives average experts per token over time |
| **Per-request patched vLLM** | Hard | Exact expert count per request via `request_metrics` |

### Data source

```
# From vLLM's /metrics (Prometheus-style)
vllm:cache_config.num_experts    → 64
vllm:cache_config.num_experts_per_token   → 2
vllm:num_tokens_total            → total tokens processed
vllm:num_tokens_to_schedule      → tokens queued

# Computed: num_tokens_to_schedule / num_tokens_total → average experts/token
```

### Implementation

```python
def get_expert_config(model_config: dict) -> dict:
    """Read MoE config from model architecture."""
    return {
        "num_experts": model_config.get("num_experts", 0),
        "num_experts_per_token": model_config.get("num_experts_per_token", 0),
        "num_local_experts": model_config.get("num_local_experts", 0),
        "moe_router_topk": model_config.get("moe_router_topk", 0),
    }
```

---

## 2. Expert Imbalance

### What it is

If all experts were used equally, each expert receives `1/N` of requests. With MoE routing, the distribution is probabilistic — some experts get swarmed, others sleep. Severe imbalance causes **straggler effects**: the last expert batch finishes first, limiting throughput.

### How to measure

#### Level 1: Batch-level proxy (no vLLM patch)

Run a batch of N requests, measure how many unique experts fire.

```python
def measure_unique_experts_per_batch(
    client: ModelClient, num_requests: int, num_experts: int
) -> dict:
    """Run N cold requests and measure expert diversity."""
    unique_per_batch = set()
    for _ in range(num_requests):
        gen = client.generate(prompt, max_tokens=1)
        # Requires vLLM patch to log expert IDs
        # Otherwise: just count if different experts fired
        ...
    return {"unique_experts": len(unique_per_batch), "total_experts": num_experts}
```

**Limitation:** Without a vLLM patch that logs expert IDs per request, this is only an estimate. The prompt diversity itself drives which experts are activated.

#### Level 2: vLLM `/metrics` histogram (no patch, partial)

If vLLM exposes expert-level metrics (it may not by default), we can scrape `vllm:expert_*` metrics.

```python
def scrape_expert_metrics(client: ModelClient, gpu_monitor: GpuMonitor) -> dict:
    """Scrape MoE metrics from vLLM's /metrics endpoint."""
    import requests as req

    resp = req.get(f"{client.base_url}/metrics", timeout=5)
    metrics_text = resp.text

    # Parse Prometheus-style metrics
    metrics = {}
    for line in metrics_text.split("\n"):
        if line.startswith("# TYPE") or not line:
            continue
        if line.startswith("vllm:expert"):
            parts = line.split()
            key = parts[0]
            value = float(parts[1]) if len(parts) > 1 else 0
            metrics[key] = value

    return metrics
```

#### Level 3: Patched vLLM (full measurement)

Patch vLLM's `MoEForCausalLM` to log expert assignments:

```python
# In MoEForCausalLM.forward()
expert_ids = torch.nonzero(expert_mask, as_tuple=False)  # (num_experts_activated, d)
self.logger.log({
    "expert_ids": expert_ids.tolist(),
    "expert_count": expert_ids.shape[0],
    "expert_load": expert_mask.float().mean(dim=-1).tolist(),  # per-expert utilization
})
```

Then compute:
- **Gini coefficient** of expert load distribution
- **Expert utilization histogram** over the benchmark window

### Implementation

```python
def compute_expert_imbalance(gpu_monitor: GpuMonitor) -> dict:
    """Compute expert imbalance from scraped vLLM metrics.
    
    Returns:
        - expert_load_imbalance: Gini coefficient of expert utilization
        - expert_balance_ratio: min/max expert utilization
        - active_experts_percent: % of experts that fired at least once
    """
    ...
```

---

## 3. Routing Overhead

### What it is

The gating network (router) is a small linear layer that decides which experts to activate:
```
Gating: d → num_experts → TopK (k) → softmax → d
```

For a model with `hidden_size=2048` and `num_experts=64`:
- Gating FLOPs per token: `2 * 2048 * 64 * 2` = ~524K FLOPs (forward + backward not needed at decode time)
- At H100 peak FLOPS (~1.979 TFLOPS): `524K / 1.979e12` = **0.0003ms per token** — theoretically negligible

The real question: is the overhead **higher than theoretical** due to:
- Kernel launch overhead
- Memory access patterns (small per-expert weights loaded from HBM)
- Load imbalance (some experts run longer, holding the SMs)

### How to measure

#### Method A: Theoretical estimate (no runtime measurement)

```python
def estimate_routing_overhead(config: dict) -> dict:
    """Compute theoretical routing overhead from model config."""
    d = config["hidden_size"]
    n_experts = config.get("num_experts", 1)
    k = config.get("num_experts_per_token", 1)
    
    # Gating: d → n_experts (linear), top-k selection, d → 1 (linear to score)
    routing_flops = 2 * d * n_experts * k  # rough upper bound
    
    return {
        "routing_flops_per_token": routing_flops,
        "theoretical_overhead_ms": routing_flops / (peak_tflops * 1e12) * 1000,
        "as_pct_of_flop_budget": routing_flops / total_flops_per_token * 100,
    }
```

#### Method B: MoE vs dense comparison (empirical)

Run the same workload with:
1. **MoE model** (35B total / 3.5B active)
2. **Equivalent dense model** (3.5B total)

The difference in TTFT and decode TPS is the routing + expert overhead.

```python
def compare_moe_vs_dense(client_moe: ModelClient, client_dense: ModelClient) -> dict:
    """Compare MoE vs dense throughput."""
    moe_results = run_decode_speed(client_moe, output_lengths=[512, 1024, 2048])
    dense_results = run_decode_speed(client_dense, output_lengths=[512, 1024, 2048])
    
    overhead = {}
    for key in moe_results:
        moe_tps = moe_results[key]["tok_per_sec_avg"]
        dense_tps = dense_results[key]["tok_per_sec_avg"]
        if dense_tps and moe_tps:
            overhead[key] = {
                "moe_tps": moe_tps,
                "dense_tps": dense_tps,
                "overhead_ratio": 1 - moe_tps / dense_tps,  # how much slower is MoE
            }
    return overhead
```

#### Method C: Per-request timing (vLLM patched)

Add a CUDA event around the routing call in `MoEForCausalLM.forward()`:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... gating network ...
end.record()
# latency = start.elapsed_time(end) in ms
```

### Implementation

```python
def compute_routing_overhead(
    moe_config: dict, gpu_specs: dict, model_name: str
) -> dict:
    """Compute both theoretical and empirical routing overhead."""
    # Theoretical
    theoretical = estimate_routing_overhead(moe_config)
    
    # Empirical (from previous benchmark runs)
    empirical = None  # Set from moe_vs_dense comparison if available
    
    return {
        "theoretical_overhead_ms": theoretical,
        "empirical_overhead_ms": empirical,
        "is_bottleneck": theoretical["theoretical_overhead_ms"] > SOME_THRESHOLD,
    }
```

---

## 4. Expert Occupancy

### What it is

In MoE models, experts run in parallel with each other (and sometimes with the base model). This creates two problems:

1. **Resource contention:** Multiple experts compete for SMs, registers, and L2 cache.
2. **Load imbalance:** If Expert A has 3× more tokens than Expert B, Expert B finishes early and sits idle while A is still running.

The question: **what fraction of the GPU's theoretical compute is actually used for expert computation** vs wasted on contention, padding, or idle time?

### How to measure

#### Method A: MoE utilization vs dense utilization

```python
def measure_expert_utilization(client: ModelClient, gpu_monitor: GpuMonitor) -> dict:
    """Compare GPU utilization between MoE and dense at same active params."""
    # Run MoE decode
    ...
    moe_gpu_util = gpu_monitor.stop_window("moe_decode")
    
    # Run dense decode (same active param count)
    ...
    dense_gpu_util = gpu_monitor.stop_window("dense_decode")
    
    return {
        "moe_gpu_util": moe_gpu_util,
        "dense_gpu_util": dense_gpu_util,
        "expert_contention_factor": moe_gpu_util / dense_gpu_util,
    }
```

#### Method B: Per-expert batch size from vLLM

If we can measure how many tokens each expert processes in a batch, we can compute:
- **Expert bandwidth utilization** = (total expert tokens) / (theoretical max expert throughput)
- **Padding overhead** = (batch_size - unique_tokens) / batch_size

#### Method C: Nsight Systems (external tool)

Use NVIDIA Nsight Systems (`nsys profile --trace cuda,osrt,ptx`) to see:
- Which kernels overlap (compute vs expert kernels)
- SM occupancy per expert kernel
- Idle time between expert executions

```bash
nsys profile --trace cuda,osrt -o nsight_output python core_runner.py --model config.yml --skip-roofline
```

Then analyze with `nsys-ui` or `nsight-sys-cli`.

### Implementation

```python
def compute_expert_occupancy(
    gpu_monitor: GpuMonitor, model_config: dict, theoretical_tps: float,
    measured_tps: float
) -> dict:
    """Compute expert occupancy from measured vs theoretical throughput."""
    # Effective utilization = measured TPS / theoretical TPS
    # Adjusted for active params (not total)
    active_params = estimate_active_params(model_config)
    total_params = model_config.get("total_params", 0)
    param_ratio = active_params / total_params if total_params > 0 else 1
    
    # Expert utilization = (measured / theoretical) adjusted for active param ratio
    expert_utilization = (measured_tps / theoretical_tps) / param_ratio
    
    return {
        "expert_utilization_pct": round(expert_utilization * 100, 1),
        "gpu_utilization_avg": gpu_monitor.gpu_utilization_avg,
        "contention_factor": round(measured_tps / theoretical_tps, 4),
    }
```

---

## 5. MoE-Specific Metrics Benchmark Function

This is the top-level function that ties everything together:

```python
def run_moe_benchmarks(
    client: ModelClient,
    model_config: dict,
    gpu_monitor: GpuMonitor | None = None,
) -> dict:
    """Run MoE-specific analysis: expert count, imbalance, routing, occupancy.
    
    Returns:
        Dict with:
        - expert_config: static MoE architecture info
        - routing_overhead: theoretical + empirical
        - expert_imbalance: Gini coefficient, utilization distribution
        - expert_occupancy: GPU utilization during expert execution
    """
    ...
```

---

## 6. Output Schema (`moe_metrics.json`)

```json
{
  "config": {
    "benchmark_version": "1.0",
    "definition": "MoE-specific metrics: expert load, routing overhead, occupancy",
    "model_name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "start_time": "..."
  },
  "expert_config": {
    "total_params": 35_000_000_000,
    "active_params": 3_500_000_000,
    "num_experts": 64,
    "num_experts_per_token": 2,
    "moe_router_topk": 2,
    "expert_load_ratio": 0.0314  // 3.5B / 35B = 3.14% of params active
  },
  "routing_overhead": {
    "theoretical_flops_per_token": 524_288,
    "theoretical_overhead_ms": 0.00026,
    "as_pct_of_total_flops": 0.0005,
    "empirical_overhead_ms": null,  // set from MoE vs dense comparison
    "is_bottleneck": false
  },
  "expert_imbalance": {
    "gini_coefficient": 0.32,  // 0 = perfectly balanced, 1 = worst imbalance
    "active_experts_percent": 58.0,  // % of experts that fired at least once
    "experts_with_zero_load": 27,
    "max_expert_load_ratio": 1.8,  // max load / average load
    "note": "Requires vLLM patch for accurate measurement"
  },
  "expert_occupancy": {
    "gpu_utilization_avg_pct": 82.3,
    "gpu_utilization_peak_pct": 95.1,
    "effective_expert_compute_pct": 68.5,  // fraction of GPU util used for experts
    "contention_factor": 0.83,  // measured / theoretical TPS
    "note": "Requires MoE vs dense comparison for precise measurement"
  },
  "end_time": "..."
}
```

---

## 7. Implementation Priority

| Metric | Effort | vLLM patch needed? | Can run now? |
|---|---|---|---|
| **Expert config** | Low | No | **Yes** — static from model YAML |
| **Routing overhead (theoretical)** | Low | No | **Yes** — math from model config |
| **Routing overhead (empirical)** | Medium | No | MoE vs dense comparison benchmark |
| **Expert imbalance** | Medium-Hard | Optional | With `/metrics` scraping (limited) |
| **Expert occupancy** | Medium | No | MoE vs dense comparison |
| **Expert imbalance (full)** | Hard | Yes | Requires `MoEForCausalLM` patch |

### Recommended implementation order

1. **Expert config + routing overhead (theoretical)** — no vLLM patch, pure static analysis
2. **Expert config in roofline** — already computed in `roofline.json` (active params ratio)
3. **MoE vs dense comparison benchmark** — runs existing decode benchmarks with both
4. **vLLM `/metrics` scraping** — extracts MoE fields from Prometheus endpoint
5. **Full expert imbalance + occupancy** — requires vLLM code patch

---

## 8. vLLM Patch Required for Full Measurement

To get expert-level metrics from vLLM, the following patch would be needed in `vllm/model_executor/parallel_utils/custom_radix_tree.py` or `MoEForCausalLM`:

```python
# In MoEForCausalLM.forward()
def forward(self, hidden_states):
    ...
    # BEFORE MoE forward
    expert_ids = torch.nonzero(expert_mask, as_tuple=False)
    self._expert_load = expert_mask.float().mean(dim=-1).tolist()
    self._num_experts_activated = len(expert_ids)
    
    # AFTER MoE forward
    # Record in request_metrics
    request_metrics["num_experts_activated"] = self._num_experts_activated
    request_metrics["expert_load_histogram"] = self._expert_load
```

Then expose via the streaming API by adding these to the `request_metrics` dict that's already included in each stream chunk.