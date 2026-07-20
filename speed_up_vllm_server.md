## How to Speed Up VLLM Server Loading

Current command:

```bash

vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name unsloth-qwen3.6-35b-a3b-nvfp4-fast \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 262144 \
  --moe-backend flashinfer_b12x \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3  
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --async-scheduling \

```

1. Add -00 level and lower max-model-len. 

