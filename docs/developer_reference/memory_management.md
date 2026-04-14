# Memory Management

SGLang-Omni inherits SGLang's GPU memory management for AR engine stages (e.g., thinker, talker). This guide explains how `mem_fraction_static` controls the memory split between model weights, KV cache, and activations — and how to tune it for different OOM scenarios.

## Memory Layout

When an AR engine stage starts, GPU memory is divided into three regions:

```
┌─────────────────────────────────────────────────────────────┐
│                    Total GPU Memory                         │
├────────────────────────────────┬────────────────────────────┤
│   Model Weights + KV Cache     │   Activations + CUDA Graph │
│                                │   Buffers                  │
│   mem_fraction_static × total  │  (1 - mem_fraction_static) │
│                                │        × total             │
└────────────────────────────────┴────────────────────────────┘
```

`mem_fraction_static` defines the fraction of **total GPU memory** allocated to model weights and KV cache. The remainder is reserved for activations and CUDA graph buffers.

## KV Cache Allocation Formula

After model weights are loaded, the engine computes the available KV cache budget:

```python
# total_gpu_memory:     GPU capacity (e.g., 80 GB for H100)
# available_gpu_memory: free memory after loading model weights
# mem_fraction_static:  fraction of total GPU memory for weights + KV cache

reserved = total_gpu_memory * (1 - mem_fraction_static)
rest_memory = available_gpu_memory - reserved
max_num_tokens = rest_memory / cell_size_per_token
```

The `reserved` amount is subtracted from the *current* free memory. If `rest_memory <= 0`, initialization fails with:

```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

## Two Types of OOM

### Init OOM: KV cache allocation fails at startup

**Symptom**: Server crashes during startup with the error above.

**Cause**: The model is large relative to GPU capacity (e.g., Qwen3-Omni 60 GB on a single 80 GB H100). After loading weights, very little free memory remains. With the default `mem_fraction_static=0.7`, the reserved region is `80 × 0.3 = 24 GB`, which exceeds the ~20 GB of free memory, making `rest_memory` negative.

**Fix**: Increase `mem_fraction_static` to shrink the reserved region.

```bash
# Speech mode (via examples script)
CUDA_VISIBLE_DEVICES=4,5,6,7 python examples/run_qwen3_omni_speech_server.py \
  --port 8008 \
  --mem-fraction-static 0.85

# Text-only mode (via CLI)
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only --port 8008 \
  stages.4.executor.args.server_args_overrides.mem_fraction_static=0.85
```

| `mem_fraction_static` | reserved (H100 80GB) | free after 60GB model | rest for KV cache | result |
|---|---|---|---|---|
| 0.70 (default) | 24.0 GB | 20 GB | -4.0 GB | Init OOM |
| 0.80 | 16.0 GB | 20 GB | 4.0 GB | OK (small KV cache) |
| 0.85 | 12.0 GB | 20 GB | 8.0 GB | OK |

### Runtime OOM: out-of-memory during inference

**Symptom**: CUDA OOM errors during request processing, not at startup.

**Cause**: The KV cache pool is too large, leaving insufficient memory for activations and CUDA graph captures during inference. This typically happens with smaller models where the KV cache consumes most of the GPU.

**Fix**: Decrease `mem_fraction_static` to give more room to activations.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only --port 8008 \
  stages.4.executor.args.server_args_overrides.mem_fraction_static=0.5
```

## Summary

| scenario | model size vs GPU | `mem_fraction_static` | direction |
|---|---|---|---|
| Init OOM (KV cache allocation fails) | large (>70% of GPU) | increase (e.g., 0.85) | `reserved` shrinks → `rest_memory` grows |
| Runtime OOM (activations overflow) | small (<50% of GPU) | decrease (e.g., 0.5) | KV cache shrinks → activation headroom grows |

## Reference

- KV cache allocation: `sglang/srt/model_executor/model_runner_kv_cache_mixin.py` lines 111-144
- `mem_fraction_static` auto-calculation: `sglang/srt/server_args.py` lines 880-1029
- Default value in SGLang-Omni: `sglang_omni/engines/ar/sglang_backend/server_args_builder.py` (0.7)
