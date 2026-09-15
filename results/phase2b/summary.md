# Phase 2B — real Transformer KV offload and restore

## Objective

Validate the actual Transformer cache path: real Qwen prefill -> GPU past_key_values -> pinned CPU snapshot -> GPU KV release and allocator reuse -> H2D restore -> greedy continuation.

## Setup

The selected model is Qwen/Qwen2.5-1.5B-Instruct (no fallback), Transformers 4.45.2, PyTorch 2.2.0/CUDA 12.1, GPU0 RTX A6000, float16, eval mode, and inference mode. A deterministic ordinary-text prompt was tokenized to exactly 4096 tokens at batch size 1. The model returns a legacy cache tuple under this Transformers/model combination; the experiment uses the public DynamicCache.from_legacy_cache() reconstruction API for continuation. One warm-up and three full measured runs were executed.

## Results

The model has 28 layers. Each layer has key/value shapes [1, 2, 4096, 128], producing a 112.00 MiB KV cache.

| quantity | median |
|---|---:|
| D2H cache snapshot | 7.61 ms (15.42 GB/s) |
| H2D cache restore | 6.29 ms (18.67 GB/s) |
| GPU allocated memory released | 112.00 MiB |
| baseline first-token decode | 22.84 ms |
| restored first-token decode | 22.79 ms |
| cold resume latency (H2D + restored first token) | 29.08 ms |
| warm resume latency (restored first token) | 22.79 ms |

## Required answers

- **Which model was used?** Qwen/Qwen2.5-1.5B-Instruct.
- **How large was the 4096-token KV cache?** 112.00 MiB across 28 layers.
- **How long did D2H/H2D take?** 7.61/6.29 ms median.
- **How much allocated GPU memory was released?** 112.00 MiB, matching the KV cache size.
- **Was GPU capacity reusable?** PASS. A float16 replacement allocation of 89.6 MiB (80% of KV bytes) succeeded after release.
- **Did all 16 continuation tokens match?** PASS for every measured run; max and mean first-step logit differences are both 0.0.
- **What was baseline first-token latency?** 22.84 ms.
- **What was cold resume latency if restored at resume?** 29.08 ms; the measured cache restore contributes 6.29 ms.
- **Was transfer timing consistent with Phase 2A?** Materially different for D2H. The 112-MiB cache D2H is 39.1% slower than a Phase 2A 256-MiB contiguous-tensor scaled estimate (and 75.6% slower than the Phase 0 piecewise estimate); H2D is 15.4% slower than the Phase 2A scaled estimate. The cache is copied as 56 separate K/V tensors, so this is expected per-tensor launch/dispatch overhead rather than a new transfer calibration.

## Anomalies and limitations

Transformers emitted a non-fatal warning about future logits dtype behavior and a tokenizers fork-parallelism warning. An initial compatibility attempt assumed a DynamicCache return object; the installed model returned a public legacy tuple instead, and the final run handles that supported representation. This remains a single model, single context length, single GPU, no queueing, and no vLLM experiment.

## Conclusion

**Result B: Real-model mechanism works, but runtime calibration needs adjustment.** The cache round trip is correct and allocator capacity is reusable, but fragmented real-KV D2H is materially slower than contiguous-tensor estimates.

Phase 2B execution status: COMPLETE.
