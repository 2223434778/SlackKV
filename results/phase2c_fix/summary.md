# Phase 2C-Fix — selective offload on a CUDA-compatible vLLM runtime

## Runtime and capability gate

- vLLM `0.23.0` (source build, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`), isolated at `/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix`.
- PyTorch `2.11.0+cu126`, CUDA runtime `12.6`, Python `3.11.16`, NVIDIA driver `550.54.14`, GPU0 RTX A6000 (49,140 MiB reported by `nvidia-smi`). FlashInfer `0.6.12` and its cubin package were installed without dependency resolution that could replace Torch; a CUDA-12.6 `lib64 -> lib` link was confined to this environment for JIT linking.
- Capability gate: vLLM import, CUDA availability, `OffloadingConnector`, and request-level `kv_transfer_params` all passed. `SamplingParams(extra_args={"kv_transfer_params": {"max_offload_tokens": 0}})` was accepted by the installed runtime.

## Experimental control

Qwen/Qwen2.5-1.5B-Instruct was served on GPU0 with prefix caching, 320 MiB GPU KV capacity, and 2 GiB CPU offload capacity. A and B used distinct exact 4096-token prefixes. The pressure workload used two distinct 9500-token prompts because one pressure request did not reliably evict B's local prefix; this is the only pressure increase and does not change admission or policy. A used normal offload; B sent `kv_transfer_params.max_offload_tokens=0`. Repetitions were restarted independently in A,B; B,A; A,B order.

## Evidence

The first-turn `vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}` delta was 117,440,512 bytes for A and 0 bytes for B in all three repetitions. After pressure, A resume produced 4,096 `external_kv_transfer` prompt tokens and 4,096 external-prefix-cache hit tokens in every repetition. B produced 0 external tokens and a median 4,106 locally recomputed prompt tokens (one-token generation accounts for the small difference). Local-prefix hit tokens at both resumes were 0 after the two-pressure workload. All six resume requests returned a streamed token and were marked generation-valid.

| admission | median resume TTFT (ms) | median external hit tokens | median recomputed prompt tokens |
|---|---:|---:|---:|
| A / OFFLOAD | 94.82 | 4096 | 10 |
| B / KEEP (`max_offload_tokens=0`) | 148.51 | 0 | 4106 |

The median A-versus-B TTFT reduction is **36.15%** (1.57× speedup). The no-pressure A control had 4,096 local-prefix-hit tokens and 93.12 ms TTFT, confirming the prefix construction and local cache path.

## Result

**Result A: Selective offload validated in real vLLM serving.** The evidence is specifically external recovery for A after local-cache pressure, no external storage for B, substantial B recomputation, and a materially lower median A resume TTFT. This phase still does **not** test deadline-aware JIT promotion.

## Limitations and anomalies

- This is a two-session, one-GPU serving validation, not QueueJIT or deadline-aware scheduling.
- vLLM exposes source counters and external-prefix counters; exact allocator-resident bytes are not exposed, so local eviction is validated by zero local-prefix-hit tokens plus successful external recovery/recomputation.
- The isolated environment emits a harmless `nvidia_cutlass_dsl_packages.pth` warning because FlashInfer was installed with dependencies constrained to preserve the CUDA-12.6 Torch runtime.

Phase 2C-Fix execution status: COMPLETE
