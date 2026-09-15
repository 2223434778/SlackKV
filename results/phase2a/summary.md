# Phase 2A — real CUDA tensor offload/prefetch runtime

## Objective

Validate the single-GPU tensor path GPU KV -> pinned CPU -> reusable GPU allocation -> JIT H2D prefetch, without a language model or serving framework.

## Setup

GPU0 (NVIDIA RTX A6000, 48 GiB), PyTorch 2.2.0 with CUDA 12.1, and float16 contiguous tensors. Tensor sizes are 256 MiB, 1 GiB, and 4 GiB. Each size uses three warm-ups and ten CUDA-event-timed measured runs on a CUDA stream; CPU buffers are pinned. No primary-run call uses torch.cuda.empty_cache(). JIT uses the measured 1-GiB H2D median, pauses 1/2/5 s, margins 0/250/500 ms, and five repetitions per cell.

## Results

| size | runtime D2H median / P95 ms | runtime H2D median / P95 ms | Phase 0 D2H / H2D P50 ms | D2H / H2D relative error |
|---|---:|---:|---:|---:|
| 256 MiB | 12.51 / 12.59 | 12.46 / 12.48 | 9.91 / 10.33 | 26.3% / 20.7% |
| 1 GiB | 43.74 / 43.95 | 44.40 / 44.40 | 41.49 / 40.86 | 5.4% / 8.7% |
| 4 GiB | 169.84 / 170.21 | 170.52 / 170.53 | 162.65 / 164.43 | 4.4% / 3.7% |

D2H/H2D match Phase 0 approximately at 1 and 4 GiB; 256 MiB is 21–26% slower under this PyTorch runtime, consistent with additional runtime overhead at the small tested size.

For 1 GiB and a 2-s pause, 0-ms margin yields median resume stall 2.59 ms and zero early residency. A 250-ms margin yields zero median stall with 247.79 ms median early residency; a 500-ms margin yields zero median stall with 497.65 ms early residency.

## Required answers

- **Did D2H/H2D match Phase 0 approximately?** Yes for 1 and 4 GiB (3.7–8.7% relative error); 256 MiB is materially slower by 20.7–26.3%.
- **Did GPU allocated memory become reusable?** PASS. After every measured D2H, memory_allocated decreased after deleting the GPU source tensor, and a same-size replacement allocation succeeded. memory_reserved was recorded but not required to decrease.
- **Did GPU->CPU->GPU preserve data?** PASS. Every measured run passed the fixed-seed 1024-position float16 sample check.
- **Did 250/500 ms margins reduce real resume stall?** Yes. For all pauses, both margins reduced the measured median stall to 0 ms; they trade this for approximately their margin in early GPU residency.

## Anomalies and limitations

An initial pre-measurement implementation used O(numel) random permutation generation to select sample positions. It was terminated before producing measurement files and corrected to fixed-seed direct index sampling; the reported run is the complete rerun. This experiment uses one tensor, one GPU, no queue contention, no prediction error, and no real model or vLLM integration.

## Conclusion

**Result A: Runtime assumptions validated.** The real tensor offload/prefetch path works, 1/4-GiB transfer timings are approximately consistent with the Phase 0 model, allocator capacity is reusable, content survives round trip, and a safety margin produces the expected stall-versus-early-residency effect.

Phase 2A execution status: COMPLETE.
