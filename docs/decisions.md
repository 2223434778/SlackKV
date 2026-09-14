# Decisions

Record material technical and experimental decisions, including context and rationale.

## 2026-09-14 — Use “peer GPU memory” in experimental documentation

- **Decision:** Phase 0 documentation uses the hardware-neutral term **peer GPU memory** (or peer VRAM), rather than calling the current platform's physical memory HBM.
- **Reason:** The measured platform is two RTX A6000 GPUs with 48 GiB GDDR6 ECC memory. Future datacenter platforms may use HBM.
- **Alternative considered:** Describing all GPU memory as HBM.
- **Consequence:** Experimental terminology remains accurate for the present A6000 platform and portable to future hardware without changing the research direction.

## 2026-09-14 — Time transfers with CUDA events

- **Decision:** Native Phase 0 transfer timing uses CUDA events around asynchronous copies; synchronized CPU wall-clock timing is used only for the concurrent makespan.
- **Reason:** CUDA events measure asynchronous GPU copy timing without timing host scheduling overhead as transfer latency.
- **Consequence:** Reported transfer latency and bandwidth are device-timed and reproducible from the committed raw CSV.
