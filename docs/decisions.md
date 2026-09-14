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

## 2026-09-14 — Phase 1A isolates queue-aware resume-slack scheduling

- **Decision:** After Phase 0 found no substantial peer-versus-CPU transfer advantage on the current `SYS` topology, Phase 1A uses CPU→GPU0 as its primary calibrated promotion path and treats peer GPU memory as an optional topology-dependent ablation.
- **Reason:** The Phase 0 P50 latency advantage for GPU1→GPU0 over pinned CPU→GPU0 was small, so assuming a universally superior peer tier would confound the scheduling question.
- **Alternative considered:** Making peer GPU memory the default faster tier in the simulator.
- **Consequence:** Phase 1A isolates resume-slack queue scheduling without changing the overall SlackKV research direction; peer behavior remains explicitly measurable through the optional source-tier switch.

## 2026-09-14 — Phase 1B uses strict TraceLab round aggregation

- **Decision:** Phase 1B maps one promotion opportunity to each valid TraceLab tool round using its latest tool emission and latest tool result; calls with incomplete timing are excluded.  `write_stdin` continuation calls remain their own rounds and are not latency-summed with a prior `exec_command`.
- **Reason:** This preserves the public dataset's observed round and continuation structure while avoiding invented timing for missing lifecycle information or double-counting overlapping wall time.
- **Consequence:** The simulator preserves joint pause, context, tool, and session characteristics but remains an oracle-deadline abstraction.  Transfer time uses the Phase 0 CPU→GPU0 P50 piecewise model with chunking above the largest measurement rather than a theoretical PCIe bandwidth.

Reference: Zhu et al., *TraceLab: Characterizing Coding Agent Workloads for LLM Serving*, 2026, arXiv:2606.30560.

## 2026-09-14 — Phase 1B-Fix uses closed-loop, target-capped replay

- **Decision:** Session replay propagates each policy's observed promotion stall to later releases in the same session.  The primary mapping is 56 KiB/token with a 131,072-token reusable-context cap; 128 KiB/token on the identical filtered events is a stress sensitivity.
- **Reason:** Fixed original timestamps create non-causal queue pressure after a simulated stall.  Uncapped contexts can exceed the target model's supported context or the practical single-GPU KV size.
- **Alternative considered:** Retaining open-loop timestamps or filtering a different event set per footprint.
- **Consequence:** Policy comparisons preserve trace ordering and causal feedback while distinguishing replay artifacts from KV-footprint stress.  `result_at` remains a conservative deadline because the public trace lacks a direct next-GPU-inference-start timestamp.
