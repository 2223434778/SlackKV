# M1 bounded cost diagnosis

See `docs/SLACKKV_SYSTEM_M1_COST_20261013.md` for full setup, evidence and limitations.

- Objective: diagnose shared reclaim CPU cost and existing unused writes; validate at most one equivalent fix, not a new policy.
- Setup: frozen shared_v1/preparation5, compiled Qwen2.5-1.5B TP1; GPU KV 512 MiB, host 224 MiB; 19 measured requests,153088 input/304 output; identical disjoint warmup, no GPU clearing. Batch-invariant content separately.
- CPU result: single 512-protected-block reclaim ~9.0 to0.375 ms; idle unaffected. Two29-group native audit outcomes identical; installed optimized manager29groupsPASS.
- GPU result: optimized vs reference M1 median E2E +0.4605%; vs root-aware StaticFair −1.8376%, DMA +93.5185%. One screen each, no stable performance advantage.
- Correctness: contentPASS, three independent cold/restored16IDs match, shared survivor reads, final512free. Current stricter oracle independently reapplied to pinned remote runner output.
- Anomalies: two CPU post-analysis failures archived (missing report evidence, stale analyzer dependency); no GPU retry. Python float-sum last bits disclosed in spotcheck. Existing cutlass/semaphore warnings retained.
- Resources: fourGPUruns0.05931228920166985GPUh; cumulative0.37888914884861963/0.5; cleanup verified. CPU0GPUh.
- Conclusion: retain equivalent ancestor-walk fix and opt-in lifecycle components; stop current dynamic variant performance tuning. StaticFair engineering reference. No M2/M3, parameter search, new policy or deployment switch.
- Next-step status: requested bounded diagnosis completed; further research requires a new opportunity definition. No running GPU process.
