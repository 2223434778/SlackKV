# Host prefix: three-question follow-up, frozen 2026-10-10

Starting main 67eff77. Questions: does partial prefix beat NoHost; does simple
fair sharing retain gains; do gains survive without explicit GPU resets?
No runtime/kernel changes. Reuse Compiled OnDemand/APC/native CPU connector.
Same Qwen1.5B FP16 TP1, 32768 maximum model length. All prescribed inputs and
outputs identical within a comparison. NoHost means max_offload_tokens=0 only.
Full stores full input. Fair stores block-aligned static equal shares of physical
CPU blocks; last session receives remainder. Distinct, nonshared prefixes.
This is admission on a static membership set, NOT a general physical quota or
immediate reclamation API. No leases/reservations or forced frees introduced.

All synthetic workloads: each session 6 turns, initial28672 input, growth64/turn,
16 output. Fixed canonical tokens, not task-quality evaluation. Background8192
input/256output, no host stores. Save exact inputs/timestamps/source snapshots.

A: single session, host256MiB, natural85% GPU allocation, reset inactive GPU
cache before each turn; start BG then issue Agent at its first token, join both.
Two independently chosen seeds773311/884422, order NoHost/Fair, Fair/NoHost.
Primary screen: both seed-pairs resume max (5 resumes, nearest-rank P95) >=10%
better vs NoHost; BG median no worse5%. Diagnostic total not natural Session E2E.

B: two sessions, host256MiB, same GPU allocation, reset each request, alternate
A/B over6 turns. Seed773311. Order Fair/Full/NoHost. Each request BG256output.
Per-session relative mean service and resume max, same complete BG work. Require
both sessions' resume max >=10% better than NoHost; no session mean service or BG
median worse5%. One matched screen, no significance. Native metadata audit checks
actual retained prefix and load/store references, cancellation, replacement of
completed membership. It does not certify GPU cancellation/preemption exhaustively.

C: four independent closed-loop sessions, .2s synthetic tool gap after each
completion, no resets. Launch together at first BG token. BG12 sequential jobs
run independently of sessions. Whole sessions finish all6 turns; measure actual
closed-loop E2E including tool gaps and contention. Main median session E2E,
max per-session relative regression, and BG12-job makespan. No claim trace realism.
Seed773311, fixed order:
  GPU-hot: natural85% GPU allocation, host256MiB; NoHost/Fair/Full.
  controlled natural-eviction pressure: GPU2GiB, host512MiB; Full/Fair/NoHost.
  ample-host: GPU2GiB, host8GiB; Fair/NoHost/Full.
These budget caps are controlled resource sensitivity, not physical RAM scarcity
or evidence of production demand. Actual GPU hits/misses/host transfers must show
pressure occurred; historical total input is not simultaneous residency.
Positive screen requires median Session E2E >=10% better vs both simple baselines,
each session E2E no worse5%, BG makespan no worse5%. No benefits required in hot or
ample controls; >5% regressions flag applicability limits. Only one seed per regime;
expand only after clear evidence. Stop a regime for correctness/liveness failure.

Record sources per request and final counters, transferred bytes, output counts,
actual GPU pool, full logs, failures and unresolved counters as UNKNOWN. Missing
dynamic transfer labels mean zero ONLY when both snapshots explicitly contain
native metric HELP/TYPE and source external=0 with known fresh process; otherwise
UNKNOWN. Recompute input hashes from raw payload, not just recorded hash labels.
Do not add timing differences to claim causal DMA/prefill decomposition.
Prior valid invariant oracle stays one-input evidence; native content unchanged.

All setup/compile/work/cleanup on GPU0 counted; <=30min/run, conservative continuous
4GPUh ledger includes autonomous_fork, autonomous_round4, host_prefix_cost and this
follow-up. Reserve .5h before each run, no service interference. Freeze runners
and preserve all failed attempts; no large matrix or tuning to favorable results.
