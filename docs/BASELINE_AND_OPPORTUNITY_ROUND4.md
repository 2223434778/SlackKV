# Round4 frozen baseline and opportunity protocol

Frozen before GPU runs, 2026-10-10; starting main d403714ea3ed91e585a2b9f80beb363023272de8.

Question: does the custom K2 first-token gate account for the historical capacity queue,
or does gate removal simply replace admission waiting with preemption and restoration?
This is a baseline intervention, not a new algorithm.

## Controls and workload

Qwen2.5-1.5B FP16, TP1, GPU0 A6000; compiled execution, batch invariant OFF,
CPU KV 8 GiB, GPU KV 2 and 4 GiB. Same isolated Round3 runtime copy for both
methods. Backfill, continuation credit, epochs, JIT promotion and full block-list
fork tracing OFF. Native offloading, asynchronous reservations, deduplication,
allocator fit checks, identity and DMA references remain. Historical resume/source
events remain ON. This is patched OnDemand, not pristine upstream vLLM.

Replay all forty exact input token arrays from
`results/autonomous_capacity/capacity-2/inputs.jsonl.gz`; original eight sessions,
five turns, finish-relative empirical tool pauses, source-prescribed output lengths.
First requests retain the original concurrent launch; actual issue order is reported,
not claimed to be an identical exogenous timestamp trace. Later token inputs remain
fixed even when generated tokens differ. This is hybrid structural replay, not a
semantic task-quality test. BGv2 retains four 9500-token/2048-output chunks (8192
outputs), full completion and positive tail overlap checks. No reduction of work.

## Independent screens and acceptance

Three independent process pairs at each budget. Frozen order: 2 GiB pairs
K2/off, off/K2, K2/off; 4 GiB pairs off/K2, K2/off, off/K2. No outcome-selected
repetitions. Each process loads, warms up and completes the full workload.
All initial screens have allocator detailed tracing OFF; separate diagnosis if needed.

Primary: per-run median full session E2E, with all eight session times reported.
Continuation TTFT median/P95/P99, mixed throughput, actual H2D/D2H, preemptions,
BG wall and TPOT, maximum session time and completion counts are secondary.
Promising baseline improvement requires >=10% session median reduction in the
median paired effect at a budget, BG wall and TPOT median paired increases <=5%,
and every pair valid with matched input/output workload/configuration. All paired
effects and adverse sessions are reported; three runs are screening evidence, not
statistical significance. H2D is explanatory; it is not a post-hoc rejection threshold.
Numerical output equivalence remains UNKNOWN without a separate invariant oracle.

Missing run/invalid lifecycle is not performance success; preserve failures and
do not aggregate an incomplete three-pair condition as passing. No automatic
replacement of failed runs. Allocator observations cannot assign unobserved queue
duration to capacity; reserved pool size is not actual owned memory-time.

## Budget, selection and stop

Continuous `results/autonomous_round4/*/execution.json` ledger includes failures,
load/compile/warmup/cleanup, one GPU per run, <=30 minutes each and <=4 GPU-hours
total. Initial twelve screens are expected below one GPU-hour. Only owned processes
are stopped; occupied port refuses launch. Preserve all historical raw files.

If gate removal eliminates opportunity, establish the stronger engineering baseline.
If it fails, diagnose actual resource dependencies before implementing one candidate.
Physical retention controls must differ from the previous logical epoch; comparisons
with Continuum/MORI/program priority are required before a novelty claim. No new
allocator/lifetime mechanism reaches GPU performance tests before safety smoke.
Stop a route when measured opportunity is small, a matched candidate fails, necessary
data remain unidentifiable within budget, or the proposed mechanism has no substantive
distinction from existing work. Do not interpret this as rejecting all Agent KV research.

## Follow-up frozen before control GPU runs

After the twelve baseline screens: one common diagnostic-runtime gate-off pair,
instrumentation OFF then ON at 2 GiB. Full inputs/BG remain matched; quantify
observer perturbation rather than counting these runs as candidate gains.
Ownership snapshots are throttled to 100 ms; actual free events are unthrottled.
Completion witnesses count physical references and shared unions, not reservation
slack or hypothetical future scheduling. Lookup None reason remains UNKNOWN.

Control: first run a separate batch-invariant local-reuse/cold-prefill, idle-expiry
and cancellation smoke. Only if valid, perform one 2-GiB common-control-runtime
triplet: OFF, program-order-only, program-order plus physical fixed-TTL pin.
Same replay/compiled/default numerical mode, first-token gate OFF in all three.
Primary screen: session median >=10% improvement against OFF, BG wall and TPOT
increases <=5%, all work/lifecycle matched, final physical retention map empty.
If either control passes, execute one reverse triplet as independent screening
replication; if neither passes, stop this specific control without TTL/age tuning.
This is a conditional preregistration, not outcome-selected exclusion of runs.

Fixed TTL and waiting-age guard both 1 second. Full hashed blocks remain physically
referenced until timeout, successful next allocation or cancellation. Actual decode
growth and BG allocation pressure revoke idle pins before preempting/deferring
those demands. Idle timer uses the existing engine probe wakeup. Mature waiting
requests override pin/program-order preference; actual allocator and native DMA
ownership remain authoritative. Background can still suffer compute interference,
so measured BG gates remain necessary. This is an intentionally bounded,
mechanism-matched control, not a full paper implementation or an innovation claim.

Before its first GPU smoke, add a conservative HOL resolver: if allocation fails
and any queued consumer owns an idle retention entry, revoke idle references and
retry. Do not rely solely on timeout to break a retained-consumer/queue-head cycle.
Delayed D2H frees from older turns cannot pin or release a newer turn's retention
state; track outstanding request generations until actual native free completes.

[Continuum v7](https://arxiv.org/html/2511.02230v7), §§4–5, combines program order
and TTL retention with a queue/reload benefit model. This control fixes TTL and
adds demand revocation/age protection; it does not reproduce that estimation.
The [author repository](https://github.com/Hanchenli/vllm-continuum) also explicitly
omits paper estimation. [MORI](https://arxiv.org/html/2606.00866v1), §4, already
combines program idleness, tier placement and capacity admission. A future novel
mechanism must demonstrate a substantive gap beyond these combinations.
