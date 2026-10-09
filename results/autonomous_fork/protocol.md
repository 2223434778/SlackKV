# Round 3 pre-registration — demand materialization at agent forks

Baseline source: 7519bda64de6f2c48051b17f60f8e21c83be564e. Historical sources and raw data stay intact.

## Question and initial rejection decisions

Does a burst of actual agent branches restoring a shared CPU prefix duplicate H2D and physical GPU KV enough to affect the workflow critical path? Ordinary singleflight is an existing technique, not a novelty claim. Investigate partial-overlap, bounded waiting and correctness/cancellation only if the baseline confirms an opportunity. No predicted tool deadlines, physical lease or speculative prefetch.

Reject implementing another TTL/priority/working-set write filter now: Continuum, MORI, Autellix, EfficientAgent, CacheScout, RR-Evict and AsymCache already cover much of this space. CPU whole-batch store rejection remains a possible engineering issue, not evidence of a new algorithm. See final report for audited sources and limitations.

## Workload and controls

Synthetic fork/join stress, not a production Agent trace or quality benchmark. Deterministic exact token IDs: eight children share a 24576-token ancestor, append 256 private tokens, generate 128 tokens, then after fixed 0.5-second tool time append a fixed 64-token continuation and generate another 128. All branches and any background calls complete. Three fork episodes use distinct seeds. Inputs fixed independently of output numerics; this is resource replay. Cache warming and deliberate GPU-only cache reset are controlled interventions outside the measured windows, identical on both methods. Record reset semantics and source hits; if no external hit, classify the CPU-fork episode invalid. Do not call this steady production demand.

Final wording clarification (no input or experiment change): the runner always appends **192 fixed canonical tokens** after the first prompt: 128 tokens representing canonical generated history plus 64 new continuation tokens. Thus second prompts are 25024 tokens, first prompts 24832; actual generated outputs are not fed back. Raw inputs and runner code are authoritative; the initial shorthand "64-token continuation" omitted the canonical history.

Compiled OnDemand, same model/dtype/stack, CPU 8 GiB, GPU KV budget, input IDs, branch count and output count. Evaluate both 4 and 8 GiB GPU KV (capacity-limited and capacity-sufficient screening, not physical hardware limits). Start with cached 1.5B for mechanism diagnosis; 7B (56 KiB/token, about 14 GiB weights) if downloadable and evidence warrants. K8 exposes ungated fork behavior; K2 is an additional existing protection baseline, not dropped. Background: one independent 8192-token prompt / 1024-token output call overlaps each fork, with no CPU stores; work never reduced. Use measured total fork E2E, per-call TTFT, transfer counters, actual block unions if instrumented, fixed-work BG time, preemption and output identity.

## Frozen acceptance / escalation

Before coding candidate: require baseline overlapping external restores for shared hashes; distinguish transfer savings from actual critical-path opportunity. Stop if overlap absent or existing protection already removes worthwhile opportunity. Candidate initial screen requires median episode E2E >=10% improvement, P95 TTFT no worse than 5%, BG E2E no worse than 5%, all work/lifecycle valid, H2D no greater. Then run reverse order and the other capacity. A single screen is not stable/significant evidence. Record all failed setup runs and all unfavorable configurations; do not tune branch count, waiting bound or deadlines after observing results. KV content oracle uses batch-invariant mode separately; default performance and invariant timings never mixed.

## Budget

Round-three cumulative ledger includes every GPU test/startup/compile/cleanup, <=4 GPU-hours; one GPU initially; <=1800 seconds per experiment including cleanup. CPU-only modeling/literature/download costs reported separately, not GPU hours. Budget is not a target. Owned subprocesses only; port health refusal if preexisting service. Original test workload preserved for regression. No historical files deleted or dependencies changed.

## Diagnostic addition after first control, before any candidate

First control's three cold-background episodes have one external materialization each, seven GPU hits: no observed stampede. Cold BG prefill consumes the batch token budget while the first restore completes. Preserve this condition and its negative evidence. Add `--background-phase decode`: start the identical BG request first and wait for its first token before releasing the child barrier. BG full E2E is still measured and all 1024 outputs remain; this distinguishes the in-flight lookup mechanism from cold-prefill batching. It is another labeled regime, never silently substituted for the original. Freeze the same metrics/gates before any candidate. If duplicates appear, test K2 as a strong protection and both 4/8 GiB; ordinary coalescing remains non-novel. Potential bounded follower wait is fixed to 100 ms and bypass to original OnDemand on timeout/cancel/failed owner; no post-result tuning.

### Mechanism audit correction before further runs

Decode-background control also has one restore per episode. Root cause is the existing native connector's `_blocks_being_loaded` lookup deferral, followed by completed GPU prefix reuse; cold BG batching alone is an incomplete explanation. No coalescing candidate will be written. Run decode-background 8 GiB / K8 and 4 GiB / K2 to delimit the negative finding, not to screen a speedup. Earlier `episodes.json` derived `source_delta` fields are INVALID (incorrect source labels produced zeros); raw source events and actual H2D counters take precedence. Preserve those files and this correction. New runner omits that misleading field.

Second opportunity screen: exact active-prefix streaming. Compute a generous *lower bound*: a shared ancestor is fetched only once per synchronized decode step, no tails/output KV, no staging or kernel overhead, perfect overlap. Use measured Phase0 pinned H2D median bandwidth and 256 required decode steps. If even this bound exceeds current fork E2E and the unique working set fits current GPU capacity, do not implement streaming for this regime. This does not reject large-model, larger-pool, CPU-attention, compression or sparse attention designs outside these assumptions.

### Correctness / regression addition before these runs

Use independent `fork-oracle-4 --batch-invariant --oracle`: three normal fork episodes followed by serial cache-salted cold Prefill of all sixteen episode-zero branch/turn inputs; confirm local/external hits zero and exact 128 output tokens. Keep its invariant timing separate from performance controls. Old BGv2/40-input/K2/2-GiB Compiled OnDemand is rerun as `legacy-regression-2`, using the immutable Round2 actual inputs and the **same Round3 execution ledger**. These are validation controls, not a candidate or threshold change.

Final capacity clarification: the initial "capacity-limited 4 GiB" description was a **duplication hypothesis**, not observed allocator pressure. Native coalescing leaves the unique fork footprint below 1 GiB; 4 and 8 GiB are both sufficient for this test. Its forced GPU-only cache reset intentionally isolates CPU restoration and cannot establish steady-state host-cache pressure.
