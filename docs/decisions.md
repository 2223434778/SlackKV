# Decisions

## Phase 4D-B — Separate logical reservation from physical feasibility

- **Decision:** Keep the 3,510-block logical future-reservation budget, but require each guarded promotion to pass an in-engine check against the block pool's exact allocatable free-block count before connector load/allocation. Lease handoff releases protection ownership; it does not mean GPU KV blocks have become free.
- **Evidence:** On the frozen large-prefix workload, exact-free G1 admitted 1,276 blocks with 2,135 free and rejected 1,277 blocks with 859 free (2/2 each). The guarded 8-job integration completed in the smoke and all three fresh-server repetitions; E06/E07/E03 were rejected before connector work and used OnDemand fallback.

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

## 2026-09-14 — Phase 1C evaluates admission before promotion scheduling

- **Decision:** Phase 1B-Fix showed that QueueJIT scheduling alone provides limited benefit because many real pauses are shorter than their KV migration cost. Phase 1C therefore evaluates transfer-normalized selective admission before promotion scheduling.
- **Mapping:** 56 KiB/token with a 131,072-token cap is primary; 128 KiB/token is only a stress sensitivity on the same retained events.

## 2026-09-17 — Phase 3A reuses vLLM connector lifecycle for native promotion

- **Decision:** The research-only native promotion path enters through a local `slackkv_prefetch_only` request, then reuses vLLM's normal prefix identity, GPU block allocation, OffloadingConnector load metadata, asynchronous worker transfer, completion bookkeeping, and prefix-cache transition. It does not introduce an independent CUDA copy engine.
- **Reason:** Reusing the pinned runtime's existing connector lifecycle preserves block ownership and cache invariants while allowing `kv_connector_no_forward` to prove that the special request schedules no transformer work and produces no token.
- **Consequence:** The Phase 3A patch remains isolated and reviewable, normal inference behavior is unchanged, and subsequent runtime work can evaluate a genuine promotion-only mechanism. The endpoint is explicitly experimental rather than a production API.
\n## 2026-09-20 — Phase 3B separates connector completion from normal forward work\n\n- Decision: active SlackKV prefetch requests receive connector-only scheduler microsteps and are polled until completion before deferred normal work is admitted.\n- Reason: Phase 3A showed ~5 ms CPU→GPU but ~90 ms endpoint completion when unrelated forward work shared a step; Phase 3B overlap reduced client completion to ~24–27 ms with zero mixed steps.\n- Consequence: native promotion correctness is preserved, but the six-session result is empirical Result B: OnDemand remains competitive; no new policy or workload dimension was introduced.\n\n## 2026-09-20 — Phase 3C uses one fixed safety service budget\n\n- Decision: evaluate exactly one `p_safe=30 ms` scheduling budget, without recalibration or a margin sweep, while reusing the Phase 3B connector-first implementation unchanged.\n- Reason: Phase 3B isolated service was ~17 ms but overlap was ~24–27 ms; a modest fixed budget tests whether uncertainty caused the observed cascade.\n- Consequence: QueueJIT-Safe30 reduced misses and P95 versus NaiveJIT-Safe30 at very low memory-time, but OnDemand remained competitive.\n\n## 2026-09-20 — Phase 3D models future resume blackouts\n\n- Decision: add one frozen `resume_guard_ms=110` blackout calendar to a generic QueueJIT feasibility simulator; do not alter connector behavior or sweep parameters.\n- Reason: Phase 3C identified C's 100-ms promotion inflation when it overlapped clustered resume compute.\n- Consequence: ResumeAware eliminated the measured conflict and long promotion while retaining low memory-time, but did not beat OnDemand latency.\n\n## 2026-09-20 — Phase 3E tests fixed resume spacing\n\n- Decision: evaluate one 40 ms gated resume spacing with latency charged to original deadlines; no spacing sweep or runtime changes.\n- Result: fixed gating did not reduce measured overlap and increased ABC/overall tail latency, so no dynamic admission design is inferred from this experiment.\n
## 2026-09-21 - Phase 3H uses a fixed two-slot in-engine pre-first gate

- Decision: generalize the existing first-token gate to one run-wide request-specified capacity and evaluate K=2 against unlimited and K=1 controls.
- Reason: K=1 removed pre-first contention but over-serialized multi-token admission; K=2 is the smallest bounded alternative that retains a useful batch.
- Consequence: the gate keeps a set of owners, asserts mixed limits within a run, and releases one owner on real FIRST_TOKEN output. Promotion and cache lifecycle remain unchanged.
## Phase 3R-FIX: instantaneous capacity is not a residency guarantee

- Decision: do not treat a target-time `kv_cache_usage_perc` headroom check as a guarantee that promoted KV remains resident until resume.
- Evidence: all twelve promotions had at least 196.13 MiB measured free space against the frozen 144-MiB requirement, but seven later entered `WAITING_FOR_REMOTE_KVS`; aggregate source conservation showed 48.01% local and 51.99% external tokens.
- Consequence: retain the negative Result B and evaluate explicit residency reservation only in a separately authorized phase. The 144-MiB threshold was not tuned.

## 2026-09-24 — Phase 4A requires an observed LLM request-start timestamp

- Decision: do not equate TraceLab v0.0.2 `tool_calls.result_at`, a next-round `tool_result`/`user_message`, or any output-side timing event with the actual start of the next LLM request. The public DuckDB does not contain an explicit next-LLM request/start event, so Phase 4A stops at `RESULT_C_NO_VALID_RESUME_SEMANTICS` rather than generating pseudo-slack or predictor errors.
- Reason: tool completion/input readiness can precede model admission or execution by an unobserved interval. Calling it resume time would invalidate a deadline-aware prediction evaluation. The released `prefix_tokens` value is cached-input-token accounting per TraceLab documentation, not a semantic KV-block identity.
- Alternative considered: the earlier Phase 4A eligibility-only attempt used latest tool result as the resume boundary. That is a useful proxy for some descriptive analyses but not the actual-resume timestamp required by the revised protocol.
- Consequence: no real-trace opportunity, causal-predictor, or structural timed-replay conclusion is drawn from this dataset. Existing Phase 1 oracle-proxy interpretations remain historical and are not rewritten.

## 2026-09-24 — Do not treat coding-agent `tool_exec_ms` as resume slack

- Decision: for the pinned `netpreme/coding_agents` implementation, do not use `tool_exec_ms` as response-completion-to-next-request slack.
- Evidence: the proxy calls `note_response_time()` when a new request arrives, immediately before forwarding; recorder stores that current timestamp and next call subtracts it. This makes the value an inter-request arrival interval. The metrics scraper separately emits E2E counter deltas averaged per completed request when multiple completions land in one polling tick, so exact request-time pairing is not guaranteed by row order.
- Alternative considered: following the README field description and using `tool_exec_ms` as ground-truth tool gap. Rejected because pinned code contradicts that description.
- Consequence: use neither `tool_exec_ms` nor poll-window E2E as an exact slack label without a verified request-ID alignment source. Phase 4B does not create timed replay jobs.

## 2026-09-25 — Bounded live HF sample does not repair timing alignment

- Decision: preserve the Phase 4B no-timed-replay conclusion after the HF endpoint became reachable; do not equate token/text turn records with tool completion timing.
- Evidence: pinned HF metadata confirms `turns: list<JSON>`. A bounded 100-row sample from each GPT-OSS config (5,550 turns total) exposed `isl`, `isl_new`, `osl` and text members, but no tool/request/timestamp/duration/latency fields. The full multi-GB data was not downloaded and the sample is not exhaustive. Pinned source independently shows `tool_exec_ms` is arrival-to-arrival, while poll-window E2E can be averaged across requests.
- Alternative considered: treat this sample as proving field absence across all nested rows. Rejected because the scan is bounded.
- Consequence: retain `RESULT_C_TIMING_NOT_RECONSTRUCTABLE`; do not report slack/replayability counts or timed candidates without per-request timing alignment.

## 2026-09-25 — Phase 4C replay is hybrid, not request-timestamp-faithful

- Decision: Future SlackKV serving experiments may use Phase 4C hybrid replay only as a hybrid workload: real token/KV structure + empirical tool-phase timing proxy. It must not be labeled request-timestamp-faithful real trace replay.
- Reason: Phase 4C reconstructs released ISL/ISL_new/OSL structure and uses empirically observed TraceLab tool-phase durations as controlled inter-request pauses; public traces do not provide request-level real resume timestamps.
- Consequence: `results/phase4c/hybrid_workload.csv` is a frozen hybrid workload interface, not a real-session timing trace. Phase 4C does not establish performance superiority.

## 2026-09-26 — Phase 4D-A derives capacity and service budget at real prefix scale

- Decision: for frozen Phase 4C boundary-0 jobs, derive reservation capacity from 75% of actual GPU KV blocks and freeze a single promotion service budget from nine real-size calibration measurements plus a 20% margin. Do not carry Phase 3's 512-block/40-ms constants into this larger-prefix workload.
- Reason: each selected prefix requires 1,271–1,444 blocks, exceeding the former 512-block budget; runtime reported 4,681 blocks, yielding a mechanically derived 3,510-block budget. Calibration produced a frozen 102-ms budget.
- Consequence: these are Phase 4D-A workload/runtime derivations, not tuned performance settings. The policy matrix did not start because NaiveJIT failed the required integration completion gate; no benefit claim follows.


## 2026-09-26 — Phase 4D-A-FIX exact lease identity and capacity return

- **Decision:** Keep logical reservation keys episode-scoped, but require an explicit episode-to-turn-qualified physical lease-key mapping and structured exact-key `LEASE_RELEASED` evidence before reclaiming logical blocks. No substring/prefix matching is permitted.
- **Evidence:** Regression tests demonstrate that logical `E04` does not equal physical `E04:turn:1`; runtime traces then observed exact `LEASE_RELEASED` for E04/E00 before logical release, followed by E06/E07 reservation promotions after capacity recycling.
- **Limitation:** The retry integration did not complete: E06/E07 native requests remained pending until the 120-s client read timeout, so the 8/8 acceptance gate failed. Naive L0/L1 were not run. The release mapping is runtime-validated, but the complete Reservation integration and Naive root cause remain unresolved. No vLLM source or serving algorithm was changed.

## Phase 4D-B — Separate logical reservation from physical feasibility

- **Decision:** Keep the 3,510-block logical future-reservation budget, but require each guarded promotion to pass an in-engine check against the block pool's exact allocatable free-block count before connector load/allocation. Lease handoff releases protection ownership; it does not mean GPU KV blocks have become free.
- **Evidence:** On the frozen large-prefix workload, exact-free G1 admitted 1,276 blocks with 2,135 free and rejected 1,277 blocks with 859 free (2/2 each). The guarded 8-job integration then completed in the smoke and all three fresh-server repetitions; E06/E07/E03 were rejected before connector work and used OnDemand fallback.

- **Scope:** This establishes guard correctness and restores liveness for the frozen integration. It does not show that capacity is the only possible cause of all NaiveJIT failures, nor make a performance or SOTA claim.

## 2026-09-27 — Physical leases require a serving-progress path

- **Decision:** Treat instantaneous exact-free-block promotion admission as insufficient by itself when successful promotions become non-evictable physical leases. Admission must preserve a future serving-progress path; Phase 4E-A-RCA confirms the need, but does not implement a new policy or fix.
- **Evidence:** In the frozen large-prefix hybrid workload, sequentially leasing E00/E01/E02 left fewer than E04's 1,271 required blocks free. E04 was the earliest unleased resume; the pinned FCFS waiting path hit `allocate_slots() -> None` and `break`, preventing leased resumes behind it from obtaining real-request ownership and releasing their leases. U3 reproduced in 2/2 no-X and 3/3 X runs; removing one lease, leaving the same prefetched set evictable, or leasing E04 itself restored progress in the prescribed controls.
- **Scope:** This is a controlled causal result for the pinned vLLM scheduler and Phase 4C first-boundary hybrid workload. It does not establish a universal vLLM deadlock or prove that a particular reservation policy is optimal.

## 2026-09-27 — Physical fit is not progress-safe fit

- **Decision:** A leased native promotion must pass both exact in-engine physical fit and PSA-1's incremental headroom check for the earliest unfinished resume. Protected progress is `ceil((Turn1 ISL + 1)/16)` blocks; the scheduler subtracts actual matching lease blocks, including the candidate's hypothetical lease when promoting the protected request itself.
- **Evidence:** At 2,073 free blocks, E02's 1,323-block promotion would physically fit but leave only 750 blocks for E04's 1,278-block progress footprint; PSA fast-rejected it. The unprotected deterministic control reproduced the RCA circular wait, while PSA completed 2/2 no-X and 3/3 X repetitions. Concurrent JIT without logical reservation completed 8/8 fresh runs with two promotions per run; the unchanged logical-reservation policy also completed 4/4.
- **Scope:** PSA-1 is validated only for the frozen eight first-boundary hybrid jobs and pinned vLLM waiting semantics. It is not proven optimal, a replacement for logical reservation, or a performance winner. The 3,510-block logical ledger and PSA showed overlapping behavior in P5, but equivalence was not established.

## 2026-09-28 — Compare only PSA-safe policies after liveness closure

- **Decision:** Phase 4F-B is the first frozen hybrid performance matrix after the lease-HOL root cause was confirmed and PSA-1 restored liveness. Exclude the unsafe physical-lease-without-PSA variant from latency comparisons; retain it only as liveness-negative evidence. Compare OnDemand, evictable guarded JIT, guarded JIT+lease+PSA, and full reservation+lease+PSA using fresh servers and one immutable workload/runtime.
- **Evidence:** All 32 measured runs were valid. The full policy retained minimum H2D and background interference within preregistered bounds, but improved median P95 over OnDemand by only 1.59%, below the 10% H1 threshold. Logical reservation reduced promotion attempts and PSA rejection, yet had limited incremental latency/traffic benefit over the safe simplified C policy here. B's evictable JIT incurred median 275.406 MiB duplicate H2D and seven promoted remote waits.
- **Scope:** This is a directional result for the frozen Phase 4C first-boundary hybrid workload. C jointly adds lease and PSA versus B, so their separate effects cannot be inferred from this matrix. Eight repetitions and descriptive bootstrap intervals are not a statistical-significance claim; no request-timestamp-faithful real-trace or SOTA conclusion follows.

## 2026-09-28 — K2 ownership follows first-output lifetime, not scheduling transitions

- **Decision:** A traced request participates in the first-token gate only while its canonical output-token count is zero. Post-first preemption/remote restoration must not reacquire a slot. Pre-first re-admission of an existing owner is idempotent; terminal cleanup always discards gate and first-token bookkeeping, including delayed KV cleanup.
- **Reason:** Phase 4G-A-FIX's failed run re-admitted E03:turn:1 with 17 output tokens and E06:turn:1 with 6. Both reacquired slots that the first-output-only release path could never release again. This is an implementation lifecycle defect, not a workload-data defect or evidence against PSA.
- **Alternatives:** Clearing all owners periodically or scanning finished requests could mask the defect and add scheduler work. The chosen repair instead enforces the invariant at admission/acquisition/termination with constant-time state checks. K remains 2; no workload, PSA, lease, reservation, connector, or service-budget tuning is performed.
- **Scope:** Validation is recorded separately in `results/k2_lifecycle_fix/`; historical incomplete matrices and their source fingerprints remain unchanged. No performance conclusion follows from the repair validation.

## 2026-09-29 — Reject guarded native prefetch on actual KV allocation failure

- **Decision:** For a capacity-guarded, prefetch-only request, `allocate_slots(None)` in the connector-first waiting path terminates the promotion with a structured zero-transfer `capacity_rejected` before connector load or lease acquisition. The real resume falls back to OnDemand. Ordinary serving requests and historical unguarded prefetches retain their existing scheduling behavior.
- **Reason:** An earlier physical-headroom ADMIT is not a guarantee that allocation still succeeds once other in-flight block reservations are accounted for. Leaving the guarded prefetch pending in connector-first mode prevented the ordinary prefills needed to free those reservations from running. The repair is an O(1) failure-path check, without a new periodic query or a changed successful-promotion path.
- **Scope:** The repaired branch occurred once in the fresh Phase 4G-B-FIX matrix, with zero transfer/lease and successful real-resume completion. The full 12-run mechanism matrix completed, but the frozen Result D is conditioned on sampled PSA-safe windows, not a general performance or continuous-time proof. See `results/phase4g_b_fix/`.

## 2026-09-30 — Use native absolute monotonic sleeps for hybrid pause injection

- **Decision:** Replace Python relative sleep plus busy-spin in the Phase 4G-C-FIX pause waiter with Linux `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`. Retry `EINTR` against the same absolute target and fail explicitly on other errors; do not fall back to spinning.
- **Reason:** The previous Phase 4G-C matrix was invalidated solely by pause-fidelity failure while multiple Python workers busy-spun under full controller/probe/background load. Absolute native sleep removes that Python spin tail and avoids cumulative relative-wait drift. The implementation records wake overshoot separately from client dispatch delay so later timing failures can be localized.
- **Evidence:** The native 100-sleep self-test passed (P95/max absolute overshoot 0.214/0.219 ms). All six full-load calibration runs passed the frozen per-run pause gates and aggregate P99 gate; all eight fresh measured runs passed (C0 absolute pause P95/max 0.507/2.533 ms; C1 1.044/6.242 ms). No SlackKV or vLLM serving mechanism changed.
- **Consequence and scope:** The timing repair permits mechanism RCA under the frozen pause-fidelity thresholds; it is not a real-time scheduling guarantee. Stage T's single 12.342-ms maximum was a host wake overshoot and remained within the frozen 30-ms maximum. The separate mechanism result was `RESULT_D_GUARD_ALLOCATOR_MISMATCH`; source recheck removed stale protected visibility (6/40 to 0/127), but candidate=protected rate moved only from 2.50% to 3.15% and median PSA reserve from 1,336 to 1,359 blocks. See `results/phase4g_c_fix/`.

## 2026-10-09 — Do not infer allocator mismatch from missing observations; inspect terminal protected heads before PSA-2

- **Decision:** Keep the current C1/PSA-1 serving policy unchanged. Treat nullable allocator fields as UNKNOWN, never zero. The Phase 4G-C-FIX mismatch result is unsupported after correction. Phase 4G-D found no sampled actionable retry window and found four exact-known F0-blocked selections recoverable under F1/F2 shadow frontiers, but this is not a safety proof for changing the live frontier.
- **Evidence:** Four fresh runs completed 128/128 boundaries; allocator observability covered 46/46 eligible selections. Of 61 exact-known protected-need rows, median PSA-1 need was 1,340 blocks and median instantaneous allocator need 1,292 (paired median gap 0). In the 31 exact-known rows with local protected cache, median gap was 1,307 blocks, while median owned blocks were 1,306 and median evictable blocks were zero. This demonstrates that local residency can reduce instantaneous allocation need while often being ownership-protected, so it cannot simply be subtracted from a future-progress reserve.
- **Consequence and scope:** Do not proceed directly to PSA-2 or partial promotion from these data. Retry was not observed as actionable in the five adequately sampled long pre-resume intervals; frontier persistence is a plausible shadow-level blocker, but any state-aware alternative must independently prove liveness under the pinned waiting-queue behavior. No runtime policy was changed. See `results/phase4g_d/`.

## 2026-10-09 — Separate direct restoration opportunity from joint scheduling opportunity

- **Decision:** Under the user's independent-research handoff, reassess end-to-end opportunity before extending the JIT/PSA architecture. Recommend a diagnostic/safety evaluation of engine-owned joint demand admission and restoration; retain the live policy unchanged pending the user's direction decision.
- **Evidence:** Raw first-output event pairs show OnDemand fixed-path P95 improvement of 0.905% when every observed remote-wait interval is deleted. Nonremote pre-admission at the original tail remains about 6.32 seconds. Most complete-system promotion calls were safely rejected (62 capacity, 17 progress, one success). Remote wait lies partly within the historical scheduler-wait metric; pure queue attribution from that metric is invalid.
- **Alternatives:** State-aware frontier may open four shadow opportunities but has no demonstrated safe E2E gain. Partial promotion reduces speculative occupancy but does not reduce standard full-attention execution footprint. Both remain candidate architectures rather than implemented changes.
- **Consequence:** Treat the deletion model as conditional on fixed observed service/arrival paths, not a global causal bound. Do not infer a 10% achievable gain from the size of queue waiting or from Frontier-dominant taxonomy. Require blocking-cause evidence, calibrated resource modeling and independent liveness validation before a small runtime trial. Full discussion: `docs/RESEARCH_REASSESSMENT.md`.

## 2026-10-09 — Reject capacity-only backfill; establish a compiled demand baseline and separate numerical correctness

- **Decision:** Keep bounded backfill disabled and stop expanding the existing JIT/Frontier matrix under the frozen scenario. Retain native/offload/ownership/reservation/PSA protections. Next research should identify and reduce actual demand KV residence and admission contention against compiled OnDemand, with a separate invariant oracle for debugging.
- **Evidence:** Real allocator failures support capacity contention; capacity-only backfill worsened P95 16.97% and caused repeated restoration. A single exact-input matched compiled/eager pair improved P95 15.81% and session median 29.94%, while fixed-path remote-wait deletion in compiled runs improves P95 only 0.870–1.051%. Compiled Full had 0/10 successful promotions and worsened P95 16.08% vs the exact-input matched compiled control. These are bounded screens, not general causal upper bounds or universal negative results.
- **Audit correction:** First execution pair and one Full comparison had future input drift; exclude them from exact-input matched aggregation. Cold no-cache requests themselves varied with batching; invariant control made 10/10 sensitive cold/resume outputs equal. Do not treat hash disagreement alone as KV corruption or infer general task-quality equivalence from two inputs.
- **Consequence:** Do not count ordinary compilation as innovation. Do not introduce a formal GPU matrix until remaining avoidable queue cost, effective concurrency and fairness are causally identified. Full GPU KV memory-time, real workload generalization and successful Graph/native-promotion correctness remain UNKNOWN. Full record: `docs/AUTONOMOUS_RESEARCH_REPORT_20261009.md`.

## 2026-10-09 — Round 2: reject both demand prototypes; treat capacity and transfer proxies separately

- Decision: leave continuation credit and soft working-set epochs disabled; stop their parameter expansion. Preserve compiled OnDemand, ownership, allocator and lease/liveness guards. Final project goal remains; prioritize capacity-efficient serving under explicit SLO/fairness instead of JIT transfer timing or cache-hit metrics.
- Evidence: fixed-input 2/4/8-GiB OnDemand probes show P95 5199.92/1163.51/445.32 ms. At the same 2-GiB budget, credit improves P95 39.90% but worsens session median 6.08%; epochs reduce actual H2D 38.60% but worsen P95 106.80% and session median 14.37%. Both fail frozen gates, no threshold or lifetime tuning.
- Measurement: epoch gating suppresses allocator checks and moves queueing before them. Fewer observed failures is not causal capacity relief. Actual transfer counters supersede source estimates; one-block mismatch and missing 8-GiB H2D series retained as UNKNOWN. Full owned KV memory-time and content/quality correctness remain unknown.
- Consequences: these single-run failures reject the tested implementations, not all scheduling/retention algorithms. No new novelty or stable speedup. Next evidence should include representative steady multi-turn arrivals, a justified model/capacity regime and actual owned-block/compute timeline before expensive matrices. Report: docs/AUTONOMOUS_RESEARCH_REPORT_ROUND2_20261009.md.

## 2026-10-09 — Round3: reject redundant materialization and full-root streaming in the tested regime

- **Evidence:** Twelve non-invariant synthetic fork episodes, matched 51 actual inputs/run, GPU KV4/8GiB and K2/K8, each restore one shared ancestor. Native APC OffloadingConnector already defers lookup on `_blocks_being_loaded`; preserve that ownership/readiness protection. H2D2016MiB/run equals actual source accounting. Ordinary coalescing does not meet its implementation premise.
- **Opportunity screen:** Root-only exact active streaming needs180.39GB across256attention iterations; historical pinned H2D reference implies6.86s versus~4.13s fork E2E and a resident unique workset. Reject only this regime/assumption, not CPU attention or all large-model tiering. Generic TTL/admission/tail eviction substantially overlaps recent work.
- **Decision:** Stop B, no accepted serving algorithm or strong-baseline algorithm speedup. New issue to establish is actual host-miss/recompute/readiness critical-path cost for long-context dynamic agents; range readiness is an unverified candidate, not an innovation claim. Retain the original final goal and all historical mechanisms/data.
- **Safety/evidence:** Independent invariant16/16cold token oracle, old40/32/BG8192regression valid,52local/15CPU-runtime tests pass. Full GPU cancel/Lease/HOL matrix and owned memory-time remain unverified. Seven GPU executions including partial failure cost0.20048675264813934GPUh; report/docs and all raw failures archived under results/autonomous_fork.

## 2026-10-09 — External advisory: resolve baseline and capacity opportunity before range readiness

- Context: review of 61139e310a5976cf4311e8bbc4badf82f56439af, requested by the user; no runtime policy change in this commit.
- Recommendation: retain Round3's scoped negative conclusion for ordinary coalescing; next investigate matched K2 versus gate-off baselines, full request latency boundaries and actual shared-block ownership. Use one suitable existing retention/admission mechanism as a control before treating a new heuristic as a research contribution.
- Rationale: the fork already restores each root once; the representative run has material unclassified client/API intervals; follower lookup deferral is outside its own remote-wait events. The old soft epoch did not physically retain KV and does not falsify all retention mechanisms.
- Alternatives: immediate range-ready implementation, more JIT/Frontier tuning, or waiting for a larger model. None currently has a verified opportunity large enough to justify being the default next step. Larger-model and realistic-workload validation remain necessary for broader claims.
- Consequences: advisory instructions and evidence are in docs/RESEARCH_REVIEW_AND_NEXT_ROUND_20261009.md. The executing Codex retains discretion within AGENTS.md budgets and safety/evidence requirements. Historical thresholds/results are unchanged; any new objective/threshold is frozen before new measurements.

## 2026-10-10 — Round4: stop the tested retention control; identify active ownership before a novel policy

- Evidence: three matched K2/off pairs at each of2/4GiB give median paired session changes−0.37/−0.63%; no large gate-removal gain. Separate actual ownership diagnosis observes175 fit failures with reservation0 and physical free-block partition checks;30/32 resumes fail before admission.173 single-request and2 two-request static completion witnesses do not identify optimal scheduling or causal delay.
- Matched physical control: order-only/session+6.87%, order+pin/session+20.09%, pin/P95+22.20%, actual H2D essentially unchanged.32 physical acquisitions all released,28 by HOL demand fallback and4 byTTL. Both miss frozen acceptance; conditional reverse replication not triggered. No TTL/age tuning.
- Decision: leave experimental control default-off, retain baseline and native ownership/DMA/liveness guards. Do not claim a new algorithm, defeat of full Continuum, or general impossibility. Shared-release-set scheduling has no demonstrated distinctive opportunity in this workload; do not manufacture a PoC.
- Next: establish representative exogenous arrivals and workflow/shared ownership where actual release-dependent execution differs from ordinary priority/cost policies, with matched strong baselines and predeclared SLO/owned-memory/fairness targets. Final research goal unchanged. No formal matrix or larger-model download requested.
- Report and all failures/UNKNOWN: docs/AUTONOMOUS_RESEARCH_ROUND4.md; results/autonomous_round4/. Cost0.5462201082GPUh,19 executions,17 complete performance+1 valid correctness+1 failed instrumentation run.

## 2026-10-10 — External advisory after Round4: stop local heuristic iteration

- Evidence: commit465c4d2240b34210e93d6d861e3ec0f9fc830774. Gate removal lacks the expected session benefit; physical retention has no successful request-allocation handoff in the observed performance run. Native active ownership limits cannot be eliminated by re-evicting blocks already counted as available.
- Recommendation: stop expanding these specific variants on the old hybrid workload. Next deliver a research-opportunity decision grounded in existing/public workload data before implementing another scheduler; do not assume shared-release hypergraphs are the required next topic.
- Rationale: previous K2/API suspicions do not explain the measured seconds-level admission delay. Conservative pin reclamation can remove the intended benefit, but deleting liveness protections or tuning TTL without a resource/fairness argument is not an acceptable remedy.
- Scope: advisory only; no production default, safety constraint, historical result, or final Agent Serving objective changed. Full Continuum/other retention methods are not falsified. Detailed limits and reproducible evidence are in docs/RESEARCH_REVIEW_ROUND4_20261010.md.
## 2026-10-10 — Opportunity screen: no new algorithm GO

- Latest handoff/main1c9c7da calls for evidence before another scheduler. Read one bounded, pinned public Claude Code workload prefix (15traces/2661requests); no old matrix or GPU execution.
- Structural completed-history prefix coverage97.83% suggests reuse importance, not measured cache loss or avoidable time. Proxy context lengths, model changes and interleaved explicit containers prevent claiming faithful legacy32K replay, real deployment capacity or exactDAG. Actual host loss/recompute cost remainsUNKNOWN.
- Decision: recommend pausing/archiving the original JIT paper claim, preserve code/data/final goal. Do not implement generic TTL/priority/write admission or shared-block accounting as new research. MORI/EfficientAgent/RR-Evict and existing AIPerf reconstruction must inform any stronger future baseline.
- At most retain one cost-identification question about loss of historically reusable prefix under natural host budgets. Advance only after measurable critical-path cost and a distinct uncovered decision; current evidence satisfies neither. Reportdocs/RESEARCH_OPPORTUNITY_DECISION.md; compactCPU evidence results/opportunity_decision_20261010/.
- 2026-10-10：用户放宽创新门槛，组件重叠不再作为否决PoC的充分理由；结果仍须有整体差异与实测/消融证据。旧JIT负结论保留，新增重点从promotion时机转向host写入放大与可持续部分前缀。
- CPU cost-tail全部门槛失败，不做权重调参；连续前缀参考给出写入线索后真实GPU验证。固定physical-cap checkpoint复用原生max_offload_tokens，不改KV内容/ownership，也不称其为独立论文算法。
- 两对受控筛选通过资源/退化门槛，但host256MiB、forcedcold、单路径适用范围严格；不代替natural workload或SOTA对照。明确H2D增加8.01倍、总bytes下降67.54%的tradeoff。
- 固定cap不是pool公平性策略；双会话反例已测。保留机制、再组合capacity-conditioned admission/公平尾部与adaptive frontiers，先验证自然聚合压力而不是直接正式矩阵。新报告docs/AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md。

- 2026-10-10 host-prefix审查：按用户放宽创新标准保留真实机制收益，不因复用组件否决。推进顺序改为NoHost、静态公平额度与自然回收证据在先，动态frontier在后；若简单方案足够，接受工程价值。当前cost-tail使用累计损失而非单块边际损失，其失败不否定所有cost-aware方法。不把人为256MiB/forced-cold结果推广成生产E2E优势。详见docs/RESEARCH_REVIEW_HOST_PREFIX_20261010.md。

- 2026-10-10 three_question_followup：NoHost补强单会话partial机制收益；静态均分保住两个会话前缀但只有8.4/8.9%P95收益；自然驱逐压力下会话E2E约−8.88%，未过10%预注册门槛。保留简单候选，不调阈值/动态权重。GPU-hot无额外命中，host充足应全存；普通Full比NoHost的大收益不归为partial算法。真实部署host不足尚未证实，不能从受控512MiB推断生产意义。详细报告与16run数据见AUTONOMOUS_RESEARCH_HOST_PREFIX_FOLLOWUP_20261010.md、results/host_prefix_followup_20261010/。安全复现使用name-prefix，始终同一连续budget。

- 2026-10-11 外部审查c62d9f6：独立复算全部16run原始requests与指标快照，确认单会话胜NoHost、多会话前缀保留及受控无reset约8.88%会话改善；未通过历史10%门槛，hot/ample单会话退化警示保留。新增GPU消耗0，未重跑payload哈希或完整GPU测试。下一自主目标为验证合理GPU/pinned-host预算下的部署价值，不追阈值调参、不规定细碎实验步骤。审查与范围docs/RESEARCH_REVIEW_FOLLOWUP_20261011.md，复算results/research_review_followup_20261011/spotcheck.json。


- 2026-10-12 deployment_opportunity：累计proxy输入不能证明同时驻留压力；离线future-needed只能定义保留目标，不能包装成在线policy或最小容量下界。模型bytes/token用真实配置计算，但proxy/目标tokenizer与endpoint/硬件时间不可直接映射。保留Full/NoHost强基线，不在未经证实的512MiB生产预算下扩展动态策略。32B TP2长上下文仅资源可行性候选，GPU池未测、需求UNKNOWN。服务器研究盘消失时不自行改挂载/安装环境/重置预算；已设计pin32实验保持NOT_STARTED而非失败或成功。详见AUTONOMOUS_RESEARCH_DEPLOYMENT_OPPORTUNITY_20261012.md。

- 2026-10-12恢复：选空闲GPU1进行host诊断，不因GPU0远程桌面占用终止其他服务；历史family缺失拒绝启动。32GiB实际pin成功证明512MiB不是本机物理必要上限；27s初始化与常驻RSS说明预算不是无成本。CPU pinned allocator缓存不能等同请求引用泄漏，不用VmPin/VmLck判pin正确性。保留未挂盘历史，不将此资源诊断包装为Serving加速；继续以合理Full为基线。

- 2026-10-12 外部审查b85027a：复算27 namespace汇总、32B KV/权重容量与pin32 RSS/系统内存差，支持报告边界；新增GPU0。32GiB分配成功不等于serving性能通过，27s是初始化成本而非逐请求成本。下一自主目标为明确部署下Full/NoHost的实际可避免成本，允许有依据的透明合成场景，不强制完整生产trace或自动转32B。交接docs/RESEARCH_REVIEW_DEPLOYMENT_20261012.md。

- 2026-10-12 large_model：自然32B/TP2 GPU池17.09GiB、host64GiB、21长历史合成session，无reset；静态前缀准入比compiled NoHost/Full E2E−16.44/−18.40%，全部session和BG makespan满足冻结退化约束。Full 448.875GiB写入未复用，Fair真实前缀恢复524,288tokens，支持继续这一候选，停止条件A。P95未达10%、第三BG平均响应后段间隔恶化、同host池未降低RSS、单seed/顺序/非生产和不完整生命周期oracle均保留；不宣称SOTA或稳定性，不删除旧负对照，不继续旧JIT调参。连续4.191988/用户批准4.5GPUh，只完成冻结组。下一轮先反序复验与少量regime验证，再决定动态额度。

- 2026-10-12 独立审查eead834：复算三方法原始请求计时/记录哈希配对和迁移来源，确认Fair相对NoHost/Full会话E2E−16.44%/−18.40%，接受初步整体正结果。保留单run、配置差异与后台输出阶段代价；未重算payload SHA/完整oracle或新增GPU。下一目标为自主反序复验和少量收益边界，遵守仅完成本组三方法的4.5GPUh授权范围。详见docs/RESEARCH_REVIEW_LARGE_MODEL_20261012.md及results/research_review_large_model_20261012/spotcheck.json。

- 2026-10-12 用户批准分阶段系统架构交接：当前优先实现动态前缀基本份额、弹性借用与原生安全尾部回收（M1），随后压力切换（M2），有明确机会再加可选预取（M3）。复用compiled/native offload，允许成熟组件，不把旧反序多种子矩阵作为开发前置条件。文档docs/SLACKKV_SYSTEM_ARCHITECTURE_20261012.md；AGENTS新增当前入口与优先级，旧证据保留。仅文档交接，未实现运行时或新增GPU；数字预算未扩大，代码/CPU验证可先自主推进。

- 2026-10-12 接纳599b89b执行侧架构审查，原子store/reclaim、逐批准入、物理/逻辑结算分离、共享降级计费和write游标约束合并至主架构§9。审查侧检查归档manager/LRU/TP scheduler，未执行新测试或GPU。M2可与M1已有全存行为合并，M3保持可选。下一实质交付为M1实现和CPU/native验证，不再开启纯文档审查循环；预算不扩展。


- 2026-10-12 M1原子适配决策：采用真实CPUOffloadingManager subclass而非另建物理缓存；在同一个store/reclaim锁入口使用actual free+net new，禁止受控store触发任意LRU驱逐。冻结请求上限可缩不可在本请求内随目标增长；不保留未物化整计划承诺。最深引用受保护时不能删祖先，过期generation通知仍完成物理结算，原生cursor不回退。理由：避免旧ceiling挤掉新成员、pending=-1误free和共享无计费Full回退。替代方案整计划reservation/单次缩额后任意LRU/热切换Full未采用。支持single full-attention LRU TP，其他布局拒绝；共享/未知/分支保守停止optional writes，read保留native身份。session end在scheduler线程显式传递，暂无TTL或生产tombstone GC；动态份额不等于延迟公平。CPU/native验证通过，但GPU内容与性能仍UNKNOWN，按新授权再做smoke。详见docs/SLACKKV_SYSTEM_M1_REPORT.md。


- 2026-10-12 M1修正交接（审查基线92e65cb）：独立执行原始控制器确认 end/finish/collect 后同代迟到请求重新获准（reason=None, ceiling=2, allowed=2）。指导执行侧保留独立结束状态、添加针对性控制器/native回归，并准备内容/生命周期与M1对StaticFair的小规模GPU验证；沿用架构，不扩M2/M3，不要求先做反序多种子。原64项及native审计未由审查侧重跑，本地pytest不可用；本次无代码修复/GPU运行。交接docs/SLACKKV_SYSTEM_M1_CORRECTION_20261012.md。新增0.5GPUh仍为待批准建议，先完成代码/CPU/脚本，若执行会话已有新授权则遵循之。


- 2026-10-13 M1终止契约修正：每session只保存latest generation及其终止bit；终止有效期独立于物理Member GC/cache reset。同代迟到请求仍走native合法读取/正常计算但停止optional store；更高代显式重启；重复/旧end不影响新代，未见/未来end no-op。原因是92e65cb在collect后丢失结束状态，新旧控制器/native回归已复现并修复。不采用逐代永久Member、TTL或提前丢弃DMA完成。smoke固定账本根/全policy共享额度、工作及清理计费，内容与性能模式分开；资源可行性先CPU核对，512MiB GPU KV配18432 max length，不沿用不适配的32K limit。M2/M3不启动，GPU内容和动态收益无实测，需新数值授权。


- 2026-10-13 M1 GPU闭环决策：修复owned服务器PATH、补齐禁用的继承scheduler模块依赖、使用vllm logger层级保证trace可见；没有修改旧数据或KV策略来强行通过。GPU内容gate只承诺本输入与正常结束路径，取消/强制交错及bitwise数据未全面验证。第一M1/Fair整体负结果保留，因首真实请求额外成本做一次反序复验并补NoHostrepeat，仍保持全部首请求计入，不给复杂SOTA/稳定性能结论。全协议M1额外224MiB terminal无重用写入是可优化候选而非本次成果；会话阶段和全协议流量分列，不换主指标隐藏成本。M1可进入下一次受控strongFull比较，先统一预注册warmup和更强身份oracle/共享prefix，再决定压力反馈或预取；M2/M3现在不启动。


- 2026-10-13 独立审查82bf363：CPU复算7个性能run原始请求/解压输入哈希及三处prom，确认M1/Fair median +2.83197%/-6.01401%、来源守恒和H2D=external×28672；核对content4原始16IDs/cold零命中/35状态快照，独立复现终止generation修复。接受M1基础实现及被测GPU路径，不接受稳定胜Full；long3 Full与M1均约42ms，末尾224MiB写入不在原会话E2E范围。下一步小幅terminal准入优化、等价warmup/不同marker oracle、单个多会话竞争筛选；共享覆盖先于无证据M3，不重写主架构、不要求大矩阵。已知新增0.5GPUh授权有效，已用0.13476591384806347，按原范围/账本执行，不重复申请已有许可。审查新增GPU0，未重跑服务器测试/native或完整manifest。详见docs/RESEARCH_REVIEW_M1_GPU_20261013.md，复算results/research_review_m1_20261013/spotcheck.json。


- 2026-10-13 M1 bounded net-benefit：采用显式、默认关闭的terminal optional-store gate；在完整观察/读取后仅拒绝未来新增写入，仍按真正完成结束。Full/StaticFair提供同信号native gate，避免通用优化归因于动态max-min。固定五形状非正式输入预热，不清GPU，首正式请求完整计入。单次四会话资源竞争中M1/Fair E2E median仅−1.79%而D2H多280MiB，M1/Full−7.40%且Full host有效恢复低；故保留M1 opt-in，不默认替换StaticFair，不扩JIT或大矩阵。共享覆盖优先于M2/M3；未测并发公平/生产/SLO和新NoHost不作收益主张。失败oracle和目录shadow bug均保留。详见SLACKKV_SYSTEM_M1_NET_BENEFIT_20261013.md；沿用0.5授权及原锁/账本，累计0.21011529948083787GPUh，未扩预算。


- 2026-10-13 独立审查375c28e：CPU复算三性能run输入哈希、warmup、原始计时、起止prom及双marker内容记录，M1/Full median−7.3951%、总DMA−35.7966%；M1/Fair median−1.7908%、总DMA+74.0057%，认可独立前缀M1工程里程碑和本筛选正结果，动态仍opt-in。独立controller复现terminal-only加入使A目标512→256→512；结合native先observe/reclaim后禁写的路径，要求补齐不新增未来保留需求语义，未执行该反例native/GPU。发现性能trace_states=true/M1有57详细快照，开销未测，下次性能测量调整，不为此单独大复验。下一实现增量优先有限共享system-prefix+私有后缀的单次物理计费/安全释放，M2/M3关闭，当前场景不再追参。审查新增GPU0，未重跑完整服务器测试或manifest；既有0.5授权累计0.21011529948083787不重置。详见docs/RESEARCH_REVIEW_M1_NET_20261013.md，results/research_review_m1_net_20261013/spotcheck.json。


- 2026-10-13 M1 bounded shared-root：采用有界共享root与private tail、多逻辑key_refs，目标按native key集合并集max-min计费，真实free/net-new gate和pending/load祖先闭包回收，避免共享重复收费或一会话结束删除其他前缀。terminal仅更新身份路径，不扩大future demand/无效转余数，完成前保留旧缓存可读；默认共享关闭，超界同私有历史/merge/fork仍fallback。性能StaticFair公共root计一次、固定private份额，不用重复计root的弱基线；详细状态只在content，性能三个方法均两个公共checkpoint。四会话单次M1/Fair median−0.87%但总DMA+93.52%，保持动态opt-in、StaticFair可用，不扩大旧矩阵。失败oracle全部保留、完整16IDs比较未放松，最后严格肯定文法拒绝否定/EOS后答案；共享内容验证是限定身份检查，不是逐字节证明。报告SLACKKV_SYSTEM_M1_SHARED_20261013.md；0.5授权累计0.3195768596469497GPUh，没有扩预算或挪旧4.5例外。


- 2026-10-13 system_m1_shared归档补充：content4固定机制路径可定位A初次512→256块尾部112MiB在首次恢复前回收，B末段192→396块补写89.25MiB后仅一次B_end全6336GPU-local/host0再结束；共201.25MiB未复用写入，原始状态/prom/来源证据在unused_write_evidence.json，不作性能全成本/生产上界/删除反事实归因。另修正共享deferred独占尾部诊断低报（含公共根target不能直接从独占数减），audit4新增回归后最终29组PASS，无分配/回收/DMA改变，旧GPU快照原样；原文件总435校验。分析器按cohort筛选，重复同policy不再隐式选最后一run；预算仍计所有family。最终86项相关测试PASS，额外GPU0，累计0.3195768596469497不变。


- 2026-10-13 独立审查c07976b：复算共享三性能run原始计时/输入哈希/起止prom/来源/checkpoint及content4三组16IDs与70状态守恒，确认M1/Fair median−0.87355%、DMA+93.51852%；M1/Full−2.39971%、DMA−13.87363%。输入/顺序/Fair规则改变，不能以旧百分比判断版本退化。接受有限共享与terminal功能，动态仅保留可选，root-aware StaticFair为工程参考。下一步只做一次有上限CPU/已有事件成本定位，核查shared重复祖先遍历，测得明显开销才做等价优化，有在线信号才提出最多一个候选；否则停止该动态变体性能优化，不扩矩阵/M2M3。201.25MiB事后未复用不是在线可免费消除的上界。审查新增GPU0，服务器完整测试/native/manifest未重跑；既有0.5授权已用0.3195768596469497。详见docs/RESEARCH_REVIEW_M1_SHARED_20261013.md及results/research_review_m1_shared_20261013/spotcheck.json。


- 2026-10-13 system_m1_cost（5495a36交接）：一次有界CPU/已有事件定位，未调参数/M2M3关闭。真实native元数据固定状态计时确认共享保护祖先重复遍历：单512保护块约9→0.375ms，空闲不变；needed与已有保护链祖先闭合，新增key-not-in-keep条件，保持分配/引用/free/event等价。原/候选29组审计完整结果一致、各75次shared回收；安装版本29组PASS；84项相关本地回归PASS。现有70状态定位1228写/回收块537.25MiB，70MiB为前缀深度推断的重写，与201.25MiB事后未复用分类重叠，不作key级/性能全归因。复用历史门控不能排除已有真实host复用的末段B扩额，未来GPU/end不可在线使用；未推出新策略。真实contentPASS，3cold/恢复完整16IDs一致、严格有限语法离线复核、共享生存与最终512free。冻结原M1/修正版/StaticFair各一次compiled同输入19请求153088/304：median5341.393/5365.992/5466.442ms；修正版/原median+0.4605%、序列+0.2862%，修正版/Fair median−1.8376%而DMA+93.5185%。CPU局部改善未转为可确认整体加速，停止当前动态变体性能调参，保留optional组件/StaticFair参考，无部署切换。CPU后分析2失败（缺报告证据、旧依赖）保存，不重跑GPU；268原文件哈希与2runner源码hash复核，跨Python求和末位公开。新GPU0.05931228920166985，原0.5账本累计0.37888914884861963、历史4.570877168576147，四新run全部COMPLETE/cleanup、两卡19MiB/0%。CPU真实GPU关键路径累计/阻塞迁移/并发后台公平性UNKNOWN。报告docs/SLACKKV_SYSTEM_M1_COST_20261013.md，复现/原数据results/system_m1_cost与原system_m1_smoke新四目录。提交并push后结束本交接。


- 2026-10-13 research_direction_after_m1（用户收尾M1后的新任务，基线2cdf515）：CPU机会判断，建议暂缓新系统，保留32BStaticFair正结果/生命周期工具/全部历史，不动默认、不开发M2M3/即时动态额度。独立重校82归档文件84782502原字节、三方法198请求/哈希/128输出及prom/时戳；固定私有初始前缀、只首次写入且两次复用的future-aware token分配参考上限524288=Fair实际external，额外0；仅该条件类的token界，不是rolling/compute/E2E全局上界。无限KV输入连续性参考1214976可复用、Fair仍重算690688；不推导可达加速。公共hash代理SHA d078503...重新统计2661调用/65container-model组，calls median20/P95220/max311，无完整>=3调用组满足旧32Kproxy输入+输出；不是Qwen实tokens/线性Agent，不能挑短片段制造新实测。读取EfficientAgent准入/实现条件、RR-Evict公平tail/结束回收、MORIsticky tier及KVFlow图预取；不把普通组件当主创新。唯一条件候选为异质长期会话持久部分前缀价值分配，缺强公平baseline损失/在线区分/成本曲线，未立项PoC；仅提出一个带事前token机会/在线信号停止门槛的CPU最小验证方案，当前不申请GPU。8项新CPU/归档回归PASS；GPU新增0，历史4.570877168576147及M1授权已用0.37888914884861963不变。报告docs/RESEARCH_DIRECTION_AFTER_M1.md，证据results/research_direction_after_m1/evidence_verified.json；无历史文件覆盖。提交push完成本阶段。


- 2026-10-16 infra_engineering_handoff：用户明确转为面向 infra 实习的完整可运行项目。新增 docs/INFRA_ENGINEERING_HANDOFF_20261016.md 并更新 AGENTS.md 当前入口；StaticFair 工程主线、NoHost/Full 对照、M1 opt-in，不继续 M2/M3。E1 统一启动/真实多轮/报告，E2 可靠性与观测/小型对比，E3 复现与面试讲解。任务允许范围内自主实现，E1 工程 smoke 自设0.5GPUh上限、单次30分钟含清理，新账本累计历史而不挪用旧专用余额。此次仅文档交接，GPU新增0，没有已实现或已测工程版本的声明。旧证据与配置不覆盖。


- 2026-10-16 infra E1（d38e1b8工程交接）：固定上游0fc695fc6完整Python树+匹配编译库隔离overlay，原生max_offload_tokens+薄terminal spec；CLI/配置/doctor/owned进程/固定槽位/真实三轮历史/本地JSON工具/报告实现。18CPU回归、6native CPU契约PASS、wheel构建成功。两真实demo，各2session×3turn且有交叠；最终host3 marker3018:3023处于host2304:4096，H2D49MiB/D2H112MiB、1792external，独立cold8192compute/零命中、完整IDs一致。host1失败为HF5 BatchEncoding长度误判，修正/预检提前并保留失败；host2弱marker检查不作为最终敏感验证。102原文件校验。5GPUrun累计0.052477931313624139/0.5工程GPUh、历史4.6233550998897712，含启动/清理、全部cleanup，未挪旧预算/未改环境。独立close仅逻辑；单Pool限制，不宣称网关/共享/生产SLA。报告docs/INFRA_E1_STATUS_20261016.md，数据results/infra_engineering_20261016。先提交E1，再按已授权交接继续E2/E3。


- 2026-10-16 infra E2（基线b6773c7）：工程固定三用途hot/host-enough/host-limited各NoHost/Full/Fair一次，无研究矩阵/调参；冻结token输入、16实际输出、同case同资源/预热、实际GPU18720tokens，九run全部COMPLETE/交叠。实际source/bytes和JSON/CSV/MD完成，hot0host恢复不归因offload；enough Full/Fair相同计数，limited Fair全序列3369ms vsFull3590/NoHost3779，单次不承诺稳定加速。可靠性真实首段SSE后断流、失败历史1system、换generation成功、scheduler idle、独立stop确认STOPPED/owner0/cleanup；server abort vs自然完成UNKNOWN。修复timeout后线程池等待reader风险，先退回owned service cleanup；PID出生身份/零PID保护，requested stop与失败区分。Windows65PASS3SKIP，Linux27PASS（含实际小owned进程与native CPU、0GPUh）；CI配置不冒充远端已绿。261raw/hash/输入/usage/source/迁移/源代码/cleanup独立复算。E2新增0.092996380345834448GPUh，工程累计0.14547431165945859/0.5，历史4.7163514802356055，15run均清理，GPU1未用。E1最终UTF8新增回归后19项（早期日志18为该补充前）；保留全部失败。报告docs/INFRA_E2_STATUS_20261016.md，数据results/infra_engineering_20261016。提交E2后继续E3纯CPU/交付文档，不加GPU。


- 2026-10-16 infra E3（E2基线c45cd25）：完成README/架构/运行排障/面试讲解/动手清单与总交付，明确上游和AI辅助边界。修复terminal已提交但pending未结算时的槽位复用；报告支持gzip指标及独立output保留历史；上游hash仅排除构建生成顶层_version.py，donor版本/目标仍严格校验。Windows68PASS3SKIP，Linux最终71PASS（native CPU禁CUDA）；最小上传遗漏旧夹具初次61PASS10FAIL，补齐后通过，失败日志保留。离线wheel隔离安装、目录外CLI/工具数据/报告/help共7检查PASS；261原文件与15run账本独立复算PASS，重建归档demo报告成功。最终runtime1806Python与SO manifest/doctor通过，两卡19MiB/0%，E3GPU新增0；工程累计0.1454743116594586/0.5、历史4.7163514802356055不变，不再启动实验。干净机从零构建全vLLM尚未重做；非全局网关/物理硬分区/即时close，UNKNOWN保留，最终E3CPU修正不冒充重跑GPU。交付docs/INFRA_DELIVERY_20261016.md，证据results/infra_engineering_20261016/e3/validation.json。分阶段提交push后结束本次工程交接。
