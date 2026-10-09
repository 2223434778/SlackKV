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
