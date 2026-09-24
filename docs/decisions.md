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
