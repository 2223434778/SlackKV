# SlackKV Phase 4C: Real-Structure Hybrid Replay Correctness

**Final result: `RESULT_A_HYBRID_REPLAY_CORRECT`**

## Objective and scope

This phase tested replay fidelity only: real coding-agent ISL/ISL_new/OSL structure, deterministic synthetic token IDs, empirical TraceLab tool-phase timing proxies, and real vLLM local prefix caching. It did not test SlackKV performance or compare serving policies.

This is a hybrid trace-driven replay using real coding-agent token/KV structure and empirically observed TraceLab tool-phase timing. It is not a request-timestamp-faithful replay of any single real Agent session.

## Source population and coverage

Pinned HF revision: `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`; TraceLab v0.0.2 SHA256: `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744` (PASS). Fixed population: offsets 0–99 in each GPT-OSS config; no rows beyond 99 were inspected.

- Pro: 87/100 eligible; 4 selected. Sessions with at least five turns: 100/100.
- Verified: 71/100 eligible; 4 selected. Sessions with at least five turns: 100/100.
- Combined eligible: 158/200 (79.0%). The selected workload is eight sessions, 40 requests, and 32 validated inter-turn boundaries.

Exclusion flags below are non-exclusive because a session can have multiple disqualifying windows: gpt-oss-120b-pro: invalid_token_accounting=13, context_limit_violation=11, osl_cap_violation=12, prefix_feasibility_violation=0, context_shrink=0; gpt-oss-120b-verified: invalid_token_accounting=29, context_limit_violation=27, osl_cap_violation=22, prefix_feasibility_violation=1, context_shrink=1.

The selected 8-session subset is a correctness workload, not a representative sample of the full coding-agent population.

## Workload construction

For each source turn, reusable source cache structure is `isl - isl_new`. With measured vLLM block size 16, the target is `floor(source_cached_tokens / 16) * 16`; alignment loss is always 0–15 tokens. The synthetic prompts contain exactly source ISL token IDs, and each continuation generates exactly source OSL tokens. Every following prompt reuses exactly the target full-block prefix from the preceding served sequence, then diverges with a deterministic unique suffix. Cross-session and older-turn accidental full-block matches were checked offline.

Selected prompt lengths: min 20336, median 21662.0, max 25527; selected OSL min 33, median 69.0, max 496.

## TraceToolProxy timing

DuckDB tables and fields were inspected. Tool-phase proxy uses, per `round_pk`, latest complete result timestamp minus earliest complete emission timestamp. Positive valid rounds: 563,809; retained 100–3000 ms rounds: 246,773 (43.8%). This bounded pool is not the full real tool-duration distribution.
Full observed distribution P50/P90/P95: 366.0/14510.2/30168.0 ms. Retained pool P50/P90/P95: 471.0/1818.0/2338.4 ms.
Across Stage B's 32 pauses, absolute injection error P50/P95/max was 5.10/7.36/7.96 ms (frozen limits 10/30 ms: PASS). These are controlled timing proxies, not observed resume slack or real resume deadlines.

## Prefix metric calibration and fidelity

Stage A0: 1024-token priming local hit = 0; 768-token shared case = 768 tokens / 48 blocks; 770-token shared case = 768 tokens / 48 blocks; external hit = 0; block size = 16. PASS.
Stage A1: 10/10 exact prompt lengths, exact OSL counts; 8/8 exact cache-block hits; both priming turns had zero hits; external hit tokens zero. PASS.
Stage B: 40/40 requests completed; exact prompts 40/40; exact generation lengths 40/40; exact boundary blocks 32/32; priming isolation zero-hit 8/8; external cached tokens zero 40/40. PASS.

The cache target is block-aligned, not raw token-level equality: source `isl - isl_new` can be unaligned, so fidelity is judged against complete 16-token blocks.

## Runtime and anomalies

The server used the existing local model snapshot, vLLM 0.23.0 / pinned source commit 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665, FP16, TP=1, GPU0, prefix caching, 4 GiB GPU KV, and max model length 32768. No external KV connector, offload, promotion, lease, reservation, K2, background request, or pressure prompt was enabled. No vLLM source file was changed.
Two server startup attempts before valid Stage A0 were corrected: an unsupported logging CLI flag, then a missing runtime PATH entry for `ninja`; neither reached a measured request stage. The valid A0/A1/B servers completed without a crash or OOM. Stage A1 attempt 1 had correct token/cache fidelity but pause errors around +14–17 ms because metric polling sat between wakeup and send; the metrics baseline was moved before the wait and the same frozen A1 then passed. The first Stage B artifact was similarly rerun after external-hit deltas were changed from a configuration assumption to measured counter deltas; the final artifact is the second fresh Stage B run.

## Interpretation

Real coding-agent token/KV structure was faithfully reconstructed at vLLM cache-block granularity. The replay uses empirically observed tool-phase durations as controlled timing proxies. It does not claim real end-to-end Agent timing or SlackKV performance superiority.
Request E2E times were recorded only for debugging/sanity and are not presented as comparative performance results.

## Next-step status

Phase 4C technically completed with hybrid replay fidelity. No Phase 4D or performance comparison was started.
