# Phase 4G-D — Protected-Head Persistence × Retry Opportunity × Frontier RCA

Final result: RESULT_B_PROTECTED_FRONTIER_POISONING_DOMINANT.
Start commit: 7bada3599076fa4fbf8e0d73120c37dbe6285ca0; workload SHA256: 9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105.

## Purpose and frozen behavior

This is a C1 source-recheck-only mechanism RCA, not a performance experiment. The current controller's candidate ranking, protected frontier, PSA-1, reservation, guard, lease, K2 and deadlines were left unchanged. Retry and F0–F4 frontier alternatives are shadow-only.

## Phase 4G-C-FIX allocator reanalysis

Total guard observations 123; exact-known 54; missing 69; known agreement 54; known mismatch 0; false-admit 0; false-reject 0.
Therefore the old `RESULT_D_GUARD_ALLOCATOR_MISMATCH` is not supported after missing-value correction. UNKNOWN samples are excluded from mismatch rates.

## Fresh runs

Valid runs 4/4; turns 160/160; boundaries 128/128.
Crash/OOM/logical-leak/lease-leak/unsafe-release/K2-violation counts: 0/0/0/0/0/0.
Source accounting uses the frozen prompt-only offload limitation: local+external=2819840 / expected=2822912 tokens; explained shortfall=3072, unexplained=0; all measured source-accounting gates pass=True. The 768-token per-run prompt-only shortfall is retained as a limitation, not treated as missing/unknown data.
Selections 126; terminal-head protection 121 (0.9603174603174603); poison episodes 128; poison-duration median/P95 4403.812685000001/6849.08935 ms.

Candidate matched protected 5/126 (0.03968253968253968); stale protected visibility / stale-source wrong candidate 0/0.
PSA-1 reserve median/P95/max 1348.0/1568.0/1584.0 blocks; protected-self H median 10.0; median actual-minus-self H 1329.0 blocks.
Terminal decisions occurred after resume start in 120 episodes and before it in 8; pre-resume/frontier windows ≥20 ms: 5, sampled: 5; samples/actionable samples: 5/0.
Duration-weighted union of poison intervals / workload span: 0.8061364156752863 (intervals are unioned per run to avoid double-counting overlap).

## Retry versus alternate frontiers

Retry-recoverable heads 0 (0.0); median failure-to-first-window None ms.
Frontier-recoverable selections F1/F2/F3/F4: 4/4/4/0.
4/7 exact-known current-F0 PSA-blocked selections were shadow-recoverable under at least one alternate frontier. This is a shadow-only counterfactual, not evidence that an alternate frontier is safe. F1/F2/F3/F4 recoveries: 4/4/4/0.
Poison episode types: Retryable=0, Frontier-only=4, Capacity-bound=1, PSA-intrinsic=0, Mixed/unknown=123.
F3 candidate-self is an upper-bound counterfactual, not a safe admission policy. The instantaneous allocator-fit proxy is not a future liveness guarantee.

## Allocator observability and need comparison

Eligible selections 46; exact-known 46; exact-unknown 0; coverage 1.0 (required ≥90%).
Exact allocator fit/reject fraction among known eligible selections: 0.2391304347826087/0.7608695652173914.
The exact allocator-fit/reject counts were 11/35 of 46 eligible selections; known legacy-guard disagreements were false-admit 2 and false-reject 0.
Need summaries use exact-known allocator rows for paired comparisons; missing values stay unknown. Instantaneous allocator need is not a future liveness guarantee and cannot replace PSA-1.

| Cohort | n | exact allocator n | Need_PSA1 median (all) | Need_PSA1 median (known) | allocator Need median | gap median | ratio median |
|---|---:|---:|---:|---:|---:|---:|---:|
| All selections | 126 | 61 | 1348.0 | 1340.0 | 1292.0 | 0.0 | 1.0 |
| candidate = protected | 5 | 3 | 18.0 | 18.0 | 1292.0 | -1274.0 | 0.01393188854489164 |
| candidate ≠ protected | 121 | 58 | 1359.0 | 1354.5 | 1288.0 | 0.5 | 1.0003681885125184 |
| no local cached KV | 93 | 30 | 1350.0 | 1343.5 | 1343.5 | 0.0 | 1.0 |
| local cached KV > 0 | 33 | 31 | 1340.0 | 1340.0 | 50.0 | 1307.0 | 26.7 |

For selections with any local protected KV, ownership snapshots are summarized below. Empty ownership observations are excluded rather than inferred.

| Local-positive ownership measure | n observed | median blocks | P90 blocks |
|---|---:|---:|---:|
| Protected local cached | 33 | 1306.0 | 1444.0 |
| Leased | 33 | 0.0 | 0.0 |
| Owned/refcounted | 33 | 1306.0 | 1444.0 |
| Evictable cached | 33 | 0.0 | 976.0 |

The local-positive group has an elevated median instantaneous PSA-minus-allocator gap, while local-zero observations have near-zero median gap. This is an observed association, not evidence that cached KV is safely reclaimable: the ownership snapshot distinguishes owned blocks from evictable cache where available.

Across the 61 exact-known paired protected-need observations, Need_PSA1 median=1340.0, allocator-need median=1292.0, gap median/P90/P95=0.0/1430.0/1444.0 blocks; PSA1/allocator ratio median/P90=1.0/191.0. Ratios are sensitive to small allocator denominators; local-positive rows show the clearest gap association.

| Metric | Value |
|---|---:|
| Selections | 126 |
| Terminal-head selections | 121 (0.9603174603174603) |
| Poison episodes; median/P95 duration (ms) | 128; 4403.812685000001/6849.08935 |
| Retry-recoverable heads; median first-window delay (ms) | 0; None |
| F1/F2/F3/F4 recoverable selections | 4/4/4/0 |
| Terminalized, later exact-feasible retry heads | 0 |

| Poison type | Episodes |
|---|---:|
| Retryable head | 0 |
| Frontier-only | 4 |
| Capacity-bound | 1 |
| PSA-intrinsic | 0 |
| Mixed/unknown | 123 |

| Allocator observability | Count |
|---|---:|
| Eligible selections | 46 |
| Exact-known / exact-unknown | 46 / 0 |
| Exact fit / reject among known | 11 / 35 |
| Known agreement / false-admit / false-reject | 44 / 2 / 0 |

## Timing and diagnostic overhead

CLOCK_NANOSLEEP_ABS pause |error| P95 range: 0.298999–0.87545 ms; max across runs: 1.08374 ms.
Combined diagnostic CPU duty by measured run: [5.297, 5.588, 5.728, 5.506]%.

## Conclusion

Priority interpretation: No continuous ≥20-ms RETRY_ACTIONABLE window was observed in the sampled data (5/5 long pre-resume/frontier intervals had probe coverage). This is absence of observed opportunity, not proof that retry can never help. 4/7 exact-known current-F0 PSA-blocked selections were shadow-recoverable under at least one alternate frontier. This is a shadow-only counterfactual, not evidence that an alternate frontier is safe. Exact allocator no-fit remains 35 of 46 eligible selections. Candidate-self PSA failed in 3/7 exact-known F0-blocked cases, below the frozen >50% PSA-intrinsic threshold; these data do not justify PSA-2.
The data may justify a focused safety evaluation of state-aware frontier alternatives before relaxing PSA, but do not establish their liveness safety. They do not by themselves justify partial promotion. No runtime retry or frontier change was tested.
The frozen classifier result is `RESULT_B_PROTECTED_FRONTIER_POISONING_DOMINANT`. Retry-first classification is applied before frontier, exact-capacity, and PSA-intrinsic explanations. No retries or frontier changes were applied to real requests. Candidate-self PSA fit is only an upper-bound counterfactual and does not prove future liveness; instantaneous allocator fit is not a future liveness guarantee.
No PSA-2, partial promotion, new reservation rule, or performance claim is made by this experiment.

## Artifacts

`phase4g_c_fix_reanalysis.json`, `protected_poison_episodes.csv`, `protected_retry_timeline.csv`, `protected_state_timeline.csv`, `frontier_shadow.csv`, `frontier_recoverability.csv`, `allocator_truth.csv`, `allocator_observability.csv`, `protected_need.csv`, `protected_need_summary.json`, `poison_classification.csv`, `example_poison_timeline.csv`, `protected_poison_timeline.png`, and `validation_report.json`.
