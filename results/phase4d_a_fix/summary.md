# Phase 4D-A-FIX — Runtime Release & Naive Liveness Audit

## Outcome

Final status: `INCOMPLETE`; conservative taxonomy: `RESULT_D_RELEASE_FIX_FAILED` because the required Reservation integration acceptance gate did not pass. The targeted exact-identity release correction itself is directly observed to work; this status does **not** mean that the old E04/E04:turn:1 key mismatch remains.

No Phase 4D-A Stage B matrix ran. Naive L0/L1 were not started because the protocol requires stopping when the Reservation integration gate fails. No Naive capacity-crossing cause is established.

## Frozen inputs and runtime

- Start commit: `86ac84a94b671514c78b69aeb3dd294e137a03a8`.
- Phase 4C workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`.
- 8 frozen jobs, 168,016 cached tokens, 10,501 blocks, minimum H2D 4,594.1875 MiB.
- Runtime re-reported block size 16 and 4,681 GPU KV blocks; reservation budget remains `floor(0.75×4681)=3510`. Frozen `P_SAFE=102 ms`, guard 110 ms, K2=2. No vLLM algorithm changes.

## Q1 — logical reservation release

The previous controller tracked logical reservations as `E04`/`E00` but physical events used `E04:turn:1`/`E00:turn:1`. Exact string lookup by the episode ID therefore missed `LEASE_RELEASED`. The fix keeps logical IDs in the ledger and maintains an explicit episode→physical lease-key mapping; a structured JSON lease parser now releases logical capacity only after an exact-key physical release with an earlier/equal monotonic timestamp. Duplicate drains are idempotent. The loop drains before every later reservation decision and after resumes.

Runtime evidence from the third fresh attempt:

| Episode | Physical lease key | Physical release | Logical release | Logical − physical | Blocks returned |
|---|---|---:|---:|---:|---:|
| E04 | `E04:turn:1` | 159.19 ms | 250.69 ms | 91.50 ms | 1,271 |
| E00 | `E00:turn:1` | 219.40 ms | 460.55 ms | 241.15 ms | 1,277 |

Both exact physical release events preceded logical release. The ledger fell from 2,548 to 1,277 and then 0. Later E06 (1,276 blocks) and E07 (1,444 blocks) were both promoted after that released capacity became available. This confirms identity matching and actual capacity recycling at the controller decision level.

However, the integration acceptance gate failed: 8/8 resumes arrived, only 3 were admitted/reached first token, and there were 0 `REQUEST_FINISHED` events. Four promotions were issued (E04, E00, E06, E07); E04/E00 returned actual `status=ok` endpoint records, while E06/E07 remained pending and timed out on the client after 120 s. The server remained alive without OOM/crash and logged 54,252 connector-only steps in the retry2 interval. The harness then stopped the server; the shutdown-time HTTP 500 is not treated as an independent root cause. Because the run was interrupted, final source conservation, full token counts, and terminal logical-reservation leak status are not evaluable; E06/E07 had 2,720 logical blocks outstanding at abort, not a proven post-completion leak.

## Q2 — NaiveJIT liveness isolation

The mechanically sorted first four jobs are E04/E05/E06/E00 with cumulative requirements N1=1,271, N2=2,545, N3=3,821, N4=5,098 blocks; thus `3821 < 4681 < 5098`. These are the planned isolated cases only. L0 and L1 were **not run** because the required Reservation integration acceptance gate failed. Therefore there are no N=1/2/3/4 Naive outcomes, no clean capacity-boundary test, and no claim that capacity crossing caused Naive non-progress.

The narrow diagnostic observation is only that, in this Reservation/lease integration, later E06/E07 native requests did not return within 120 s while two connector-only requests continued to be scheduled. This is not the specified unleased Naive experiment and does not isolate its root cause.

## Service-envelope audit

`P_SAFE=102 ms` was not changed. Of the two completed runtime endpoint records, issue→T6 was 98.50 ms (E04) and 107.24 ms (E00); one of two observed records exceeded the budget by 5.24 ms. E06/E07 had no returned T6 values. Accordingly the frozen status is `P_SAFE_102_INTEGRATION_ENVELOPE_FAIL`; no recalibration was performed. This service finding is separate from the logical release ordering result.

## Tests and interpretation

Focused tests: 35/35 passed, including exact turn-qualified mapping, E04/E040 non-match, physical-before-logical order, no-release/no-drain, duplicate-drain idempotence, capacity reuse, frozen workload/capacity, watchdog/crash/OOM classification, and preventing Stage B invocation. The benign environment Python emits a missing optional `nvidia_cutlass_dsl` `.pth` warning; tests and compilation succeed.

No performance conclusion is made. The liveness root cause remains unisolated, and Naive L0/L1 were deliberately withheld by the integration gate.

Artifacts: `results/phase4d_a_fix/`.
