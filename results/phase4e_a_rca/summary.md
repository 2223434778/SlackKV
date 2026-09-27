# Phase 4E-A-RCA — lease-induced head-of-line liveness diagnosis

Result: `RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED`. This is a controlled mechanism diagnosis, not a performance comparison.

Frozen Phase 4C hybrid workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; GPU KV: 4,681 blocks; first-boundary jobs and deadlines unchanged.

The pinned vLLM waiting path calls `allocate_slots()`, and when it returns `None`, executes `break` (not `continue`). The RCA patch adds bounded structured logs immediately before that existing break. It changes no scheduler branch, queue order, allocator, lease, guard, or K2 semantics.

| Stage | Condition | Reps | Classification | E04 first allocation failure free blocks | Finished requests |
|---|---|---:|---|---|---|
| R1 | U3_UNLEASED_HEAD_LEASE3 | 2 | LEASE_HOL_NONPROGRESS, LEASE_HOL_NONPROGRESS | 750, 750 | 0, 0 |
| R1 | H2_HEADROOM_LEASE2 | 2 | PROGRESS, PROGRESS | NA, NA | 8, 8 |
| R1 | N3_EVICTABLE_PREFETCH3 | 2 | PROGRESS, PROGRESS | NA, NA | 8, 8 |
| R1 | S3_HEAD_LEASED | 2 | PROGRESS, PROGRESS | NA, NA | 8, 8 |
| R2 | U3_UNLEASED_HEAD_LEASE3 | 3 | LEASE_HOL_NONPROGRESS, LEASE_HOL_NONPROGRESS, LEASE_HOL_NONPROGRESS | 154, 154, 154 | 0, 0, 0 |
| R2 | S3_HEAD_LEASED | 3 | PROGRESS, PROGRESS, PROGRESS | NA, NA, NA | 8, 8, 8 |

## Direct scheduler evidence

- R1 rep 1: E04 `WAITING_ALLOCATE_NONE`, free=750 < required=1271, matching_lease=False, active leases=3 / unique leased blocks=3930, will_break=True.
- R1 rep 2: E04 `WAITING_ALLOCATE_NONE`, free=750 < required=1271, matching_lease=False, active leases=3 / unique leased blocks=3930, will_break=True.
- R2 rep 1: E04 `WAITING_ALLOCATE_NONE`, free=154 < required=1271, matching_lease=False, active leases=3 / unique leased blocks=3930, will_break=True.
- R2 rep 2: E04 `WAITING_ALLOCATE_NONE`, free=154 < required=1271, matching_lease=False, active leases=3 / unique leased blocks=3930, will_break=True.
- R2 rep 3: E04 `WAITING_ALLOCATE_NONE`, free=154 < required=1271, matching_lease=False, active leases=3 / unique leased blocks=3930, will_break=True.

The U3 precondition sequentially leases E00/E01/E02 (3,930 blocks) before t0, while E04 is not promoted or leased and has the earliest 120-ms deadline. In U3, E04 reaches the existing waiting-loop allocation failure and break; later leased resumes do not acquire real-request ownership, the leases do not release, and headroom does not recover during the 12-s observation. This conclusion is limited to the observed pinned scheduler/workload and the control outcomes below.

At/after t0+8 s in the R2 U3 repetitions, later rate-limited E04 allocator-failure traces still show free blocks 750 (<1,271), despite initial X pressure subsiding. Exact X finish timestamps were not separately persisted, so this is supporting evidence rather than an X-finish-time measurement.

- U3_UNLEASED_HEAD_LEASE3: 0/2 complete all eight requests; leases=3930 blocks.
- H2_HEADROOM_LEASE2: 2/2 complete all eight requests; leases=2607 blocks.
- N3_EVICTABLE_PREFETCH3: 2/2 complete all eight requests; leases=0 blocks.
- S3_HEAD_LEASED: 2/2 complete all eight requests; leases=3822 blocks.

All 5 U3 runs observed 8 Agent arrivals and 0 admission, 0 first token, 0 completion, and 0 `LEASE_RELEASED` at the 12-s checkpoint; all servers remained alive without OOM or server ERROR. The later leased E00/E01/E02 requests therefore never reached real-request ownership behind E04.

In S3, E04 had `matching_lease=true` at its first waiting attempt; E04 `LEASE_RELEASED` preceded `RESUME_ADMITTED` in 5/5 runs, and all 5 runs completed 8/8 requests.

## Interpretation

The controlled comparison supports lease-induced waiting-head circular wait: U3 leaves E04 unleased and without enough allocatable blocks; the pinned waiting-loop break prevents leased requests behind E04 from acquiring ownership and releasing their leases. H2 restores headroom, N3 makes the same prefetched KV evictable, and S3 gives E04 a matching lease; each control restores progress. The X-background repetitions reproduce U3 and S3 outcomes. Instantaneous promotion fit alone does not guarantee a future resume progress path.

No P95 policy winner, H1/H2/H3, progress-reserve fix, or Phase 4E performance matrix was run.
