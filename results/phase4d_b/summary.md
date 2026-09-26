# Phase 4D-B — Runtime Physical-Headroom Admission

**Result: `RESULT_A_PHYSICAL_HEADROOM_RESTORES_LIVENESS`**

The exact-free-block guard passed both physical-boundary cases and the frozen hybrid integration completed successfully with real capacity rejections. This is a mechanism/liveness validation only; it is not a performance comparison and does not establish that capacity is the sole possible cause of every prior NaiveJIT failure.

## Frozen inputs and mechanism

- SlackKV start commit: `c7921d93f873ceb6ec7d6103574607bcd2d3acdf`.
- Frozen Phase 4C workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; 8 boundary-0 jobs, 168,016 cached-source tokens, 10,501 blocks, 4,594.1875 MiB minimum H2D.
- Runtime remained Qwen2.5-1.5B-Instruct, vLLM 0.23.0, FP16, TP=1, 32K context, 2 GiB GPU KV, 8 GiB CPU tier, block size 16, K2=2. Runtime reported 4,681 GPU blocks; logical future budget remained 3,510 blocks; P_SAFE remained 102 ms.
- Authoritative physical API: `Scheduler.kv_cache_manager.block_pool.get_num_free_blocks()`. In vLLM's `BlockPool`, `num_gpu_blocks` includes the non-allocatable null block; the free queue count is the allocatable headroom. That explains the observed one-block difference from simple total-minus-owned arithmetic and is why admission uses the API's returned free count directly.
- The guard runs in `Scheduler.schedule()` before connector-first preflight/load and before request allocation. It validates prompt alignment and `required_blocks == len(prompt_token_ids)/16`; then it admits iff exact free blocks are at least required blocks. Rejection finishes the research prefetch request through the prefetch completion channel without a model token or connector load.

## G1 physical threshold

| Case | Repetitions | Exact free blocks | Required | Decision/status | Return | Transfer / rejected lease |
|---|---:|---:|---:|---|---:|---:|
| Safe side, E06 after E04+E05 leases | 2/2 | 2,135 | 1,276 | ADMIT / `ok` | 56.30–58.66 ms | Candidate lease acquired as expected |
| Unsafe side, E00 after E04+E05+E06 leases | 2/2 | 859 | 1,277 | REJECT / `capacity_rejected` | 21.29–22.86 ms | 0 bytes / 0 rejected-request leases |

Both sides met the predeclared gates. For both rejected E00 calls: external tokens 0, transfer bytes 0, forwards 0, generated tokens 0, and no lease.

## G2 frozen integration

The one smoke plus three fresh-server repetitions all passed (4/4 runs). Each completed 8/8 resumes and token counts, preserved source conservation, stayed within K2, had no server crash/OOM, and ended with no pending guarded prefetch, logical reservation leak, physical lease leak, unsafe release, or reservation/lease overcommit. Across the four valid runs, 8 promotions were admitted and 12 were rejected; the three measured repetitions alone had 6 admits and 9 rejects.

- E04 and E00 were admitted on every run.
- E06 was rejected every run: required 1,276 blocks; exact free count 81–83.
- E07 was rejected every run: required 1,444; exact free count 76.
- E03 was rejected every run: required 1,306; exact free count 169.
- Every capacity rejection immediately released its logical reservation and followed OnDemand fallback. Rejected request lifecycle checks found no `WAITING_FOR_REMOTE_KVS`, connector-only step, lease, transfer, or forward.
- The prior 50k+ connector-only non-progress pattern did not recur in the valid guarded runs. Successful promotions produced only 15–16 connector-only step records per run and reached T6; rejected request IDs contributed none. No prefetch exceeded the 5-second watchdog.
- K2 maximum pre-first concurrency across the four valid runs was `2` (limit 2).

## Service envelope

The 8 successful promotions in the valid integration runs had issue→T6 min/median/P95/max of **77.51/84.78/92.22/92.22 ms** (empirical nearest-rank P95). **0/8** exceeded frozen P_SAFE=102 ms, so `P_SAFE_102_INTEGRATION_ENVELOPE_PASS`. No recalibration was performed.

## Interpretation and scope

The runtime guard prevented promotion requests from entering connector load when the engine's exact allocatable KV headroom was insufficient; in this frozen workload those sessions proceeded through OnDemand fallback and all requests completed. This demonstrates that logical reservation capacity alone does not imply physical free capacity after lease handoff. It does not establish that physical capacity is the only possible cause of all NaiveJIT non-progress, nor that active occupancy is solved for all workloads.

No Phase 4D-A Stage B or performance matrix ran. No latency/throughput superiority claim is made. Failed harness attempts are retained with `_attempt*_incomplete` / `_attempt*_validation_incomplete` names and excluded from valid G2 counts.
