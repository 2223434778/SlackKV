# Phase 4D-A — integration gate stopped

**INCOMPLETE. No Stage B runs; no RESULT_A/B/C assigned.** The required integration gate did not pass because NaiveJIT-Async-K2 failed to finish all eight requests.

## Frozen setup

- Phase 4C workload SHA256: 9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105.
- Boundary 0, 8 episodes; cached tokens 168016, blocks 10501, minimum H2D 4594.1875 MiB.
- Runtime capacity: 4681 blocks; frozen 75% budget 3510 blocks (1535.625 MiB); at most 2 frozen jobs.
- Real-size calibration: 9/9 valid; max issue-to-T6 84.493 ms; frozen P_SAFE 102 ms. Stage A0 prefix calibration passed.
- X-only baseline: 3/3; median decode 3632.514 ms; median TPOT 14.245 ms.

## Integration only

- OnDemand: valid, 8/8 tokens, local/external 0/168016, K2 and X checks pass.
- Reservation: valid, 8/8 tokens, local/external 40768/127248, promotions 2, peak logical/physical 2548/2548 blocks, lease leaks 0.
- Naive: latest run had 5/8 completed native calls and 3 pending calls errored on shutdown; scheduler observed 8 arrivals, 3 first tokens, 0 finishes and 53935 connector-only steps before 120-second client timeout. This is consistent with unprotected residency non-progress, not a latency measurement.

## Stop decision

The frozen protocol requires eight completed resumes for every integration policy before Stage B. Naive did not progress to request completion; therefore no measured matrix or comparative performance claim is valid. This is not classified as a SlackKV lease failure or an invalid runtime environment.
