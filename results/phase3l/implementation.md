# Phase 3L implementation

Phase 3L measures the existing native `/slackkv/prefetch` endpoint under four frozen decode-load states. The target prefix is offloaded and pressured out of GPU-local cache; ordinary background requests X1–X3 are admitted without `slackkv_first_token_gate`, and the promotion is submitted after their first tokens are observed.

No vLLM source or scheduling algorithm was modified. Endpoint response fields are copied directly into `calibration_metrics.csv`; non-copy overhead is derived as client E2E minus CPU→GPU time (and server non-copy as server E2E minus the same copy counter). Each state has five fresh-server repetitions. The budget is reported, not applied as a policy.
