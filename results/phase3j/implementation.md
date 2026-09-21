Phase 3J reuses the Phase 3H generalized K-capacity gate and the Phase 3B connector-first native prefetch endpoint without source changes.

Stage A measures one 4096-token native promotion after external offload. L0 has no normal decode. L1 starts the ordinary 128-token X decode, waits for its first token, then immediately starts the promotion and verifies that X remains active at both promotion boundaries.

The busy budget is frozen as ceil((max L1 client E2E + 10 ms) / 5 ms) * 5 ms. Stage B uses that one fixed budget in the existing latest_feasible_start scheduler for A/B/C and runs FirstTokenGate K=2 only. Native endpoint response fields are copied directly into native_prefetch_metrics.csv.

The K2 C-GATE_BLOCKED condition is treated as a required exercise check. In the observed validation, promotion was clean and K2 reached two owners, but C was not gate-blocked in every repetition because server intake timing admitted it after an owner had already emitted its first token. The phase is therefore reported as Result D for incomplete K2 exercise, without tuning the workload.
