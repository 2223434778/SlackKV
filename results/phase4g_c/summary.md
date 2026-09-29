# Phase 4G-C: Source Recheck × PSA × Allocator-Truth RCA

Final status: EXPERIMENT_INVALID_MATRIX. The run stopped during the measured matrix after the frozen pause-fidelity gate failed. No causal or performance comparison is valid.

Start commit: c1761e980b16eec5d5381e8cf0a6edd4d245bf09. Frozen workload SHA256: 9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105.

Failure detail: Rep4/C1 pause P95 exceeded the frozen limit. Six valid runs completed; seven of eight planned measured runs were attempted.

## Execution outcome

- Stage A smoke: C0 and C1 both passed (2/2 fresh runs).
- Measured matrix: 8 planned runs; 6 valid runs completed; 7 runs were attempted. Rep4/C1 failed the frozen pause-fidelity gate; Rep4/C0 was not started.
- The 6 valid runs contribute 240/320 valid turns and 192/256 valid boundaries. Including the invalid but completed Rep4/C1 run, 280 turns and 224 boundaries executed. No partial C0/C1 matrix aggregation was performed.

## Pause-fidelity failure

- Frozen limits: nearest-rank P95 absolute pause error <= 10 ms and maximum <= 30 ms.
- Rep4/C1: P95 absolute error 12.211563 ms; maximum 18.885849 ms. Three of 32 boundaries exceeded 10 ms: E01:b1 (260 ms proxy, +18.885849 ms), E04:b2 (439 ms, +12.211563 ms), and E04:b1 (297 ms, +11.469409 ms).
- The maximum gate passed, but the P95 gate failed, so this run and the 8-run matrix are invalid. These are client monotonic issue-time deviations; the data establish pause-injection jitter, not a specific kernel scheduler root cause.

## Other observations

- The failed run still completed 40/40 turns and 32/32 boundaries; maximum pre-first concurrency was 2. BGv2 generated 8,192 tokens and remained active 97,588.074 ms beyond the last Agent first token. The server stayed alive with no OOM, crash, logical leak, or lease leak.
- Source accounting showed a 768-token full-prefix shortfall explained by the pinned prompt-only offload behavior; this is retained as a separate limitation and is not used to make a causal claim.
- No final result A-E classification, causal funnel, or C0/C1 aggregate is reported. Partial valid runs must not be interpreted as a comparison.

## Validation and artifacts

- Phase tests: 49 passed. One optional Python site-package .pth warning about missing nvidia_cutlass_dsl appeared; it did not affect the test result.
- Failure evidence: failure_summary.json, validation_report.json, and the compact server log for measured_rep41_d. Full per-run JSON remains in the server workspace and is excluded from the commit as bulky raw diagnostics.
- Final status: EXPERIMENT_INVALID_MATRIX. The experiment did not establish whether source recheck, PSA-1, or allocator capacity dominates.
