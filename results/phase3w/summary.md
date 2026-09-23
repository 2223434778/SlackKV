# Phase 3W — stopped at mixed-error integration gate

## Status

INCOMPLETE. The zero-error regression passed (2/2; Oracle and Raw decisions matched); X-only controls passed (3/3). Mixed-error integration attempted all three policies. Oracle passed its strict reservation check. Safe had no timing/residency/lease errors but produced 5 promotions and 7 skips, below the required 6/6. Raw integration failed because the client stream ended without a generated token; its server-side root cause cannot be verified because the shared log path was overwritten. Therefore the integration gate failed and, as required, the 9-run measured matrix was not started.

| Policy | Promote/skip | Useful rate | Internal misses | Remote waits | AL P95 (ms) | Duplicate H2D (MiB) | Gate | Scope |
|---|---:|---:|---:|---:|---:|---:|---|---|
| OracleDynamicBlock512-K2 | 6/6 | 1.0 | 0 | 0 | 87.23 | 0.00 | PASS | integration only, not measured |
| SafePred200-DynamicBlock512-K2 | 5/7 | 1.0 | 0 | 0 | 86.15 | 0.00 | FAIL | integration only, not measured |
| RawPredDynamicBlock512-K2 | NA | NA | NA | NA | NA | NA | FAIL | Client stream returned without a generated token; server root cause unverified because log path was overwritten |

## Evidence and interpretation

- X-only controls: 3/3, median decode duration 3663.63 ms; median TPOT 14.37 ms.
- Zero-error regression: True; both policies made 6 promote / 6 skip decisions, 6/6 useful, no misses/waits/leaks, source 24,576 local + 24,576 external, 1344 MiB total H2D and 0 duplicate H2D.
- Oracle integration: PASS; 6 promotes / 6 skips, all six useful, no misses/waits/leaks, 512-block maximum.
- Safe integration: FAIL; 5 promotes / 7 skips, all five useful, zero promotion deadline misses, zero remote waits, zero lease leaks, maximum reservation 512 blocks. The frozen Safe gate required 6/6, so the measured phase was blocked. This is capacity contention from conservative reservation windows, not a T6 service miss.
- Raw integration: REQUEST_FAILURE_ROOT_CAUSE_LOG_NOT_PRESERVED. The client recorded `stream completed without a generated token`. A server exception was observed in the live run, but the shared Phase 3V-derived log filename was overwritten by the following Safe integration; the exact Raw exception cannot now be independently verified from saved artifacts. No Raw performance statistics are claimed.
- Prediction buckets are kept in `mixed_error/prediction_metrics.csv` for the two valid integration runs. No three-repetition error-bucket result exists.
- The actual resume deadlines, 200-ms margin, reservation expiry, Async40, 110-ms resume guard, 512 blocks, K2, model, cache sizes, and policy were not tuned. No vLLM source was modified.

## Requested questions

1. Raw prediction error reduced useful promotions: **not validly quantifiable** because Raw's integration process crashed; the server error is preserved.
2. Whether +200 ms caused missed promotions: **not measurable** beyond the observed Raw crash.
3. Whether −50 ms caused excessive early residency: Safe's measured integration-only promotion lead exposure is 120.35 MiB·s; it did not yield the required 6/6 reservation split.
4. Whether +120 ms remained safe: the one Safe integration had 0 T6 misses for its five promoted jobs; this is not a repeated result.
5. Safe promotion usefulness was 100% among its five issued promotions.
6. Safe stayed within the 512-block controller budget and physical lease peak; no leak occurred.
7. Safe reservation expiry covered each admitted actual resume.
8. Safe had zero internal-ready misses, zero remote waits, and zero lease leaks in its one integration.
9. Lowest observed integration AL P95 among valid records was Oracle/Safe as shown above; it is not a policy comparison.
10. No latency price is estimated from a single integration.
11. Duplicate H2D was 0 MiB in the valid Oracle and Safe integrations; Raw has no valid traffic accounting.
12. Valid-run cache/source accounting passed; no full measured source-conservation count exists.
13. The fixed safety margin was not tuned. Integration shows the Safe reservation split criterion failed even though promoted jobs met T6.
14. Repeatability and medians are unavailable because the matrix was blocked.

**Result:** no A/B/C algorithm result assigned; experiment is **INCOMPLETE / EXPERIMENT_INVALID** at the predeclared integration gate. The 9 measured runs were not run, and no next phase was started.
