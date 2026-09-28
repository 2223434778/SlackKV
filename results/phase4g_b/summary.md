# Phase 4G-B — incomplete mechanism RCA

Final status: **INCOMPLETE_INVALID_RUN**. The frozen 12-run matrix stopped at its first invalid measured run. One valid OnDemand run is retained as an execution record only. No full-matrix aggregate, policy comparison, or A–E bottleneck classification is claimed.

## Frozen inputs and gates

- Start commit: `3c887a710cca06142a00c4ca35ca152e7b590bc0`
- Phase 4C workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`
- Instrumentation smoke: True (`smoke_rep2_M2`), probe CPU duty 0.03619659621395638 (<5% gate).
- Measured sequence: M0/A/no-background repetition 1 passed; M1/D/no-background repetition 2 did not complete. Ten later runs were not started.

## First invalid run evidence

The M1 FullSlackKV+PSA1 no-background run's driver timed out waiting for a request future after 300 seconds. The server log recorded 253,044 capacity-guard checks and 253,045 connector-only steps, with no native-prefetch completion event. Only 2 resumes arrived (E01:turn:1, E03:turn:1); neither arrival had scheduled tokens. Shutdown then produced EngineDeadError messages; the log does not show an independent OOM or engine crash before controlled teardown.

For E01, a `SOURCE_READY` event preceded its promotion attempt. The readiness probe found 1330 local blocks out of 1330, with 0 external blocks needed. At the first guarded attempt, exact free blocks were 1563 for 1330 required; PSA's incremental reserve was 7, so the candidate was admitted. The same E01 request then repeated in connector-only scheduling without completing.

E03 shows the contrasting PSA decision: free blocks 1563 were enough for its 1306-block physical fit, but its protected-progress reserve was 1337; PSA emitted `PROGRESS_REJECT`. This one run exposes an interaction between source readiness, JIT promotion, and protected headroom, but does not establish which factor is sufficient or generalize beyond this observation.

The inherited native-prefetch client uses a 5-second HTTP timeout, while the lifecycle driver awaited worker futures for 300 seconds before teardown. The failed run and compact event summaries are retained under `failures/`; the 315-MiB raw server log remains outside the repository.

## Interpretation and limitations

This is not a completed mechanism matrix: there is one valid run and one invalid run, no repeated M1 condition, and no background-X condition. Do not infer a policy result or attribute the non-progress solely to source readiness, physical headroom, PSA, or JIT timing. No serving policy, workload, scheduler algorithm, or threshold was changed after observing this run, and no further matrix runs were attempted.

Artifacts: `partial_run_inventory.csv`, `validation_report.json`, `failures/measured_rep2_d_diagnostic.json`, `failures/source_probe_summary.csv`, and `failures/capacity_guard_summary.csv`.
