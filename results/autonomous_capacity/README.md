# Autonomous Research Round 2

Report: `docs/AUTONOMOUS_RESEARCH_REPORT_ROUND2_20261009.md`.
Protocol: `protocol.md`; observed commands: `commands.sh`.

Seven valid full lifecycles, one pre-workload failure. GPU0 only; conservative
budget 0.2332685893 GPU-hours, longest 130.446 seconds. Both algorithms rejected,
default disabled. Credit: P95 -39.90%, session median +6.08%; soft epochs:
P95 +106.80%, session median +14.37%, actual H2D -38.60%. No accepted novel mechanism.

Capacity probes at 2/4/8 GiB: P95 5199.92/1163.51/445.32 ms, observed failed
resumes 31/9/0 of 32. Resource intervention is not algorithm acceleration.

## Evidence and reproduction

- `*/execution.json`: settings, actual server command, code fingerprints, cumulative
  budget, full wall time and failures.
- `*/runs/measured_rep1_a/run.json.gz`, `inputs.jsonl.gz`, `output_tokens.jsonl.gz`:
  complete results, actual inputs and outputs. Complete logs are lossless `.log.gz`.
- `metrics_after_timed_workload.prom`: actual H2D/D2H/preemption. Eight-GiB H2D
  series missing means UNKNOWN; source restore separately zero. Credit candidate
  has an unexplained one-block actual/source mismatch; counter is primary.
- `analysis.json`: original source estimate; `round_summary.json`: current matched
  gates; `independent_audit.json`: independent arithmetic and raw gate observations.
- `SHA256SUMS.json`: 87 logical export hashes, decompress gzip before hashing.
  Regenerated summary is separate from `exported_round_summary.json`, the exact
  frozen copy used to verify the original summary hash. Raw evidence is unchanged.
  Added reports/audits are outside this original export manifest.
- `runtime_scheduler_final.py.gz`: final runtime snapshot, hash in `environment.json`.
  Apply `experiments/autonomous_capacity/continuation.patch` then `epochs.patch`
  to the round-one copy. Pinned upstream/diff/dev API snapshots are preserved in
  `results/autonomous_admission/`; allocator unchanged.
- `runtime-validation/`: both 15-test CPU-only regressions; `analysis_failures.txt`:
  preserved offline corrections; `footprint_model.json`: sizing, not actual trajectory.

```sh
python -m experiments.autonomous_capacity.summary results/autonomous_capacity
python -m experiments.autonomous_capacity.audit results/autonomous_capacity
python -m pytest -q tests/test_autonomous_capacity.py tests/test_autonomous_admission.py tests/test_research_reassessment.py --basetemp=.pytest_cache/round2-reproduce
```

40 tests pass, including raw archive/identity/counter checks. Both algorithm pairs
have 40 identical inputs; output hashes differ. New invariant content oracle,
owned KV memory-time, production quality and general fairness remain UNKNOWN.

Original uncompressed data/logs remain at
`/media/a6000/D/ZhaoBinHe/SlackKV/results/autonomous_capacity`. New runtime is
`/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round2-source`; historical runtime
unchanged. Fresh GPU reruns must use new names and the same cumulative ledger,
never rename/move the old ledger to reset budget. The failed 4-GiB command used
the historical fixed-pool validator; only the autonomous wrapper was corrected.
Fixed-input resource replay replaces synthetic output feedback, retains model
work and causal finish-relative pauses, and does not validate semantic Agent quality.
