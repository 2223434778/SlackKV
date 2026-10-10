# CPU research opportunity decision

Source main1c9c7da; GPU hours0; no serving changes or new speedup claim.
Report: [RESEARCH_OPPORTUNITY_DECISION.md](../../docs/RESEARCH_OPPORTUNITY_DECISION.md).

One published workload family, pinned SemiAnalysis Weka v7 revision23f152f6f0f9399a85901b89a6458def0ef16729.
Preselected first16 complete file rows /64MiB cap yielded15 complete rows,46,447,908bytes.
No full1.85GB corpus download; no result-dependent filtering or random-sample claim.
Raw source remains under ignored `.tmp/`; compact2661-request CSV and summaries are committed.

2661requests,47explicit subagent groups,892nested requests; input proxy median118208/P95624320,
output median460/P955053. Same-model completed input history structural prefix coverage97.8255%,
other-container coverage3.4640%; these are infinite-cache input-only structure, not physical hits.
53negative same-container end gaps flag interleaving; no proven DAG or true sequential agent chain.

Decision: no accepted new algorithm opportunity; stop old JIT/gate/pin variants. At most retain the
question of actual host-loss/recompute cost under natural long-context budgets, conditional on a
measured gap beyond EfficientAgent/MORI/RR-Evict. Such cost and deployment feasibility remain UNKNOWN.

Reproduce CPU only from repository root (public fetch, at most64MiB; fails on source drift):

```sh
python -m experiments.opportunity_decision.fetch --output .tmp/opportunity_trace_prefix.jsonl
python -m experiments.opportunity_decision.analyze --input .tmp/opportunity_trace_prefix.jsonl --output results/opportunity_decision_20261010
python -m experiments.opportunity_decision.audit --input .tmp/opportunity_trace_prefix.jsonl --results results/opportunity_decision_20261010
python -m pytest -q tests/test_opportunity_decision.py
```

Actual local command used E:/Software/Anaconda/python.exe;4unit tests passed. Independent CSV arithmetic
checks confirmed all reported coverage fractions and input/output quantiles.92 deterministic sampled
LCPs are checked against a separate direct previous-completed-input comparison in `audit.json`.
Server public-source metadata probe timed out; local HTTPS acquisition succeeded. No GPU cost/failure.
Schema, source and caveats are in source_manifest.json /structure.json; no prompt text or original IDs saved.
