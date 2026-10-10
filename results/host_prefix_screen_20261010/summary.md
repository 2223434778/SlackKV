# CPU host-prefix screening

Report: [host-prefix research](../../docs/AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md).
Protocol: [frozen CPU gate](../../docs/HOST_PREFIX_SCREEN_PROTOCOL.md).
Source pinned publicWeka15traces/2661requests; same SHA as opportunity_decision.
7capacities×4policies,27namespaces,74,508records; no synchronization across traces.
Host-only counterfactual; fixed original event times; completion-time synchronous
writes. No GPU state, actual DMA or changed scheduling; not a latency simulator.

Cost-tail failed all7capacity gates. At524288 proxytokens total uncached+96.46%
andP95+33.67%; at1048576 respectively+114.84%/+208.33%. Do not tune this candidate.
Prefix-LRU uncached savings about1%, but write proxy drops93.13% at262144; led to
separately frozen actual GPU checkpoint test. No CPU-result speedup claim.

Initial implementation interrupted for redundant scans; originals infailed_attempt1.
Exact native manager hit/order differential320requests passed both before/after
optimization; native refcount/cancel checks passed. CPU GPU-hours0.
All request data arithmetic and paired lengths independently checked byassess.py.

```sh
python -m experiments.opportunity_decision.fetch --output .tmp/opportunity_trace_prefix.jsonl
python -m experiments.host_prefix.run --input .tmp/opportunity_trace_prefix.jsonl --output <new-directory>
python -m experiments.host_prefix.assess --directory results/host_prefix_screen_20261010/retry
```

No runtime changes, future lookup, real cache-hit or task-quality claim.
