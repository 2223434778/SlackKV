# Host prefix screening: frozen before execution

2026-10-10, base685ac5c. User relaxed novelty requirements: existing methods may
be reused; overall difference and measured value, rather than each novel component,
are required. This supersedes excluding a PoC solely for overlap in the opportunity report.

Question: do prefix-closed tail eviction and a length-cost balancing variant reduce
lost reusable input in the already frozen15 public traces? This is CPU screening,
not a faithful serving simulator or proof of actual host misses.

Four methods, all capacities, no outcome selection: native-inspired block LRU
(reverse touch, atomic32-block stores); prefix-closed leaf LRU; round-robin leaf
tail eviction; cost-balanced leaf tail eviction. Last candidate chooses smallest
observed-length-squared minus surviving-depth-squared loss after one block eviction.
It combines ordinary fair-tail reclamation with a nonlinear cost proxy; novelty
is not established. RR model is inspired by RR-Evict, not the full SGLang method:
no reliable agent chain/completion signals,64-token leaves, no active GPU state.

Capacities per trace/model namespace:16384,65536,131072,262144,524288,1048576,
2097152 proxy tokens. At old28672bytes/token these correspond to0.4375 through
56GiB. Unit projection only; not actual Claude/Qwen tokenization or production
working-set budgets. Traces are not synchronized into an invented concurrent pool.

Inputs, event timestamps and endpoint durations identical. Writes occur synchronously
at recorded request completion, not during prefill; no generated hashes, GPU reuse,
DMA refcounts, interference, output quality or changed execution schedule modeled.
Ordering at equal timestamps: complete then arrival; a zero-duration call is rejected
as unsupported if present. Hash identity isolates trace/model and ordered ancestors.
Offline topology is an index only: policy never examines unseen children/future
lengths/next use. Metadata extent updates only after an observed completion.

Metrics: all per-request hit/uncached proxy tokens, writes, P95/P99 uncached,
quadratic attention proxy, per-namespace regressions. Candidate may advance only if
uncached total and P95 improve>=10% over block LRU and it is not worse than RR
on either by>5%, at two adjacent budgets; writes not>10% worse; at least20/27
namespaces have no>10% total-uncached regression. Otherwise reject this candidate,
retain useful reference method results without performance claims. No parameter tuning.

Next GPU experiment only after a concrete regime/decision is supported. Main
performance baseline remains matched Compiled OnDemand; conditional structural
results are never reported as an end-to-end speedup. New exploration GPU ceiling4h,
each run<=30min; CPU screen costs0GPUh.

Commands from repository root:

```sh
python -m pytest -q tests/test_host_prefix.py
python -m experiments.host_prefix.run --input .tmp/opportunity_trace_prefix.jsonl --output results/host_prefix_screen_20261010
```
