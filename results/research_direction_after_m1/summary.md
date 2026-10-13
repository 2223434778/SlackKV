# Research direction after M1

Recommendation: defer a new system pending evidence of avoidable loss beyond strong fair-prefix baselines. Stop current dynamic M1/M2/M3 work; preserve all implementations, defaults and historical results.

CPU evidence:
- 82 archived32B files /84,782,502 rawbytes verified;198request inputs/outputs and raw counters/times rechecked.
- Fixed private persistent-prefix future-aware token oracle at64GiB:524288savedtokens, exactly observedStaticFair; extra0. Conditional token bound, not rolling-cache/global latency optimum.
- Unlimited usable prior-input KV reference:1214976reusabletokens;Fair remaining690688. Not achievable-policy/E2E prediction.
- Existing public proxy:2661requests,65container/modelgroups;calls median20/P95220/max311. No complete >=3-call group fits old32K proxy input+output boundary; proxy data is not Qwen tokenization or ground-truth linear agent timing.

At most one conditional candidate: value-aware durable partial prefixes for heterogeneous long-lived sessions. No policy implementation. Needs trace/cost evidence and fair/standard-admission baselines first; proposed minimal next validation is CPU-only and pre-gated, not a GPU matrix.

GPUhours added0; prior ledgers unchanged. No changes to serving defaults or historical data. Full report:docs/RESEARCH_DIRECTION_AFTER_M1.md. Reproduction:python -m experiments.research_direction_after_m1 --large-model results/large_model_20261012 --proxy .tmp/opportunity_trace_prefix.jsonl --output <new-output.json>.
