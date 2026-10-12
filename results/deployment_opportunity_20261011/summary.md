# Deployment opportunity audit

Objective: evaluate credible resources before expanding partial-prefix GPU experiments. Baseline `834ed5a`, work begun 2026-10-11 and resumed 2026-10-12.

CPU audit completed: fixed15 traces/2661 requests/27 isolated namespaces. Cumulative unique proxy input median/P95/max222720/1454080/2760192; active input peak165248/704768/769344; active+offline future-needed retention peak165248/800000/1332032. Endpoint peak inflight median2/P955/max5. These are proxy structural quantities, not Qwen tokens, GPU measurements or an online policy.

Resource arithmetic: FP16 KV bytes/token28672 for cached1.5B,262144 for official32B,128KiB/rank underTP2. 32B weights61.0275GiB; twoA6000 at85% minus weights leaves at most20.5526GiB before runtime overhead. Host snapshot125.65GiB total/~118.87GiB available; no512MiB process limit found. Native host budget aggregates across TP workers.

38 related tests passed. Pin32 probe NOT_STARTED_ENVIRONMENT: original research disk path disappeared after server state change; repo, runtime and ledger unavailable. SSH/read-only diagnostics succeeded. No mount/env modifications, no new GPU performance comparison.

Conclusion: partial-prefix deployment value UNKNOWN. Reasonable current1.5B budgets do not establish double pressure;32B long contexts are a feasibility candidate only. Keep Full/NoHost, avoid arbitrary tiny host caps. Await existing research volume restoration before host measurement. No new correctness/fairness/performance claim. GPU added0; continuous used1.2500769679056248/4h.

Full report and commands: `docs/AUTONOMOUS_RESEARCH_DEPLOYMENT_OPPORTUNITY_20261012.md`. Frozen protocol: `docs/DEPLOYMENT_OPPORTUNITY_PROTOCOL_20261011.md`.
