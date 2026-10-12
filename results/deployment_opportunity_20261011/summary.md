# Deployment opportunity audit

Objective: evaluate credible resources before expanding partial-prefix GPU experiments. Baseline `834ed5a`, work begun 2026-10-11 and resumed 2026-10-12.

CPU audit completed: fixed15 traces/2661 requests/27 isolated namespaces. Cumulative unique proxy input median/P95/max222720/1454080/2760192; active input peak165248/704768/769344; active+offline future-needed retention peak165248/800000/1332032. Endpoint peak inflight median2/P955/max5. These are proxy structural quantities, not Qwen tokens, GPU measurements or an online policy.

Resource arithmetic: FP16 KV bytes/token28672 for cached1.5B,262144 for official32B,128KiB/rank underTP2. 32B weights61.0275GiB; twoA6000 at85% minus weights leaves at most20.5526GiB before runtime overhead. Host snapshot125.65GiB total/~118.87GiB available; no512MiB process limit found. Native host budget aggregates across TP workers.

Restored-state summary: pin32_20261012 PASS,40 related tests passed;32GiB real pinned allocation27.01465s and process-exit release verified. Historical NOT_STARTED_ENVIRONMENT occurred when the research disk disappeared; those diagnostics remain archived. No mount/env modifications were made by the agent.

Conclusion at this audit stage: partial-prefix deployment value UNKNOWN. Reasonable1.5B budgets do not establish double pressure;32B long contexts remain a feasibility candidate. Keep Full/NoHost, avoid arbitrary tiny host caps. Pin probe complete, no serving-performance claim in this audit. AddedGPU0.01574406685; cumulative1.2658210347572914/4h. Subsequent32B validation is separately recorded in docs/AUTONOMOUS_RESEARCH_LARGE_MODEL_20261012.md and results/large_model_20261012/; this audit does not retroactively claim those results.

Full report and commands: `docs/AUTONOMOUS_RESEARCH_DEPLOYMENT_OPPORTUNITY_20261012.md`. Frozen protocol: `docs/DEPLOYMENT_OPPORTUNITY_PROTOCOL_20261011.md`.

Restoration addendum: user restored original disk; pin32_20261012 PASS on GPU1. Real32GiB pinned zero allocation27.01465s, worker lifecycle55.67864s. RSS increased32.104GiB; del+GC retained pool, process exit restored system available memory.40 related tests pass. New cost0.015744066851666676GPUh, continuous1.2658210347572914/4h. Earlier NOT_STARTED and GPU0 descriptions are historical, preserved as evidence. No serving-performance/SLO claims; feasible32GiB host allocation does not justify an arbitrary512MiB deployment cap.
