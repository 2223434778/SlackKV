# Round 2: bounded negative result

Objective: identify avoidable system delay against Compiled OnDemand and screen
two architectures without reducing model or background work.

Setup: Qwen2.5-1.5B FP16, one A6000, 2/4/8-GiB probes, CPU 8 GiB, 40 calls /
32 resumes / 8192 BG tokens per valid run. Both matched algorithm pairs at 2 GiB.

Results: capacity P95 5199.92→1163.51→445.32 ms. Credit P95 -39.90% but session
median +6.08%; epochs actual H2D -38.60% but P95 +106.80%, session +14.37%.
Neither passes preregistered gates. Background changes <0.2% in algorithm pairs.
Seven valid lifecycles plus one preserved setup failure; 0.233269 GPU-hours.

Anomalies: varying initial order/baseline P95, one-block transfer mismatch,
missing 8-GiB H2D counter, output differences without a new content oracle,
owned KV memory-time UNKNOWN. No stable speedup or universal impossibility claim.

Conclusion: stop both candidates without tuning. Lower transfers/failure counts
or one better tail statistic do not establish overall improvement. Keep compiled
OnDemand and safety guards. Scientific stopping condition B in this limited scope;
all owned servers stopped. Full report and evidence in README.md.
