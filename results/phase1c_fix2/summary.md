# Phase 1C-Fix2

QueueJIT now advances to safe-idle expiry and wakes for D2H-ready/session events; D2H residency continues through D2H completion. The frozen Phase 1C-Fix matrix was rerun.

C0: PASS. C1: PASS. C2: PASS.

A = Strong GO.

| M | Policy | miss | P95 stall | normalized memory |
|---:|---|---:|---:|---:|
| 32 | OffloadAll-QueueJIT | 61.8282% | 508.8 ms | 0.0124 |
| 32 | FixedTTL-0.5 | 0.1789% | 0.0 ms | 0.0084 |
| 32 | FixedTTL-1 | 0.0000% | 0.0 ms | 0.0098 |
| 32 | FixedTTL-2 | 0.0000% | 0.0 ms | 0.0182 |
| 32 | SlackRatio-2 | 0.0000% | 0.0 ms | 0.0095 |
| 32 | SlackRatio-3 | 0.0000% | 0.0 ms | 0.0134 |
| 32 | SlackRatio-4 | 0.0000% | 0.0 ms | 0.0159 |
| 64 | OffloadAll-QueueJIT | 63.9719% | 526.0 ms | 0.0118 |
| 64 | FixedTTL-0.5 | 0.1838% | 0.0 ms | 0.0081 |
| 64 | FixedTTL-1 | 0.0000% | 0.0 ms | 0.0091 |
| 64 | FixedTTL-2 | 0.0000% | 0.0 ms | 0.0157 |
| 64 | SlackRatio-2 | 0.0111% | 0.0 ms | 0.0090 |
| 64 | SlackRatio-3 | 0.0000% | 0.0 ms | 0.0116 |
| 64 | SlackRatio-4 | 0.0000% | 0.0 ms | 0.0140 |

Q1: corrected timing materially reduced P95 and local-memory time versus Fix; see `fix_delta.csv`. Q2: selective offload remains far better than OffloadAll. Q3/Q4: C1/C2 pass. Q5/Q6: see TTL and SlackRatio Pareto CSVs; SlackRatio remains a qualifying frontier extension.

Limitations: oracle-deadline simulator, optimistic non-contended D2H, serialized H2D, no real KV tensors or LLM serving.

Phase 1C-Fix2 execution status: COMPLETE
