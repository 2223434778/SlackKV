# Phase 1D
Synthetic prediction-noise simulation; primary 56 KiB/token, M=32, five seeds. Predicted deadlines drive admission/scheduling; actual deadlines drive stall and residency.

| sigma | policy | miss | P95 | memory | offload |
|---:|---|---:|---:|---:|---:|
| 0 | KeepAll | 0.00% | 0.0 ms | 1.0000 | 0.000 |
| 0 | OffloadAll-QueueJIT | 61.87% | 516.3 ms | 0.0187 | 1.000 |
| 0 | FixedTTL-0.5 | 0.18% | 0.0 ms | 0.0115 | 0.371 |
| 0 | FixedTTL-1 | 0.00% | 0.0 ms | 0.0130 | 0.321 |
| 0 | SlackRatio-1.25 | 0.09% | 0.0 ms | 0.0115 | 0.391 |
| 0 | SlackRatio-2 | 0.00% | 0.0 ms | 0.0127 | 0.359 |
| 0.1 | KeepAll | 0.00% | 0.0 ms | 1.0000 | 0.000 |
| 0.1 | OffloadAll-QueueJIT | 81.13% | 742.1 ms | 0.0583 | 1.000 |
| 0.1 | FixedTTL-0.5 | 19.77% | 693.9 ms | 0.0506 | 0.371 |
| 0.1 | FixedTTL-1 | 17.03% | 693.9 ms | 0.0520 | 0.317 |
| 0.1 | SlackRatio-1.25 | 20.50% | 693.9 ms | 0.0504 | 0.392 |
| 0.1 | SlackRatio-2 | 18.77% | 693.9 ms | 0.0517 | 0.358 |
| 0.25 | KeepAll | 0.00% | 0.0 ms | 1.0000 | 0.000 |
| 0.25 | OffloadAll-QueueJIT | 79.13% | 1384.8 ms | 0.1105 | 1.000 |
| 0.25 | FixedTTL-0.5 | 18.20% | 1354.7 ms | 0.1044 | 0.366 |
| 0.25 | FixedTTL-1 | 16.84% | 1354.7 ms | 0.1082 | 0.311 |
| 0.25 | SlackRatio-1.25 | 19.67% | 1354.7 ms | 0.1044 | 0.388 |
| 0.25 | SlackRatio-2 | 18.33% | 1354.7 ms | 0.1064 | 0.356 |
| 0.5 | KeepAll | 0.00% | 0.0 ms | 1.0000 | 0.000 |
| 0.5 | OffloadAll-QueueJIT | 79.13% | 3294.9 ms | 0.2436 | 1.000 |
| 0.5 | FixedTTL-0.5 | 18.48% | 3294.9 ms | 0.2359 | 0.353 |
| 0.5 | FixedTTL-1 | 16.99% | 3294.9 ms | 0.2395 | 0.289 |
| 0.5 | SlackRatio-1.25 | 20.11% | 3294.9 ms | 0.2361 | 0.375 |
| 0.5 | SlackRatio-2 | 18.62% | 3294.9 ms | 0.2382 | 0.328 |

TTL-0.5 beats SlackRatio-1.25 in miss rate for sigma 0.10/0.25/0.50: 19.77/18.20/18.48% versus 20.50/19.67/20.11%.
TTL-1 beats SlackRatio-2: 17.03/16.84/16.99% versus 18.77/18.33/18.62%.

Final conclusion: Result C — Prediction uncertainty does not support a SlackRatio advantage.
This remains simulation with synthetic prediction noise.
Phase 1D status: COMPLETE
