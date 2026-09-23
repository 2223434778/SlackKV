# Phase 3V — trace-derived heterogeneous reservation

## Status

INCOMPLETE — workload eligibility gate failed before calibration or serving runs.

## Objective

Test oracle JIT promotion, heterogeneous reservation, physical leases, and K2 on six TraceLab-derived multi-turn episodes without prediction error.

## Dataset validation

The exact TraceLab v0.0.1 DuckDB was downloaded and its SHA256 matched `97715265367cc72376475f5d444c8e1900b88cab1482aa7b9a742894d9f15619`. The 96 MiB external database is excluded from Git.

## Blocking finding

The released schema was inspected rather than assumed. With `session_id` as the logical-session key, `input_tokens_total` as specified, four consecutive tool-driven boundaries, 100–3000 ms pauses, 1024–8192 reusable-token range, next input at most 9000, and non-decreasing context, the dataset contains:

| Median size class | Eligible episodes | Required |
|---|---:|---:|
| small [1024,3072) | 0 | 2 |
| medium [3072,5632) | 11 | 2 |
| large [5632,8192] | 2 | 2 |

The deterministic six-episode 2/2/2 selection is therefore mathematically impossible under the frozen Phase 3V criteria. A diagnostic substitution with `prefix_tokens` also yields 0/5/1 and does not help.

## Scientific decision

No size class, pause bound, consecutiveness rule, or context-monotonicity rule was relaxed. No episodes were cherry-picked. Service calibration, integration runs, and the nine measured runs were not started because they would operate on a non-conforming workload.

No Result A/B/C/D is assigned: those classes presuppose a valid 6-episode/24-job workload. This is an experiment-definition/data-eligibility blocker, not a reservation, service-model, or lease failure.
