# Phase 3R-FIX

One fresh Stage-A capacity-audit run; no Stage B.

Experiment valid: True. Capacity metric sane: True. Decisions: 12/12.
PROMOTE/SKIP: 12/0. Internal-ready misses: 0. Promoted remote waits: 7 (E, F, G, I, J, K, L).
Source local/external/total: 23600/25552/49152; fractions 48.01%/51.99%; conservation PASS.
Capacity query ms median/P95/max: 21.29/21.94/22.14.
Issue lateness ms median/P95/max: 21.35/22.01/22.24.
Issue-to-T6 ms median/P95/max: 25.62/31.23/31.51.
Target-to-T6 ms median/P95/max: 47.41/52.42/52.51.
The three idle/setup checkpoints read zero, but the twelve target-time samples varied from 0.254 to 0.808, proving the live metric changed during the measured workload.
All 12 decisions had measured free capacity above 144 MiB, yet seven later entered WAITING_FOR_REMOTE_KVS. Instantaneous free capacity is therefore insufficient to guarantee future residency.
Result B; next direction RESIDENCY_RESERVATION. No threshold tuning or next phase was run.
