# Phase 3K summary

Phase 3K is the clean retry of Phase 3I with p_busy=70 ms and resume_guard=110 ms. The Phase 3H K-capacity gate and connector-first promotion were reused unchanged.

|Policy|ABC P95 TTFT|Pre-first overlap|Gate wait|ABC decode tok/s|X slowdown|Combined tok/s|Makespan|Memory|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|ResumeAware-K3|59.53|32.50|0.00|187.55|1.041|115.81|557.85|45.32|
|FirstTokenGate-K1|80.76|0.00|48.61|176.16|1.046|115.24|586.04|48.90|
|FirstTokenGate-K2|59.91|32.59|0.00|187.72|1.042|115.70|554.79|45.68|

Promotion-clean runs: 15/15.
K1 max pre-first=1 in every run: YES.
K2 max pre-first=2 in every run: YES.
Busy-load ranking by ABC P95 TTFT: ResumeAware-K3 < FirstTokenGate-K2 < FirstTokenGate-K1.
Light-load ranking from Phase 3H: K2 < K3 < K1.
Ranking changed: YES.
K1 protects X more than K2: NO.
K1 improves combined throughput over K2: NO.
Dynamic K justified: NO.
Result: Result C.
Next direction: DYNAMIC_K if Result A; KEEP_K2 if Result B; RETHINK_ADMISSION if Result C.

The X request generated 128 tokens in every run and remained active through C first token. No dynamic K or Phase 3L work was started.
