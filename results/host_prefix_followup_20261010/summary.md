# Three-question host prefix screen

16 real GPU runs, Compiled OnDemand, unchanged native runtime, Qwen1.5B FP16.
Frozen definition: docs/HOST_PREFIX_FOLLOWUP_PROTOCOL.md, commit158778f.

- Partial vs NoHost: two forced-GPU-cold single-session pairs, resume max/P95
  −17.93%/−20.09%, BG median −7.91%/−9.35%; screen passes, no significance claim.
- Static sharing: two alternating sessions, same256MiB host,4672/4688tokens;
  both preserve all five resumes. P95 −8.40%/−8.90% vs NoHost, BG median−4.76%;
  fails preregistered10% threshold. Reset/join diagnostic is not natural Session E2E.
- No reset: four closed-loop sessions under2GiB GPU/512MiB host, Session E2E
  median−8.88% vs NoHost,−8.87% vs Full; BG makespan−5.84%/−5.61%. All four
  session E2Es improve; one resume P95 worsens1.41%. Fails10% main threshold.
- GPU-hot: identical576000GPU hits,0host reads; no extra benefit, one session
  E2E worsens8.52%. Ample host: effective fair admission equalsFull; ordinary
  complete Offload gains75.02% vs NoHost, not a partial-policy innovation.
  Fair vs equivalentFull median+0.94%, one session+5.46%; single concurrent runs
  cannot resolve noise versus small overhead. Preserve negative control flags.

All matched raw hashes/output work, timestamp arithmetic, source conservation,
H2D and actual pool checked after lossless archive extraction on local computer.
30 tests pass, native metadata references/cancel/retired replacement pass.
Only previous one-input invariant tensor-content oracle; no new GPU multi-session
content/preemption oracle. Tool failures retained, including INVALID empty summary.

GPU0 new0.3645630388854382h,16complete/0GPUfailed,max117.63s/run; continuous total
1.250076967905625/4h. GPU1 unused. Originals retained server-side. Conservative
budget includes setup/compile/warmup/cleanup. Raw inputs xz, requests/logs gzip.
No real production trace or natural host-RAM shortage established. No dynamic
architecture, SOTA superiority, SLO concurrency, or owned-memory-time claims.
Keep simple candidate; do not tune threshold or launch formal matrix.

Full Chinese report: docs/AUTONOMOUS_RESEARCH_HOST_PREFIX_FOLLOWUP_20261010.md.
Recompute: python -m experiments.host_prefix.followup_analyze --directory results/host_prefix_followup_20261010
