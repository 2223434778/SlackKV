# SlackKV infra engineering — 2026-10-16

Objective: runnable single-node prototype on native vLLM, not a new inference
engine or a paper speedup claim. E1/E2 GPU functional validation is complete;
E3 is packaging, documentation and CPU verification only.

Environment: Qwen2.5-1.5B-Instruct, RTX A6000 GPU0, Python3.11.16,
torch2.11.0+cu126, vLLM distribution0.23.0+cu126 / upstream0fc695fc6,
transformers5.17.0. GPU KV512MiB, host224MiB except host-enough1GiB.

- Two real 2-session ×3-turn demos with generated histories and local JSON tool.
- Final discriminative host oracle:1792 external tokens,49MiB H2D,112MiB D2H,
  full output IDs equal independent zero-hit cold; no GPU reset.
- Nine fixed-work benchmarks:3 scenarios ×3 policies, all COMPLETE.
  Host-limited sequence ms:NoHost3778.93 /Full3590.18 /Fair3369.38.
  Host-enough Full/Fair source and traffic counters identical; hot has no host
  restore and its timing gap is not attributed to offload. One cohort only.
- Real first-content disconnect, atomic history, new generation, scheduler idle;
  external owner stop is STOPPED/cleanup confirmed. Native abort vs normal finish
  remains UNKNOWN. Global counters are not per-request attribution.
-15 runs incl1 early failure, all cleaned; engineering0.1454743116594586/0.5GPUh,
  historical continuous4.7163514802356055GPUh. E3 uses0GPUh, GPU1unused.
-261 archived original artifacts independently verified; previous manifests and
  failed/weak checks retained. E3 harness missing-fixture failure also retained.

See [delivery](../../docs/INFRA_DELIVERY_20261016.md),
[E1](../../docs/INFRA_E1_STATUS_20261016.md),
[E2](../../docs/INFRA_E2_STATUS_20261016.md), e2_verified.json, comparison-*.json,
e3/validation.json and per-run JSON/CSV/Markdown. All GPU argv and source hashes
are in execution.json and provenance. Reproduction commands are in the delivery
and operations documents. No additional GPU experiments are scheduled.
