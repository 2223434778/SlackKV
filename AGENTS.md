# SlackKV Project Rules

## Current implementation direction — 2026-10-12

The user explicitly accepted the system design and asked to upload it so the
execution Codex can implement it step by step. Read
`docs/SLACKKV_SYSTEM_ARCHITECTURE_20261012.md` first for this work.
The execution-side review in
`docs/SLACKKV_SYSTEM_ARCHITECTURE_EXECUTION_REVIEW_20261012.md` is accepted and
merged into section 9 of the architecture. Apply those native lifecycle contracts;
proceed to M1 implementation and CPU/native validation rather than another
review-only cycle.
Prioritize milestone M1: dynamic host-prefix allocation, elastic borrowing, safe
tail reclamation, and native on-demand offload. Reuse the existing compiled
runtime and lifecycle mechanisms. Leave speculative prefetch off initially.
The user permits reuse of mature methods and prioritizes implementation progress;
repeating the old reverse-order/multi-seed performance matrix is not a
prerequisite for developing this system. Preserve correctness and truthful
evidence; this does not retroactively validate or change historical results.
Choose implementation details autonomously and deliver a concrete runnable
milestone with its tested scope before expanding to M2/M3.
The latest architecture supersedes earlier review recommendations that put
repeatability experiments first. It does not grant a new numeric GPU budget:
the recorded 4.5 GPU-hour exception was restricted to the prior matched triple.
Complete authorized code/CPU validation and prepare a concrete GPU smoke plan
before requesting additional GPU allowance if none has since been granted.

## Current authority override — 2026-10-09

The user explicitly authorized autonomous algorithm research, architecture changes,
implementation, experiment design, and iterative GPU validation. This supersedes
the historical restrictions below requiring externally assigned hypotheses,
preserving JIT/PSA/Frontier as a fixed research direction, or stopping after each
phase. The final objective remains safe, repeatable end-to-end improvements for
representative multi-turn agent workloads; routine runtime optimization alone is
not a novel research contribution.

Each experiment is limited to 30 minutes and each autonomous exploration round to
4 GPU-hours, counting GPUs actually used. Seek user approval before exceeding
these budgets, deleting historical evidence irreversibly, disrupting other
services/environment, undertaking a costly formal matrix, or changing the final
objective. Preserve matched baselines, failures and UNKNOWN, maintain
`docs/AUTONOMOUS_RESEARCH_LOG.md`, and commit/push meaningful work. All remaining
safety, evidence-quality, reproducibility and ownership/liveness requirements
below still apply. Keep historical rules below as provenance.

## 1. Project role

This repository is a research project named **SlackKV**.

You are primarily responsible for:

* implementing experiments specified by the user;
* writing reliable and readable research code;
* running experiments;
* debugging implementation/runtime issues;
* collecting complete experimental results;
* producing reproducible artifacts;
* committing and pushing completed work to GitHub.

Do not independently redefine the research problem or significantly change the experimental hypothesis.

The research strategy, interpretation of results, and decision about the next experiment are handled externally.

When a requested experiment is complete, stop after documenting and pushing the results. Do not automatically start the next research phase unless explicitly instructed.

---

## 2. Current research direction

SlackKV studies KV-cache management for agentic LLM serving.

The current main direction is:

**resume-slack-aware KV placement and just-in-time promotion across local GPU HBM, peer GPU HBM, and CPU memory.**

Important concepts include:

* agent/tool-call idle intervals;
* predicted remaining time until the next LLM turn;
* local GPU KV cache;
* direct peer-GPU HBM as a nearline KV tier;
* CPU pinned-memory tier;
* GPU-to-GPU P2P transfers;
* asynchronous CUDA transfers;
* transfer/compute overlap;
* just-in-time KV promotion;
* resume latency;
* HBM pressure;
* throughput and tail latency.

Do not silently replace this research direction with generic KV offloading, generic caching, speculative decoding, model parallelism, or another topic.

---

## 3. Experimental discipline

Treat every experiment as a scientific experiment.

Before running an experiment, identify:

* objective;
* hypothesis;
* independent variables;
* controlled variables;
* baselines;
* metrics;
* expected output files.

Do not modify the experiment definition simply because an initial result is poor.

Never hide, discard, or selectively omit negative results.

If an experiment fails, distinguish between:

1. implementation failure;
2. environment/hardware failure;
3. invalid experimental setup;
4. genuine negative research result.

Do not automatically tune parameters until a desired result appears.

Preserve enough information to reproduce every meaningful experiment.

---

## 4. Do not make unsupported research decisions

You may make normal software-engineering decisions when implementing a clearly specified experiment.

You must NOT independently make major research decisions such as:

* changing the main hypothesis;
* removing a baseline because it performs well;
* replacing a metric because results look unfavorable;
* changing workloads in order to improve SlackKV;
* changing hardware assumptions;
* proceeding to a new experimental phase;
* claiming an experimental conclusion not supported by measurements.

If an ambiguity could materially affect scientific validity, document the ambiguity and choose the most conservative implementation possible. Clearly report the choice.

---

## 5. Result directory convention

Every experimental phase should store persistent results under:

`results/<phase>/`

Prefer the following structure:

`results/<phase>/summary.md`
`results/<phase>/metrics.csv`
`results/<phase>/config.json`
`results/<phase>/environment.json`
`results/<phase>/plots/`

When useful, also include:

`results/<phase>/commands.sh`
`results/<phase>/manifest.json`

Do not commit massive raw traces solely for archival purposes.

Commit compact data sufficient to independently inspect the conclusions.

---

## 6. summary.md requirements

Every completed experiment must have a human-readable `summary.md`.

It should contain:

### Objective

What was tested and why.

### Setup

Hardware, software, parameters, workload, and relevant configuration.

### Compared methods

All baselines and SlackKV variants actually executed.

### Results

Key numerical results. Do not only describe plots.

### Anomalies

Errors, instability, unexpected behavior, outliers, or limitations.

### Conclusion

State only conclusions directly supported by the experiment.

### Next-step status

Do not propose or execute the next research phase unless requested. State whether the current experiment technically completed successfully.

---

## 7. Machine-readable results

Important measurements must also be stored in CSV or JSON form.

Plots alone are insufficient.

For benchmarks, preserve per-configuration measurements whenever reasonably small.

Examples include:

* payload size;
* direction;
* latency;
* P50;
* P95;
* bandwidth;
* iteration count;
* transfer mode;
* compute interference;
* GPU IDs;
* peer-access status.

A future researcher must be able to regenerate plots from committed numerical data.

---

## 8. Reproducibility

For every experiment, record:

* exact command used;
* relevant random seeds;
* CUDA version;
* NVIDIA driver;
* PyTorch version;
* Python version;
* GPU names;
* GPU topology where relevant;
* relevant package versions;
* experiment configuration.

Avoid undocumented manual steps.

Prefer scripts over sequences of interactive commands.

If a benchmark requires warm-up, synchronization, pinned memory, CUDA events, or a specific stream configuration, make these explicit in code.

---

## 9. Measurement correctness

For GPU benchmarks:

* use proper synchronization;
* separate warm-up from measured iterations;
* avoid timing asynchronous CUDA operations with naive CPU wall-clock timing;
* use CUDA events when appropriate;
* perform enough repetitions to characterize variance;
* report at least median and P95 where applicable;
* validate copied data for correctness;
* distinguish allocated bandwidth from effective bandwidth;
* record failures rather than silently retrying until success.

When measuring concurrent compute and transfers, explicitly document CUDA streams and synchronization dependencies.

Correctness is more important than producing favorable performance numbers.

---

## 10. Hardware constraints

The target research hardware is:

* 2 × NVIDIA RTX A6000
* 48 GB VRAM per GPU

Never assume NVLink exists merely because the GPUs support it.

Determine the actual topology experimentally using tools such as:

`nvidia-smi topo -m`

and CUDA peer-access checks.

Use measured transfer performance rather than theoretical NVIDIA bandwidth when constructing later experimental models.

---

## 11. Code quality

Research code should remain simple and inspectable.

Prefer:

* small modules;
* clear CLI arguments;
* deterministic configuration;
* explicit assertions;
* documented units;
* type hints when useful;
* minimal dependencies.

Avoid unnecessary abstractions and large frameworks during feasibility experiments.

Do not optimize code before establishing measurement correctness.

---

## 12. Validation before declaring completion

Before declaring a task complete:

1. Run the implemented experiment or test.
2. Verify expected result files exist.
3. Inspect results for NaN/Inf/impossible values.
4. Verify plots can be regenerated from committed numerical data.
5. Check `git diff`.
6. Check `git status`.
7. Ensure no secrets or large unwanted artifacts are staged.
8. Update documentation.
9. Commit.
10. Push to `origin/main`.
11. Confirm the pushed commit exists.

A task is not complete merely because the code was written.

---

## 13. Git workflow

Repository:

`git@github.com:2223434778/SlackKV.git`

Default branch:

`main`

After every meaningful completed experiment or implementation milestone:

* update `docs/experiment_log.md`;
* update `docs/decisions.md` if a material implementation decision was made;
* commit relevant code and results;
* push to `origin/main`.

Use descriptive commit messages, for example:

`phase0: benchmark peer HBM transfers`

`phase0: add compute-transfer overlap benchmark`

`phase1: implement trace-driven SlackKV simulator`

Do not:

* force push;
* rewrite history;
* amend old commits unless explicitly instructed;
* delete previous experimental results merely because newer results exist.

---

## 14. experiment_log.md

For every meaningful experiment append an entry to:

`docs/experiment_log.md`

Include:

* date;
* experiment ID;
* objective;
* configuration;
* command;
* main metrics;
* result path;
* important observations;
* conclusion;
* commit hash when available.

Maintain chronological history.

---

## 15. decisions.md

Use:

`docs/decisions.md`

for important technical decisions that affect interpretation of later experiments.

Examples:

* choosing CUDA events instead of CPU timers;
* determining that NVLink is or is not present;
* changing peer-memory implementation;
* changing the KV representation;
* selecting a particular simulator assumption.

Record:

* decision;
* reason;
* alternatives considered;
* consequences.

Do not use this file for routine implementation details.

---

## 16. Large files and secrets

Never commit:

* model checkpoints;
* downloaded model weights;
* Hugging Face caches;
* large datasets;
* SSH keys;
* API keys;
* access tokens;
* credentials;
* environment secrets;
* enormous raw logs.

Before every commit, inspect staged files for accidental large or sensitive content.

CSV, JSON, Markdown, small diagnostic logs, and experiment plots should normally be committed.

---

## 17. Failure behavior

Do not hide errors by weakening checks.

If an experiment cannot run:

* capture the exact error;
* identify the likely cause;
* attempt reasonable implementation-level fixes;
* preserve useful diagnostics;
* document unresolved problems.

Do not fabricate benchmark data.

Do not substitute simulated data for real measurements unless the experiment explicitly calls for simulation.

---

## 18. Communication after each task

At the end of every requested task, report concisely:

* what was implemented;
* what was actually executed;
* whether execution succeeded;
* key numerical results;
* result file paths;
* important limitations or anomalies;
* commit hash;
* whether push to GitHub succeeded.

Do not report an experiment as successful if only the code was implemented but the experiment did not run successfully.

---

## 19. Research phase control

SlackKV will be developed incrementally.

A typical progression may include:

* hardware feasibility;
* trace-driven simulation;
* real GPU KV-like tensor runtime;
* real model integration;
* resume-time prediction;
* serving-framework integration;
* full-system evaluation.

However, this list is context, not permission to execute all phases.

Only implement and execute the phase explicitly requested in the current task.

---

## 20. Priority order

When requirements conflict, optimize in this order:

1. experimental correctness;
2. reproducibility;
3. truthful reporting;
4. clean implementation;
5. performance;
6. convenience.

Never sacrifice the first three merely to obtain a better benchmark result.
